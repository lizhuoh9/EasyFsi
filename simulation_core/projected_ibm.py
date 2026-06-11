from __future__ import annotations

from dataclasses import dataclass
import math

from .fluid import (
    CG_PRECONDITIONER_CHOICES,
    CartesianFluidSolver,
    FluidImpulseReport,
    VelocityConstraintReport,
)
from .fsi_coupling import (
    ForceBalanceReport,
    RegionPairInterfaceReactionTarget,
    action_reaction_balance,
    region_pair_interface_reaction_forces,
)
from .tri_surface import TriSurfaceDiagnosticReport, TriSurfaceForcePairReport, TriSurfaceRegionDiagnostics


def _vector3(value: tuple[float, float, float], *, name: str) -> tuple[float, float, float]:
    try:
        components = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly 3 components") from exc
    if len(components) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    result = (float(components[0]), float(components[1]), float(components[2]))
    if any(not math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain only finite components")
    return result


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    try:
        numeric = float(value)
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric != float(result) or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _non_negative_int(value: object, *, name: str) -> int:
    try:
        numeric = float(value)
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(numeric) or numeric != float(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _grid_nodes(value: tuple[int, int, int], *, name: str) -> tuple[int, int, int]:
    try:
        nodes = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly 3 positive integers") from exc
    if len(nodes) != 3:
        raise ValueError(f"{name} must contain exactly 3 positive integers")
    return tuple(_positive_int(node, name=name) for node in nodes)


def _add_vector3(
    lhs: tuple[float, float, float],
    rhs: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(lhs, rhs, strict=True))


@dataclass(frozen=True)
class ProjectedIbmRegionPairStepConfig:
    primary_region_id: int
    secondary_region_id: int
    primary_velocity_mps: tuple[float, float, float]
    secondary_velocity_mps: tuple[float, float, float]
    dt_s: float
    ibm_correction_iterations: int
    projection_iterations: int
    pressure_outlet_zmin: bool
    velocity_constraint_blend: float
    constraint_force_scale: float
    density_kgm3: float
    viscosity_pa_s: float
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    grid_nodes: tuple[int, int, int]
    fluid_substeps: int = 1
    fluid_advection_scheme: str = "euler"
    constraint_force_solid_mobility_ratio: float = 0.0
    primary_constraint_force_solid_mobility_ratio: float | None = None
    secondary_constraint_force_solid_mobility_ratio: float | None = None
    velocity_target_solid_mobility_ratio: float = 0.0
    primary_velocity_target_solid_mobility_ratio: float | None = None
    secondary_velocity_target_solid_mobility_ratio: float | None = None
    velocity_constraint_solid_mobility_ratio: float = 0.0
    divergence_cleanup_iterations: int = 0
    divergence_cleanup_relaxation: float = 0.7
    reset_pressure_each_projection: bool = True
    pressure_solver: str = "jacobi"
    multigrid_cycles: int | None = None
    cg_tolerance: float = 1.0e-6
    cg_preconditioner: str = "auto"
    read_full_report: bool = True
    primary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    secondary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    primary_pressure_robin_impedance_ns_m: float = 0.0
    secondary_pressure_robin_impedance_ns_m: float = 0.0
    primary_pressure_robin_reference_pa: float = 0.0
    secondary_pressure_robin_reference_pa: float = 0.0
    primary_interface_area_m2: float = 0.0
    secondary_interface_area_m2: float = 0.0

    def __post_init__(self) -> None:
        primary_region_id = int(self.primary_region_id)
        secondary_region_id = int(self.secondary_region_id)
        if primary_region_id == secondary_region_id:
            raise ValueError(
                "primary_region_id and secondary_region_id must be different; "
                "identical ids would double-count the same interface region"
            )
        object.__setattr__(self, "primary_region_id", primary_region_id)
        object.__setattr__(self, "secondary_region_id", secondary_region_id)
        dt_s = _finite_float(self.dt_s, name="dt_s")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        object.__setattr__(self, "dt_s", dt_s)
        object.__setattr__(
            self,
            "fluid_substeps",
            _positive_int(self.fluid_substeps, name="fluid_substeps"),
        )
        advection_scheme = str(self.fluid_advection_scheme)
        if advection_scheme not in {"euler", "rk2"}:
            raise ValueError("fluid_advection_scheme must be one of: euler, rk2")
        object.__setattr__(self, "fluid_advection_scheme", advection_scheme)
        object.__setattr__(
            self,
            "ibm_correction_iterations",
            _positive_int(self.ibm_correction_iterations, name="ibm_correction_iterations"),
        )
        object.__setattr__(
            self,
            "projection_iterations",
            _positive_int(self.projection_iterations, name="projection_iterations"),
        )
        velocity_constraint_blend = _finite_float(
            self.velocity_constraint_blend,
            name="velocity_constraint_blend",
        )
        if not 0.0 <= velocity_constraint_blend <= 1.0:
            raise ValueError("velocity_constraint_blend must be in [0, 1]")
        object.__setattr__(self, "velocity_constraint_blend", velocity_constraint_blend)
        velocity_constraint_solid_mobility_ratio = _finite_float(
            self.velocity_constraint_solid_mobility_ratio,
            name="velocity_constraint_solid_mobility_ratio",
        )
        if velocity_constraint_solid_mobility_ratio < 0.0:
            raise ValueError(
                "velocity_constraint_solid_mobility_ratio must be non-negative"
            )
        object.__setattr__(
            self,
            "velocity_constraint_solid_mobility_ratio",
            velocity_constraint_solid_mobility_ratio,
        )
        constraint_force_scale = _finite_float(
            self.constraint_force_scale,
            name="constraint_force_scale",
        )
        object.__setattr__(self, "constraint_force_scale", constraint_force_scale)
        constraint_force_solid_mobility_ratio = _finite_float(
            self.constraint_force_solid_mobility_ratio,
            name="constraint_force_solid_mobility_ratio",
        )
        if constraint_force_solid_mobility_ratio < 0.0:
            raise ValueError(
                "constraint_force_solid_mobility_ratio must be non-negative"
            )
        object.__setattr__(
            self,
            "constraint_force_solid_mobility_ratio",
            constraint_force_solid_mobility_ratio,
        )
        primary_constraint_force_solid_mobility_ratio = (
            constraint_force_solid_mobility_ratio
            if self.primary_constraint_force_solid_mobility_ratio is None
            else _finite_float(
                self.primary_constraint_force_solid_mobility_ratio,
                name="primary_constraint_force_solid_mobility_ratio",
            )
        )
        if primary_constraint_force_solid_mobility_ratio < 0.0:
            raise ValueError(
                "primary_constraint_force_solid_mobility_ratio must be non-negative"
            )
        secondary_constraint_force_solid_mobility_ratio = (
            constraint_force_solid_mobility_ratio
            if self.secondary_constraint_force_solid_mobility_ratio is None
            else _finite_float(
                self.secondary_constraint_force_solid_mobility_ratio,
                name="secondary_constraint_force_solid_mobility_ratio",
            )
        )
        if secondary_constraint_force_solid_mobility_ratio < 0.0:
            raise ValueError(
                "secondary_constraint_force_solid_mobility_ratio must be non-negative"
            )
        object.__setattr__(
            self,
            "primary_constraint_force_solid_mobility_ratio",
            primary_constraint_force_solid_mobility_ratio,
        )
        object.__setattr__(
            self,
            "secondary_constraint_force_solid_mobility_ratio",
            secondary_constraint_force_solid_mobility_ratio,
        )
        velocity_target_solid_mobility_ratio = _finite_float(
            self.velocity_target_solid_mobility_ratio,
            name="velocity_target_solid_mobility_ratio",
        )
        if velocity_target_solid_mobility_ratio < 0.0:
            raise ValueError("velocity_target_solid_mobility_ratio must be non-negative")
        object.__setattr__(
            self,
            "velocity_target_solid_mobility_ratio",
            velocity_target_solid_mobility_ratio,
        )
        primary_velocity_target_solid_mobility_ratio = (
            velocity_target_solid_mobility_ratio
            if self.primary_velocity_target_solid_mobility_ratio is None
            else _finite_float(
                self.primary_velocity_target_solid_mobility_ratio,
                name="primary_velocity_target_solid_mobility_ratio",
            )
        )
        if primary_velocity_target_solid_mobility_ratio < 0.0:
            raise ValueError(
                "primary_velocity_target_solid_mobility_ratio must be non-negative"
            )
        secondary_velocity_target_solid_mobility_ratio = (
            velocity_target_solid_mobility_ratio
            if self.secondary_velocity_target_solid_mobility_ratio is None
            else _finite_float(
                self.secondary_velocity_target_solid_mobility_ratio,
                name="secondary_velocity_target_solid_mobility_ratio",
            )
        )
        if secondary_velocity_target_solid_mobility_ratio < 0.0:
            raise ValueError(
                "secondary_velocity_target_solid_mobility_ratio must be non-negative"
            )
        object.__setattr__(
            self,
            "primary_velocity_target_solid_mobility_ratio",
            primary_velocity_target_solid_mobility_ratio,
        )
        object.__setattr__(
            self,
            "secondary_velocity_target_solid_mobility_ratio",
            secondary_velocity_target_solid_mobility_ratio,
        )
        density_kgm3 = _finite_float(self.density_kgm3, name="density_kgm3")
        if density_kgm3 <= 0.0:
            raise ValueError("density_kgm3 must be positive")
        object.__setattr__(self, "density_kgm3", density_kgm3)
        viscosity_pa_s = _finite_float(self.viscosity_pa_s, name="viscosity_pa_s")
        if viscosity_pa_s < 0.0:
            raise ValueError("viscosity_pa_s must be non-negative")
        object.__setattr__(self, "viscosity_pa_s", viscosity_pa_s)
        bounds_min = _vector3(self.bounds_min_m, name="bounds_min_m")
        bounds_max = _vector3(self.bounds_max_m, name="bounds_max_m")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)
        object.__setattr__(self, "grid_nodes", _grid_nodes(self.grid_nodes, name="grid_nodes"))
        object.__setattr__(
            self,
            "divergence_cleanup_iterations",
            _non_negative_int(
                self.divergence_cleanup_iterations,
                name="divergence_cleanup_iterations",
            ),
        )
        divergence_cleanup_relaxation = _finite_float(
            self.divergence_cleanup_relaxation,
            name="divergence_cleanup_relaxation",
        )
        if not 0.0 <= divergence_cleanup_relaxation <= 1.0:
            raise ValueError("divergence_cleanup_relaxation must be in [0, 1]")
        object.__setattr__(
            self,
            "divergence_cleanup_relaxation",
            divergence_cleanup_relaxation,
        )
        object.__setattr__(
            self,
            "primary_velocity_mps",
            _vector3(self.primary_velocity_mps, name="primary_velocity_mps"),
        )
        object.__setattr__(
            self,
            "secondary_velocity_mps",
            _vector3(self.secondary_velocity_mps, name="secondary_velocity_mps"),
        )
        primary_impedance_force = _vector3(
            self.primary_interface_impedance_force_n,
            name="primary_interface_impedance_force_n",
        )
        secondary_impedance_force = _vector3(
            self.secondary_interface_impedance_force_n,
            name="secondary_interface_impedance_force_n",
        )
        object.__setattr__(
            self,
            "primary_interface_impedance_force_n",
            primary_impedance_force,
        )
        object.__setattr__(
            self,
            "secondary_interface_impedance_force_n",
            secondary_impedance_force,
        )
        primary_area_m2 = float(self.primary_interface_area_m2)
        secondary_area_m2 = float(self.secondary_interface_area_m2)
        if not math.isfinite(primary_area_m2) or primary_area_m2 < 0.0:
            raise ValueError("primary_interface_area_m2 must be a finite non-negative number")
        if not math.isfinite(secondary_area_m2) or secondary_area_m2 < 0.0:
            raise ValueError("secondary_interface_area_m2 must be a finite non-negative number")
        if any(component != 0.0 for component in primary_impedance_force) and primary_area_m2 <= 0.0:
            raise ValueError("primary_interface_area_m2 must be positive when primary impedance force is nonzero")
        if any(component != 0.0 for component in secondary_impedance_force) and secondary_area_m2 <= 0.0:
            raise ValueError("secondary_interface_area_m2 must be positive when secondary impedance force is nonzero")
        primary_pressure_robin_impedance = _finite_float(
            self.primary_pressure_robin_impedance_ns_m,
            name="primary_pressure_robin_impedance_ns_m",
        )
        secondary_pressure_robin_impedance = _finite_float(
            self.secondary_pressure_robin_impedance_ns_m,
            name="secondary_pressure_robin_impedance_ns_m",
        )
        if primary_pressure_robin_impedance < 0.0:
            raise ValueError("primary_pressure_robin_impedance_ns_m must be non-negative")
        if secondary_pressure_robin_impedance < 0.0:
            raise ValueError("secondary_pressure_robin_impedance_ns_m must be non-negative")
        primary_pressure_robin_reference = _finite_float(
            self.primary_pressure_robin_reference_pa,
            name="primary_pressure_robin_reference_pa",
        )
        secondary_pressure_robin_reference = _finite_float(
            self.secondary_pressure_robin_reference_pa,
            name="secondary_pressure_robin_reference_pa",
        )
        if primary_pressure_robin_impedance > 0.0 and primary_area_m2 <= 0.0:
            raise ValueError(
                "primary_interface_area_m2 must be positive when primary pressure Robin impedance is nonzero"
            )
        if secondary_pressure_robin_impedance > 0.0 and secondary_area_m2 <= 0.0:
            raise ValueError(
                "secondary_interface_area_m2 must be positive when secondary pressure Robin impedance is nonzero"
            )
        object.__setattr__(
            self,
            "primary_pressure_robin_impedance_ns_m",
            primary_pressure_robin_impedance,
        )
        object.__setattr__(
            self,
            "secondary_pressure_robin_impedance_ns_m",
            secondary_pressure_robin_impedance,
        )
        object.__setattr__(
            self,
            "primary_pressure_robin_reference_pa",
            primary_pressure_robin_reference,
        )
        object.__setattr__(
            self,
            "secondary_pressure_robin_reference_pa",
            secondary_pressure_robin_reference,
        )
        object.__setattr__(self, "primary_interface_area_m2", primary_area_m2)
        object.__setattr__(self, "secondary_interface_area_m2", secondary_area_m2)
        pressure_solver = str(self.pressure_solver)
        if pressure_solver not in {"jacobi", "compact_jacobi", "fv_jacobi", "fv_multigrid", "fv_cg"}:
            raise ValueError(f"unsupported pressure_solver: {self.pressure_solver!r}")
        object.__setattr__(self, "pressure_solver", pressure_solver)
        if self.multigrid_cycles is not None and int(self.multigrid_cycles) <= 0:
            raise ValueError("multigrid_cycles must be positive")
        if self.multigrid_cycles is not None:
            object.__setattr__(self, "multigrid_cycles", int(self.multigrid_cycles))
        cg_tolerance = float(self.cg_tolerance)
        if not math.isfinite(cg_tolerance) or cg_tolerance < 0.0:
            raise ValueError("cg_tolerance must be a finite non-negative number")
        object.__setattr__(self, "cg_tolerance", cg_tolerance)
        cg_preconditioner = str(self.cg_preconditioner)
        if cg_preconditioner not in CG_PRECONDITIONER_CHOICES:
            raise ValueError(f"unsupported cg_preconditioner: {self.cg_preconditioner!r}")
        object.__setattr__(self, "cg_preconditioner", cg_preconditioner)


@dataclass(frozen=True)
class ProjectedIbmRegionPairStepReport:
    divergence: dict[str, float]
    pressure_outlet_report: dict[str, float]
    force_report: TriSurfaceDiagnosticReport | TriSurfaceForcePairReport
    impulse_report: FluidImpulseReport | None
    velocity_constraint_report: VelocityConstraintReport | None
    velocity_constraint_spread_report: TriSurfaceDiagnosticReport | None
    primary_equivalent_fluid_force_n: tuple[float, float, float]
    secondary_equivalent_fluid_force_n: tuple[float, float, float]
    primary_velocity_constraint_impulse_n_s: tuple[float, float, float]
    secondary_velocity_constraint_impulse_n_s: tuple[float, float, float]
    primary_velocity_constraint_equivalent_fluid_force_n: tuple[float, float, float]
    secondary_velocity_constraint_equivalent_fluid_force_n: tuple[float, float, float]
    interface_reaction_target: RegionPairInterfaceReactionTarget
    primary_interface_reaction_balance: ForceBalanceReport
    secondary_interface_reaction_balance: ForceBalanceReport
    ibm_correction_iterations: int
    ibm_correction_dt_s: float
    fluid_substeps: int
    fluid_substep_dt_s: float
    fluid_advection_scheme: str = "euler"
    pressure_projection_cg_project_calls: int = 0
    pressure_projection_cg_iterations_total: int = 0
    pressure_projection_cg_iterations_max: int = 0
    pressure_projection_cg_host_residual_checks: int = 0
    pressure_projection_cg_mean_projection_count: int = 0
    pressure_projection_cg_restart_count: int = 0
    pressure_projection_cg_restart_count_measured: bool = False
    pressure_projection_cg_restart_policy: str = "not_applicable_non_cg"
    pressure_projection_cg_converged_all: bool = True
    pressure_projection_cg_max_relative_residual: float = 0.0
    pressure_projection_cg_max_initial_relative_residual: float = 0.0
    pressure_projection_cg_breakdown_count: int = 0
    pressure_interface_matrix_diagonal_integral: float = 0.0
    pressure_interface_matrix_rhs_integral: float = 0.0
    pressure_interface_matrix_max_abs_diagonal: float = 0.0
    pressure_interface_matrix_active_cells: int = 0


def advance_projected_ibm_region_pair_fluid_step(
    fluid: CartesianFluidSolver,
    surface_diagnostics: TriSurfaceRegionDiagnostics,
    config: ProjectedIbmRegionPairStepConfig,
) -> ProjectedIbmRegionPairStepReport:
    """Advance one fluid step with projected-IBM forcing for two FSI regions.

    The function is case-free: callers provide region ids, target region
    velocities, and constraint scaling. `fluid_substeps` splits the physical
    step into smaller fluid advances; each substep runs predictor, IBM forcing,
    and pressure projection with the substep time scale.
    """
    fluid_substeps = max(1, int(config.fluid_substeps))
    fluid_advection_scheme = str(config.fluid_advection_scheme)
    correction_iterations = max(1, int(config.ibm_correction_iterations))
    step_reads_full_report = bool(config.read_full_report)
    fluid_substep_dt_s = float(config.dt_s) / float(fluid_substeps)
    correction_dt_s = fluid_substep_dt_s / float(correction_iterations)
    spacing_m = (float(fluid.dx), float(fluid.dy), float(fluid.dz))
    probe_distance_m = min(spacing_m)
    primary_fluid_impulse_n_s = [0.0, 0.0, 0.0]
    secondary_fluid_impulse_n_s = [0.0, 0.0, 0.0]
    pressure_projection_cg_project_calls = 0
    pressure_projection_cg_iterations_total = 0
    pressure_projection_cg_iterations_max = 0
    pressure_projection_cg_host_residual_checks = 0
    pressure_projection_cg_mean_projection_count = 0
    pressure_projection_cg_restart_count = 0
    pressure_projection_cg_restart_count_measured = False
    pressure_projection_cg_restart_policy = "not_applicable_non_cg"
    pressure_projection_cg_converged_all = True
    pressure_projection_cg_max_relative_residual = 0.0
    pressure_projection_cg_max_initial_relative_residual = 0.0
    pressure_projection_cg_breakdown_count = 0
    pressure_interface_matrix_diagonal_integral = 0.0
    pressure_interface_matrix_rhs_integral = 0.0
    pressure_interface_matrix_max_abs_diagonal = 0.0
    pressure_interface_matrix_active_cells = 0
    device_velocity_constraint_impulse = (
        config.velocity_constraint_blend > 0.0
        and all(
            hasattr(fluid, name)
            for name in (
                "reset_velocity_constraint_impulse_accumulator",
                "velocity_constraint_impulse_report",
            )
        )
    )
    device_force_impulse = all(
        hasattr(surface_diagnostics, name)
        for name in (
            "reset_force_impulse_accumulator",
            "accumulate_force_impulse",
            "force_impulse_report",
        )
    )
    if device_force_impulse:
        surface_diagnostics.reset_force_impulse_accumulator()
    if device_velocity_constraint_impulse:
        fluid.reset_velocity_constraint_impulse_accumulator()
    fluid.snapshot_pressure()
    if config.reset_pressure_each_projection:
        fluid.clear_pressure()
    fsi_pressure_field = fluid.fsi_pressure

    def accumulate_pressure_projection_cg_stats() -> None:
        nonlocal pressure_projection_cg_project_calls
        nonlocal pressure_projection_cg_iterations_total
        nonlocal pressure_projection_cg_iterations_max
        nonlocal pressure_projection_cg_host_residual_checks
        nonlocal pressure_projection_cg_mean_projection_count
        nonlocal pressure_projection_cg_restart_count
        nonlocal pressure_projection_cg_restart_count_measured
        nonlocal pressure_projection_cg_restart_policy
        nonlocal pressure_projection_cg_converged_all
        nonlocal pressure_projection_cg_max_relative_residual
        nonlocal pressure_projection_cg_max_initial_relative_residual
        nonlocal pressure_projection_cg_breakdown_count

        project_calls = int(getattr(fluid, "last_project_cg_project_calls", 0))
        if project_calls <= 0:
            return
        pressure_projection_cg_project_calls += project_calls
        pressure_projection_cg_iterations_total += int(
            getattr(fluid, "last_project_cg_iterations_total", 0)
        )
        pressure_projection_cg_iterations_max = max(
            pressure_projection_cg_iterations_max,
            int(getattr(fluid, "last_project_cg_iterations_max", 0)),
        )
        pressure_projection_cg_host_residual_checks += int(
            getattr(fluid, "last_project_cg_host_residual_checks", 0)
        )
        pressure_projection_cg_mean_projection_count += int(
            getattr(fluid, "last_project_cg_mean_projection_count", 0)
        )
        pressure_projection_cg_restart_count += int(
            getattr(fluid, "last_project_cg_restart_count", 0)
        )
        pressure_projection_cg_restart_count_measured = (
            pressure_projection_cg_restart_count_measured
            or bool(getattr(fluid, "last_project_cg_restart_count_measured", False))
        )
        pressure_projection_cg_restart_policy = str(
            getattr(
                fluid,
                "last_project_cg_restart_policy",
                pressure_projection_cg_restart_policy,
            )
        )
        pressure_projection_cg_converged_all = (
            pressure_projection_cg_converged_all
            and bool(getattr(fluid, "last_project_cg_converged_all", True))
        )
        pressure_projection_cg_max_relative_residual = max(
            pressure_projection_cg_max_relative_residual,
            float(getattr(fluid, "last_project_cg_relative_residual_max", 0.0)),
        )
        pressure_projection_cg_max_initial_relative_residual = max(
            pressure_projection_cg_max_initial_relative_residual,
            float(getattr(fluid, "last_project_cg_initial_relative_residual_max", 0.0)),
        )
        pressure_projection_cg_breakdown_count += int(
            getattr(fluid, "last_project_cg_breakdown_count", 0)
        )

    def apply_correction_pass(*, read_full_reports: bool, read_force_pair_report: bool):
        nonlocal pressure_interface_matrix_diagonal_integral
        nonlocal pressure_interface_matrix_rhs_integral
        nonlocal pressure_interface_matrix_max_abs_diagonal
        nonlocal pressure_interface_matrix_active_cells
        fluid.clear_force()
        fluid.clear_volume_source()
        if hasattr(fluid, "clear_pressure_interface_matrix_terms"):
            fluid.clear_pressure_interface_matrix_terms()
        pressure_robin_matrix_enabled = (
            config.primary_pressure_robin_impedance_ns_m > 0.0
            or config.secondary_pressure_robin_impedance_ns_m > 0.0
        )
        if pressure_robin_matrix_enabled:
            surface_diagnostics.spread_pressure_interface_matrix_terms(
                fluid.pressure_interface_matrix_diagonal,
                fluid.pressure_interface_matrix_rhs,
                fluid.obstacle,
                grid_fields=fluid,
                primary_region_id=int(config.primary_region_id),
                secondary_region_id=int(config.secondary_region_id),
                primary_pressure_robin_impedance_ns_m=float(
                    config.primary_pressure_robin_impedance_ns_m
                ),
                secondary_pressure_robin_impedance_ns_m=float(
                    config.secondary_pressure_robin_impedance_ns_m
                ),
                primary_pressure_robin_reference_pa=float(
                    config.primary_pressure_robin_reference_pa
                ),
                secondary_pressure_robin_reference_pa=float(
                    config.secondary_pressure_robin_reference_pa
                ),
                primary_interface_area_m2=float(config.primary_interface_area_m2),
                secondary_interface_area_m2=float(config.secondary_interface_area_m2),
                density_kgm3=float(config.density_kgm3),
                dt_s=correction_dt_s,
                probe_distance_m=probe_distance_m,
                bounds_min_m=config.bounds_min_m,
                bounds_max_m=config.bounds_max_m,
                spacing_m=spacing_m,
                grid_nodes=config.grid_nodes,
            )
            if hasattr(fluid, "pressure_interface_matrix_terms_report"):
                matrix_report = fluid.pressure_interface_matrix_terms_report()
                pressure_interface_matrix_diagonal_integral = float(
                    matrix_report.get("diagonal_integral", 0.0)
                )
                pressure_interface_matrix_rhs_integral = float(
                    matrix_report.get("rhs_integral", 0.0)
                )
                pressure_interface_matrix_max_abs_diagonal = max(
                    pressure_interface_matrix_max_abs_diagonal,
                    float(matrix_report.get("max_abs_diagonal", 0.0)),
                )
                pressure_interface_matrix_active_cells = max(
                    pressure_interface_matrix_active_cells,
                    int(matrix_report.get("active_cells", 0)),
                )
        pass_force_report = surface_diagnostics.spread_fsi_forces(
            fluid.velocity,
            fsi_pressure_field,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=int(config.primary_region_id),
            secondary_region_id=int(config.secondary_region_id),
            primary_velocity_mps=config.primary_velocity_mps,
            secondary_velocity_mps=config.secondary_velocity_mps,
            probe_distance_m=probe_distance_m,
            density_kgm3=float(config.density_kgm3),
            viscosity_pa_s=float(config.viscosity_pa_s),
            dt_s=correction_dt_s,
            constraint_force_scale=float(config.constraint_force_scale),
            constraint_force_solid_mobility_ratio=float(
                config.constraint_force_solid_mobility_ratio
            ),
            primary_constraint_force_solid_mobility_ratio=float(
                config.primary_constraint_force_solid_mobility_ratio
            ),
            secondary_constraint_force_solid_mobility_ratio=float(
                config.secondary_constraint_force_solid_mobility_ratio
            ),
            velocity_target_solid_mobility_ratio=float(
                config.velocity_target_solid_mobility_ratio
            ),
            primary_velocity_target_solid_mobility_ratio=float(
                config.primary_velocity_target_solid_mobility_ratio
            ),
            secondary_velocity_target_solid_mobility_ratio=float(
                config.secondary_velocity_target_solid_mobility_ratio
            ),
            primary_interface_impedance_force_n=config.primary_interface_impedance_force_n,
            secondary_interface_impedance_force_n=config.secondary_interface_impedance_force_n,
            primary_interface_area_m2=float(config.primary_interface_area_m2),
            secondary_interface_area_m2=float(config.secondary_interface_area_m2),
            bounds_min_m=config.bounds_min_m,
            bounds_max_m=config.bounds_max_m,
            spacing_m=spacing_m,
            grid_nodes=config.grid_nodes,
            read_full_report=read_full_reports,
            read_force_pair_report=read_force_pair_report,
        )
        if device_force_impulse:
            surface_diagnostics.accumulate_force_impulse(correction_dt_s)
        pass_interface_reaction_target = None
        pass_primary_interface_reaction_balance = None
        pass_secondary_interface_reaction_balance = None
        if pass_force_report is not None:
            pass_interface_reaction_target = region_pair_interface_reaction_forces(
                primary_fluid_force_n=pass_force_report.primary_fluid_force_n,
                secondary_fluid_force_n=pass_force_report.secondary_fluid_force_n,
            )
            pass_primary_interface_reaction_balance = action_reaction_balance(
                pass_force_report.primary_fluid_force_n,
                pass_interface_reaction_target.primary_force_n,
            )
            pass_secondary_interface_reaction_balance = action_reaction_balance(
                pass_force_report.secondary_fluid_force_n,
                pass_interface_reaction_target.secondary_force_n,
            )
        pass_impulse_report = fluid.apply_body_force(
            dt_s=correction_dt_s,
            read_report=read_full_reports,
        )
        pass_velocity_constraint_report = None
        pass_velocity_constraint_spread_report = None
        if config.velocity_constraint_blend > 0.0:
            fluid.clear_velocity_constraints()
            pass_velocity_constraint_spread_report = surface_diagnostics.spread_fsi_velocity_constraints(
                fluid.velocity_constraint_sum,
                fluid.velocity_constraint_weight,
                grid_fields=fluid,
                primary_region_id=int(config.primary_region_id),
                secondary_region_id=int(config.secondary_region_id),
                primary_velocity_mps=config.primary_velocity_mps,
                secondary_velocity_mps=config.secondary_velocity_mps,
                probe_distance_m=probe_distance_m,
                bounds_min_m=config.bounds_min_m,
                bounds_max_m=config.bounds_max_m,
                spacing_m=spacing_m,
                grid_nodes=config.grid_nodes,
                read_full_report=read_full_reports,
            )
            pass_velocity_constraint_report = fluid.apply_velocity_constraints(
                blend=float(config.velocity_constraint_blend),
                solid_mobility_ratio=float(
                    config.velocity_constraint_solid_mobility_ratio
                ),
                read_report=False,
            )
        pass_divergence = fluid.project(
            iterations=int(config.projection_iterations),
            pressure_outlet_zmin=bool(config.pressure_outlet_zmin),
            dt_s=correction_dt_s,
            preserve_velocity_constraints=config.velocity_constraint_blend > 0.0,
            velocity_constraint_blend=float(config.velocity_constraint_blend),
            velocity_constraint_solid_mobility_ratio=float(
                config.velocity_constraint_solid_mobility_ratio
            ),
            divergence_cleanup_iterations=int(config.divergence_cleanup_iterations),
            divergence_cleanup_relaxation=float(config.divergence_cleanup_relaxation),
            reset_pressure=bool(config.reset_pressure_each_projection),
            pressure_solver=str(config.pressure_solver),
            multigrid_cycles=config.multigrid_cycles,
            cg_tolerance=float(config.cg_tolerance),
            cg_preconditioner=str(config.cg_preconditioner),
            read_report=read_full_reports,
        )
        accumulate_pressure_projection_cg_stats()
        if pressure_robin_matrix_enabled:
            pass_force_report = surface_diagnostics.diagnose_fsi_forces_from_fields(
                fluid.velocity,
                fluid.pressure,
                fluid.force,
                fluid.volume_source_s,
                fluid.obstacle,
                grid_fields=fluid,
                primary_region_id=int(config.primary_region_id),
                secondary_region_id=int(config.secondary_region_id),
                primary_velocity_mps=config.primary_velocity_mps,
                secondary_velocity_mps=config.secondary_velocity_mps,
                probe_distance_m=probe_distance_m,
                density_kgm3=float(config.density_kgm3),
                viscosity_pa_s=float(config.viscosity_pa_s),
                dt_s=correction_dt_s,
                constraint_force_scale=float(config.constraint_force_scale),
                constraint_force_solid_mobility_ratio=float(
                    config.constraint_force_solid_mobility_ratio
                ),
                primary_constraint_force_solid_mobility_ratio=float(
                    config.primary_constraint_force_solid_mobility_ratio
                ),
                secondary_constraint_force_solid_mobility_ratio=float(
                    config.secondary_constraint_force_solid_mobility_ratio
                ),
                velocity_target_solid_mobility_ratio=float(
                    config.velocity_target_solid_mobility_ratio
                ),
                primary_velocity_target_solid_mobility_ratio=float(
                    config.primary_velocity_target_solid_mobility_ratio
                ),
                secondary_velocity_target_solid_mobility_ratio=float(
                    config.secondary_velocity_target_solid_mobility_ratio
                ),
                primary_interface_impedance_force_n=config.primary_interface_impedance_force_n,
                secondary_interface_impedance_force_n=config.secondary_interface_impedance_force_n,
                primary_interface_area_m2=float(config.primary_interface_area_m2),
                secondary_interface_area_m2=float(config.secondary_interface_area_m2),
                bounds_min_m=config.bounds_min_m,
                bounds_max_m=config.bounds_max_m,
                spacing_m=spacing_m,
                grid_nodes=config.grid_nodes,
            )
            pass_interface_reaction_target = region_pair_interface_reaction_forces(
                primary_fluid_force_n=pass_force_report.primary_fluid_force_n,
                secondary_fluid_force_n=pass_force_report.secondary_fluid_force_n,
            )
            pass_primary_interface_reaction_balance = action_reaction_balance(
                pass_force_report.primary_fluid_force_n,
                pass_interface_reaction_target.primary_force_n,
            )
            pass_secondary_interface_reaction_balance = action_reaction_balance(
                pass_force_report.secondary_fluid_force_n,
                pass_interface_reaction_target.secondary_force_n,
            )
        pass_pressure_outlet_report = None
        if read_full_reports:
            pass_pressure_outlet_report = fluid.pressure_outlet_fv_flux_report(
                dt_s=correction_dt_s,
            )
        if read_full_reports and config.velocity_constraint_blend > 0.0:
            pass_velocity_constraint_report = fluid.velocity_constraint_report()
        return (
            pass_divergence,
            pass_pressure_outlet_report,
            pass_force_report,
            pass_impulse_report,
            pass_velocity_constraint_report,
            pass_velocity_constraint_spread_report,
            pass_interface_reaction_target,
            pass_primary_interface_reaction_balance,
            pass_secondary_interface_reaction_balance,
        )

    for _fluid_substep in range(fluid_substeps):
        fluid.predict(
            dt_s=fluid_substep_dt_s,
            advection_scheme=fluid_advection_scheme,
        )
        for _correction_iteration in range(correction_iterations):
            final_correction_pass = (
                _fluid_substep == fluid_substeps - 1
                and _correction_iteration == correction_iterations - 1
            )
            read_full_reports = (
                step_reads_full_report
                and final_correction_pass
            )
            read_force_pair_report = (
                not device_force_impulse
                or final_correction_pass
                or read_full_reports
            )
            (
                divergence,
                pressure_outlet_report,
                force_report,
                impulse_report,
                velocity_constraint_report,
                velocity_constraint_spread_report,
                interface_reaction_target,
                primary_interface_reaction_balance,
                secondary_interface_reaction_balance,
            ) = apply_correction_pass(
                read_full_reports=read_full_reports,
                read_force_pair_report=read_force_pair_report,
            )
            if not device_force_impulse:
                if force_report is None:
                    raise RuntimeError("projected IBM force pass did not return a force report")
                for component in range(3):
                    primary_fluid_impulse_n_s[component] += (
                        float(force_report.primary_fluid_force_n[component]) * correction_dt_s
                    )
                    secondary_fluid_impulse_n_s[component] += (
                        float(force_report.secondary_fluid_force_n[component]) * correction_dt_s
                    )
        if _fluid_substep < fluid_substeps - 1 and not config.reset_pressure_each_projection:
            fluid.snapshot_pressure()
            fsi_pressure_field = fluid.fsi_pressure

    if device_force_impulse:
        primary_fluid_impulse_n_s, secondary_fluid_impulse_n_s = (
            surface_diagnostics.force_impulse_report()
        )
    if force_report is None:
        raise RuntimeError("projected IBM step did not produce a final force report")
    primary_equivalent_fluid_force_n = tuple(
        impulse / float(config.dt_s) for impulse in primary_fluid_impulse_n_s
    )
    secondary_equivalent_fluid_force_n = tuple(
        impulse / float(config.dt_s) for impulse in secondary_fluid_impulse_n_s
    )
    primary_velocity_constraint_impulse_n_s = (0.0, 0.0, 0.0)
    secondary_velocity_constraint_impulse_n_s = (0.0, 0.0, 0.0)
    if device_velocity_constraint_impulse:
        (
            primary_velocity_constraint_impulse_n_s,
            secondary_velocity_constraint_impulse_n_s,
        ) = fluid.velocity_constraint_impulse_report()
        primary_velocity_constraint_impulse_n_s = _vector3(
            primary_velocity_constraint_impulse_n_s,
            name="primary_velocity_constraint_impulse_n_s",
        )
        secondary_velocity_constraint_impulse_n_s = _vector3(
            secondary_velocity_constraint_impulse_n_s,
            name="secondary_velocity_constraint_impulse_n_s",
        )
    primary_velocity_constraint_equivalent_fluid_force_n = tuple(
        impulse / float(config.dt_s)
        for impulse in primary_velocity_constraint_impulse_n_s
    )
    secondary_velocity_constraint_equivalent_fluid_force_n = tuple(
        impulse / float(config.dt_s)
        for impulse in secondary_velocity_constraint_impulse_n_s
    )
    primary_total_equivalent_fluid_force_n = _add_vector3(
        primary_equivalent_fluid_force_n,
        primary_velocity_constraint_equivalent_fluid_force_n,
    )
    secondary_total_equivalent_fluid_force_n = _add_vector3(
        secondary_equivalent_fluid_force_n,
        secondary_velocity_constraint_equivalent_fluid_force_n,
    )
    interface_reaction_target = region_pair_interface_reaction_forces(
        primary_fluid_force_n=primary_total_equivalent_fluid_force_n,
        secondary_fluid_force_n=secondary_total_equivalent_fluid_force_n,
    )
    primary_interface_reaction_balance = action_reaction_balance(
        primary_total_equivalent_fluid_force_n,
        interface_reaction_target.primary_force_n,
    )
    secondary_interface_reaction_balance = action_reaction_balance(
        secondary_total_equivalent_fluid_force_n,
        interface_reaction_target.secondary_force_n,
    )

    if step_reads_full_report:
        if pressure_outlet_report is None:
            raise RuntimeError("projected IBM step did not produce a final pressure outlet report")
        if impulse_report is None:
            raise RuntimeError("projected IBM step did not produce a final impulse report")
    elif pressure_outlet_report is None:
        pressure_outlet_report = {}

    return ProjectedIbmRegionPairStepReport(
        divergence=divergence,
        pressure_outlet_report=pressure_outlet_report,
        force_report=force_report,
        impulse_report=impulse_report,
        velocity_constraint_report=velocity_constraint_report,
        velocity_constraint_spread_report=velocity_constraint_spread_report,
        primary_equivalent_fluid_force_n=primary_equivalent_fluid_force_n,
        secondary_equivalent_fluid_force_n=secondary_equivalent_fluid_force_n,
        primary_velocity_constraint_impulse_n_s=primary_velocity_constraint_impulse_n_s,
        secondary_velocity_constraint_impulse_n_s=secondary_velocity_constraint_impulse_n_s,
        primary_velocity_constraint_equivalent_fluid_force_n=(
            primary_velocity_constraint_equivalent_fluid_force_n
        ),
        secondary_velocity_constraint_equivalent_fluid_force_n=(
            secondary_velocity_constraint_equivalent_fluid_force_n
        ),
        interface_reaction_target=interface_reaction_target,
        primary_interface_reaction_balance=primary_interface_reaction_balance,
        secondary_interface_reaction_balance=secondary_interface_reaction_balance,
        ibm_correction_iterations=correction_iterations,
        ibm_correction_dt_s=correction_dt_s,
        fluid_substeps=fluid_substeps,
        fluid_substep_dt_s=fluid_substep_dt_s,
        fluid_advection_scheme=fluid_advection_scheme,
        pressure_projection_cg_project_calls=pressure_projection_cg_project_calls,
        pressure_projection_cg_iterations_total=pressure_projection_cg_iterations_total,
        pressure_projection_cg_iterations_max=pressure_projection_cg_iterations_max,
        pressure_projection_cg_host_residual_checks=pressure_projection_cg_host_residual_checks,
        pressure_projection_cg_mean_projection_count=(
            pressure_projection_cg_mean_projection_count
        ),
        pressure_projection_cg_restart_count=pressure_projection_cg_restart_count,
        pressure_projection_cg_restart_count_measured=(
            pressure_projection_cg_restart_count_measured
        ),
        pressure_projection_cg_restart_policy=pressure_projection_cg_restart_policy,
        pressure_projection_cg_converged_all=pressure_projection_cg_converged_all,
        pressure_projection_cg_max_relative_residual=(
            pressure_projection_cg_max_relative_residual
        ),
        pressure_projection_cg_max_initial_relative_residual=(
            pressure_projection_cg_max_initial_relative_residual
        ),
        pressure_projection_cg_breakdown_count=pressure_projection_cg_breakdown_count,
        pressure_interface_matrix_diagonal_integral=(
            pressure_interface_matrix_diagonal_integral
        ),
        pressure_interface_matrix_rhs_integral=pressure_interface_matrix_rhs_integral,
        pressure_interface_matrix_max_abs_diagonal=(
            pressure_interface_matrix_max_abs_diagonal
        ),
        pressure_interface_matrix_active_cells=pressure_interface_matrix_active_cells,
    )
