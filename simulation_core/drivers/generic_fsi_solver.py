from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from simulation_core.coupling.iqn_ils import (
    IqnIlsAccelerator,
    IqnIlsConfig,
    IqnIlsUpdate,
)


RuntimeFactory = Callable[
    ["FsiProblem", "FsiSolverConfig", "DiagnosticsConfig"],
    "FsiRuntime",
]


@dataclass(frozen=True)
class FluidDomain:
    domain_id: str
    coordinate_model: str
    grid_nodes: tuple[int, int, int]
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    boundary_conditions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.domain_id, name="domain_id")
        _require_non_empty(self.coordinate_model, name="coordinate_model")
        if len(self.grid_nodes) != 3:
            raise ValueError("grid_nodes must contain three positive integers")
        for value in self.grid_nodes:
            _strict_positive_integer(value, name="grid_nodes")
        if len(self.bounds_m) != 2:
            raise ValueError("bounds_m must contain min and max points")
        bounds_min = _vector3(self.bounds_m[0], name="bounds_m")
        bounds_max = _vector3(self.bounds_m[1], name="bounds_m")
        if any(
            min_value >= max_value
            for min_value, max_value in zip(bounds_min, bounds_max, strict=True)
        ):
            raise ValueError(
                "bounds_m min point must be strictly less than max point on every axis"
            )


@dataclass(frozen=True)
class SolidBody:
    body_id: str
    material: Mapping[str, Any]
    initial_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.body_id, name="body_id")


@dataclass(frozen=True)
class SurfaceRegion:
    region_id: str
    marker_count: int = 0
    fluid_side_normal_sign: float | None = None
    reference_pressure_pa: float = 0.0

    def __post_init__(self) -> None:
        _require_non_empty(self.region_id, name="region_id")
        _strict_non_negative_integer(self.marker_count, name="marker_count")
        if self.fluid_side_normal_sign is not None:
            sign = float(self.fluid_side_normal_sign)
            if sign not in (-1.0, 1.0):
                raise ValueError("fluid_side_normal_sign must be -1.0 or 1.0")
        _finite_float(self.reference_pressure_pa, name="reference_pressure_pa")


@dataclass(frozen=True)
class InterfaceSurface:
    surface_id: str
    regions: tuple[SurfaceRegion, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.surface_id, name="surface_id")
        if not self.regions:
            raise ValueError("regions must contain at least one SurfaceRegion")


@dataclass(frozen=True)
class SurfaceRegionPolicy:
    region_id: str
    fluid_side_normal_sign: float
    reference_pressure_pa: float = 0.0

    def __post_init__(self) -> None:
        _require_non_empty(self.region_id, name="region_id")
        sign = float(self.fluid_side_normal_sign)
        if sign not in (-1.0, 1.0):
            raise ValueError("fluid_side_normal_sign must be -1.0 or 1.0")
        _finite_float(self.reference_pressure_pa, name="reference_pressure_pa")


@dataclass(frozen=True)
class OneSidedPressurePolicy:
    region_policies: tuple[SurfaceRegionPolicy, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.region_policies)

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "region_policies": [
                {
                    "region_id": policy.region_id,
                    "fluid_side_normal_sign": float(policy.fluid_side_normal_sign),
                    "reference_pressure_pa": float(policy.reference_pressure_pa),
                }
                for policy in self.region_policies
            ],
        }


@dataclass(frozen=True)
class PressureSamplePairProvider:
    mode: str
    pair_source_status: str = "runtime_generated"
    source: str = ""

    def __post_init__(self) -> None:
        supported_modes = {
            "runtime_anchored_cell_pair",
            "normal_ladder",
            "replay_from_diagnostics",
        }
        if self.mode not in supported_modes:
            raise ValueError(f"unsupported pressure sample pair mode: {self.mode}")
        _require_non_empty(self.pair_source_status, name="pair_source_status")

    @property
    def transition_backed(self) -> bool:
        return self.mode == "replay_from_diagnostics" or self.pair_source_status not in {
            "runtime_generated",
            "not_required",
        }

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pair_source_status": self.pair_source_status,
            "source": self.source,
            "transition_backed": self.transition_backed,
        }


@dataclass(frozen=True)
class PressureSamplingConfig:
    pair_provider: PressureSamplePairProvider
    sample_pair_fallback_count_max: int = 0

    def __post_init__(self) -> None:
        _strict_non_negative_integer(
            self.sample_pair_fallback_count_max,
            name="sample_pair_fallback_count_max",
        )

    def as_diagnostics(self) -> dict[str, Any]:
        payload = self.pair_provider.as_diagnostics()
        payload["sample_pair_fallback_count_max"] = int(
            self.sample_pair_fallback_count_max
        )
        return payload


@dataclass(frozen=True)
class TractionConfig:
    pressure_sampling: PressureSamplingConfig
    one_sided_pressure: OneSidedPressurePolicy = field(
        default_factory=OneSidedPressurePolicy
    )
    include_viscous: bool = False

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "pressure_sampling": self.pressure_sampling.as_diagnostics(),
            "one_sided_pressure": self.one_sided_pressure.as_diagnostics(),
            "include_viscous": bool(self.include_viscous),
        }


@dataclass(frozen=True)
class FsiCouplingConfig:
    max_iterations: int = 16
    relative_tolerance: float = 1.0e-3
    absolute_tolerance_mps: float = 0.0
    initial_relaxation: float = 0.5
    history_limit: int = 8
    iqn_svd_relative_cutoff: float = 1.0e-10
    iqn_max_condition_number: float = 1.0e10
    iqn_max_coefficient_norm: float | None = None
    iqn_max_update_ratio: float | None = 2.0

    def __post_init__(self) -> None:
        iterations = _strict_positive_integer(
            self.max_iterations,
            name="max_iterations",
        )
        if iterations < 2:
            raise ValueError("max_iterations must be at least 2")
        _finite_positive_float(
            self.relative_tolerance,
            name="relative_tolerance",
        )
        absolute_tolerance = _finite_float(
            self.absolute_tolerance_mps,
            name="absolute_tolerance_mps",
        )
        if absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance_mps must be non-negative")
        relaxation = _finite_float(
            self.initial_relaxation,
            name="initial_relaxation",
        )
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("initial_relaxation must be in (0, 1]")
        _strict_positive_integer(self.history_limit, name="history_limit")
        IqnIlsConfig(
            history_limit=int(self.history_limit),
            initial_picard_relaxation=relaxation,
            svd_relative_cutoff=self.iqn_svd_relative_cutoff,
            max_condition_number=self.iqn_max_condition_number,
            max_coefficient_norm=self.iqn_max_coefficient_norm,
            max_update_ratio=self.iqn_max_update_ratio,
        )


@dataclass(frozen=True)
class FsiSolverConfig:
    step_count: int
    time_step_s: float
    coupling: FsiCouplingConfig = field(default_factory=FsiCouplingConfig)
    completed_step_offset: int = 0

    def __post_init__(self) -> None:
        _strict_positive_integer(self.step_count, name="step_count")
        _finite_positive_float(self.time_step_s, name="time_step_s")
        _strict_non_negative_integer(
            self.completed_step_offset,
            name="completed_step_offset",
        )


@dataclass(frozen=True)
class FsiStepContext:
    step: int
    step_index: int
    time_s: float
    dt_s: float


@dataclass(frozen=True)
class FsiTrialResult:
    marker_velocity_mps: Any
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FsiCouplingReport:
    iterations: int
    converged: bool
    relative_residual: float
    absolute_residual_mps: float
    max_marker_residual_mps: float
    relative_residual_history: tuple[float, ...]
    absolute_residual_history_mps: tuple[float, ...]
    update_modes: tuple[str, ...]
    iqn_rank_history: tuple[int, ...] = ()
    iqn_condition_number_history: tuple[float | None, ...] = ()
    iqn_fallback_reasons: tuple[str | None, ...] = ()
    iqn_update_limited_history: tuple[bool, ...] = ()
    iqn_fallback_count: int = 0


class FsiCouplingConvergenceError(RuntimeError):
    def __init__(self, context: FsiStepContext, report: FsiCouplingReport) -> None:
        super().__init__(
            "FSI marker-velocity coupling did not converge at step "
            f"{context.step}: iterations={report.iterations}, "
            f"relative_residual={report.relative_residual:.6e}, "
            f"absolute_residual_mps={report.absolute_residual_mps:.6e}"
        )
        self.context = context
        self.report = report


class FsiRuntime(Protocol):
    def begin_step(self, context: FsiStepContext) -> Any:
        """Capture rollback state before any mutating step preparation."""
        ...

    def evaluate_trial(
        self,
        context: FsiStepContext,
        marker_velocity_guess_mps: np.ndarray,
    ) -> FsiTrialResult:
        ...

    def commit_step(
        self,
        context: FsiStepContext,
        trial: FsiTrialResult,
        coupling: FsiCouplingReport,
    ) -> Mapping[str, Any]:
        ...

    def rollback_step(self, context: FsiStepContext) -> None:
        ...

    def finalize_run(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class DiagnosticsConfig:
    output_root: str
    export_history: bool = True
    export_comparable_csv: bool = True

    def __post_init__(self) -> None:
        _require_non_empty(self.output_root, name="output_root")


@dataclass(frozen=True)
class FsiProblem:
    problem_id: str
    fluid_domain: FluidDomain
    solid_bodies: tuple[SolidBody, ...]
    interface_surfaces: tuple[InterfaceSurface, ...]
    traction_config: TractionConfig
    runtime_factory: RuntimeFactory
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.problem_id, name="problem_id")
        if not self.solid_bodies:
            raise ValueError("solid_bodies must contain at least one body")
        if not self.interface_surfaces:
            raise ValueError("interface_surfaces must contain at least one surface")
        if not callable(self.runtime_factory):
            raise ValueError("runtime_factory must be callable")

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "fluid_domain": {
                "domain_id": self.fluid_domain.domain_id,
                "coordinate_model": self.fluid_domain.coordinate_model,
                "grid_nodes": list(self.fluid_domain.grid_nodes),
                "bounds_m": [
                    list(point) for point in self.fluid_domain.bounds_m
                ],
                "boundary_conditions": dict(
                    self.fluid_domain.boundary_conditions
                ),
            },
            "solid_body_count": len(self.solid_bodies),
            "interface_surface_count": len(self.interface_surfaces),
            "traction_config": self.traction_config.as_diagnostics(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FsiRunResult:
    problem_id: str
    run_status: str
    requested_step_count: int
    completed_step_count: int
    history: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    artifacts: Mapping[str, str]
    raw_report: Mapping[str, Any]


@dataclass(frozen=True)
class FsiRuntimeRun:
    history: tuple[Mapping[str, Any], ...]
    finalization: Mapping[str, Any]


def solve_fsi(
    problem: FsiProblem,
    solver_config: FsiSolverConfig,
    diagnostics_config: DiagnosticsConfig,
) -> FsiRunResult:
    runtime = problem.runtime_factory(problem, solver_config, diagnostics_config)
    runtime_run = solve_fsi_runtime(runtime, solver_config)
    history = runtime_run.history
    raw = dict(runtime_run.finalization)
    diagnostics = {
        **dict(raw.get("diagnostics", {})),
        "generic_api_invoked": True,
        "problem": problem.as_diagnostics(),
        "pressure_pair_policy": (
            problem.traction_config.pressure_sampling.pair_provider.as_diagnostics()
        ),
        "one_sided_pressure_policy": (
            problem.traction_config.one_sided_pressure.as_diagnostics()
        ),
        "interface_unknown": "marker_velocity_mps",
        "coupling_accelerator": "iqn_ils",
        "coupling_max_iterations": int(solver_config.coupling.max_iterations),
    }
    return FsiRunResult(
        problem_id=problem.problem_id,
        run_status="completed",
        requested_step_count=int(solver_config.step_count),
        completed_step_count=len(history),
        history=history,
        diagnostics=diagnostics,
        artifacts={
            str(key): str(value)
            for key, value in dict(raw.get("artifacts", {})).items()
        },
        raw_report=dict(raw.get("report", raw)),
    )


def solve_fsi_runtime(
    runtime: FsiRuntime,
    solver_config: FsiSolverConfig,
) -> FsiRuntimeRun:
    """Run the sole physical-step loop for an initialized FSI runtime."""

    history_rows: list[dict[str, Any]] = []
    completed_step_offset = int(solver_config.completed_step_offset)
    for local_step_index in range(int(solver_config.step_count)):
        step_index = completed_step_offset + local_step_index
        context = FsiStepContext(
            step=step_index + 1,
            step_index=step_index,
            time_s=float(step_index + 1) * float(solver_config.time_step_s),
            dt_s=float(solver_config.time_step_s),
        )
        trial, coupling = solve_fsi_step(
            runtime,
            context,
            solver_config.coupling,
        )
        try:
            runtime_row = dict(runtime.commit_step(context, trial, coupling))
        except Exception as failure:
            _rollback_after_failure(runtime, context, failure)
            raise
        committed_row = {
            **runtime_row,
            "step": context.step,
            "time_s": context.time_s,
            "fsi_coupling_iterations": coupling.iterations,
            "fsi_coupling_converged": coupling.converged,
            "fsi_coupling_relative_residual": coupling.relative_residual,
            "fsi_coupling_absolute_residual_mps": (
                coupling.absolute_residual_mps
            ),
            "fsi_coupling_max_marker_residual_mps": (
                coupling.max_marker_residual_mps
            ),
            "fsi_coupling_first_relative_residual": (
                coupling.relative_residual_history[0]
            ),
            "fsi_coupling_first_absolute_residual_mps": (
                coupling.absolute_residual_history_mps[0]
            ),
            "fsi_coupling_relative_residual_history": list(
                coupling.relative_residual_history
            ),
            "fsi_coupling_absolute_residual_history_mps": list(
                coupling.absolute_residual_history_mps
            ),
            "fsi_coupling_update_modes": list(coupling.update_modes),
            "fsi_iqn_rank_history": list(coupling.iqn_rank_history),
            "fsi_iqn_condition_number_history": list(
                coupling.iqn_condition_number_history
            ),
            "fsi_iqn_fallback_reasons": list(
                coupling.iqn_fallback_reasons
            ),
            "fsi_iqn_update_limited_history": list(
                coupling.iqn_update_limited_history
            ),
            "fsi_iqn_fallback_count": coupling.iqn_fallback_count,
        }
        history_rows.append(committed_row)
        publish_step = getattr(runtime, "publish_step", None)
        if publish_step is not None:
            # Publication observes an already committed physical state. A file
            # or observer failure stops the run but must not roll that state
            # back and leave persisted evidence ahead of the solver.
            publish_step(context, committed_row)

    return FsiRuntimeRun(
        history=tuple(history_rows),
        finalization=dict(runtime.finalize_run()),
    )


def solve_fsi_step(
    runtime: FsiRuntime,
    context: FsiStepContext,
    coupling_config: FsiCouplingConfig,
) -> tuple[FsiTrialResult, FsiCouplingReport]:
    """Solve one physical step with the canonical marker-velocity fixed point."""

    return _solve_marker_velocity_step(runtime, context, coupling_config)


def _solve_marker_velocity_step(
    runtime: FsiRuntime,
    context: FsiStepContext,
    config: FsiCouplingConfig,
) -> tuple[FsiTrialResult, FsiCouplingReport]:
    try:
        initial_marker_velocity = runtime.begin_step(context)
        guess = _marker_velocity_array(
            initial_marker_velocity,
            name="initial marker velocity",
        )
    except Exception as failure:
        _rollback_after_failure(runtime, context, failure)
        raise
    relative_history: list[float] = []
    absolute_history: list[float] = []
    max_marker_history: list[float] = []
    update_modes: list[str] = []
    iqn_updates: list[IqnIlsUpdate] = []
    accelerator = IqnIlsAccelerator(
        IqnIlsConfig(
            history_limit=int(config.history_limit),
            initial_picard_relaxation=float(config.initial_relaxation),
            svd_relative_cutoff=float(config.iqn_svd_relative_cutoff),
            max_condition_number=float(config.iqn_max_condition_number),
            max_coefficient_norm=config.iqn_max_coefficient_norm,
            max_update_ratio=config.iqn_max_update_ratio,
        )
    )
    last_trial: FsiTrialResult | None = None

    try:
        for _iteration_index in range(int(config.max_iterations)):
            trial = runtime.evaluate_trial(context, guess.copy())
            candidate = _marker_velocity_array(
                trial.marker_velocity_mps,
                name="candidate marker velocity",
            )
            if candidate.shape != guess.shape:
                raise ValueError(
                    "candidate marker velocity shape changed within one FSI step: "
                    f"{candidate.shape} != {guess.shape}"
                )
            residual = candidate - guess
            metrics = _marker_velocity_residual_metrics(candidate, residual)
            relative_history.append(metrics["relative_residual"])
            absolute_history.append(metrics["absolute_residual_mps"])
            max_marker_history.append(metrics["max_marker_residual_mps"])
            last_trial = trial

            absolute_hit = bool(
                float(config.absolute_tolerance_mps) > 0.0
                and metrics["absolute_residual_mps"]
                <= float(config.absolute_tolerance_mps)
            )
            relative_hit = bool(
                metrics["relative_residual"] <= float(config.relative_tolerance)
            )
            if relative_hit or absolute_hit:
                report = _coupling_report(
                    converged=True,
                    relative_history=relative_history,
                    absolute_history=absolute_history,
                    max_marker_history=max_marker_history,
                    update_modes=update_modes,
                    iqn_updates=iqn_updates,
                )
                return last_trial, report

            if _iteration_index + 1 == int(config.max_iterations):
                break
            iqn_update = accelerator.update(guess, candidate)
            guess = iqn_update.next_guess
            iqn_updates.append(iqn_update)
            update_modes.append(iqn_update.mode)
    except Exception as failure:
        _rollback_after_failure(runtime, context, failure)
        raise

    report = _coupling_report(
        converged=False,
        relative_history=relative_history,
        absolute_history=absolute_history,
        max_marker_history=max_marker_history,
        update_modes=update_modes,
        iqn_updates=iqn_updates,
    )
    failure = FsiCouplingConvergenceError(context, report)
    _rollback_after_failure(runtime, context, failure)
    raise failure


def _rollback_after_failure(
    runtime: FsiRuntime,
    context: FsiStepContext,
    failure: Exception,
) -> None:
    try:
        runtime.rollback_step(context)
    except Exception as rollback_failure:
        raise failure from rollback_failure


def _coupling_report(
    *,
    converged: bool,
    relative_history: Sequence[float],
    absolute_history: Sequence[float],
    max_marker_history: Sequence[float],
    update_modes: Sequence[str],
    iqn_updates: Sequence[IqnIlsUpdate],
) -> FsiCouplingReport:
    return FsiCouplingReport(
        iterations=len(relative_history),
        converged=bool(converged),
        relative_residual=float(relative_history[-1]),
        absolute_residual_mps=float(absolute_history[-1]),
        max_marker_residual_mps=float(max_marker_history[-1]),
        relative_residual_history=tuple(float(value) for value in relative_history),
        absolute_residual_history_mps=tuple(
            float(value) for value in absolute_history
        ),
        update_modes=tuple(str(value) for value in update_modes),
        iqn_rank_history=tuple(int(update.rank) for update in iqn_updates),
        iqn_condition_number_history=tuple(
            update.condition_number for update in iqn_updates
        ),
        iqn_fallback_reasons=tuple(
            update.fallback_reason for update in iqn_updates
        ),
        iqn_update_limited_history=tuple(
            bool(update.update_limited) for update in iqn_updates
        ),
        iqn_fallback_count=sum(
            update.fallback_reason is not None for update in iqn_updates
        ),
    )


def _marker_velocity_residual_metrics(
    candidate: np.ndarray,
    residual: np.ndarray,
) -> dict[str, float]:
    squared_marker_residual = np.sum(residual * residual, axis=1)
    squared_marker_velocity = np.sum(candidate * candidate, axis=1)
    absolute_residual = float(np.sqrt(np.mean(squared_marker_residual)))
    velocity_scale = float(np.sqrt(np.mean(squared_marker_velocity)))
    return {
        "absolute_residual_mps": absolute_residual,
        "max_marker_residual_mps": float(
            np.sqrt(squared_marker_residual).max(initial=0.0)
        ),
        "relative_residual": absolute_residual / max(velocity_scale, 1.0e-30),
    }


def _marker_velocity_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (marker_count, 3)")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _require_non_empty(value: str, *, name: str) -> None:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be non-empty")


def _strict_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _strict_positive_integer(value: Any, *, name: str) -> int:
    integer = _strict_integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be a strictly positive integer")
    return integer


def _strict_non_negative_integer(value: Any, *, name: str) -> int:
    integer = _strict_integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _finite_float(value: Any, *, name: str) -> float:
    try:
        finite_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(finite_value):
        raise ValueError(f"{name} must be finite")
    return finite_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number, not bool")
    finite_value = _finite_float(value, name=name)
    if finite_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return finite_value


def _vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    raw_values = tuple(values)
    if len(raw_values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = tuple(_finite_float(value, name=name) for value in raw_values)
    return (vector[0], vector[1], vector[2])
