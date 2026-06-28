from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


RuntimeExecutor = Callable[
    ["FsiProblem", "FsiSolverConfig", "DiagnosticsConfig"],
    Mapping[str, Any],
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
        if len(self.grid_nodes) != 3 or any(int(value) <= 0 for value in self.grid_nodes):
            raise ValueError("grid_nodes must contain three positive integers")
        if len(self.bounds_m) != 2:
            raise ValueError("bounds_m must contain min and max points")
        for point in self.bounds_m:
            _vector3(point, name="bounds_m")


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
        if int(self.marker_count) < 0:
            raise ValueError("marker_count must be non-negative")
        if self.fluid_side_normal_sign is not None:
            sign = float(self.fluid_side_normal_sign)
            if sign not in (-1.0, 1.0):
                raise ValueError("fluid_side_normal_sign must be -1.0 or 1.0")


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
        if int(self.sample_pair_fallback_count_max) < 0:
            raise ValueError("sample_pair_fallback_count_max must be non-negative")

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
class FsiSolverConfig:
    step_count: int
    time_step_s: float
    solver_name: str = "generic-fsi-solver"

    def __post_init__(self) -> None:
        if int(self.step_count) <= 0:
            raise ValueError("step_count must be positive")
        if float(self.time_step_s) <= 0.0:
            raise ValueError("time_step_s must be positive")
        _require_non_empty(self.solver_name, name="solver_name")


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
    runtime_executor: RuntimeExecutor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.problem_id, name="problem_id")
        if not self.solid_bodies:
            raise ValueError("solid_bodies must contain at least one body")
        if not self.interface_surfaces:
            raise ValueError("interface_surfaces must contain at least one surface")

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "fluid_domain": {
                "domain_id": self.fluid_domain.domain_id,
                "coordinate_model": self.fluid_domain.coordinate_model,
                "grid_nodes": list(self.fluid_domain.grid_nodes),
            },
            "solid_body_count": len(self.solid_bodies),
            "interface_surface_count": len(self.interface_surfaces),
            "traction_config": self.traction_config.as_diagnostics(),
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


def solve_fsi(
    problem: FsiProblem,
    solver_config: FsiSolverConfig,
    diagnostics_config: DiagnosticsConfig,
) -> FsiRunResult:
    raw = dict(problem.runtime_executor(problem, solver_config, diagnostics_config))
    history = tuple(dict(row) for row in raw.get("history", ()))
    diagnostics = dict(raw.get("diagnostics", {}))
    diagnostics.setdefault("generic_api_invoked", True)
    diagnostics.setdefault("problem", problem.as_diagnostics())
    diagnostics.setdefault(
        "pressure_pair_policy",
        problem.traction_config.pressure_sampling.pair_provider.as_diagnostics(),
    )
    diagnostics.setdefault(
        "one_sided_pressure_policy",
        problem.traction_config.one_sided_pressure.as_diagnostics(),
    )
    return FsiRunResult(
        problem_id=problem.problem_id,
        run_status=str(raw.get("run_status", "completed" if history else "unknown")),
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


def _require_non_empty(value: str, *, name: str) -> None:
    if not str(value):
        raise ValueError(f"{name} must be non-empty")


def _vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (vector[0], vector[1], vector[2])
