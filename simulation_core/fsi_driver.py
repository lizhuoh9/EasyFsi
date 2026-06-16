from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .fsi_coupling import ForceBalanceReport, action_reaction_balance


Vector = tuple[float, ...]


class FluidModel(Protocol):
    def advance(self, context: FsiStepContext) -> Mapping[str, Any] | None:
        ...

    def interface_force_n(self) -> Sequence[float]:
        ...

    def apply_interface_displacement(self, displacement_m: Sequence[float]) -> None:
        ...


class SolidModel(Protocol):
    def apply_interface_force(self, force_n: Sequence[float]) -> None:
        ...

    def advance(self, context: FsiStepContext) -> Mapping[str, Any] | None:
        ...

    def interface_displacement_m(self) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class FsiCaseSpec:
    case_id: str
    source_url: str
    coordinate_model: str
    geometry: Mapping[str, Any]
    fluid: Mapping[str, Any]
    solid: Mapping[str, Any]
    boundary_conditions: Mapping[str, Any]
    reference_results: Mapping[str, float]
    acceptance_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.source_url:
            raise ValueError("source_url must be non-empty")
        if not 0.0 < float(self.acceptance_tolerance) < 1.0:
            raise ValueError("acceptance_tolerance must be in (0, 1)")


@dataclass(frozen=True)
class FsiStepContext:
    step_index: int
    time_s: float
    dt_s: float


@dataclass(frozen=True)
class FsiStepReport:
    step_index: int
    time_s: float
    fluid_force_n: Vector
    solid_reaction_force_n: Vector
    solid_displacement_m: Vector
    action_reaction: ForceBalanceReport
    fluid_report: Mapping[str, Any]
    solid_report: Mapping[str, Any]


@dataclass(frozen=True)
class FsiRunReport:
    case_id: str
    step_count: int
    final_results: Mapping[str, float]
    reference_results: Mapping[str, float]
    relative_errors: Mapping[str, float]
    step_reports: tuple[FsiStepReport, ...]


class FsiDriver:
    """Small case-agnostic partitioned FSI driver for injected models."""

    def __init__(
        self,
        *,
        case_spec: FsiCaseSpec,
        fluid_model: FluidModel,
        solid_model: SolidModel,
        dt_s: float = 1.0,
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.case_spec = case_spec
        self.fluid_model = fluid_model
        self.solid_model = solid_model
        self.dt_s = float(dt_s)

    def run(self, *, step_count: int) -> FsiRunReport:
        if step_count <= 0:
            raise ValueError("step_count must be positive")
        step_reports: list[FsiStepReport] = []
        max_displacement_m = 0.0

        for step_index in range(step_count):
            context = FsiStepContext(
                step_index=step_index,
                time_s=float(step_index) * self.dt_s,
                dt_s=self.dt_s,
            )
            fluid_report = dict(self.fluid_model.advance(context) or {})
            fluid_force = _vector(self.fluid_model.interface_force_n(), name="fluid_force_n")
            solid_reaction = tuple(-component for component in fluid_force)
            self.solid_model.apply_interface_force(solid_reaction)
            solid_report = dict(self.solid_model.advance(context) or {})
            displacement = _vector(
                self.solid_model.interface_displacement_m(),
                name="solid_displacement_m",
            )
            self.fluid_model.apply_interface_displacement(displacement)
            balance = action_reaction_balance(fluid_force, solid_reaction)
            max_displacement_m = max(max_displacement_m, _norm(displacement))
            step_reports.append(
                FsiStepReport(
                    step_index=step_index,
                    time_s=context.time_s,
                    fluid_force_n=fluid_force,
                    solid_reaction_force_n=solid_reaction,
                    solid_displacement_m=displacement,
                    action_reaction=balance,
                    fluid_report=fluid_report,
                    solid_report=solid_report,
                )
            )

        final_results = {"max_displacement_m": max_displacement_m}
        relative_errors = _relative_errors(
            final_results=final_results,
            reference_results=self.case_spec.reference_results,
        )
        return FsiRunReport(
            case_id=self.case_spec.case_id,
            step_count=step_count,
            final_results=final_results,
            reference_results=dict(self.case_spec.reference_results),
            relative_errors=relative_errors,
            step_reports=tuple(step_reports),
        )


def _vector(values: Sequence[float], *, name: str) -> Vector:
    vector = tuple(float(value) for value in values)
    if len(vector) == 0:
        raise ValueError(f"{name} must contain at least one component")
    if any(not math.isfinite(component) for component in vector):
        raise ValueError(f"{name} components must be finite")
    return vector


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _relative_errors(
    *,
    final_results: Mapping[str, float],
    reference_results: Mapping[str, float],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key, reference in reference_results.items():
        if key not in final_results:
            continue
        reference_value = float(reference)
        if abs(reference_value) <= 1.0e-30:
            continue
        errors[key] = abs(float(final_results[key]) - reference_value) / abs(reference_value)
    return errors
