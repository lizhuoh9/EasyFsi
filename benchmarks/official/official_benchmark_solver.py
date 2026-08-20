from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

BenchmarkRunner = Callable[[Any], Mapping[str, object]]


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
class OfficialBenchmarkRunSpec:
    case_spec: FsiCaseSpec
    solver_family: str
    case_metadata: Mapping[str, Any]
    boundary_conditions: Mapping[str, Any]
    config: Any
    runner: BenchmarkRunner

    def __post_init__(self) -> None:
        if not self.solver_family:
            raise ValueError("solver_family must be non-empty")


def run_official_fsi_benchmark(spec: OfficialBenchmarkRunSpec) -> dict[str, object]:
    """Run one official FSI benchmark through the shared case-agnostic entrypoint."""

    raw_report = dict(spec.runner(spec.config))
    report = {
        **raw_report,
        "case": raw_report.get("case", spec.case_spec.case_id),
        "solver_family": spec.solver_family,
        "case_metadata": raw_report.get("case_metadata", dict(spec.case_metadata)),
        "boundary_conditions": raw_report.get(
            "boundary_conditions",
            dict(spec.boundary_conditions),
        ),
        "acceptance_tolerance": raw_report.get(
            "acceptance_tolerance",
            spec.case_spec.acceptance_tolerance,
        ),
        "reference_results": raw_report.get(
            "reference_results",
            dict(spec.case_spec.reference_results),
        ),
    }
    _validate_report(report, spec)
    return report


def _validate_report(
    report: Mapping[str, object],
    spec: OfficialBenchmarkRunSpec,
) -> None:
    if report["case"] != spec.case_spec.case_id:
        raise ValueError(
            f"benchmark runner returned case={report['case']!r}; "
            f"expected {spec.case_spec.case_id!r}"
        )
    if "computed_result_sources" not in report:
        raise ValueError("benchmark report must include computed_result_sources")
