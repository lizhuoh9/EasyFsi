from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from simulation_core.drivers.case_spec import FsiCaseSpec


BenchmarkRunner = Callable[[Any], Mapping[str, object]]


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
    authoritative = {
        "case": spec.case_spec.case_id,
        "solver_family": spec.solver_family,
        "case_metadata": dict(spec.case_metadata),
        "boundary_conditions": dict(spec.boundary_conditions),
        "acceptance_tolerance": spec.case_spec.acceptance_tolerance,
        "reference_results": dict(spec.case_spec.reference_results),
    }
    for field, expected in authoritative.items():
        if field in raw_report and raw_report[field] != expected:
            raise ValueError(
                f"benchmark runner cannot override authoritative {field}: "
                f"{raw_report[field]!r} != {expected!r}"
            )
    report = {**raw_report, **authoritative}
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
    sources = report.get("computed_result_sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError(
            "benchmark report computed_result_sources must be a non-empty mapping"
        )
