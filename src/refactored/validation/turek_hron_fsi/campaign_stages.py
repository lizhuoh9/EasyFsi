"""Pure frozen formal-stage contracts for the Turek--Hron FSI campaign."""

from __future__ import annotations

import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

from .references import (
    FEATFLOW_RAW_FSI2_SOURCE_ID,
    FEATFLOW_RAW_FSI3_SOURCE_ID,
    canonical_fsi1_metric_values,
    reference_metric,
)


_FSI1_S0_SPEC = {
    "stage": "fsi1-s0",
    "preset": "fsi1",
    "grid_nodes": (4, 48, 288),
    "dt_s": 0.005,
    "step_count": 1600,
    "markers_per_side": "auto",
    "markers_per_tip": "auto",
    "ib_anisotropic_envelope": True,
    "classify_far_internal_nodes": True,
    "flow_cg_preconditioner": "fv_multigrid",
    "flow_predictor_substeps": 1,
    "fluid_advection_scheme": "rk2",
    "flow_projection_iterations": 4000,
    "flow_cg_tolerance": 1.0e-6,
    "flow_reprojection_iterations": 1200,
    "flow_reprojection_cg_tolerance": 1.0e-4,
    "fsi_coupling_absolute_tolerance_mps": 1.0e-4,
    "solid_substeps": 100,
    "velocity_damping": 1.0,
    "marker_reseed_interval_steps": None,
}


def _with(base: Mapping[str, Any], **updates: Any) -> Mapping[str, Any]:
    return MappingProxyType({**base, **updates})


_DYNAMIC_BASE = _with(
    _FSI1_S0_SPEC,
    dt_s=0.001,
    step_count=35000,
    solid_substeps=100,
    fsi_coupling_tolerance=1.0e-3,
    fsi_coupling_absolute_tolerance_mps=0.0,
)
_STAGE_SPECS = MappingProxyType(
    {
        "fsi1-s0": MappingProxyType(dict(_FSI1_S0_SPEC)),
        "fsi1-m0": _with(
            _FSI1_S0_SPEC, stage="fsi1-m0", grid_nodes=(4, 96, 576), solid_substeps=200
        ),
        "fsi1-m1": _with(
            _FSI1_S0_SPEC, stage="fsi1-m1", grid_nodes=(4, 96, 576), dt_s=0.0025,
            step_count=3200, solid_substeps=100,
        ),
        "fsi1-f0": _with(
            _FSI1_S0_SPEC, stage="fsi1-f0", grid_nodes=(4, 144, 864), dt_s=0.0025,
            step_count=3200, solid_substeps=200,
        ),
        "fsi2-h0": _with(_DYNAMIC_BASE, stage="fsi2-h0", preset="fsi2", grid_nodes=(4, 48, 288), step_count=3000),
        "fsi2-m0": _with(_DYNAMIC_BASE, stage="fsi2-m0", preset="fsi2", grid_nodes=(4, 96, 576)),
        "fsi2-m1": _with(_DYNAMIC_BASE, stage="fsi2-m1", preset="fsi2", grid_nodes=(4, 96, 576), dt_s=0.0005, step_count=70000),
        "fsi2-f0": _with(_DYNAMIC_BASE, stage="fsi2-f0", preset="fsi2", grid_nodes=(4, 144, 864)),
        "fsi3-h0": _with(_DYNAMIC_BASE, stage="fsi3-h0", preset="fsi3", grid_nodes=(4, 48, 288), step_count=3000),
        "fsi3-m0": _with(_DYNAMIC_BASE, stage="fsi3-m0", preset="fsi3", grid_nodes=(4, 96, 576)),
        "fsi3-m0-tight": _with(_DYNAMIC_BASE, stage="fsi3-m0-tight", preset="fsi3", grid_nodes=(4, 96, 576), fsi_coupling_tolerance=1.0e-4),
        "fsi3-m1": _with(_DYNAMIC_BASE, stage="fsi3-m1", preset="fsi3", grid_nodes=(4, 96, 576), dt_s=0.0005, step_count=70000),
        "fsi3-f0": _with(_DYNAMIC_BASE, stage="fsi3-f0", preset="fsi3", grid_nodes=(4, 144, 864), solid_substeps=200),
    }
)
STAGE_NAMES = tuple(_STAGE_SPECS)
FSI1_S0_SPEC = dict(_STAGE_SPECS["fsi1-s0"])

_PREREQUISITES = MappingProxyType(
    {
        "fsi1-s0": (),
        "fsi1-m0": ("fsi1-s0",),
        "fsi1-m1": ("fsi1-s0", "fsi1-m0"),
        "fsi1-f0": ("fsi1-s0", "fsi1-m0", "fsi1-m1"),
        "fsi2-h0": ("fsi1-f0",),
        "fsi2-m0": ("fsi1-f0", "fsi2-h0"),
        "fsi2-m1": ("fsi1-f0", "fsi2-h0", "fsi2-m0"),
        "fsi2-f0": ("fsi1-f0", "fsi2-h0", "fsi2-m0", "fsi2-m1"),
        "fsi3-h0": ("fsi2-f0",),
        "fsi3-m0": ("fsi2-f0", "fsi3-h0"),
        "fsi3-m0-tight": ("fsi2-f0", "fsi3-h0", "fsi3-m0"),
        "fsi3-m1": ("fsi2-f0", "fsi3-h0", "fsi3-m0"),
        "fsi3-f0": ("fsi2-f0", "fsi3-h0", "fsi3-m0", "fsi3-m1", "fsi3-m0-tight"),
    }
)
_FSI1_FIELDS = tuple(canonical_fsi1_metric_values())
_DYNAMIC_FIELDS = (
    "point_a_uy_amplitude_m",
    "point_a_uy_frequency_hz",
    "total_drag_midrange_n",
    "total_lift_amplitude_n",
)
_DISPLACEMENT_FIELDS = _FSI1_FIELDS[:2]
_FORCE_FIELDS = _FSI1_FIELDS[2:]
_DYNAMIC_POINT_A_FIELDS = _DYNAMIC_FIELDS[:2]
_DYNAMIC_FORCE_FIELDS = _DYNAMIC_FIELDS[2:]


def stage_spec(name: str) -> dict[str, Any]:
    """Return a mutable copy of one internally frozen formal-stage spec."""

    if name not in _STAGE_SPECS:
        raise ValueError(f"unknown formal Turek--Hron stage {name!r}")
    return dict(_STAGE_SPECS[name])


def is_periodic_stage(name: str) -> bool:
    if name not in _STAGE_SPECS:
        raise ValueError(f"unknown formal Turek--Hron stage {name!r}")
    return name.startswith(("fsi2-", "fsi3-"))


def stage_prerequisites(name: str) -> tuple[str, ...]:
    if name not in _PREREQUISITES:
        raise ValueError(f"unknown formal Turek--Hron stage {name!r}")
    return _PREREQUISITES[name]


def _dynamic_references(case: str) -> dict[str, float]:
    source = FEATFLOW_RAW_FSI2_SOURCE_ID if case == "fsi2" else FEATFLOW_RAW_FSI3_SOURCE_ID
    return {
        "point_a_uy_amplitude_m": reference_metric(source, case, "point_a_uy", "amplitude").value,
        "point_a_uy_frequency_hz": reference_metric(source, case, "point_a_uy", "frequency").value,
        "total_drag_midrange_n": reference_metric(source, case, "total_drag", "midrange").value,
        "total_lift_amplitude_n": reference_metric(source, case, "total_lift", "amplitude").value,
    }


def _references_for(stage: str) -> dict[str, float]:
    if stage.startswith("fsi1-"):
        return dict(canonical_fsi1_metric_values())
    return _dynamic_references(stage[:4])


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _metric_ledger(
    metrics: Mapping[str, object], references: Mapping[str, float], violations: list[str]
) -> dict[str, dict[str, float]]:
    ledger: dict[str, dict[str, float]] = {}
    for field, reference in references.items():
        observed = _finite_number(metrics.get(field))
        if observed is None:
            violations.append(field)
            continue
        error = abs(observed - reference) / abs(reference)
        if not math.isfinite(error):
            violations.append(field)
            continue
        ledger[field] = {
            "reference": float(reference),
            "observed": observed,
            "relative_error": error,
        }
    return ledger


def _predecessors(
    stage: str, reports: Mapping[str, Mapping[str, Any]], violations: list[str]
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for predecessor in stage_prerequisites(stage):
        report = reports.get(predecessor)
        cross_case = predecessor in {"fsi1-f0", "fsi2-f0"}
        passed = bool(
            isinstance(report, Mapping)
            and report.get("benchmark_quality_passed" if cross_case else "stage_gate_passed") is True
        )
        result[predecessor] = passed
        if not passed:
            violations.append(f"prerequisite:{predecessor}")
    return result


def _require_error_limits(
    ledger: Mapping[str, Mapping[str, float]], limits: Mapping[str, float], violations: list[str]
) -> bool:
    passed = True
    for field, limit in limits.items():
        entry = ledger.get(field)
        if entry is None or not float(entry["relative_error"]) < limit:
            violations.append(f"reference:{field}")
            passed = False
    return passed


def _pair_delta(
    current: Mapping[str, Mapping[str, float]], previous: Mapping[str, Any] | None,
    fields: tuple[str, ...], limits: Mapping[str, float], label: str, violations: list[str],
) -> tuple[dict[str, float], bool]:
    prior_ledger = previous.get("metric_ledger") if isinstance(previous, Mapping) else None
    values: dict[str, float] = {}
    passed = True
    for field in fields:
        current_entry = current.get(field)
        prior_entry = prior_ledger.get(field) if isinstance(prior_ledger, Mapping) else None
        if not isinstance(current_entry, Mapping) or not isinstance(prior_entry, Mapping):
            violations.append(f"{label}:{field}")
            passed = False
            continue
        observed = _finite_number(current_entry.get("observed"))
        previous_observed = _finite_number(prior_entry.get("observed"))
        reference = _finite_number(current_entry.get("reference"))
        if observed is None or previous_observed is None or reference is None or reference == 0.0:
            violations.append(f"{label}:{field}")
            passed = False
            continue
        delta = abs(observed - previous_observed) / abs(reference)
        values[field] = delta
        if not delta < limits[field]:
            violations.append(f"{label}:{field}")
            passed = False
    return values, passed


def _fsi1_assessment(
    stage: str, ledger: Mapping[str, Mapping[str, float]], reports: Mapping[str, Mapping[str, Any]],
    violations: list[str],
) -> tuple[dict[str, dict[str, float]], bool, bool]:
    pairs: dict[str, dict[str, float]] = {}
    passed = True
    exploratory = False
    if stage in {"fsi1-m0", "fsi1-m1"}:
        passed &= _require_error_limits(
            ledger, {**{field: 0.10 for field in _DISPLACEMENT_FIELDS}, **{field: 0.15 for field in _FORCE_FIELDS}}, violations
        )
    if stage == "fsi1-m0":
        deltas, improved = _pair_delta(
            ledger, reports.get("fsi1-s0"), _FSI1_FIELDS,
            {field: float("inf") for field in _FSI1_FIELDS}, "improvement", violations,
        )
        pairs["m0_vs_s0_error"] = deltas
        previous_ledger = (
            reports.get("fsi1-s0", {}).get("metric_ledger", {})
            if isinstance(reports.get("fsi1-s0"), Mapping)
            else {}
        )
        for field in _FSI1_FIELDS:
            prior = previous_ledger.get(field, {}).get("relative_error") if field in deltas else None
            if prior is None or not ledger[field]["relative_error"] < prior:
                violations.append(f"improvement:{field}")
                improved = False
        passed &= improved
    if stage == "fsi1-m1":
        deltas, pair_passed = _pair_delta(
            ledger, reports.get("fsi1-m0"), _FSI1_FIELDS,
            {field: 0.05 for field in _FSI1_FIELDS}, "temporal_delta", violations,
        )
        pairs["m1_vs_m0"] = deltas
        passed &= pair_passed
        exploratory = passed
    if stage == "fsi1-f0":
        passed &= _require_error_limits(
            ledger, {**{field: 0.05 for field in _DISPLACEMENT_FIELDS}, **{field: 0.10 for field in _FORCE_FIELDS}}, violations
        )
        deltas, pair_passed = _pair_delta(
            ledger, reports.get("fsi1-m1"), _FSI1_FIELDS,
            {**{field: 0.03 for field in _DISPLACEMENT_FIELDS}, **{field: 0.05 for field in _FORCE_FIELDS}}, "spatial_delta", violations,
        )
        pairs["f0_vs_m1"] = deltas
        passed &= pair_passed
    return pairs, bool(passed), exploratory


def _periodic_assessment(
    stage: str, ledger: Mapping[str, Mapping[str, float]], reports: Mapping[str, Mapping[str, Any]],
    violations: list[str],
) -> tuple[dict[str, dict[str, float]], bool, bool]:
    pairs: dict[str, dict[str, float]] = {}
    if stage.endswith("-h0"):
        return pairs, True, False
    benchmark_limits = {
        **{field: 0.05 for field in _DYNAMIC_POINT_A_FIELDS},
        **{field: 0.10 for field in _DYNAMIC_FORCE_FIELDS},
    }
    exploratory_limits = {
        **{field: 0.10 for field in _DYNAMIC_POINT_A_FIELDS},
        **{field: 0.15 for field in _DYNAMIC_FORCE_FIELDS},
    }
    passed = _require_error_limits(
        ledger, exploratory_limits if stage.endswith("-m0") and not stage.endswith("-m0-tight") else benchmark_limits, violations
    )
    if stage.endswith("-m1"):
        deltas, pair_passed = _pair_delta(
            ledger, reports.get(f"{stage[:4]}-m0"), _DYNAMIC_FIELDS,
            {**{field: 0.03 for field in _DYNAMIC_POINT_A_FIELDS}, **{field: 0.05 for field in _DYNAMIC_FORCE_FIELDS}}, "temporal_delta", violations,
        )
        pairs["m1_vs_m0"] = deltas
        passed &= pair_passed
    if stage.endswith("-m0-tight"):
        deltas, pair_passed = _pair_delta(
            ledger, reports.get("fsi3-m0"), _DYNAMIC_FIELDS,
            {**{field: 0.03 for field in _DYNAMIC_POINT_A_FIELDS}, **{field: 0.05 for field in _DYNAMIC_FORCE_FIELDS}}, "tight_delta", violations,
        )
        pairs["tight_vs_m0"] = deltas
        passed &= pair_passed
    if stage.endswith("-f0"):
        # M0 may enter refinement at exploratory accuracy, but final
        # benchmark quality requires every M0/M1/F0 run at strict accuracy.
        # Recompute from observed values rather than trusting stored errors.
        for prior_stage in (f"{stage[:4]}-m0", f"{stage[:4]}-m1"):
            prior_report = reports.get(prior_stage, {})
            prior_ledger = prior_report.get("metric_ledger", {})
            observed = {
                field: entry.get("observed")
                for field, entry in prior_ledger.items()
                if isinstance(entry, Mapping)
            } if isinstance(prior_ledger, Mapping) else {}
            prior_violations: list[str] = []
            recomputed = _metric_ledger(
                observed, _references_for(stage), prior_violations
            )
            prior_passed = _require_error_limits(
                recomputed, benchmark_limits, prior_violations
            )
            violations.extend(
                f"benchmark:{prior_stage}:{reason}" for reason in prior_violations
            )
            passed &= prior_passed
        deltas, pair_passed = _pair_delta(
            ledger, reports.get(f"{stage[:4]}-m0"), _DYNAMIC_FIELDS,
            {**{field: 0.03 for field in _DYNAMIC_POINT_A_FIELDS}, **{field: 0.05 for field in _DYNAMIC_FORCE_FIELDS}}, "spatial_delta", violations,
        )
        pairs["f0_vs_m0"] = deltas
        passed &= pair_passed
    return pairs, bool(passed), stage.endswith("-m0") and bool(passed)


def assess_stage(
    stage: str,
    *,
    metrics: Mapping[str, float],
    numerical_health_passed: bool,
    settled: bool,
    prerequisite_reports: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assess one frozen stage without touching artifacts, solvers, or lineage."""

    if stage not in _STAGE_SPECS:
        raise ValueError(f"unknown formal Turek--Hron stage {stage!r}")
    violations: list[str] = []
    reports = prerequisite_reports if isinstance(prerequisite_reports, Mapping) else {}
    predecessors = _predecessors(stage, reports, violations)
    health = numerical_health_passed is True
    if not health:
        violations.append("numerical_health")
    settled_required = not stage.endswith("-h0")
    settled_passed = settled is True
    if settled_required and not settled_passed:
        violations.append("settled")
    references = {} if stage.endswith("-h0") else _references_for(stage)
    ledger = _metric_ledger(metrics if isinstance(metrics, Mapping) else {}, references, violations)
    if len(ledger) != len(references):
        metric_passed = False
        pairs: dict[str, dict[str, float]] = {}
        exploratory = False
    elif stage.startswith("fsi1-"):
        pairs, metric_passed, exploratory = _fsi1_assessment(stage, ledger, reports, violations)
    else:
        pairs, metric_passed, exploratory = _periodic_assessment(stage, ledger, reports, violations)
    stage_gate_passed = bool(health and (not settled_required or settled_passed) and all(predecessors.values()) and metric_passed)
    benchmark_quality_passed = stage_gate_passed and stage.endswith("-f0")
    return {
        "stage": stage,
        "status": "passed" if stage_gate_passed else "failed",
        "numerical_health_passed": health,
        "settled_required": settled_required,
        "settled": settled_passed,
        "prerequisites": predecessors,
        "metric_ledger": ledger,
        "pair_deltas": pairs,
        "stage_gate_passed": stage_gate_passed,
        "exploratory_passed": bool(stage_gate_passed and exploratory),
        "benchmark_quality_passed": benchmark_quality_passed,
        "violations": tuple(violations),
    }


__all__ = [
    "FSI1_S0_SPEC", "STAGE_NAMES", "assess_stage", "is_periodic_stage",
    "stage_prerequisites", "stage_spec",
]
