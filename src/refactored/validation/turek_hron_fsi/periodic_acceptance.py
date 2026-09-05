"""Solver-free periodic Turek--Hron health, cycle and accepted-work evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .acceptance import (
    Fsi1AcceptanceConfig,
    NumericalHealthPolicy,
    TurekHronAcceptanceError,
    read_numerical_history_csv,
)
from .limit_cycle import LimitCycleValidationError, analyze_limit_cycle


_WORK_FIELDS = (
    "coupling_trial_count", "coupling_rejected_trial_count",
    "pressure_cg_iterations_total", "pressure_matvec_count_total",
    "fluid_solve_count", "solid_macro_solve_count",
    "mpm_substeps_executed_total", "iqn_fallback_count",
)


def dynamic_numerical_policy(relative_tolerance: float) -> NumericalHealthPolicy:
    """Dynamic stages certify their frozen relative tolerance with no absolute floor."""
    return NumericalHealthPolicy(
        mechanism_probe_required=False,
        enforce_absolute_coupling_limits=False,
        coupling_relative_residual_max=relative_tolerance,
        required_finite_fields=("max_displacement_m", "fluid_speed_max_mps", "fsi_coupling_initial_relaxation",
                                "fsi_coupling_iterations_used"),
    )


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cycle_payload(report: Any, times: np.ndarray, signals: Mapping[str, np.ndarray]) -> dict:
    start, end = report.cycles[0].start_s, report.cycles[-1].end_s
    dt = float(times[1] - times[0])
    sample_times = start + np.arange(int(np.floor((end - start) / dt + 1e-9)) + 1) * dt
    spectra = {}
    for name, values in signals.items():
        sampled = np.interp(sample_times, times, values)
        spectra[name] = {
            "frequency_hz": np.fft.rfftfreq(len(sampled), d=dt),
            "amplitude": 2.0 * np.abs(np.fft.rfft(sampled - np.mean(sampled))) / len(sampled),
            "dominant_frequency_hz": report.spectral_frequency_hz[name],
        }
    return _finite_json({
        "crossing_times_s": report.crossing_times_s,
        "cycles": [{
            "start_s": cycle.start_s, "end_s": cycle.end_s,
            "period_s": cycle.period_s, "frequency_hz": 1.0 / cycle.period_s,
            "signals": {key: asdict(value) for key, value in cycle.signals.items()},
        } for cycle in report.cycles],
        "primary_period_s": report.primary_period_s,
        "primary_frequency_hz": report.primary_frequency_hz,
        "spectral_frequency_hz": report.spectral_frequency_hz,
        "spectra": spectra,
        "stability": {
            **{key: value for key, value in vars(report.stability).items() if key != "checks"},
            "checks": dict(report.stability.checks),
        },
    })


def _matches(observed: Any, expected: Any, label: str) -> None:
    actual, reference = np.asarray(observed), np.asarray(expected)
    if actual.shape != reference.shape or not np.all(np.isfinite(actual)) or not np.allclose(
        actual, reference, rtol=32 * np.finfo(np.float64).eps,
        atol=32 * np.finfo(np.float64).eps,
    ):
        raise TurekHronAcceptanceError(f"accepted {label} does not match history")


def validate_record_history_pair(record: Mapping[str, Any], row: Mapping[str, Any], *, span_m: float) -> None:
    """Bind accepted arrays to the same one-based CSV state and slab force units."""
    _matches(record["accepted_step"], row["step"], "step")
    _matches(record["accepted_time_s"], row["time_s"], "time")
    _matches(record["point_a_displacement_turek_xy_m"],
             [row["tip_ux_turek_hron_m"], row["tip_uy_turek_hron_m"]], "Point A displacement")
    force = np.asarray(record["total_force_solver_xyz_n"])
    _matches([-force[2] / span_m, force[1] / span_m],
             [row["total_drag_per_span_n_per_m"], row["total_lift_per_span_n_per_m"]], "total force")
    if "fsi_coupling_residual" in row:
        count = int(np.asarray(record["coupling_trial_count"]))
        _matches(count, row["fsi_coupling_iterations_used"], "coupling trial count")
        _matches(np.asarray(record["coupling_relative_residual_history"])[count - 1],
                 row["fsi_coupling_residual"], "relative residual")
        _matches(np.asarray(record["coupling_absolute_residual_history_mps"])[count - 1],
                 row["fsi_coupling_absolute_residual_mps"], "absolute residual")


def _marker_spacing(record: Mapping[str, Any], marker_segments: Any = None) -> dict:
    position = np.asarray(record["marker_current_position_m"])
    regions, order = np.asarray(record["marker_region_id"]), np.asarray(record["marker_order"])
    distances = []
    if marker_segments is not None:
        segments = np.asarray(marker_segments, dtype=np.int64).reshape((-1, 2))
        distances.extend(np.linalg.norm(position[segments[:, 1]] - position[segments[:, 0]], axis=1))
    else:
        for region in np.unique(regions):
            indices = np.flatnonzero(regions == region)
            indices = indices[np.argsort(order[indices])]
            distances.extend(np.linalg.norm(np.diff(position[indices], axis=0), axis=1))
    values = np.asarray(distances)
    positive = values[values > 0]
    return {
        "max_min_ratio": float(np.max(positive) / np.min(positive)) if len(positive) and np.all(values > 0) else None,
        "nonpositive_spacing_count": int(np.sum(values <= 0)),
        "interpretation": "adjacent markers within each disconnected ordered surface region",
    }


def _telemetry(record: Mapping[str, Any], row: Mapping[str, Any], cycles: tuple,
               marker_segments: Any = None) -> dict:
    count = int(np.asarray(record["coupling_trial_count"]))
    relative = np.asarray(record["coupling_relative_residual_history"])[:count]
    absolute = np.asarray(record["coupling_absolute_residual_history_mps"])[:count]
    cycle_index, phase = None, None
    for index, cycle in enumerate(cycles):
        if cycle.start_s <= row["time_s"] <= cycle.end_s:
            cycle_index, phase = index, (row["time_s"] - cycle.start_s) / cycle.period_s
            break
    updates = max(0, count - 1)
    payload = {
        "step": row["step"], "time_s": row["time_s"],
        "cycle_index": cycle_index, "cycle_phase": phase,
        "point_a_ux_m": row["tip_ux_turek_hron_m"],
        "point_a_uy_m": row["tip_uy_turek_hron_m"],
        "point_a_uy_velocity_mps": float(np.asarray(record["point_a_velocity_turek_xy_mps"])[1]),
        "total_drag_n": row["total_drag_per_span_n_per_m"],
        "total_lift_n": row["total_lift_per_span_n_per_m"],
        "relative_residual_first": float(relative[0]),
        "relative_residual_second": float(relative[1]) if count > 1 else None,
        "coupling_relative_residual_history": relative,
        "coupling_absolute_residual_history_mps": absolute,
        "coupling_converged": row["fsi_coupling_converged"],
        "coupling_residual_measured": row["fsi_coupling_residual_measured"],
        "coupling_max_marker_residual_mps": row["fsi_coupling_max_marker_residual_mps"],
        "initial_picard_relaxation": row["fsi_coupling_initial_relaxation"],
        "fixed_root_max_displacement_m": row["fixed_root_max_displacement_m"],
        "maximum_solid_displacement_m": row["max_displacement_m"],
        "maximum_marker_displacement_m": float(np.max(np.linalg.norm(
            record["marker_material_displacement_m"], axis=1))),
        "marker_spacing": _marker_spacing(record, marker_segments),
        "marker_coverage": {
            "expected": row["stress_expected_marker_count"],
            "stress_valid": row["stress_valid_marker_count"],
            "velocity_valid": row["post_solid_no_slip_valid_marker_count"],
        },
        **{key: int(np.asarray(record[key])) for key in _WORK_FIELDS},
        **{key: np.asarray(record[key])[:updates] for key in (
            "coupling_update_mode_history", "iqn_rank_history",
            "iqn_condition_number_history", "iqn_fallback_reason_history",
            "iqn_update_limited_history")},
    }
    return _finite_json(payload)


def assess_periodic_history_csv(
    history_csv: str | Path,
    config: Fsi1AcceptanceConfig,
    *,
    coupling_relative_residual_max: float,
    accepted_records: Iterable[Mapping[str, Any]],
    span_m: float,
    health_only: bool = False,
    marker_segments: Any = None,
) -> dict[str, Any]:
    """Assess fixed-length accepted evidence; no runtime, resume or early stop.

    The caller supplies integrity-validated accepted records in order. This
    function additionally binds their state, force and residuals to the CSV.
    Full trial histories remain in the chunks; the last three complete cycles
    are emitted with actual Point A velocity, work and IQN diagnostics.
    """
    if not np.isfinite(span_m) or span_m <= 0:
        raise ValueError("span_m must be finite and positive")
    policy = dynamic_numerical_policy(coupling_relative_residual_max)
    rows, violations = read_numerical_history_csv(
        history_csv, config, policy=policy,
    )
    times = np.asarray([row["time_s"] for row in rows])
    signals = {
        "point_a_ux_m": np.asarray([row["tip_ux_turek_hron_m"] for row in rows]),
        "point_a_uy_m": np.asarray([row["tip_uy_turek_hron_m"] for row in rows]),
        "total_drag_n": np.asarray([row["total_drag_per_span_n_per_m"] for row in rows]),
        "total_lift_n": np.asarray([row["total_lift_per_span_n_per_m"] for row in rows]),
    }
    limit_cycle, cycle_error = None, None
    if not health_only:
        try:
            limit_cycle = analyze_limit_cycle(times, *signals.values())
        except LimitCycleValidationError as error:
            cycle_error = str(error)
    cycles = () if limit_cycle is None else limit_cycle.cycles
    work_totals = dict.fromkeys(_WORK_FIELDS, 0)
    trial_distribution: Counter = Counter()
    telemetry = []
    count = 0
    for count, record in enumerate(accepted_records, start=1):
        if count > len(rows):
            raise TurekHronAcceptanceError("accepted record count exceeds history")
        row = rows[count - 1]
        validate_record_history_pair(record, row, span_m=span_m)
        for field in _WORK_FIELDS:
            work_totals[field] += int(np.asarray(record[field]))
        trial_distribution[int(np.asarray(record["coupling_trial_count"]))] += 1
        if health_only or (cycles and cycles[0].start_s <= row["time_s"] <= cycles[-1].end_s):
            telemetry.append(_telemetry(record, row, cycles, marker_segments))
    if count != len(rows):
        raise TurekHronAcceptanceError("accepted record count does not match history")
    metrics = {}
    if limit_cycle is not None:
        last = cycles[-1]
        metrics = {
            "point_a_uy_amplitude_m": last.signals["point_a_uy_m"].amplitude,
            "point_a_uy_frequency_hz": limit_cycle.primary_frequency_hz,
            "total_drag_midrange_n": last.signals["total_drag_n"].midrange,
            "total_lift_amplitude_n": last.signals["total_lift_n"].amplitude,
        }
    stable = limit_cycle is not None and limit_cycle.stability.passed
    status = ("numerical_contract_failed" if violations else
              "health_only_passed" if health_only else
              "insufficient_complete_cycles" if cycle_error else
              "limit_cycle_stable" if stable else "limit_cycle_unstable")
    return {
        "status": status, "history_csv": str(history_csv),
        "completed_steps": len(rows), "dt_s": config.expected_dt_s,
        "observed_end_time_s": float(times[-1]), "history_schema_version": 4,
        "numerical_contract_passed": not violations,
        "numerical_health_policy": asdict(policy),
        "health_only": health_only, "limit_cycle_stable": stable,
        "acceptance_passed": not violations and (health_only or stable),
        "violations": [*violations, *([] if health_only or stable else [cycle_error or "limit-cycle stability gate failed"])],
        "metrics": metrics,
        "limit_cycle": None if limit_cycle is None else _cycle_payload(limit_cycle, times, signals),
        "work_totals": work_totals,
        "coupling_trial_distribution": {str(key): value for key, value in sorted(trial_distribution.items())},
        "cycle_telemetry": telemetry,
        "force_semantics": {
            "history": "total drag/lift per span in N/m; cylinder plus beam",
            "accepted": "solver xyz force in N for the actual slab span",
            "slab_span_m": span_m, "reference_span_m": 1.0,
            "reference": "2D unit-span force in N, numerically equal to N/m history",
        },
        "relaxation_semantics": {
            "method": "fixed initial Picard relaxation; subsequent IQN updates use their recorded limiter",
            "adaptive_relaxation": False,
            "update_limiting_field": "iqn_update_limited_history",
        },
        "quality_boundary": "periodic health and cycle evidence; reference and cross-run gates are assessed separately",
    }
