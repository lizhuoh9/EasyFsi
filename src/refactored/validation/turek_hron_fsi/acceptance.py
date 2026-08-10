"""Fail-closed, offline acceptance analysis for Turek--Hron FSI1 histories.

This module deliberately has no solver imports.  It consumes only the committed
per-step CSV certificate, so running it cannot initialize Taichi, mutate a run,
or conceal a missing physical/numerical certificate behind a summary value.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class TurekHronAcceptanceError(ValueError):
    """Raised when a history cannot serve as acceptance evidence."""


@dataclass(frozen=True)
class Fsi1AcceptanceConfig:
    """Acceptance contract and explicitly reviewable physical thresholds."""

    expected_steps: int
    ramp_duration_s: float = 2.0
    settling_duration_s: float = 2.0
    steady_window_duration_s: float = 2.0
    fixed_root_max_displacement_m: float = 1.0e-8
    flux_imbalance_rel_max: float = 1.0e-2
    flux_imbalance_rel_mean_max: float = 5.0e-3
    cg_relative_residual_max: float = 1.0e-3
    projection_l2_max: float = 1.0e-1
    projection_max_abs_max: float = 10.0
    coupling_absolute_residual_mps_max: float = 1.0e-4
    coupling_max_marker_residual_mps_max: float = 1.0e-3
    scatter_action_reaction_residual_n_max: float = 1.0e-6
    steady_window_mean_drift_rel_max: float = 1.0e-2
    steady_p05_p95_span_rel_max: float = 5.0e-2
    steady_slope_change_rel_max: float = 1.0e-2
    canonical_relative_error_max: float = 5.0e-2

    def __post_init__(self) -> None:
        if (
            isinstance(self.expected_steps, bool)
            or not isinstance(self.expected_steps, Integral)
            or int(self.expected_steps) <= 0
        ):
            raise ValueError("expected_steps must be a positive integer")
        positive = (
            "ramp_duration_s",
            "settling_duration_s",
            "steady_window_duration_s",
        )
        nonnegative = tuple(
            name
            for name in self.__dataclass_fields__
            if name.endswith(("_max", "_m", "_mps", "_n_max"))
            and name not in positive
        )
        for name in positive:
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"{name} must be a real number")
            value = float(raw_value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in nonnegative:
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"{name} must be a real number")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")


METRIC_FIELDS = (
    "tip_ux_turek_hron_m",
    "tip_uy_turek_hron_m",
    "total_drag_per_span_n_per_m",
    "total_lift_per_span_n_per_m",
)

HISTORY_SCHEMA_VERSION = 3

CANONICAL_FSI1_REFERENCE: Mapping[str, float] = {
    "tip_ux_turek_hron_m": 2.27e-5,
    "tip_uy_turek_hron_m": 8.209e-4,
    "total_drag_per_span_n_per_m": 14.295,
    "total_lift_per_span_n_per_m": 0.7638,
}

LOCAL_LS_DYNA_REFERENCE: Mapping[str, float] = {
    "tip_ux_turek_hron_m": 1.7e-5,
    "tip_uy_turek_hron_m": 8.6e-4,
    "total_drag_per_span_n_per_m": 14.26,
    "total_lift_per_span_n_per_m": 0.73,
}

LOCAL_LS_DYNA_UNCERTAINTY: Mapping[str, float] = {
    "total_lift_per_span_n_per_m": 0.30,
}

_INTEGER_FIELDS = (
    "step",
    "history_schema_version",
    "stress_valid_marker_count",
    "stress_invalid_marker_count",
    "stress_viscous_gradient_invalid_marker_count",
    "stress_one_sided_pressure_marker_count",
    "stress_expected_marker_count",
    "projection_cg_breakdown_count",
    "post_solid_projection_cg_project_calls",
    "post_solid_projection_cg_breakdown_count",
    "post_solid_no_slip_valid_marker_count",
    "post_solid_no_slip_invalid_marker_count",
    "marker_total_count",
    "mpm_scatter_active_marker_count",
    "mpm_scatter_invalid_marker_count",
    "mpm_scatter_active_pair_count",
)

_FLOAT_FIELDS = (
    "time_s",
    "ramp_factor",
    *METRIC_FIELDS,
    "fixed_root_max_displacement_m",
    "fsi_coupling_absolute_residual_mps",
    "flux_imbalance_rel",
    "projection_cg_relative_residual_max",
    "post_solid_projection_l2",
    "post_solid_projection_max_abs",
    "post_solid_projection_cg_relative_residual_max",
    "post_solid_no_slip_max_mps",
    "post_solid_no_slip_l2_mps",
    "mpm_scatter_action_reaction_residual_n",
    "fsi_coupling_max_marker_residual_mps",
)

_BOOLEAN_FIELDS = (
    "fsi_coupling_residual_measured",
    "fsi_coupling_converged",
    "projection_cg_converged_all",
    "post_solid_projection_applied",
    "post_solid_projection_report_available",
    "post_solid_projection_cg_converged_all",
    "post_solid_projection_pressure_solve_failed",
    "post_solid_projection_physical_failure",
    "post_solid_no_slip_report_available",
    "mechanism_probe_enabled",
    "mechanism_probe_triggered",
)

_STRING_FIELDS = ("post_solid_projection_pressure_solver",)

REQUIRED_FIELDS = frozenset(
    (*_INTEGER_FIELDS, *_FLOAT_FIELDS, *_BOOLEAN_FIELDS, *_STRING_FIELDS)
)


def _strict_bool(value: object, *, field: str, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise TurekHronAcceptanceError(
        f"row {row_number} field {field!r} must be a strict boolean; got {value!r}"
    )


def _finite_float(value: object, *, field: str, row_number: int) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TurekHronAcceptanceError(
            f"row {row_number} field {field!r} must be a finite numeric value; "
            f"got {value!r}"
        ) from exc
    if not np.isfinite(converted):
        raise TurekHronAcceptanceError(
            f"row {row_number} field {field!r} must be a finite numeric value; "
            f"got {value!r}"
        )
    return converted


def _typed_history_rows(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        handle = path.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise TurekHronAcceptanceError(f"cannot read history CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise TurekHronAcceptanceError("history CSV has no header")
        duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if duplicates:
            raise TurekHronAcceptanceError(
                f"history CSV has duplicate columns: {duplicates}"
            )
        missing = sorted(REQUIRED_FIELDS.difference(fieldnames))
        if missing:
            raise TurekHronAcceptanceError(
                f"history CSV has missing required columns: {missing}"
            )
        raw_rows = tuple(reader)
    if not raw_rows:
        raise TurekHronAcceptanceError("history CSV has no data rows")

    typed: list[dict[str, Any]] = []
    for row_number, raw in enumerate(raw_rows, start=2):
        floats = {
            field: _finite_float(raw.get(field), field=field, row_number=row_number)
            for field in _FLOAT_FIELDS
        }
        integers: dict[str, int] = {}
        for field in _INTEGER_FIELDS:
            value = _finite_float(raw.get(field), field=field, row_number=row_number)
            if not value.is_integer():
                raise TurekHronAcceptanceError(
                    f"row {row_number} field {field!r} must be an integer; got {value!r}"
                )
            integers[field] = int(value)
        booleans = {
            field: _strict_bool(raw.get(field), field=field, row_number=row_number)
            for field in _BOOLEAN_FIELDS
        }
        strings = {field: str(raw.get(field, "")).strip() for field in _STRING_FIELDS}
        typed.append({**floats, **integers, **booleans, **strings})
    return tuple(typed)


def _validate_history_identity(
    rows: Sequence[Mapping[str, Any]], config: Fsi1AcceptanceConfig
) -> float:
    if len(rows) != int(config.expected_steps):
        raise TurekHronAcceptanceError(
            f"history row count {len(rows)} does not match expected_steps "
            f"{config.expected_steps}"
        )
    steps = tuple(int(row["step"]) for row in rows)
    expected = tuple(range(1, int(config.expected_steps) + 1))
    if steps != expected:
        mismatch_index = next(
            index
            for index, pair in enumerate(zip(steps, expected))
            if pair[0] != pair[1]
        )
        raise TurekHronAcceptanceError(
            "history must contain continuous steps 1..expected_steps exactly; "
            f"first mismatch is at CSV row {mismatch_index + 2}"
        )
    schema_steps = [
        int(row["step"])
        for row in rows
        if int(row["history_schema_version"]) != HISTORY_SCHEMA_VERSION
    ]
    if schema_steps:
        raise TurekHronAcceptanceError(
            f"FSI1 acceptance requires history schema version 3 on every row; "
            f"mismatch at steps {_step_preview(schema_steps)}"
        )
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise TurekHronAcceptanceError("history time_s must be strictly increasing")
    dt_s = float(times[0])
    expected_times = np.arange(1, len(rows) + 1, dtype=np.float64) * dt_s
    tolerance = max(1.0e-12, abs(dt_s) * 1.0e-8)
    if dt_s <= 0.0 or not np.allclose(times, expected_times, rtol=1.0e-9, atol=tolerance):
        raise TurekHronAcceptanceError(
            "history time_s must be uniform and consistent with one-based steps"
        )
    return dt_s


def _step_preview(steps: Iterable[int]) -> str:
    values = tuple(int(value) for value in steps)
    preview = ", ".join(str(value) for value in values[:8])
    return f"[{preview}{', ...' if len(values) > 8 else ''}]"


def _failed_steps(
    rows: Sequence[Mapping[str, Any]], predicate: Any
) -> tuple[int, ...]:
    return tuple(int(row["step"]) for row in rows if predicate(row))


def _numerical_contract_violations(
    rows: Sequence[Mapping[str, Any]], config: Fsi1AcceptanceConfig
) -> tuple[str, ...]:
    expected_marker_count = int(rows[0]["stress_expected_marker_count"])
    checks = (
        (
            "mechanism probe",
            lambda row: not row["mechanism_probe_enabled"]
            or row["mechanism_probe_triggered"],
        ),
        (
            "inlet ramp",
            lambda row: row["ramp_factor"] < 0.0
            or row["ramp_factor"] > 1.0 + 1.0e-12
            or (
                row["time_s"] >= config.ramp_duration_s
                and row["ramp_factor"] < 1.0 - 1.0e-9
            ),
        ),
        (
            "coupling convergence",
            lambda row: not row["fsi_coupling_residual_measured"]
            or not row["fsi_coupling_converged"]
            or row["fsi_coupling_absolute_residual_mps"] < 0.0
            or row["fsi_coupling_absolute_residual_mps"]
            > config.coupling_absolute_residual_mps_max
            or row["fsi_coupling_max_marker_residual_mps"] < 0.0
            or row["fsi_coupling_max_marker_residual_mps"]
            > config.coupling_max_marker_residual_mps_max,
        ),
        (
            "main projection CG",
            lambda row: not row["projection_cg_converged_all"]
            or row["projection_cg_breakdown_count"] != 0
            or row["projection_cg_relative_residual_max"] < 0.0
            or row["projection_cg_relative_residual_max"]
            > config.cg_relative_residual_max,
        ),
        (
            "post-solid projection CG",
            lambda row: not row["post_solid_projection_applied"]
            or not row["post_solid_projection_report_available"]
            or row["post_solid_projection_pressure_solver"] != "fv_cg"
            or row["post_solid_projection_cg_project_calls"] < 1
            or not row["post_solid_projection_cg_converged_all"]
            or row["post_solid_projection_cg_breakdown_count"] != 0
            or row["post_solid_projection_cg_relative_residual_max"] < 0.0
            or row["post_solid_projection_cg_relative_residual_max"]
            > config.cg_relative_residual_max
            or row["post_solid_projection_pressure_solve_failed"]
            or row["post_solid_projection_physical_failure"]
            or row["post_solid_projection_l2"] < 0.0
            or row["post_solid_projection_l2"] > config.projection_l2_max
            or row["post_solid_projection_max_abs"] < 0.0
            or row["post_solid_projection_max_abs"] > config.projection_max_abs_max,
        ),
        (
            "near-wall marker sampling coverage",
            lambda row: not row["post_solid_no_slip_report_available"]
            or row["post_solid_no_slip_valid_marker_count"]
            != row["stress_expected_marker_count"]
            or row["post_solid_no_slip_invalid_marker_count"] != 0
            or row["post_solid_no_slip_max_mps"] < 0.0
            or row["post_solid_no_slip_l2_mps"] < 0.0,
        ),
        (
            "stress marker certificate",
            lambda row: row["stress_expected_marker_count"]
            != expected_marker_count
            or row["stress_expected_marker_count"] <= 0
            or row["stress_valid_marker_count"] != row["stress_expected_marker_count"]
            or row["stress_invalid_marker_count"] != 0
            or row["stress_viscous_gradient_invalid_marker_count"] != 0
            or row["stress_one_sided_pressure_marker_count"]
            != row["stress_expected_marker_count"]
            or row["marker_total_count"] != row["stress_expected_marker_count"],
        ),
        (
            "fixed-root displacement",
            lambda row: row["fixed_root_max_displacement_m"] < 0.0
            or row["fixed_root_max_displacement_m"]
            > config.fixed_root_max_displacement_m,
        ),
        (
            "scatter marker/action-reaction certificate",
            lambda row: row["mpm_scatter_active_marker_count"]
            != row["stress_expected_marker_count"]
            or row["mpm_scatter_invalid_marker_count"] != 0
            or row["mpm_scatter_active_pair_count"]
            < row["mpm_scatter_active_marker_count"]
            or abs(row["mpm_scatter_action_reaction_residual_n"])
            > config.scatter_action_reaction_residual_n_max,
        ),
        (
            "flux imbalance",
            lambda row: row["ramp_factor"] >= 1.0 - 1.0e-9
            and abs(row["flux_imbalance_rel"])
            > config.flux_imbalance_rel_max,
        ),
    )
    violations: list[str] = []
    for label, predicate in checks:
        steps = _failed_steps(rows, predicate)
        if steps:
            violations.append(f"{label} gate failed at steps {_step_preview(steps)}")
    full_load_flux = tuple(
        abs(float(row["flux_imbalance_rel"]))
        for row in rows
        if float(row["ramp_factor"]) >= 1.0 - 1.0e-9
    )
    if (
        full_load_flux
        and float(np.mean(full_load_flux)) > config.flux_imbalance_rel_mean_max
    ):
        violations.append(
            "full-load mean flux imbalance gate failed: "
            f"{float(np.mean(full_load_flux)):.6g} > "
            f"{config.flux_imbalance_rel_mean_max:.6g}"
        )
    return tuple(violations)


def _window_stats(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = times - float(np.mean(times))
        denominator = float(np.dot(centered, centered))
        mean = float(np.mean(values))
        slope = (
            float(np.dot(centered, values - mean) / denominator)
            if denominator > 0.0
            else 0.0
        )
        p05, p95 = np.percentile(values, (5.0, 95.0))
        statistics = {
            "mean": mean,
            "p05": float(p05),
            "p95": float(p95),
            "p05_p95_span": float(p95 - p05),
            "slope_per_s": slope,
        }
    if not all(np.isfinite(value) for value in statistics.values()):
        raise TurekHronAcceptanceError(
            "finite input values produced non-finite derived window statistics"
        )
    return {
        **statistics,
        "sample_count": int(values.size),
    }


def _metric_report(
    *,
    field: str,
    previous_times: np.ndarray,
    previous_values: np.ndarray,
    late_times: np.ndarray,
    late_values: np.ndarray,
    config: Fsi1AcceptanceConfig,
) -> dict[str, Any]:
    previous = _window_stats(previous_times, previous_values)
    late = _window_stats(late_times, late_values)
    scale = abs(float(CANONICAL_FSI1_REFERENCE[field]))
    drift_abs = abs(float(late["mean"]) - float(previous["mean"]))
    drift_rel = drift_abs / scale
    span_rel = float(late["p05_p95_span"]) / scale
    slope_change_rel = (
        abs(float(late["slope_per_s"])) * float(config.steady_window_duration_s) / scale
    )
    if not all(np.isfinite(value) for value in (drift_abs, drift_rel, span_rel, slope_change_rel)):
        raise TurekHronAcceptanceError(
            f"finite input values produced non-finite derived metrics for {field}"
        )
    stability_violations: list[str] = []
    if drift_rel > config.steady_window_mean_drift_rel_max:
        stability_violations.append("adjacent-window mean drift")
    if span_rel > config.steady_p05_p95_span_rel_max:
        stability_violations.append("late-window p05-p95 span")
    if slope_change_rel > config.steady_slope_change_rel_max:
        stability_violations.append("late-window slope")
    return {
        **late,
        "previous_window": previous,
        "window_mean_drift_abs": drift_abs,
        "window_mean_drift_rel": drift_rel,
        "p05_p95_span_rel": span_rel,
        "slope_change_rel_per_window": slope_change_rel,
        "stable": not stability_violations,
        "stability_violations": tuple(stability_violations),
    }


def _reference_ledger(
    metrics: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, float],
    uncertainties: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    uncertainty_map = uncertainties or {}
    ledger: dict[str, dict[str, float]] = {}
    for field in METRIC_FIELDS:
        mean = float(metrics[field]["mean"])
        reference = float(references[field])
        signed_error = mean - reference
        relative_error_percent = abs(signed_error) / abs(reference) * 100.0
        if not all(
            np.isfinite(value)
            for value in (mean, reference, signed_error, relative_error_percent)
        ):
            raise TurekHronAcceptanceError(
                f"non-finite derived reference comparison for {field}"
            )
        entry = {
            "reference": reference,
            "mean": mean,
            "signed_error": signed_error,
            "relative_error_percent": relative_error_percent,
        }
        if field in uncertainty_map:
            entry["uncertainty"] = float(uncertainty_map[field])
        ledger[field] = entry
    return ledger


def assess_fsi1_history_csv(
    history_csv: str | Path,
    config: Fsi1AcceptanceConfig,
) -> dict[str, Any]:
    """Assess one immutable CSV artifact and return a JSON-serializable report.

    Malformed or incomplete evidence raises :class:`TurekHronAcceptanceError`.
    Valid evidence that fails a numerical, steady, or canonical-reference gate
    returns a report with ``acceptance_passed=False`` and explicit violations.
    """

    path = Path(history_csv)
    rows = _typed_history_rows(path)
    dt_s = _validate_history_identity(rows, config)
    observed_end_time_s = float(rows[-1]["time_s"])
    numerical_violations = _numerical_contract_violations(rows, config)
    base: dict[str, Any] = {
        "history_csv": str(path),
        "completed_steps": len(rows),
        "dt_s": dt_s,
        "observed_end_time_s": observed_end_time_s,
        "history_schema_version": HISTORY_SCHEMA_VERSION,
        "numerical_contract_passed": not numerical_violations,
        "near_wall_marker_velocity_sample": {
            # The legacy report samples the cell-centred velocity at the
            # geometric marker with obstacle cells omitted and the surviving
            # trilinear weights renormalized.  At a sharp embedded boundary
            # this is a one-sided near-wall fluid sample, not the extrapolated
            # boundary value enforced by the reconstructed Dirichlet row.
            # Gate sampling coverage/finite values above, but never relabel the
            # magnitude as a formal no-slip residual.
            "formal_no_slip_gate": False,
            "max_mps": max(
                float(row["post_solid_no_slip_max_mps"]) for row in rows
            ),
            "l2_max_mps": max(
                float(row["post_solid_no_slip_l2_mps"]) for row in rows
            ),
            "interpretation": (
                "diagnostic truncated one-sided near-wall fluid sample; "
                "not the reconstructed sharp-interface boundary residual"
            ),
        },
        "steady_window_requirement": {
            "ramp_duration_s": float(config.ramp_duration_s),
            "settling_duration_s": float(config.settling_duration_s),
            "window_duration_s": float(config.steady_window_duration_s),
            "adjacent_window_count": 2,
            "minimum_end_time_s": float(
                config.ramp_duration_s
                + config.settling_duration_s
                + 2.0 * config.steady_window_duration_s
            ),
        },
    }
    if numerical_violations:
        return {
            **base,
            "status": "numerical_contract_failed",
            "acceptance_passed": False,
            "violations": numerical_violations,
        }

    minimum_end_time_s = float(
        config.ramp_duration_s
        + config.settling_duration_s
        + 2.0 * config.steady_window_duration_s
    )
    if observed_end_time_s + 0.5 * dt_s < minimum_end_time_s:
        return {
            **base,
            "status": "insufficient_steady_history",
            "acceptance_passed": False,
            "violations": (
                "history ends before two adjacent post-ramp, post-settling windows are available",
            ),
        }

    window = float(config.steady_window_duration_s)
    late_end = observed_end_time_s
    late_start = late_end - window
    previous_end = late_start
    previous_start = previous_end - window
    settled_start = float(config.ramp_duration_s + config.settling_duration_s)
    if previous_start + 0.5 * dt_s < settled_start:
        raise TurekHronAcceptanceError(
            "internal steady-window selection crossed the required settling interval"
        )

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    # Use identical (start, end] sampling on both adjacent windows.  This
    # avoids double-counting their shared boundary and keeps sample counts
    # equal for a uniform time step.
    previous_mask = (times > previous_start + 0.25 * dt_s) & (
        times <= previous_end + 0.25 * dt_s
    )
    late_mask = (times > late_start + 0.25 * dt_s) & (
        times <= late_end + 0.25 * dt_s
    )
    if int(previous_mask.sum()) < 2 or int(late_mask.sum()) < 2:
        raise TurekHronAcceptanceError("steady windows contain fewer than two samples")

    metrics: dict[str, dict[str, Any]] = {}
    for field in METRIC_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        metrics[field] = _metric_report(
            field=field,
            previous_times=times[previous_mask],
            previous_values=values[previous_mask],
            late_times=times[late_mask],
            late_values=values[late_mask],
            config=config,
        )
    stability_violations = tuple(
        f"{field}: {violation}"
        for field, metric in metrics.items()
        for violation in metric["stability_violations"]
    )
    ledgers = {
        "canonical": _reference_ledger(metrics, CANONICAL_FSI1_REFERENCE),
        "local_ls_dyna": _reference_ledger(
            metrics, LOCAL_LS_DYNA_REFERENCE, LOCAL_LS_DYNA_UNCERTAINTY
        ),
    }
    canonical_violations = tuple(
        f"canonical reference error for {field} is {entry['relative_error_percent']:.6g}%"
        for field, entry in ledgers["canonical"].items()
        if float(entry["relative_error_percent"])
        > 100.0 * float(config.canonical_relative_error_max)
    )
    if stability_violations:
        status = "steady_stability_failed"
        violations = stability_violations
    elif canonical_violations:
        status = "canonical_reference_failed"
        violations = canonical_violations
    else:
        status = "passed"
        violations = ()
    return {
        **base,
        "status": status,
        "acceptance_passed": status == "passed",
        "violations": violations,
        "steady_windows": {
            "previous": {
                "start_s": previous_start,
                "end_s": previous_end,
                "sample_count": int(previous_mask.sum()),
            },
            "late": {
                "start_s": late_start,
                "end_s": late_end,
                "sample_count": int(late_mask.sum()),
            },
        },
        "metrics": metrics,
        "reference_ledgers": ledgers,
        "canonical_reference_passed": not canonical_violations,
        "local_ls_dyna_is_diagnostic_only": True,
    }


__all__ = [
    "CANONICAL_FSI1_REFERENCE",
    "Fsi1AcceptanceConfig",
    "LOCAL_LS_DYNA_REFERENCE",
    "TurekHronAcceptanceError",
    "assess_fsi1_history_csv",
]
