"""Fail-closed validation for one R24C no-commit threshold probe."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .oracle_threshold_common import (
    OracleThresholdContractError,
    THRESHOLD_ALPHAS,
    require,
)


_PROBE_HISTORY_FIELDS = (
    "relative_residual_history",
    "absolute_residual_history_mps",
    "candidate_velocity_rms_history_mps",
    "max_marker_residual_history_mps",
    "relative_tolerance_equivalent_history_mps",
    "effective_tolerance_history_mps",
    "residual_to_effective_tolerance_history",
)
_PROBE_UPDATE_FIELDS = (
    "update_mode_history",
    "iqn_rank_history",
    "iqn_condition_number_history",
    "iqn_fallback_reasons",
    "iqn_update_limited_history",
)
_PROBE_WORK_COUNT_FIELDS = (
    "trial_count",
    "fluid_solve_count",
    "solid_macro_solve_count",
    "feedback_consumed_trial_count",
    "cg_iterations_total",
    "flow_momentum_advection_substeps_total",
    "flow_sst_transport_substeps_total",
    "solid_substeps_executed_total",
)
_PROBE_WORK_TIME_FIELDS = (
    "flow_wall_time_s_total",
    "hibm_wall_time_s_total",
    "solid_wall_time_s_total",
)
_FROZEN_DT_S = 5.0e-4
_FROZEN_RELATIVE_TOLERANCE = 1.0e-3


def _number(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise OracleThresholdContractError(f"{label} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OracleThresholdContractError(f"{label} must be numeric") from exc
    require(math.isfinite(converted), f"{label} must be finite")
    if minimum is not None:
        require(converted >= minimum, f"{label} must be at least {minimum}")
    return converted


def positive_integer(value: object, *, label: str) -> int:
    converted = _number(value, label=label, minimum=1.0)
    integer = int(converted)
    require(converted == integer, f"{label} must be an integer")
    return integer


def _nonnegative_integer(value: object, *, label: str) -> int:
    converted = _number(value, label=label, minimum=0.0)
    integer = int(converted)
    require(converted == integer, f"{label} must be an integer")
    return integer


def _numeric_history(
    row: Mapping[str, Any],
    field: str,
    *,
    expected_length: int,
    label: str,
) -> list[float]:
    raw = row.get(field)
    require(isinstance(raw, list), f"{label} {field} must be a list")
    require(
        len(raw) == expected_length,
        f"{label} {field} length disagrees with iterations",
    )
    return [
        _number(value, label=f"{label} {field}[{index}]")
        for index, value in enumerate(raw)
    ]


def _validate_solid_trial_reports(
    reports: object,
    *,
    iterations: int,
    work: Mapping[str, Any],
    arm_label: str,
) -> None:
    require(
        isinstance(reports, list) and len(reports) == iterations,
        f"{arm_label} solid trial reports disagree with iterations",
    )
    total_substeps = 0
    total_wall_time_s = 0.0
    for index, raw in enumerate(reports):
        label = f"{arm_label} solid trial {index}"
        require(isinstance(raw, Mapping), f"{label} report is missing")
        requested = _number(raw.get("requested_macro_dt_s"), label=f"{label} dt")
        accepted = _number(raw.get("solid_accepted_time_s"), label=f"{label} accepted time")
        remaining = _number(
            raw.get("solid_remaining_unadvanced_time_s"),
            label=f"{label} remaining time",
        )
        selected = positive_integer(
            raw.get("solid_substeps_selected"), label=f"{label} selected substeps"
        )
        accepted_substeps = positive_integer(
            raw.get("solid_accepted_substep_count"),
            label=f"{label} accepted substeps",
        )
        executed = positive_integer(
            raw.get("solid_substeps_executed_total"),
            label=f"{label} executed substeps",
        )
        substep_dt = _number(
            raw.get("solid_substep_dt_s"),
            label=f"{label} substep dt",
            minimum=0.0,
        )
        rejected = _nonnegative_integer(
            raw.get("solid_rejected_trial_count"),
            label=f"{label} rejected trials",
        )
        retries = _nonnegative_integer(
            raw.get("solid_retry_count"), label=f"{label} retry count"
        )
        wall_time = _number(
            raw.get("solid_wall_time_s"), label=f"{label} wall time", minimum=0.0
        )
        require(
            math.isclose(requested, _FROZEN_DT_S, rel_tol=0.0, abs_tol=1.0e-15)
            and math.isclose(accepted, _FROZEN_DT_S, rel_tol=0.0, abs_tol=1.0e-15)
            and abs(remaining) <= 1.0e-14,
            f"{label} solid trial physical time is incomplete",
        )
        require(
            selected == accepted_substeps == executed
            and rejected == 0
            and retries == 0
            and math.isclose(
                substep_dt * accepted_substeps,
                _FROZEN_DT_S,
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            ),
            f"{label} solid trial substep accounting disagrees",
        )
        require(
            raw.get("solid_wall_time_synchronized") is True,
            f"{label} solid wall time is not synchronized",
        )
        total_substeps += executed
        total_wall_time_s += wall_time
    require(
        total_substeps == int(work["solid_substeps_executed_total"]),
        f"{arm_label} solid trial substeps disagree with trial work",
    )
    require(
        math.isclose(
            total_wall_time_s,
            float(work["solid_wall_time_s_total"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        f"{arm_label} solid trial wall time disagrees with trial work",
    )


def _validate_probe_row(
    row: Mapping[str, Any],
    *,
    expected_alpha: float,
    arm_label: str,
) -> dict[str, Any]:
    alpha = _number(row.get("alpha"), label=f"{arm_label} alpha")
    require(
        math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=1.0e-15),
        f"{arm_label} alpha sequence mismatch",
    )
    require(
        row.get("baseline_mode") == "carry_forward",
        f"{arm_label} baseline must be carry_forward",
    )
    require(row.get("converged") is True, f"{arm_label} probe did not converge")
    iterations = positive_integer(
        row.get("iterations"),
        label=f"{arm_label} iterations",
    )
    require(iterations <= 16, f"{arm_label} iterations exceed the frozen limit")
    histories = {
        field: _numeric_history(
            row,
            field,
            expected_length=iterations,
            label=arm_label,
        )
        for field in _PROBE_HISTORY_FIELDS
    }
    absolute = histories["absolute_residual_history_mps"]
    relative = histories["relative_residual_history"]
    candidate = histories["candidate_velocity_rms_history_mps"]
    maximum = histories["max_marker_residual_history_mps"]
    relative_equivalent = histories[
        "relative_tolerance_equivalent_history_mps"
    ]
    effective = histories["effective_tolerance_history_mps"]
    ratios = histories["residual_to_effective_tolerance_history"]
    require(
        all(value >= 0.0 for value in (*absolute, *relative, *candidate, *maximum))
        and all(value > 0.0 for value in (*relative_equivalent, *effective))
        and all(value >= 0.0 for value in ratios),
        f"{arm_label} residual or tolerance history invalid",
    )
    for index, (
        residual,
        relative_residual,
        candidate_rms,
        relative_tolerance,
        tolerance,
        ratio,
    ) in enumerate(
        zip(
            absolute,
            relative,
            candidate,
            relative_equivalent,
            effective,
            ratios,
        )
    ):
        candidate_scale = max(candidate_rms, 1.0e-30)
        require(
            math.isclose(
                relative_residual,
                residual / candidate_scale,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ),
            f"{arm_label} relative residual mismatch at {index}",
        )
        expected_relative_tolerance = (
            _FROZEN_RELATIVE_TOLERANCE * candidate_scale
        )
        require(
            math.isclose(
                relative_tolerance,
                expected_relative_tolerance,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                tolerance,
                relative_tolerance,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            f"{arm_label} relative tolerance contract mismatch at {index}",
        )
        require(
            math.isclose(
                ratio,
                residual / tolerance,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ),
            f"{arm_label} residual/tolerance ratio mismatch at {index}",
        )
    require(
        all(ratio > 1.0 for ratio in ratios[:-1])
        and ratios[-1] <= 1.0 + 1.0e-12
        and relative[-1] <= _FROZEN_RELATIVE_TOLERANCE + 1.0e-12,
        f"{arm_label} terminal convergence is not reverified",
    )
    first_absolute = _number(
        row.get("first_absolute_residual_mps"),
        label=f"{arm_label} first absolute residual",
    )
    first_relative = _number(
        row.get("first_relative_residual"),
        label=f"{arm_label} first relative residual",
    )
    require(
        first_absolute == absolute[0] and first_relative == relative[0],
        f"{arm_label} first residual fields disagree",
    )
    if iterations == 1:
        require(
            row.get("second_absolute_residual_mps") is None
            and row.get("second_relative_residual") is None,
            f"{arm_label} one-trial probe has a second residual",
        )
    else:
        second_absolute = _number(
            row.get("second_absolute_residual_mps"),
            label=f"{arm_label} second absolute residual",
        )
        second_relative = _number(
            row.get("second_relative_residual"),
            label=f"{arm_label} second relative residual",
        )
        require(
            second_absolute == absolute[1] and second_relative == relative[1],
            f"{arm_label} second residual fields disagree",
        )

    update_length = max(iterations - 1, 0)
    for field in _PROBE_UPDATE_FIELDS:
        history = row.get(field)
        require(isinstance(history, list), f"{arm_label} {field} must be a list")
        require(
            len(history) == update_length,
            f"{arm_label} {field} length disagrees with iterations",
        )
    require(
        all(isinstance(mode, str) and mode for mode in row["update_mode_history"]),
        f"{arm_label} update modes are invalid",
    )
    for index, rank in enumerate(row["iqn_rank_history"]):
        _nonnegative_integer(rank, label=f"{arm_label} IQN rank {index}")
    for index, condition in enumerate(row["iqn_condition_number_history"]):
        if condition is not None:
            _number(
                condition,
                label=f"{arm_label} IQN condition {index}",
                minimum=0.0,
            )
    require(
        all(isinstance(value, bool) for value in row["iqn_update_limited_history"]),
        f"{arm_label} IQN update-limited history is invalid",
    )
    fallback_reasons = row.get("iqn_fallback_reasons")
    require(
        isinstance(fallback_reasons, list)
        and all(
            value is None or isinstance(value, str) and bool(value)
            for value in fallback_reasons
        ),
        f"{arm_label} IQN fallback reasons are invalid",
    )
    fallback_count = _nonnegative_integer(
        row.get("iqn_fallback_count"),
        label=f"{arm_label} IQN fallback count",
    )
    require(
        fallback_count == sum(value is not None for value in fallback_reasons),
        f"{arm_label} IQN fallback evidence disagrees",
    )

    work = row.get("trial_work")
    require(isinstance(work, Mapping), f"{arm_label} trial work is missing")
    work_counts = {
        field: _nonnegative_integer(
            work.get(field),
            label=f"{arm_label} trial work {field}",
        )
        for field in _PROBE_WORK_COUNT_FIELDS
    }
    for field in _PROBE_WORK_TIME_FIELDS:
        _number(
            work.get(field),
            label=f"{arm_label} trial work {field}",
            minimum=0.0,
        )
    require(
        work_counts["trial_count"] == iterations
        and work_counts["fluid_solve_count"] == iterations
        and work_counts["solid_macro_solve_count"] == iterations,
        f"{arm_label} trial work disagrees with iterations",
    )
    require(
        work_counts["feedback_consumed_trial_count"] == iterations,
        f"{arm_label} feedback work disagrees with iterations",
    )
    require(
        all(
            work_counts[field] > 0
            for field in (
                "cg_iterations_total",
                "flow_momentum_advection_substeps_total",
                "flow_sst_transport_substeps_total",
                "solid_substeps_executed_total",
            )
        ),
        f"{arm_label} trial work must be positive",
    )
    _validate_solid_trial_reports(
        row.get("solid_trial_reports"),
        iterations=iterations,
        work=work,
        arm_label=arm_label,
    )
    require(
        row.get("rollback_host_macro_step_state_equal") is True
        and row.get("rollback_host_macro_step_state_mismatch_fields") == [],
        f"{arm_label} rollback equality failed",
    )
    return dict(row)


def validate_probe_report(
    report: Mapping[str, Any],
    *,
    omega: float,
    target_step: int,
) -> list[dict[str, Any]]:
    """Validate one terminal arm and return its complete ordered rows."""

    label = f"omega {omega} target {target_step}"
    require(
        report.get("status") == "research_probe_terminal"
        and report.get("research_probe_terminal") is True,
        f"{label} is not a terminal research probe",
    )
    require(
        report.get("offline_oracle") is True
        and report.get("deployable") is False,
        f"{label} oracle boundary is invalid",
    )
    require(
        report.get("accepted_step_count") == target_step - 1,
        f"{label} accepted step count changed",
    )
    accepted_time = _number(
        report.get("accepted_time_s"),
        label=f"{label} accepted time",
    )
    require(
        math.isclose(
            accepted_time,
            (target_step - 1) * 5.0e-4,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        f"{label} accepted time changed",
    )
    require(
        report.get("research_probe_all_rollbacks_equal") is True,
        f"{label} aggregate rollback equality failed",
    )
    require(
        report.get("research_probe_sweep_state_equal") is True
        and report.get("research_probe_sweep_state_mismatch_fields") == [],
        f"{label} sweep rollback equality failed",
    )
    raw_rows = report.get("research_probe_rows")
    require(
        isinstance(raw_rows, list) and len(raw_rows) == len(THRESHOLD_ALPHAS),
        f"{label} alpha rows are incomplete",
    )
    rows = [
        _validate_probe_row(
            row,
            expected_alpha=alpha,
            arm_label=f"{label} alpha {alpha}",
        )
        for alpha, row in zip(THRESHOLD_ALPHAS, raw_rows)
        if isinstance(row, Mapping)
    ]
    require(len(rows) == len(THRESHOLD_ALPHAS), f"{label} alpha row type invalid")
    return rows


__all__ = ("positive_integer", "validate_probe_report")
