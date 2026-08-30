"""Strict fine50 identity for accepted-state Kalman and IQN history reuse.

The predictor changes only the iteration-zero interface guess.  Kalman state
and IQN secant history must advance through accepted FSI states; solver physics,
pressure gates, and the cross-model diagnostic threshold remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .current_iqn_adaptive_fine_contracts import (
    CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY, EXPECTED_STEPS,
    CurrentIqnAdaptiveFineContractError, _history_common, _integer, _mapping,
    _number, _validate_iqn_adaptive_fine50,
)
from .material_reference_fine_contracts import (
    MATERIAL_AUDIT_FIELDS, _validate_material_reference_fine50)
from .native_fine_final_contracts import (
    FINAL_FINE_EXPORT_IDENTITY, _identity_values_equal)

PROFILE_ID = "kalman_iqn_reuse_fine50_v2"
MATERIAL_PROFILE_ID = "kalman_iqn_reuse_material_reference_fine50_v2"
KALMAN_WARMUP_ACCEPTED_STATES = 6
IQN_HISTORY_LIMIT = 8
IQN_INITIAL_PICARD_RELAXATION = 0.5
IQN_SVD_RELATIVE_CUTOFF = 1.0e-10
IQN_MAX_CONDITION_NUMBER = 1.0e10
IQN_MAX_UPDATE_RATIO = 2.0
IQN_REUSE_RESIDUAL_GROWTH_LIMIT_FACTOR = 4.0
MARKER_TARGET_ABSOLUTE_TOLERANCE_MPS = 1.0e-4
MARKER_TARGET_CLOSURE_TOLERANCE_MPS = 1.1e-6
KALMAN_CONFIG = {
    "initial_rate_variance": [
        0.00013558624595909623,
        0.00013558624595909623,
        0.0027202529931253747,
    ],
    "initial_value_variance": [
        3.3896561489774054e-11,
        3.3896561489774054e-11,
        6.800632482813436e-10,
    ],
    "measurement_variance": [
        3.3896561489774054e-11,
        3.3896561489774054e-11,
        6.800632482813436e-10,
    ],
    "rate_process_noise_spectral_density": [
        0.0,
        46395.47219818346,
        121181.523554833,
    ],
    "warmup_accepted_states": KALMAN_WARMUP_ACCEPTED_STATES,
}
KALMAN_IQN_REUSE_FINE_CONFIG_IDENTITY = {
    **CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY,
    "preflow_steps": 0,
    "preflow_snapshot_input_path": None,
    "initial_guess_mode": "kalman",
    "initial_guess_kalman_config": KALMAN_CONFIG,
    "initial_guess_oracle_path": None,
    "iqn_reuse_previous_step_history": True,
}
MATERIAL_KALMAN_IQN_REUSE_FINE_CONFIG_IDENTITY = {
    **KALMAN_IQN_REUSE_FINE_CONFIG_IDENTITY,
    "surface_transfer_method": "cartesian_reference_adjoint_v1",
    "preserve_marker_area_during_surface_feedback": True,
}
COMPATIBILITY_IQN_REUSE_RESET_REASONS = frozenset((
    "layout_identity_unavailable",
    "layout_identity_mismatch",
    "marker_shape_mismatch",
    "dt_mismatch",
    "config_mismatch",
))
NUMERICAL_IQN_REUSE_RESET_REASONS = frozenset((
    "least_squares_failure",
    "zero_rank_history",
    "rank_deficient_history",
    "ill_conditioned_history",
    "coefficient_norm_limit",
    "reuse_update_limited",
))
KNOWN_IQN_REUSE_RESET_REASONS = (
    COMPATIBILITY_IQN_REUSE_RESET_REASONS
    | NUMERICAL_IQN_REUSE_RESET_REASONS
    | {"source_step_mismatch"}
    | {"residual_growth_limit"}
)
IQN_REUSE_STATE_MACHINE_IDENTITY = {
    "schema": "accepted_iqn_reuse_raw_replay_v1",
    "history_limit": IQN_HISTORY_LIMIT,
    "initial_picard_relaxation": IQN_INITIAL_PICARD_RELAXATION,
    "svd_relative_cutoff": IQN_SVD_RELATIVE_CUTOFF,
    "max_condition_number": IQN_MAX_CONDITION_NUMBER,
    "max_coefficient_norm": None,
    "max_update_ratio": IQN_MAX_UPDATE_RATIO,
    "residual_growth_limit_factor": IQN_REUSE_RESIDUAL_GROWTH_LIMIT_FACTOR,
    "reuse_mode_update_indices": [0],
    "retained_matrix_update_indices": list(range(IQN_HISTORY_LIMIT)),
    "strict_pre_update_reset_reasons": ["residual_growth_limit"],
    "strict_numerical_reset_evidence": "independent_raw_trial_replay",
    "replayed_numerical_reset_reasons": sorted(
        NUMERICAL_IQN_REUSE_RESET_REASONS - {"coefficient_norm_limit"}
    ),
    "rejected_completed_chain_reset_reasons": sorted(
        COMPATIBILITY_IQN_REUSE_RESET_REASONS
        | {"source_step_mismatch", "coefficient_norm_limit"}
    ),
    "compatibility_resets_in_completed_official_chain": [],
    "growth_priority": (
        "strictly_greater_than_factor_discards_retained_before_"
        "convergence_or_update"
    ),
    "retained_fallback_policy": (
        "first_replayed_numerical_fallback_discards_retained;"
        "later_updates_are_local_only"
    ),
    "fallback_count_policy": "exact_replayed_numerical_fallback_count",
    "rank_policy": "exact_replayed_matrix_rank",
    "condition_policy": "replayed_singular_value_ratio",
    "next_guess_policy": "replayed_raw_trial_guess",
}
MARKER_TARGET_CLOSURE_IDENTITY = {
    "enabled": True,
    "absolute_tolerance_mps": MARKER_TARGET_ABSOLUTE_TOLERANCE_MPS,
    "closure_tolerance_mps": MARKER_TARGET_CLOSURE_TOLERANCE_MPS,
    "bounded_residual_fields": [
        "final_max_residual_mps", "final_max_adjustable_residual_mps",
        "final_max_immutable_residual_mps", "projection_only_max_residual_mps",
    ],
    "projection_only_invalid_axis_count": 0,
}
PROFILE_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    {
        "profile": PROFILE_ID,
        "config": KALMAN_IQN_REUSE_FINE_CONFIG_IDENTITY,
        "export": FINAL_FINE_EXPORT_IDENTITY,
        "iqn_reuse_state_machine": IQN_REUSE_STATE_MACHINE_IDENTITY,
        "marker_target_closure": MARKER_TARGET_CLOSURE_IDENTITY,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()
MATERIAL_PROFILE_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    {
        "profile": MATERIAL_PROFILE_ID,
        "config": MATERIAL_KALMAN_IQN_REUSE_FINE_CONFIG_IDENTITY,
        "material_audit_fields": MATERIAL_AUDIT_FIELDS,
        "base_profile_contract_sha256": PROFILE_CONTRACT_SHA256,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()


class KalmanIqnReuseFineContractError(CurrentIqnAdaptiveFineContractError):
    """Artifacts do not establish accepted-state Kalman/IQN reuse."""


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise KalmanIqnReuseFineContractError(label)


def _prior_evidence(
    reuse: Mapping[str, Any], step: int,
) -> tuple[int | None, float | None]:
    raw_source = reuse.get("source_step")
    raw_residual = reuse.get("prior_initial_residual_norm")
    _require(
        (raw_source is None) == (raw_residual is None),
        f"IQN reuse prior evidence is incomplete at step {step}",
    )
    if raw_source is None:
        return None, None
    source = _integer(raw_source, "IQN reuse source step", minimum=1)
    _require(source < step, f"IQN reuse source is not prior to step {step}")
    residual = _number(raw_residual, "IQN prior initial residual")
    _require(residual >= 0.0, f"IQN prior residual is negative at step {step}")
    return source, residual


def _validate_source_state(
    *,
    step: int,
    prior_reports: Sequence[Mapping[str, Any]],
    source_step: int | None,
    prior_residual: float | None,
    imported: int,
    reset_reason: Any,
) -> None:
    if step == 1:
        _require(
            not prior_reports
            and source_step is None
            and prior_residual is None
            and imported == 0,
            "IQN reuse step 1 must start without prior accepted history",
        )
        _require(reset_reason is None, "IQN reuse reset lacks prior evidence at step 1")
        return

    _require(
        len(prior_reports) == step - 1,
        f"IQN reuse accepted-report chain is incomplete at step {step}",
    )
    previous = prior_reports[-1]
    previous_retained = previous["retained_pair_count"]
    if previous_retained == 0:
        _require(
            source_step is None
            and prior_residual is None
            and imported == 0
            and reset_reason is None,
            f"IQN reuse retained a phantom source after step {step - 1}",
        )
        return

    _require(
        source_step is not None and prior_residual is not None,
        f"IQN reuse lost prior accepted history at step {step}",
    )
    assert source_step is not None and prior_residual is not None
    source_report = prior_reports[source_step - 1]
    _require(
        source_report["retained_pair_count"] > 0
        and _identity_values_equal(
            prior_residual, source_report["first_residual_norm"],
        ),
        f"IQN reuse prior residual chain changed at step {step}",
    )
    _require(
        source_step == step - 1,
        f"IQN reuse source is not the previous accepted official step {step}",
    )
    _require(
        imported == previous_retained,
        f"IQN reuse imported pair count changed at step {step}",
    )


def _accepted_secants(trace: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(
        trace["_trial_candidate"], dtype=np.float64,
    ).reshape(trace["T"], -1)
    residuals = np.asarray(
        trace["_trial_residual"], dtype=np.float64,
    ).reshape(trace["T"], -1)
    first = max(0, trace["T"] - 1 - IQN_HISTORY_LIMIT)
    candidate_columns = [
        candidates[index + 1] - candidates[index]
        for index in range(first, trace["T"] - 1)
    ]
    residual_columns = [
        residuals[index + 1] - residuals[index]
        for index in range(first, trace["T"] - 1)
    ]
    dof = candidates.shape[1]
    return (
        np.column_stack(residual_columns)
        if residual_columns else np.empty((dof, 0)),
        np.column_stack(candidate_columns)
        if candidate_columns else np.empty((dof, 0)),
    )


def _history_matrix(
    residuals: np.ndarray,
    candidates: np.ndarray,
    update_index: int,
    retained: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    local_count = min(IQN_HISTORY_LIMIT, update_index)
    local_residual = [
        residuals[index + 1] - residuals[index]
        for index in range(update_index)
    ]
    local_candidate = [
        candidates[index + 1] - candidates[index]
        for index in range(update_index)
    ]
    residual_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    retained_count = 0
    if retained is not None:
        retained_count = min(
            IQN_HISTORY_LIMIT - local_count, retained[0].shape[1],
        )
        if retained_count:
            residual_parts.append(retained[0][:, -retained_count:])
            candidate_parts.append(retained[1][:, -retained_count:])
    if local_count:
        residual_parts.append(np.column_stack(local_residual[-local_count:]))
        candidate_parts.append(np.column_stack(local_candidate[-local_count:]))
    dof = residuals.shape[1]
    return (
        np.column_stack(residual_parts)
        if residual_parts else np.empty((dof, 0)),
        np.column_stack(candidate_parts)
        if candidate_parts else np.empty((dof, 0)),
        retained_count > 0,
    )


def _condition_matches(actual: Any, expected: float | None, step: int) -> None:
    if expected is None:
        _require(actual is None, f"IQN condition history changed at step {step}")
        return
    value = _number(actual, "IQN condition number")
    _require(
        math.isclose(value, expected, rel_tol=1.0e-10, abs_tol=1.0e-12),
        f"IQN condition history changed at step {step}",
    )


def _replay_iqn_updates(
    history: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    step: int,
    retained: tuple[np.ndarray, np.ndarray] | None,
    growth_reset: bool,
) -> dict[str, Any]:
    guesses = np.asarray(
        trace["_trial_guess"], dtype=np.float64,
    ).reshape(trace["T"], -1)
    candidates = np.asarray(
        trace["_trial_candidate"], dtype=np.float64,
    ).reshape(trace["T"], -1)
    residuals = np.asarray(
        trace["_trial_residual"], dtype=np.float64,
    ).reshape(trace["T"], -1)
    modes = history.get("hibm_fsi_coupling_update_mode_history")
    ranks = history.get("hibm_fsi_coupling_iqn_rank_history")
    conditions = history.get("hibm_fsi_coupling_iqn_condition_number_history")
    assert isinstance(modes, list) and isinstance(ranks, list)
    _require(
        isinstance(conditions, list) and len(conditions) == len(modes),
        f"IQN condition history has invalid length at step {step}",
    )
    active_retained = None if growth_reset else retained
    expected_modes: list[str] = []
    expected_ranks: list[int] = []
    expected_conditions: list[float | None] = []
    fallback_reasons: list[str | None] = []
    pair_counts: list[int] = []
    update_limited: list[bool] = []
    expected_reset = "residual_growth_limit" if growth_reset else None

    for update_index in range(trace["T"] - 1):
        residual_matrix, candidate_matrix, contains_retained = _history_matrix(
            residuals, candidates, update_index, active_retained,
        )
        pair_count = residual_matrix.shape[1]
        guess = guesses[update_index]
        candidate = candidates[update_index]
        residual = residuals[update_index]
        mode = "picard"
        rank = 0
        condition: float | None = None
        fallback_reason: str | None = None
        limited = False
        next_guess = guess + IQN_INITIAL_PICARD_RELAXATION * residual

        if pair_count:
            try:
                coefficients, _, raw_rank, singular_values = np.linalg.lstsq(
                    residual_matrix,
                    residual,
                    rcond=IQN_SVD_RELATIVE_CUTOFF,
                )
            except np.linalg.LinAlgError:
                fallback_reason = "least_squares_failure"
            else:
                rank = int(raw_rank)
                if rank == 0:
                    fallback_reason = "zero_rank_history"
                else:
                    retained_singular_values = np.asarray(
                        singular_values[:rank], dtype=np.float64,
                    )
                    condition = float(
                        retained_singular_values[0]
                        / retained_singular_values[-1]
                    )
                    if rank < pair_count:
                        fallback_reason = "rank_deficient_history"
                    elif (
                        not math.isfinite(condition)
                        or condition > IQN_MAX_CONDITION_NUMBER
                    ):
                        fallback_reason = "ill_conditioned_history"
                    else:
                        proposal = candidate - candidate_matrix @ coefficients
                        update = proposal - guess
                        update_norm = float(np.linalg.norm(update))
                        maximum_norm = (
                            IQN_MAX_UPDATE_RATIO * float(np.linalg.norm(residual))
                        )
                        if update_norm > maximum_norm and update_norm != 0.0:
                            limited = True
                            proposal = (
                                guess.copy()
                                if maximum_norm == 0.0
                                else guess + update * (maximum_norm / update_norm)
                            )
                        if contains_retained and limited:
                            fallback_reason = "reuse_update_limited"
                        else:
                            next_guess = proposal
                            mode = (
                                "iqn_ils_reuse"
                                if contains_retained and update_index == 0
                                else "iqn_ils"
                            )
        if fallback_reason is not None and contains_retained:
            active_retained = None
            if expected_reset is None:
                expected_reset = fallback_reason
        expected_modes.append(mode)
        expected_ranks.append(rank)
        expected_conditions.append(condition)
        fallback_reasons.append(fallback_reason)
        pair_counts.append(pair_count)
        update_limited.append(limited)
        _require(
            np.allclose(
                guesses[update_index + 1], next_guess,
                rtol=1.0e-10, atol=1.0e-12,
            ),
            f"IQN replayed next guess changed at step {step} update {update_index}",
        )

    _require(
        modes == expected_modes,
        f"IQN update mode disagrees with raw replay at step {step}",
    )
    _require(
        ranks == expected_ranks,
        f"IQN rank disagrees with replayed matrix pair count at step {step}",
    )
    for actual, expected in zip(conditions, expected_conditions):
        _condition_matches(actual, expected, step)
    fallback_count = _integer(
        history.get("hibm_fsi_coupling_iqn_fallback_count"),
        "IQN fallback count",
    )
    _require(
        fallback_count == sum(reason is not None for reason in fallback_reasons),
        f"IQN fallback count disagrees with raw replay at step {step}",
    )
    return {
        "expected_reset_reason": expected_reset,
        "fallback_count": fallback_count,
        "fallback_reason_history": fallback_reasons,
        "pair_count_history": pair_counts,
        "update_limited_history": update_limited,
    }


def _validate_marker_target_closure(
    history: Mapping[str, Any], step: int,
) -> None:
    canonical = _mapping(
        history.get("canonical_velocity_dirichlet_report"),
        f"history {step} canonical velocity Dirichlet report",
    )
    closure = _mapping(
        canonical.get("marker_target_closure"),
        f"history {step} marker-target closure",
    )
    _require(
        closure.get("enabled") is True,
        f"marker-target closure is disabled at step {step}",
    )
    absolute_tolerance = _number(
        closure.get("absolute_tolerance_mps"),
        "marker-target absolute tolerance",
    )
    closure_tolerance = _number(
        closure.get("closure_tolerance_mps"),
        "marker-target closure tolerance",
    )
    _require(
        _identity_values_equal(
            absolute_tolerance, MARKER_TARGET_ABSOLUTE_TOLERANCE_MPS,
        ) and _identity_values_equal(
            closure_tolerance, MARKER_TARGET_CLOSURE_TOLERANCE_MPS,
        ),
        f"marker-target closure tolerance changed at step {step}",
    )
    for key in (
        "final_max_residual_mps",
        "final_max_adjustable_residual_mps",
        "final_max_immutable_residual_mps",
        "projection_only_max_residual_mps",
    ):
        residual = _number(closure.get(key), f"marker-target closure {key}")
        _require(
            0.0 <= residual <= closure_tolerance,
            f"marker-target closure residual exceeds tolerance at step {step}",
        )
    _require(
        _integer(
            closure.get("projection_only_invalid_axis_count"),
            "marker-target projection-only invalid axis count",
        ) == 0,
        f"marker-target projection-only closure is invalid at step {step}",
    )


def _requires_growth_reset(
    *,
    imported_pair_count: int,
    first_residual: float,
    prior_residual: float | None,
) -> bool:
    return bool(
        imported_pair_count > 0
        and prior_residual is not None
        and first_residual
        > IQN_REUSE_RESIDUAL_GROWTH_LIMIT_FACTOR * prior_residual
    )


def _validate_reuse_report(
    history: Mapping[str, Any], trace: Mapping[str, Any], step: int, *,
    prior_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reuse = _history_common(
        history,
        trace,
        step,
        allowed_update_modes=frozenset(("picard", "iqn_ils", "iqn_ils_reuse")),
    )
    _validate_marker_target_closure(history, step)
    _require(reuse.get("enabled") is True, f"IQN reuse is not enabled at step {step}")
    used = reuse.get("used")
    _require(isinstance(used, bool), f"IQN reuse used flag is invalid at step {step}")
    imported = _integer(reuse.get("imported_pair_count"), "IQN imported pair count")
    local = _integer(reuse.get("local_pair_count"), "IQN local pair count")
    retained = _integer(reuse.get("retained_pair_count"), "IQN retained pair count")
    _require(
        imported <= IQN_HISTORY_LIMIT
        and local == min(IQN_HISTORY_LIMIT, max(0, trace["T"] - 1))
        and retained == local,
        f"IQN reuse pair counts are inconsistent at step {step}",
    )
    first_residual = _number(reuse.get("first_residual_norm"), "IQN first residual")
    _require(first_residual >= 0.0, f"IQN first residual is negative at step {step}")
    _require(
        _identity_values_equal(first_residual, trace["first_residual_l2"]),
        f"IQN reuse first residual disagrees with raw trials at step {step}",
    )
    modes = history.get("hibm_fsi_coupling_update_mode_history")
    assert isinstance(modes, list)
    _require(
        all(mode != "iqn_ils_reuse" for mode in modes[1:]),
        f"IQN reuse mode appeared after the first update at step {step}",
    )
    first_mode = reuse.get("first_update_mode")
    _require(
        (bool(modes) and first_mode == modes[0])
        or (not modes and first_mode is None),
        f"IQN reuse first update mode disagrees at step {step}",
    )
    reset_reason = reuse.get("reset_reason")
    _require(
        reset_reason is None or isinstance(reset_reason, str),
        f"IQN reuse reset reason has invalid type at step {step}",
    )
    _require(
        reset_reason not in COMPATIBILITY_IQN_REUSE_RESET_REASONS
        and reset_reason != "source_step_mismatch"
        and reset_reason != "coefficient_norm_limit",
        f"IQN reset lacks evidence in the completed official H3 identity at step {step}",
    )
    _require(
        reset_reason is None or reset_reason in KNOWN_IQN_REUSE_RESET_REASONS,
        f"IQN reuse reset reason is unknown at step {step}",
    )
    source_step, prior_residual = _prior_evidence(reuse, step)
    _validate_source_state(
        step=step,
        prior_reports=prior_reports,
        source_step=source_step,
        prior_residual=prior_residual,
        imported=imported,
        reset_reason=reset_reason,
    )
    previous_secants = (
        None
        if not prior_reports or imported == 0
        else (
            prior_reports[-1]["_delta_residual"],
            prior_reports[-1]["_delta_candidate"],
        )
    )
    growth_reset = _requires_growth_reset(
        imported_pair_count=imported,
        first_residual=first_residual,
        prior_residual=prior_residual,
    )
    replay = _replay_iqn_updates(
        history, trace, step=step,
        retained=previous_secants, growth_reset=growth_reset,
    )
    _require(
        reset_reason == replay["expected_reset_reason"],
        f"IQN reset reason disagrees with raw replay at step {step}",
    )
    expected_used = bool(modes and modes[0] == "iqn_ils_reuse")
    _require(
        used is expected_used,
        f"IQN reuse used flag disagrees with raw replay at step {step}",
    )
    delta_residual, delta_candidate = _accepted_secants(trace)
    return {
        "step": step,
        "used": used,
        "source_step": source_step,
        "reset_reason": reset_reason,
        "imported_pair_count": imported,
        "local_pair_count": local,
        "retained_pair_count": retained,
        "first_residual_norm": first_residual,
        "fallback_count": replay["fallback_count"],
        "fallback_reason_history": replay["fallback_reason_history"],
        "pair_count_history": replay["pair_count_history"],
        "update_limited_history": replay["update_limited_history"],
        "_delta_residual": delta_residual,
        "_delta_candidate": delta_candidate,
    }


def _validate_controller_report(
    report: Mapping[str, Any], *, step: int, label: str,
) -> bool:
    prediction_used = step >= KALMAN_WARMUP_ACCEPTED_STATES
    mode_used = "kalman" if prediction_used else "carry_forward"
    fallback_reason = None if prediction_used else "kalman_warmup"
    for key, expected in (
        ("accepted_step_count", step),
        ("begin_count", step),
        ("discard_count", 0),
        ("kalman_accepted_state_count", step + 1),
        ("oracle_replay_cursor", 0),
    ):
        _require(_integer(report.get(key), f"{label} {key}") == expected,
                 f"{label} {key} changed")
    for key, expected in (
        ("deployable", True),
        ("has_active_step", False),
        ("kalman_prediction_used", prediction_used),
        ("kalman_ready", step + 1 >= KALMAN_WARMUP_ACCEPTED_STATES),
        ("offline_oracle", False),
    ):
        _require(report.get(key) is expected, f"{label} {key} changed")
    _require(
        report.get("mode") == "kalman"
        and report.get("mode_used") == mode_used
        and report.get("fallback_reason") == fallback_reason,
        f"{label} initial guess mode or warmup boundary changed",
    )
    return prediction_used


def _validate_kalman_history(history: Mapping[str, Any], step: int) -> dict[str, Any]:
    report = _mapping(history.get("initial_guess_report"), "initial guess report")
    prediction_used = _validate_controller_report(
        report, step=step, label=f"history {step}",
    )
    expected_mode = "kalman" if prediction_used else "carry_forward"
    expected_fallback = None if prediction_used else "kalman_warmup"
    _require(
        history.get("initial_guess_mode_requested") == "kalman"
        and history.get("initial_guess_mode_used") == expected_mode
        and history.get("initial_guess_fallback_reason") == expected_fallback,
        f"Kalman initial guess history changed at step {step}",
    )
    _require(
        history.get("kalman_writeback_mode") == "off"
        and history.get("kalman_modified_physics") is False,
        f"Kalman predictor modified solver physics at step {step}",
    )
    metrics = {}
    for report_key, history_key, nonnegative in (
        ("last_prediction_rms_mps", "initial_guess_prediction_rms_mps", True),
        ("last_prediction_bias", "initial_guess_prediction_bias_mps", False),
        ("last_nis_mean", "initial_guess_kalman_nis_mean", True),
    ):
        value = _number(history.get(history_key), f"history {step} {history_key}")
        _require(not nonnegative or value >= 0.0,
                 f"history {step} {history_key} must be nonnegative")
        _require(_identity_values_equal(report.get(report_key), value),
                 f"Kalman report disagrees with {history_key} at step {step}")
        metrics[history_key] = value
    return {"step": step, "prediction_used": prediction_used, **metrics}


def validate_kalman_iqn_reuse_fine50(
    manifest: Mapping[str, Any], summary: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    trial_frames: Sequence[Mapping[str, Any]] | Callable[[int], Mapping[str, Any]],
    *, pressure_semantics_mode: str,
) -> dict[str, Any]:
    reuse_reports: list[dict[str, Any]] = []
    kalman_reports: list[dict[str, Any]] = []

    def validate_history(
        history: Mapping[str, Any], trace: Mapping[str, Any], step: int,
    ) -> None:
        reuse_reports.append(_validate_reuse_report(
            history, trace, step,
            prior_reports=reuse_reports,
        ))
        kalman_reports.append(_validate_kalman_history(history, step))

    base = _validate_iqn_adaptive_fine50(
        manifest,
        summary,
        histories,
        trial_frames,
        pressure_semantics_mode=pressure_semantics_mode,
        config_identity=KALMAN_IQN_REUSE_FINE_CONFIG_IDENTITY,
        profile_id=PROFILE_ID,
        profile_contract_sha256=PROFILE_CONTRACT_SHA256,
        schema="kalman_iqn_reuse_fine50_identity_v2",
        history_validator=validate_history,
    )
    _require(summary.get("initial_guess_mode") == "kalman",
             "summary Kalman initial guess mode changed")
    summary_report = _mapping(summary.get("initial_guess_summary"),
                              "summary initial guess report")
    _validate_controller_report(
        summary_report, step=EXPECTED_STEPS, label="summary",
    )
    public_reuse_reports = [
        {
            key: value for key, value in row.items()
            if not key.startswith("_")
        }
        for row in reuse_reports
    ]
    final_kalman = kalman_reports[-1]
    for summary_key, history_key in (
        ("last_prediction_rms_mps", "initial_guess_prediction_rms_mps"),
        ("last_prediction_bias", "initial_guess_prediction_bias_mps"),
        ("last_nis_mean", "initial_guess_kalman_nis_mean"),
    ):
        _require(
            _identity_values_equal(
                summary_report.get(summary_key), final_kalman[history_key],
            ),
            f"summary Kalman {summary_key} disagrees with step 50 history",
        )
    used_steps = [row["step"] for row in reuse_reports if row["used"]]
    reset_steps = [
        row["step"] for row in reuse_reports if row["reset_reason"] is not None
    ]
    prediction_steps = [
        row["step"] for row in kalman_reports if row["prediction_used"]
    ]
    return {
        **base,
        "reuse_used_step_count": len(used_steps),
        "reuse_used_steps": used_steps,
        "reuse_reset_steps": reset_steps,
        "reuse_history_reports": public_reuse_reports,
        "kalman_prediction_steps": prediction_steps,
        "kalman_history_reports": kalman_reports,
        "summary_initial_guess_report": dict(summary_report),
    }


def validate_kalman_iqn_reuse_material_reference_fine50(
    manifest: Mapping[str, Any], summary: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    trial_frames: Sequence[Mapping[str, Any]] | Callable[[int], Mapping[str, Any]],
    *, pressure_semantics_mode: str,
) -> dict[str, Any]:
    return _validate_material_reference_fine50(
        manifest,
        summary,
        histories,
        trial_frames,
        pressure_semantics_mode=pressure_semantics_mode,
        base_validator=validate_kalman_iqn_reuse_fine50,
        profile_id=MATERIAL_PROFILE_ID,
        profile_contract_sha256=MATERIAL_PROFILE_CONTRACT_SHA256,
        schema="kalman_iqn_reuse_material_reference_fine50_identity_v2",
    )
