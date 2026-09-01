"""R24C displacement and oracle-threshold evidence contracts.

This module derives new evidence from an already-validated R24B Q0/Q3 pair.
It deliberately leaves the sealed R24B implementation and artifacts unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .kalman_oracle_headroom_contracts import (
    EXPECTED_STEPS,
    OracleHeadroomContractError,
    _load_run,
    _validate_pair,
)
from .oracle_threshold_common import (
    OracleThresholdContractError,
    THRESHOLD_ALPHAS,
    THRESHOLD_OMEGAS,
    THRESHOLD_TARGET_STEPS,
    require as _require,
)
from .oracle_threshold_probe_contracts import (
    positive_integer,
    validate_probe_report,
)


DISPLACEMENT_NRMSE_MAX = 5.0e-3
_DISPLACEMENT_SCALE_FLOOR_M = 1.0e-12
_SOLID_SHAPE = (5120, 3)
_TIP_MASK_SHAPE = (5120,)
_INTERFACE_COMPONENT_COUNT = 3


def _finite_array(
    value: np.ndarray,
    *,
    label: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == expected_shape, f"{label} has invalid shape")
    _require(np.issubdtype(array.dtype, np.number), f"{label} must be numeric")
    _require(bool(np.all(np.isfinite(array))), f"{label} must be finite")
    return np.asarray(array, dtype=np.float64)


def _finite_float(config: Mapping[str, Any], key: str) -> float:
    try:
        value = float(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleThresholdContractError(
            f"marker reference config {key} must be numeric"
        ) from exc
    _require(math.isfinite(value), f"marker reference config {key} must be finite")
    return value


def _marker_reference_positions(config: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct the frozen dual-face preflow marker positions."""

    _require(
        config.get("traction_marker_layout") == "dual_physical_faces",
        "marker reference requires the dual physical-face layout",
    )
    try:
        markers_per_face = int(config["marker_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleThresholdContractError(
            "marker reference marker_count must be an integer"
        ) from exc
    _require(markers_per_face == 64, "marker reference requires 64 markers per face")
    offset = _finite_float(config, "traction_marker_face_offset_cells")
    _require(offset == 0.0, "marker reference requires zero physical-face offset")

    duct_length = _finite_float(config, "duct_length_m")
    span = _finite_float(config, "span_m")
    flap_height = _finite_float(config, "flap_height_m")
    streamwise_min = _finite_float(config, "flap_streamwise_min_m")
    streamwise_max = _finite_float(config, "flap_streamwise_max_m")
    _require(span > 0.0 and flap_height > 0.0, "marker geometry must be positive")
    _require(
        0.0 <= streamwise_min < streamwise_max <= duct_length,
        "marker streamwise bounds are invalid",
    )

    solver_z_min = duct_length - streamwise_max
    solver_z_max = duct_length - streamwise_min
    y = (
        np.arange(markers_per_face, dtype=np.float64) + 0.5
    ) * flap_height / markers_per_face
    x = np.full(markers_per_face, 0.5 * span, dtype=np.float64)
    primary = np.column_stack(
        (x, y, np.full(markers_per_face, solver_z_max))
    )
    secondary = np.column_stack(
        (x, y, np.full(markers_per_face, solver_z_min))
    )
    # The runtime reference is captured from float32 marker fields.
    return np.asarray(np.vstack((primary, secondary)), dtype=np.float32)


def displacement_error_metric(
    *,
    reference_position_m: np.ndarray,
    candidate_position_m: np.ndarray,
    rest_position_m: np.ndarray,
) -> dict[str, Any]:
    """Measure candidate error relative to the accepted reference displacement."""

    reference = np.asarray(reference_position_m, dtype=np.float64)
    candidate = np.asarray(candidate_position_m, dtype=np.float64)
    rest = np.asarray(rest_position_m, dtype=np.float64)
    _require(
        reference.ndim == 2
        and reference.shape[1] == _INTERFACE_COMPONENT_COUNT
        and candidate.shape == reference.shape
        and rest.shape == reference.shape,
        "displacement metric requires matching (count, 3) arrays",
    )
    for label, array in (
        ("reference position", reference),
        ("candidate position", candidate),
        ("rest position", rest),
    ):
        _require(bool(np.all(np.isfinite(array))), f"{label} must be finite")

    reference_displacement = reference - rest
    candidate_displacement = candidate - rest
    error = candidate_displacement - reference_displacement
    error_rmse = float(np.sqrt(np.mean(np.square(error))))
    reference_rms = float(
        np.sqrt(np.mean(np.square(reference_displacement)))
    )
    position_error_rmse = float(np.sqrt(np.mean(np.square(candidate - reference))))
    position_rms = float(np.sqrt(np.mean(np.square(reference))))
    error_rmse_per_axis = np.sqrt(np.mean(np.square(error), axis=0))
    reference_rms_per_axis = np.sqrt(
        np.mean(np.square(reference_displacement), axis=0)
    )
    nrmse_per_axis = error_rmse_per_axis / np.maximum(
        reference_rms_per_axis,
        _DISPLACEMENT_SCALE_FLOOR_M,
    )
    return {
        "nrmse": error_rmse / max(reference_rms, _DISPLACEMENT_SCALE_FLOOR_M),
        "position_nrmse": position_error_rmse
        / max(position_rms, _DISPLACEMENT_SCALE_FLOOR_M),
        "reference_displacement_rms_m": reference_rms,
        "error_rmse_m": error_rmse,
        "reference_displacement_rms_per_axis_m": (
            reference_rms_per_axis.tolist()
        ),
        "error_rmse_per_axis_m": error_rmse_per_axis.tolist(),
        "nrmse_per_axis": nrmse_per_axis.tolist(),
        "max_position_error_m": float(
            np.max(np.linalg.norm(candidate - reference, axis=1))
        ),
    }


def _load_solid_reference_arrays(
    frame_path: Path,
    *,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(frame_path, allow_pickle=False) as frame:
            _require(
                "solid_rest_position_m" in frame.files,
                f"step {step} missing solid rest positions",
            )
            _require(
                "solid_tip_mask" in frame.files,
                f"step {step} missing solid tip mask",
            )
            rest = np.array(frame["solid_rest_position_m"], copy=True)
            tip_mask = np.array(frame["solid_tip_mask"], copy=True)
    except (OSError, ValueError) as exc:
        raise OracleThresholdContractError(
            f"step {step} invalid displacement frame"
        ) from exc
    rest_f64 = _finite_array(
        rest,
        label=f"step {step} solid rest",
        expected_shape=_SOLID_SHAPE,
    )
    _require(
        tip_mask.shape == _TIP_MASK_SHAPE and tip_mask.dtype == np.bool_,
        f"step {step} solid tip mask has invalid shape or dtype",
    )
    _require(bool(np.any(tip_mask)), f"step {step} solid tip mask is empty")
    return rest_f64, tip_mask


def _tip_metrics(
    *,
    q0_position_m: np.ndarray,
    q3_position_m: np.ndarray,
    rest_position_m: np.ndarray,
    tip_mask: np.ndarray,
) -> dict[str, Any]:
    q0_displacement = q0_position_m - rest_position_m
    q3_displacement = q3_position_m - rest_position_m
    q0_tip = np.mean(q0_displacement[tip_mask], axis=0)
    q3_tip = np.mean(q3_displacement[tip_mask], axis=0)
    error = q3_tip - q0_tip
    return {
        "q0_tip_displacement_vector_m": q0_tip.tolist(),
        "q3_tip_displacement_vector_m": q3_tip.tolist(),
        "tip_displacement_vector_error_m": float(np.linalg.norm(error)),
        "tip_displacement_amplitude_error_m": abs(
            float(np.linalg.norm(q3_tip)) - float(np.linalg.norm(q0_tip))
        ),
    }


def analyze_accepted_displacements(
    q0_root: Path | str,
    q3_root: Path | str,
) -> dict[str, Any]:
    """Audit exact-8 Q0/Q3 positions on the physical displacement scale."""

    try:
        q0 = _load_run(q0_root, expected_mode="carry_forward")
        q3 = _load_run(q3_root, expected_mode="oracle_replay")
        _validate_pair(q0, q3)
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    return _analyze_loaded_accepted_displacements(q0, q3)


def _analyze_loaded_accepted_displacements(q0: Any, q3: Any) -> dict[str, Any]:
    """Derive displacement metrics from an already-validated exact8 pair."""

    marker_reference = np.asarray(
        _marker_reference_positions(q0.config),
        dtype=np.float64,
    )
    _require(
        marker_reference.shape == (128, 3),
        "marker reference must contain 128 interface rows",
    )

    rows: list[dict[str, Any]] = []
    canonical_rest: np.ndarray | None = None
    canonical_tip_mask: np.ndarray | None = None
    for q0_step, q3_step in zip(q0.steps, q3.steps):
        step = int(q0_step.step)
        _require(step == q3_step.step, "Q0/Q3 step mismatch")
        q0_rest, q0_tip_mask = _load_solid_reference_arrays(
            q0_step.frame_path,
            step=step,
        )
        q3_rest, q3_tip_mask = _load_solid_reference_arrays(
            q3_step.frame_path,
            step=step,
        )
        _require(
            np.array_equal(q0_rest, q3_rest),
            f"step {step} solid rest positions differ between Q0 and Q3",
        )
        _require(
            np.array_equal(q0_tip_mask, q3_tip_mask),
            f"step {step} solid tip mask differs between Q0 and Q3",
        )
        if canonical_rest is None:
            canonical_rest = q0_rest.copy()
            canonical_tip_mask = q0_tip_mask.copy()
        else:
            _require(
                np.array_equal(q0_rest, canonical_rest),
                f"step {step} solid rest positions drifted",
            )
            assert canonical_tip_mask is not None
            _require(
                np.array_equal(q0_tip_mask, canonical_tip_mask),
                f"step {step} solid tip mask drifted",
            )

        q0_marker = np.asarray(
            q0_step.arrays["marker_position_m"],
            dtype=np.float64,
        )
        q3_marker = np.asarray(
            q3_step.arrays["marker_position_m"],
            dtype=np.float64,
        )
        q0_solid = np.asarray(
            q0_step.arrays["solid_position_m"],
            dtype=np.float64,
        )
        q3_solid = np.asarray(
            q3_step.arrays["solid_position_m"],
            dtype=np.float64,
        )
        marker_metric = displacement_error_metric(
            reference_position_m=q0_marker,
            candidate_position_m=q3_marker,
            rest_position_m=marker_reference,
        )
        solid_metric = displacement_error_metric(
            reference_position_m=q0_solid,
            candidate_position_m=q3_solid,
            rest_position_m=q0_rest,
        )
        rows.append(
            {
                "step": step,
                "marker_position_nrmse_absolute_coordinates": marker_metric[
                    "position_nrmse"
                ],
                "marker_displacement_nrmse": marker_metric["nrmse"],
                "marker_displacement_nrmse_per_axis": marker_metric[
                    "nrmse_per_axis"
                ],
                "marker_reference_displacement_rms_per_axis_m": marker_metric[
                    "reference_displacement_rms_per_axis_m"
                ],
                "marker_error_rmse_per_axis_m": marker_metric[
                    "error_rmse_per_axis_m"
                ],
                "max_marker_position_error_m": marker_metric[
                    "max_position_error_m"
                ],
                "solid_position_nrmse_absolute_coordinates": solid_metric[
                    "position_nrmse"
                ],
                "solid_displacement_nrmse": solid_metric["nrmse"],
                "solid_displacement_nrmse_per_axis": solid_metric[
                    "nrmse_per_axis"
                ],
                "solid_reference_displacement_rms_per_axis_m": solid_metric[
                    "reference_displacement_rms_per_axis_m"
                ],
                "solid_error_rmse_per_axis_m": solid_metric[
                    "error_rmse_per_axis_m"
                ],
                "max_solid_position_error_m": solid_metric[
                    "max_position_error_m"
                ],
                **_tip_metrics(
                    q0_position_m=q0_solid,
                    q3_position_m=q3_solid,
                    rest_position_m=q0_rest,
                    tip_mask=q0_tip_mask,
                ),
            }
        )

    _require(len(rows) == EXPECTED_STEPS, "displacement audit requires exact8")
    marker_nrmse_max = max(
        float(row["marker_displacement_nrmse"]) for row in rows
    )
    solid_nrmse_max = max(
        float(row["solid_displacement_nrmse"]) for row in rows
    )
    gate = (
        marker_nrmse_max <= DISPLACEMENT_NRMSE_MAX
        and solid_nrmse_max <= DISPLACEMENT_NRMSE_MAX
    )
    return {
        "schema_version": 1,
        "campaign": "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c",
        "classification": (
            "PASS_ACCEPTED_DISPLACEMENT_AUDIT"
            if gate
            else "FAIL_ACCEPTED_DISPLACEMENT_AUDIT"
        ),
        "deployable": False,
        "q0_root": str(q0.root),
        "q3_root": str(q3.root),
        "identity": {
            "physical_marker_count_per_face": 64,
            "physical_marker_count_total": 128,
            "interface_state_row_count": int(marker_reference.shape[0]),
        },
        "thresholds": {
            "displacement_nrmse_max": DISPLACEMENT_NRMSE_MAX,
            "displacement_scale_floor_m": _DISPLACEMENT_SCALE_FLOOR_M,
        },
        "gates": {"accepted_displacement_contract": gate},
        "aggregate": {
            "marker_displacement_nrmse_max": marker_nrmse_max,
            "solid_displacement_nrmse_max": solid_nrmse_max,
            "max_marker_position_error_m": max(
                float(row["max_marker_position_error_m"]) for row in rows
            ),
            "max_solid_position_error_m": max(
                float(row["max_solid_position_error_m"]) for row in rows
            ),
            "tip_displacement_vector_error_m_max": max(
                float(row["tip_displacement_vector_error_m"]) for row in rows
            ),
            "tip_displacement_amplitude_error_m_max": max(
                float(row["tip_displacement_amplitude_error_m"]) for row in rows
            ),
        },
        "steps": rows,
    }


def summarize_threshold_matrix(
    probe_reports: Mapping[tuple[float, int], Mapping[str, Any]],
    carry_iterations: Mapping[tuple[float, int], int],
    carry_cg_iterations: Mapping[tuple[float, int], int],
) -> dict[str, Any]:
    """Validate and summarize the frozen nine-arm no-commit probe matrix."""

    expected_arms = {
        (omega, target)
        for omega in THRESHOLD_OMEGAS
        for target in THRESHOLD_TARGET_STEPS
    }
    _require(set(probe_reports) == expected_arms, "threshold matrix arms mismatch")
    _require(set(carry_iterations) == expected_arms, "carry matrix arms mismatch")
    _require(
        set(carry_cg_iterations) == expected_arms,
        "carry CG matrix arms mismatch",
    )

    arms: list[dict[str, Any]] = []
    for omega in THRESHOLD_OMEGAS:
        for target in THRESHOLD_TARGET_STEPS:
            rows = validate_probe_report(
                probe_reports[(omega, target)],
                omega=omega,
                target_step=target,
            )
            _require(
                len(rows) == len(THRESHOLD_ALPHAS),
                f"omega {omega} target {target} alpha row type invalid",
            )
            carry = positive_integer(
                carry_iterations[(omega, target)],
                label=f"omega {omega} target {target} carry iterations",
            )
            carry_cg = positive_integer(
                carry_cg_iterations[(omega, target)],
                label=f"omega {omega} target {target} carry CG iterations",
            )
            alpha_to_two = next(
                (
                    float(row["alpha"])
                    for row in rows
                    if int(row["iterations"]) <= 2
                ),
                None,
            )
            alpha_to_one = next(
                (
                    float(row["alpha"])
                    for row in rows
                    if int(row["iterations"]) == 1
                ),
                None,
            )
            arms.append(
                {
                    "omega": omega,
                    "target_step": target,
                    "carry_iterations": carry,
                    "carry_cg_iterations": carry_cg,
                    "alpha_3_to_2": alpha_to_two,
                    "alpha_2_to_1": alpha_to_one,
                    "rows": rows,
                }
            )

    omega_summary: list[dict[str, Any]] = []
    for omega in THRESHOLD_OMEGAS:
        omega_arms = [row for row in arms if row["omega"] == omega]
        baseline_not_worse = all(
            int(row["carry_iterations"])
            <= int(
                next(
                    reference["carry_iterations"]
                    for reference in arms
                    if reference["omega"] == 0.5
                    and reference["target_step"] == row["target_step"]
                )
            )
            for row in omega_arms
        )
        reduced = [
            row
            for row in omega_arms
            if row["carry_iterations"] == 3 and row["alpha_3_to_2"] is not None
        ]
        thresholds = [float(row["alpha_3_to_2"]) for row in reduced]
        threshold_indices = [THRESHOLD_ALPHAS.index(value) for value in thresholds]
        one_thresholds = [
            float(row["alpha_2_to_1"])
            for row in omega_arms
            if row["alpha_2_to_1"] is not None
        ]
        one_indices = [THRESHOLD_ALPHAS.index(value) for value in one_thresholds]
        carry_total = sum(int(row["carry_iterations"]) for row in omega_arms)
        carry_cg_total = sum(
            int(row["carry_cg_iterations"]) for row in omega_arms
        )
        safe = baseline_not_worse and len(reduced) >= 2
        omega_summary.append(
            {
                "omega": omega,
                "safe": safe,
                "carry_not_worse_than_omega_0_5": baseline_not_worse,
                "reduced_target_count": len(reduced),
                "alpha_3_to_2_min": min(thresholds) if thresholds else None,
                "alpha_3_to_2_max": max(thresholds) if thresholds else None,
                "alpha_3_to_2_mean": (
                    float(np.mean(thresholds)) if thresholds else None
                ),
                "alpha_2_to_1_target_count": len(one_thresholds),
                "alpha_2_to_1_max": (
                    max(one_thresholds) if one_thresholds else None
                ),
                "carry_iterations_total": carry_total,
                "carry_cg_iterations_total": carry_cg_total,
                "selection_rank_key": [
                    -len(reduced),
                    max(threshold_indices, default=len(THRESHOLD_ALPHAS)),
                    sum(threshold_indices),
                    -len(one_thresholds),
                    max(one_indices, default=len(THRESHOLD_ALPHAS)),
                    sum(one_indices),
                    carry_total,
                    carry_cg_total,
                    THRESHOLD_OMEGAS.index(omega),
                ],
            }
        )

    safe_omegas = [row for row in omega_summary if row["safe"]]
    higher_safe_omega_exists = any(
        float(row["omega"]) in (0.75, 1.0) for row in safe_omegas
    )
    best = (
        min(
            safe_omegas,
            key=lambda row: tuple(row["selection_rank_key"]),
        )
        if safe_omegas
        else None
    )
    if best is None:
        reuse_branch = {
            "authorized": False,
            "status": "reuse_matrix_not_authorized",
            "reason": "no_safe_omega",
        }
        predictor_decision = "academic_offline_feasibility_only"
    elif higher_safe_omega_exists:
        reuse_branch = {
            "authorized": True,
            "status": "reuse_matrix_authorized",
            "reason": "safe_higher_first_picard_relaxation",
        }
        predictor_decision = "threshold_supports_first_update_mechanism_follow_up"
    elif float(best["alpha_3_to_2_max"]) <= 0.99:
        reuse_branch = {
            "authorized": True,
            "status": "reuse_matrix_authorized",
            "reason": "best_safe_threshold_at_or_below_0.9900",
        }
        predictor_decision = "threshold_supports_first_update_mechanism_follow_up"
    else:
        reuse_branch = {
            "authorized": False,
            "status": "reuse_matrix_not_authorized",
            "reason": "best_safe_omega_requires_alpha_above_0.9900",
        }
        predictor_decision = "academic_offline_feasibility_only"

    if best is not None and float(best["alpha_3_to_2_max"]) >= 0.995:
        predictor_decision = "academic_offline_feasibility_only"
    return {
        "schema_version": 1,
        "campaign": "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c",
        "classification": "PASS_ORACLE_THRESHOLD_MATRIX",
        "deployable": False,
        "identity": {
            "target_steps": list(THRESHOLD_TARGET_STEPS),
            "omegas": list(THRESHOLD_OMEGAS),
            "alphas": list(THRESHOLD_ALPHAS),
        },
        "gates": {"complete_fail_closed_matrix": True},
        "arms": arms,
        "omega_summary": omega_summary,
        "best_safe_omega": None if best is None else float(best["omega"]),
        "reuse_branch": reuse_branch,
        "predictor_decision": predictor_decision,
    }


__all__ = (
    "DISPLACEMENT_NRMSE_MAX",
    "OracleThresholdContractError",
    "THRESHOLD_ALPHAS",
    "THRESHOLD_OMEGAS",
    "THRESHOLD_TARGET_STEPS",
    "_analyze_loaded_accepted_displacements",
    "analyze_accepted_displacements",
    "displacement_error_metric",
    "summarize_threshold_matrix",
)
