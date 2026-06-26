from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TEMPORAL_LAST_WINDOW_STEPS = 5
TEMPORAL_SOFT_ALLOWED_POST_WARMUP_FAILURES = 2

TEMPORAL_STRICT = "temporal_strict"
TEMPORAL_SOFT = "temporal_soft"
TEMPORAL_FAILED = "temporal_failed"
TEMPORAL_NOT_APPLICABLE = "temporal_not_applicable"

FLOW_TEMPORAL_STRICT = "flow_temporal_strict"
FLOW_TEMPORAL_SOFT = "flow_temporal_soft"
FLOW_TEMPORAL_FAILED = "flow_temporal_failed"
FLOW_TEMPORAL_NOT_APPLICABLE = "flow_temporal_not_applicable"

COUPLING_SETTLED = "coupling_settled"
COUPLING_SETTLED_LATE = "coupling_settled_late"
COUPLING_UNSETTLED = "coupling_unsettled"
COUPLING_NOT_APPLICABLE = "coupling_not_applicable"

PROMOTION_READY = "promotion_ready"
PROMOTION_NOT_READY = "not_promotion_candidate"
PROMOTION_NOT_APPLICABLE = "promotion_not_applicable"


@dataclass(frozen=True)
class TemporalGateProfile:
    name: str
    last_window_steps: int = TEMPORAL_LAST_WINDOW_STEPS
    allowed_post_warmup_failures: int = TEMPORAL_SOFT_ALLOWED_POST_WARMUP_FAILURES
    min_warmup_steps: int = 5
    ramp_warmup_extra_steps: int = 2
    p999_min_mps: float = 20.0
    p999_max_mps: float | None = 29.0
    peak_max_mps: float = 40.0
    outlet_min: float = 0.75
    outlet_max: float = 1.25
    last_window_outlet_min: float = 0.80
    last_window_outlet_max: float = 1.20
    allow_soft_flow: bool = True
    combined_checks_coupling: bool = True


STEP20_COUPLED_PROFILE = TemporalGateProfile(name="step20_coupled")
STEP30_FIXED_SOLID_PROFILE = TemporalGateProfile(
    name="step30_fixed_solid",
    last_window_steps=10,
    allowed_post_warmup_failures=0,
    allow_soft_flow=False,
    combined_checks_coupling=False,
)
STEP20_PREFLOW_RELEASE_PROFILE = TemporalGateProfile(
    name="step20_preflow_release",
    last_window_steps=10,
)


def classify_flow_temporal(
    row: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    profile: TemporalGateProfile = STEP20_COUPLED_PROFILE,
) -> dict[str, Any]:
    warmup_steps = _warmup_steps(row, profile)
    evaluation_start_step = warmup_steps + 1
    base: dict[str, Any] = {
        "flow_temporal_status": FLOW_TEMPORAL_NOT_APPLICABLE,
        "flow_temporal_fail_reasons": [],
        "flow_post_warmup_failed_step_count": "",
        "flow_last_window_failed_step_count": "",
        "flow_last_window_min_p999_mps": "",
        "flow_last_window_mean_outlet_ratio": "",
    }
    if row.get("run_status") != "completed":
        return {**base, "flow_temporal_fail_reasons": ["run_not_completed"]}
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return {
            **base,
            "flow_temporal_fail_reasons": ["diagnostic_full_field_reset"],
        }
    if not history:
        return {**base, "flow_temporal_fail_reasons": ["missing_history"]}

    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    if not post_warmup:
        return {
            **base,
            "flow_temporal_fail_reasons": ["missing_post_warmup_history"],
        }

    last_window = history[-profile.last_window_steps :]
    post_failures = flow_temporal_failures(
        post_warmup,
        profile=profile,
        last_window=False,
    )
    last_failures = flow_temporal_failures(
        last_window,
        profile=profile,
        last_window=True,
    )
    last_p999_values = [
        value
        for value in (_float_or_none(item.get("velocity_p999_mps")) for item in last_window)
        if value is not None
    ]
    last_outlet_values = [
        value
        for value in (
            _float_or_none(item.get("velocity_outlet_flux_ratio"))
            for item in last_window
        )
        if value is not None
    ]
    if len(post_failures) == 0 and len(last_failures) == 0:
        status = FLOW_TEMPORAL_STRICT
    elif (
        profile.allow_soft_flow
        and len(post_failures) <= profile.allowed_post_warmup_failures
        and len(last_failures) == 0
    ):
        status = FLOW_TEMPORAL_SOFT
    else:
        status = FLOW_TEMPORAL_FAILED
    return {
        **base,
        "flow_temporal_status": status,
        "flow_temporal_fail_reasons": _unique_reasons(post_failures + last_failures),
        "flow_post_warmup_failed_step_count": len(post_failures),
        "flow_last_window_failed_step_count": len(last_failures),
        "flow_last_window_min_p999_mps": (
            min(last_p999_values) if last_p999_values else ""
        ),
        "flow_last_window_mean_outlet_ratio": (
            sum(last_outlet_values) / len(last_outlet_values)
            if last_outlet_values
            else ""
        ),
    }


def classify_combined_temporal(
    row: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    profile: TemporalGateProfile = STEP20_COUPLED_PROFILE,
) -> dict[str, Any]:
    warmup_steps = _warmup_steps(row, profile)
    evaluation_start_step = warmup_steps + 1
    base: dict[str, Any] = {
        "temporal_warmup_steps": warmup_steps,
        "temporal_evaluation_start_step": evaluation_start_step,
        "temporal_last_window_steps": profile.last_window_steps,
        "temporal_candidate_status": TEMPORAL_NOT_APPLICABLE,
        "temporal_fail_reasons": [],
        "temporal_post_warmup_failed_step_count": "",
        "temporal_last_window_failed_step_count": "",
        "temporal_last_window_min_p999_mps": "",
        "temporal_last_window_mean_velocity_outlet_flux_ratio": "",
        "temporal_last_window_force_sign_ok": "",
        "temporal_last_window_tip_sign_ok": "",
    }
    if row.get("run_status") != "completed":
        return {**base, "temporal_fail_reasons": ["run_not_completed"]}
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return {**base, "temporal_fail_reasons": ["diagnostic_full_field_reset"]}
    if not history:
        return {**base, "temporal_fail_reasons": ["missing_history"]}

    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    if not post_warmup:
        return {**base, "temporal_fail_reasons": ["missing_post_warmup_history"]}

    last_window = history[-profile.last_window_steps :]
    post_failures = combined_temporal_failures(
        post_warmup,
        profile=profile,
        last_window=False,
    )
    last_failures = combined_temporal_failures(
        last_window,
        profile=profile,
        last_window=True,
    )
    last_p999_values = [
        value
        for value in (_float_or_none(item.get("velocity_p999_mps")) for item in last_window)
        if value is not None
    ]
    last_outlet_values = [
        value
        for value in (
            _float_or_none(item.get("velocity_outlet_flux_ratio"))
            for item in last_window
        )
        if value is not None
    ]
    last_force_sign_ok = all(
        _negative_value(item.get("marker_force_z_N")) for item in last_window
    )
    last_tip_sign_ok = all(_negative_value(item.get("tip_dz_m")) for item in last_window)
    if len(post_failures) == 0 and len(last_failures) == 0:
        status = TEMPORAL_STRICT
    elif (
        len(post_failures) <= profile.allowed_post_warmup_failures
        and len(last_failures) == 0
    ):
        status = TEMPORAL_SOFT
    else:
        status = TEMPORAL_FAILED
    return {
        **base,
        "temporal_candidate_status": status,
        "temporal_fail_reasons": _unique_reasons(post_failures + last_failures),
        "temporal_post_warmup_failed_step_count": len(post_failures),
        "temporal_last_window_failed_step_count": len(last_failures),
        "temporal_last_window_min_p999_mps": (
            min(last_p999_values) if last_p999_values else ""
        ),
        "temporal_last_window_mean_velocity_outlet_flux_ratio": (
            sum(last_outlet_values) / len(last_outlet_values)
            if last_outlet_values
            else ""
        ),
        "temporal_last_window_force_sign_ok": last_force_sign_ok,
        "temporal_last_window_tip_sign_ok": last_tip_sign_ok,
    }


def classify_coupling_settling(
    row: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    profile: TemporalGateProfile = STEP20_COUPLED_PROFILE,
) -> dict[str, Any]:
    warmup_steps = _warmup_steps(row, profile)
    evaluation_start_step = warmup_steps + 1
    base: dict[str, Any] = {
        "coupling_settling_status": COUPLING_NOT_APPLICABLE,
        "coupling_first_permanently_negative_force_step": "",
        "coupling_first_permanently_negative_tip_step": "",
        "coupling_first_permanently_valid_step": "",
        "coupling_longest_consecutive_pass_steps": "",
        "coupling_last_window_force_sign_ok": "",
        "coupling_last_window_tip_sign_ok": "",
    }
    if row.get("run_status") != "completed":
        return base
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return base
    if not history:
        return base

    last_window = history[-profile.last_window_steps :]
    force_first = first_permanently_negative_step(history, "marker_force_z_N")
    tip_first = first_permanently_negative_step(history, "tip_dz_m")
    valid_first = first_permanently_valid_coupling_step(history)
    longest = longest_consecutive_coupling_pass(history)
    last_force_ok = all(_negative_value(item.get("marker_force_z_N")) for item in last_window)
    last_tip_ok = all(_negative_value(item.get("tip_dz_m")) for item in last_window)
    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    if post_warmup and all(coupling_step_passes(item) for item in post_warmup):
        status = COUPLING_SETTLED
    elif valid_first != "" and last_force_ok and last_tip_ok:
        status = COUPLING_SETTLED_LATE
    else:
        status = COUPLING_UNSETTLED
    return {
        **base,
        "coupling_settling_status": status,
        "coupling_first_permanently_negative_force_step": force_first,
        "coupling_first_permanently_negative_tip_step": tip_first,
        "coupling_first_permanently_valid_step": valid_first,
        "coupling_longest_consecutive_pass_steps": longest,
        "coupling_last_window_force_sign_ok": last_force_ok,
        "coupling_last_window_tip_sign_ok": last_tip_ok,
    }


def promotion_status(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return PROMOTION_NOT_APPLICABLE
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return PROMOTION_NOT_APPLICABLE
    if (
        row.get("candidate_status") == "candidate"
        and row.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}
    ):
        return PROMOTION_READY
    return PROMOTION_NOT_READY


def select_flow_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("candidate_status") == "candidate"
        and row.get("flow_temporal_status")
        in {FLOW_TEMPORAL_STRICT, FLOW_TEMPORAL_SOFT}
    ]
    if not candidates:
        return None
    return min(candidates, key=flow_temporal_candidate_penalty)


def select_promotion_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("candidate_status") == "candidate"
        and row.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}
    ]
    if not candidates:
        return None
    return min(candidates, key=combined_temporal_candidate_penalty)


def flow_temporal_candidate_penalty(row: dict[str, Any]) -> float:
    strict_penalty = (
        0.0 if row.get("flow_temporal_status") == FLOW_TEMPORAL_STRICT else 1.0
    )
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    max_peak = _float_or_zero(row.get("max_velocity_peak_mps"))
    last_p999 = _float_or_zero(row.get("flow_last_window_min_p999_mps"))
    last_outlet = _float_or_none(row.get("flow_last_window_mean_outlet_ratio"))
    outlet_penalty = 5.0 if last_outlet is None else abs(last_outlet - 1.0) * 2.0
    return (
        strict_penalty
        + abs(p999 - 24.5)
        + max(0.0, 20.0 - last_p999) * 2.0
        + max(0.0, max_peak - 40.0) * 2.0
        + outlet_penalty
    )


def combined_temporal_candidate_penalty(row: dict[str, Any]) -> float:
    strict_penalty = 0.0 if row.get("temporal_candidate_status") == TEMPORAL_STRICT else 1.0
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    max_peak = _float_or_zero(row.get("max_velocity_peak_mps"))
    last_p999 = _float_or_zero(row.get("temporal_last_window_min_p999_mps"))
    last_outlet = _float_or_none(
        row.get("temporal_last_window_mean_velocity_outlet_flux_ratio")
    )
    outlet_penalty = 5.0 if last_outlet is None else abs(last_outlet - 1.0) * 2.0
    return (
        strict_penalty
        + abs(p999 - 24.5)
        + max(0.0, 20.0 - last_p999) * 2.0
        + max(0.0, max_peak - 40.0) * 2.0
        + outlet_penalty
    )


def flow_temporal_failures(
    rows: list[dict[str, Any]],
    *,
    profile: TemporalGateProfile,
    last_window: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in rows:
        reasons = flow_temporal_step_fail_reasons(
            item,
            profile=profile,
            last_window=last_window,
        )
        if reasons:
            failures.append({"step": int(item.get("step") or 0), "reasons": reasons})
    return failures


def combined_temporal_failures(
    rows: list[dict[str, Any]],
    *,
    profile: TemporalGateProfile,
    last_window: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in rows:
        reasons = flow_temporal_step_fail_reasons(
            item,
            profile=profile,
            last_window=last_window,
        )
        if profile.combined_checks_coupling:
            if not _negative_value(item.get("marker_force_z_N")):
                reasons.append("marker_force_z_nonnegative")
            if not _negative_value(item.get("tip_dz_m")):
                reasons.append("tip_dz_nonnegative")
        if reasons:
            failures.append({"step": int(item.get("step") or 0), "reasons": reasons})
    return failures


def flow_temporal_step_fail_reasons(
    row: dict[str, Any],
    *,
    profile: TemporalGateProfile,
    last_window: bool,
) -> list[str]:
    reasons: list[str] = []
    p999 = _float_or_none(row.get("velocity_p999_mps"))
    if p999 is None or p999 < profile.p999_min_mps:
        reasons.append(_threshold_reason("p999_below", profile.p999_min_mps))
    if p999 is not None and profile.p999_max_mps is not None and p999 > profile.p999_max_mps:
        reasons.append(_threshold_reason("p999_above", profile.p999_max_mps))
    peak = _float_or_none(row.get("velocity_peak_mps"))
    if peak is None or peak > profile.peak_max_mps:
        reasons.append(_threshold_reason("peak_above", profile.peak_max_mps))
    outlet_ratio = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    outlet_min = profile.last_window_outlet_min if last_window else profile.outlet_min
    outlet_max = profile.last_window_outlet_max if last_window else profile.outlet_max
    if outlet_ratio is None or outlet_ratio < outlet_min or outlet_ratio > outlet_max:
        reasons.append(
            f"velocity_outlet_ratio_outside_{outlet_min:.2f}_{outlet_max:.2f}"
        )
    if any(
        int(float(row.get(key) or 0)) != 0
        for key in (
            "stress_invalid_marker_count",
            "scatter_invalid_marker_count",
            "feedback_invalid_marker_count",
        )
    ):
        reasons.append("invalid_marker_count_nonzero")
    return reasons


def first_permanently_negative_step(
    rows: list[dict[str, Any]],
    key: str,
) -> int | str:
    for index, row in enumerate(rows):
        if all(_negative_value(item.get(key)) for item in rows[index:]):
            return int(row.get("step") or 0)
    return ""


def first_permanently_valid_coupling_step(rows: list[dict[str, Any]]) -> int | str:
    for index, row in enumerate(rows):
        if all(coupling_step_passes(item) for item in rows[index:]):
            return int(row.get("step") or 0)
    return ""


def longest_consecutive_coupling_pass(rows: list[dict[str, Any]]) -> int:
    longest = 0
    current = 0
    for row in rows:
        if coupling_step_passes(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def coupling_step_passes(row: dict[str, Any]) -> bool:
    return _negative_value(row.get("marker_force_z_N")) and _negative_value(
        row.get("tip_dz_m")
    )


def _warmup_steps(row: dict[str, Any], profile: TemporalGateProfile) -> int:
    ramp_steps = int(row.get("source_ramp_steps") or 0)
    return max(ramp_steps + profile.ramp_warmup_extra_steps, profile.min_warmup_steps)


def _threshold_reason(prefix: str, value: float) -> str:
    if float(value).is_integer():
        return f"{prefix}_{int(value)}"
    return f"{prefix}_{value:g}".replace(".", "p")


def _unique_reasons(failures: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    reasons: list[str] = []
    for failure in failures:
        for reason in failure.get("reasons", []):
            if reason not in seen:
                seen.add(reason)
                reasons.append(reason)
    return reasons


def _negative_value(value: Any) -> bool:
    parsed = _float_or_none(value)
    return parsed is not None and parsed < 0.0


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
