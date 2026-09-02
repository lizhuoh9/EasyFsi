"""Bottom-up R25B live-probe work accounting and frozen classifications."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .candidate_bundle import EXPECTED_ARM_IDS

EXPECTED_TARGET_STEPS = (7, 8)
EXPECTED_DT_S = 5.0e-4


class LiveAnalysisError(ValueError):
    """Live-probe evidence is incomplete, inconsistent, or transaction-unsafe."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise LiveAnalysisError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise LiveAnalysisError(f"{label} is below its minimum")
    return result


def _target_step(report: Mapping[str, Any]) -> int:
    target = _integer(report.get("target_step"), label="target_step", minimum=1)
    if target not in EXPECTED_TARGET_STEPS:
        raise LiveAnalysisError("target_step must be 7 or 8")
    return target


def _validated_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise LiveAnalysisError("research probe row must be a mapping")
    arm_id = row.get("arm_id")
    if arm_id not in EXPECTED_ARM_IDS:
        raise LiveAnalysisError(f"unexpected live-probe arm {arm_id!r}")
    if row.get("rollback_host_macro_step_state_equal") is not True or row.get(
        "rollback_host_macro_step_state_mismatch_fields"
    ) != []:
        raise LiveAnalysisError(f"{arm_id}: rollback equality failed")
    requested_sha = row.get("requested_candidate_sha256")
    actual_sha = row.get("actual_first_guess_sha256")
    if (
        not _is_sha256(requested_sha)
        or not _is_sha256(actual_sha)
        or requested_sha != actual_sha
        or row.get("actual_first_guess_equals_requested") is not True
    ):
        raise LiveAnalysisError(f"{arm_id}: actual first guess differs from requested")
    converged_value = row.get("converged")
    if not isinstance(converged_value, (bool, np.bool_)):
        raise LiveAnalysisError(f"{arm_id}: converged must be Boolean")
    converged = bool(converged_value)
    iterations = _integer(row.get("iterations"), label=f"{arm_id} iterations", minimum=1)
    rejected_trials = _integer(
        row.get("coupling_rejected_trial_count"),
        label=f"{arm_id} rejected trials",
        minimum=0,
    )
    if rejected_trials != iterations - int(converged):
        raise LiveAnalysisError(f"{arm_id}: rejected trial count is inconsistent")
    work = row.get("trial_work")
    if not isinstance(work, Mapping):
        raise LiveAnalysisError(f"{arm_id}: trial_work is missing")
    trial_count = _integer(
        work.get("trial_count"), label=f"{arm_id} trial_count", minimum=1
    )
    if trial_count != iterations:
        raise LiveAnalysisError(f"{arm_id}: trial_count differs from iterations")
    cg_iterations = _integer(
        work.get("cg_iterations_total"),
        label=f"{arm_id} pressure CG",
        minimum=0,
    )
    pressure_matvec = _integer(
        work.get("pressure_matvec_count_total"),
        label=f"{arm_id} pressure matvec",
        minimum=1,
    )
    if pressure_matvec < cg_iterations:
        raise LiveAnalysisError(
            f"{arm_id}: pressure matvec count is below CG iterations"
        )
    for field in (
        "first_absolute_residual_mps",
        "second_absolute_residual_mps",
    ):
        value = row.get(field)
        if value is not None and (
            isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value))
        ):
            raise LiveAnalysisError(f"{arm_id}: {field} is non-finite")
    return {
        **dict(row),
        "arm_id": str(arm_id),
        "converged": converged,
        "iterations": iterations,
        "coupling_rejected_trial_count": rejected_trials,
        "trial_work": {
            **dict(work),
            "trial_count": trial_count,
            "cg_iterations_total": cg_iterations,
            "pressure_matvec_count_total": pressure_matvec,
        },
    }


def validate_live_probe_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one target report without trusting its aggregate booleans alone."""

    if not isinstance(report, Mapping):
        raise LiveAnalysisError("live-probe report must be a mapping")
    target = _target_step(report)
    if (
        report.get("status") != "research_candidate_probe_terminal"
        or report.get("research_candidate_probe_terminal") is not True
    ):
        raise LiveAnalysisError("live-probe report is not terminal")
    accepted_step = _integer(
        report.get("accepted_step_count"), label="accepted_step_count", minimum=0
    )
    if accepted_step != target - 1:
        raise LiveAnalysisError("target step was committed or accepted prefix changed")
    accepted_time = report.get("accepted_time_s")
    if isinstance(accepted_time, (bool, np.bool_)) or not math.isfinite(
        float(accepted_time)
    ):
        raise LiveAnalysisError("accepted_time_s is non-finite")
    expected_time = (target - 1) * EXPECTED_DT_S
    if not np.isclose(
        float(accepted_time), expected_time, rtol=0.0, atol=1.0e-15
    ):
        raise LiveAnalysisError("accepted time changed during candidate probe")
    if report.get("research_probe_all_rollbacks_equal") is not True:
        raise LiveAnalysisError("live-probe aggregate rollback equality failed")
    if (
        report.get("research_probe_sweep_state_equal") is not True
        or report.get("research_probe_sweep_state_mismatch_fields") != []
    ):
        raise LiveAnalysisError("live-probe final sweep rollback equality failed")
    rows = report.get("research_probe_rows")
    if not isinstance(rows, list):
        raise LiveAnalysisError("live-probe rows are missing")
    validated_rows = tuple(_validated_row(row) for row in rows)
    if tuple(row["arm_id"] for row in validated_rows) != EXPECTED_ARM_IDS:
        raise LiveAnalysisError("live-probe rows do not match the exact 13-arm order")
    anchor_refresh_delta = _integer(
        report.get("research_probe_anchor_refresh_delta"),
        label="research probe anchor refresh delta",
        minimum=0,
    )
    expected_anchor_refresh_delta = sum(
        int(row["trial_work"]["trial_count"]) for row in validated_rows
    )
    if anchor_refresh_delta != expected_anchor_refresh_delta:
        raise LiveAnalysisError(
            "research probe anchor refresh delta differs from trial work"
        )
    return {
        **dict(report),
        "target_step": target,
        "accepted_step_count": accepted_step,
        "accepted_time_s": float(accepted_time),
        "research_probe_rows": validated_rows,
        "research_probe_anchor_refresh_delta": anchor_refresh_delta,
    }


def _work_totals(
    reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm_id in EXPECTED_ARM_IDS:
        rows = [
            next(
                row
                for row in reports[target]["research_probe_rows"]
                if row["arm_id"] == arm_id
            )
            for target in EXPECTED_TARGET_STEPS
        ]
        result[arm_id] = {
            "trials": sum(int(row["iterations"]) for row in rows),
            "rejected_trials": sum(
                int(row["coupling_rejected_trial_count"]) for row in rows
            ),
            "pressure_cg": sum(
                int(row["trial_work"]["cg_iterations_total"]) for row in rows
            ),
            "pressure_matvec": sum(
                int(row["trial_work"]["pressure_matvec_count_total"])
                for row in rows
            ),
            "converged": all(bool(row["converged"]) for row in rows),
            "trials_by_target": {
                str(target): int(row["iterations"])
                for target, row in zip(EXPECTED_TARGET_STEPS, rows, strict=True)
            },
            "pressure_cg_by_target": {
                str(target): int(row["trial_work"]["cg_iterations_total"])
                for target, row in zip(EXPECTED_TARGET_STEPS, rows, strict=True)
            },
            "rejected_trials_by_target": {
                str(target): int(row["coupling_rejected_trial_count"])
                for target, row in zip(EXPECTED_TARGET_STEPS, rows, strict=True)
            },
            "pressure_matvec_by_target": {
                str(target): int(row["trial_work"]["pressure_matvec_count_total"])
                for target, row in zip(EXPECTED_TARGET_STEPS, rows, strict=True)
            },
        }
    return result


def _effect_values(
    totals: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    field: str,
) -> dict[str, int]:
    c0 = int(totals["C0"][field])
    k1 = int(totals["K1"][field])
    gdelta = int(totals[f"GDelta-M-seed{seed}"][field])
    gk1 = int(totals[f"GK1-seed{seed}"][field])
    return {
        "delta_g": c0 - gdelta,
        "delta_k": c0 - k1,
        "delta_g_given_k": k1 - gk1,
        "delta_kinfo": gdelta - gk1,
        "interaction": gdelta + k1 - c0 - gk1,
    }


def _no_target_trial_worsening(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    return all(
        int(candidate["trials_by_target"][str(target)])
        <= int(reference["trials_by_target"][str(target)])
        for target in EXPECTED_TARGET_STEPS
    )


def _standalone_seed_pass(
    totals: Mapping[str, Mapping[str, Any]], seed: int
) -> bool:
    reference = totals["C0"]
    candidate = totals[f"GDelta-M-seed{seed}"]
    if not candidate["converged"] or not reference["converged"]:
        return False
    if int(candidate["trials"]) <= int(reference["trials"]) - 1:
        return True
    return bool(
        int(candidate["trials"]) == int(reference["trials"])
        and int(candidate["pressure_cg"]) <= 0.90 * int(reference["pressure_cg"])
        and _no_target_trial_worsening(candidate, reference)
    )


def _gk_seed_pass(totals: Mapping[str, Mapping[str, Any]], seed: int) -> bool:
    hybrid = totals[f"GK1-seed{seed}"]
    kalman = totals["K1"]
    control = totals[f"GDelta-M-seed{seed}"]
    if not all(row["converged"] for row in (hybrid, kalman, control)):
        return False
    if (
        int(hybrid["trials"]) < int(kalman["trials"])
        and int(hybrid["trials"]) < int(control["trials"])
    ):
        return True
    return bool(
        int(hybrid["trials"])
        == int(kalman["trials"])
        == int(control["trials"])
        and int(hybrid["pressure_cg"]) <= 0.90 * int(kalman["pressure_cg"])
        and int(hybrid["pressure_cg"]) <= 0.90 * int(control["pressure_cg"])
    )


def analyze_live_probe_reports(
    reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute work effects and classifications from step-7/8 raw rows."""

    if not isinstance(reports, Mapping) or set(reports) != set(EXPECTED_TARGET_STEPS):
        raise LiveAnalysisError("analysis requires exactly target steps 7 and 8")
    validated = {
        target: validate_live_probe_report(reports[target])
        for target in EXPECTED_TARGET_STEPS
    }
    totals = _work_totals(validated)
    effects = {
        f"seed{seed}": {
            "trials": _effect_values(totals, seed=seed, field="trials"),
            "pressure_cg": _effect_values(
                totals, seed=seed, field="pressure_cg"
            ),
        }
        for seed in (0, 1, 2)
    }
    standalone_passes = [
        seed for seed in (0, 1, 2) if _standalone_seed_pass(totals, seed)
    ]
    gk_passes = [seed for seed in (0, 1, 2) if _gk_seed_pass(totals, seed)]
    ar_pass = bool(
        totals["AR"]["converged"]
        and totals["C0"]["converged"]
        and int(totals["AR"]["trials"]) <= int(totals["C0"]["trials"]) - 1
    )
    standalone_classification = (
        "PASS_G0_MATCHED_LIVE_VALUE"
        if len(standalone_passes) >= 2
        else "FAIL_G0_MATCHED_LIVE_VALUE"
    )
    gk_classification = (
        "PASS_GK1_INCREMENTAL_LIVE_VALUE"
        if len(gk_passes) >= 2
        else "FAIL_GK1_INCREMENTAL_LIVE_VALUE"
    )
    ar_classification = (
        "PASS_POD_AR_LIVE_VALUE" if ar_pass else "FAIL_POD_AR_LIVE_VALUE"
    )
    neural_pass = len(standalone_passes) >= 2 or len(gk_passes) >= 2
    if neural_pass:
        overall = "PASS_NEURAL_LIVE_SIGNAL"
    elif ar_pass:
        overall = "FAIL_NEURAL_LIVE_VALUE_POD_AR_PASS"
    else:
        overall = "FAIL_NO_LIVE_SOLVER_WORK_REDUCTION"
    return {
        "target_steps": list(EXPECTED_TARGET_STEPS),
        "work_totals": totals,
        "effects": effects,
        "passing_seeds": {
            "standalone_gru": standalone_passes,
            "gk1_incremental": gk_passes,
        },
        "classifications": {
            "standalone_gru": standalone_classification,
            "gk1_incremental": gk_classification,
            "pod_ar": ar_classification,
            "overall": overall,
        },
    }


__all__ = [
    "LiveAnalysisError",
    "analyze_live_probe_reports",
    "validate_live_probe_report",
]
