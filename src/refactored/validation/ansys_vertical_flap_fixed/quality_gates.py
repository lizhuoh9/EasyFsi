from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "inlet_u": 7.0,
    "max_mass_imbalance_rel": 0.05,
    "max_divergence_l2_warn": 100.0,
    "max_divergence_linf_warn": 10000.0,
    "max_poisson_residual_linf_warn": 1.0e8,
    "max_poisson_residual_linf_relative": 1.0e-3,
}


def load_solver_history(path: str | Path) -> list[dict[str, float]]:
    return _load_numeric_csv(path)


def load_mass_balance(path: str | Path) -> list[dict[str, float]]:
    return _load_numeric_csv(path)


def evaluate_quality_gates(
    history_rows: list[dict[str, float]],
    mass_rows: list[dict[str, float]],
    final_summary: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if config:
        thresholds.update(config.get("quality_gates", config))

    last_history = history_rows[-1] if history_rows else {}
    last_mass = mass_rows[-1] if mass_rows else {}

    # Mass-conservation, divergence, and residual evidence must never
    # silently default to 0.0 -- a 0.0 divergence/residual/imbalance reads
    # as "perfect", so a missing metric would otherwise pass every gate
    # below for free. Every gate driven by these five metrics is a
    # "smaller is better, must be <= threshold" comparison, so +inf is a
    # safe sentinel for "missing": it can never accidentally satisfy a pass
    # condition, and it still formats cleanly in the diagnostic strings.
    missing_metrics: list[str] = []

    def _required(label: str, *sources: tuple[dict[str, Any], str]) -> float:
        for source, key in sources:
            if key in source:
                parsed = _finite_or_none(source[key])
                if parsed is not None:
                    return parsed
        missing_metrics.append(label)
        return math.inf

    mass_imbalance_rel = _required(
        "mass_imbalance_rel",
        (final_summary, "mass_imbalance_rel_corrected"),
        (final_summary, "mass_imbalance_rel"),
        (last_mass, "mass_imbalance_rel"),
    )
    divergence_l2 = _required("divergence_l2", (last_history, "divergence_l2"))
    divergence_linf = _required("divergence_linf", (last_history, "divergence_linf"))
    poisson_residual_linf = _required(
        "poisson_residual_linf", (last_history, "poisson_residual_linf")
    )
    poisson_residual_linf_relative = _required(
        "poisson_residual_linf_relative",
        (last_history, "poisson_residual_linf_relative"),
    )

    metrics = {
        "max_u": _metric(final_summary, last_history, "max_u"),
        "max_speed": _metric(final_summary, last_history, "max_speed"),
        "p99_speed": _metric(final_summary, last_history, "p99_speed"),
        "centerline_max_u": float(final_summary.get("centerline_max_u", 0.0)),
        "mass_imbalance_rel": mass_imbalance_rel,
        "mass_imbalance_rel_raw": float(
            final_summary.get(
                "mass_imbalance_rel_raw",
                last_mass.get("mass_imbalance_rel_raw", last_mass.get("mass_imbalance_rel", 0.0)),
            )
        ),
        "mass_imbalance_rel_corrected": float(
            final_summary.get(
                "mass_imbalance_rel_corrected",
                last_mass.get("mass_imbalance_rel_corrected", last_mass.get("mass_imbalance_rel", 0.0)),
            )
        ),
        "divergence_linf": divergence_linf,
        "divergence_l2": divergence_l2,
        "divergence_linf_excluding_near_solid": float(
            last_history.get("divergence_linf_excluding_near_solid", 0.0)
        ),
        "divergence_l2_excluding_near_solid": float(
            last_history.get("divergence_l2_excluding_near_solid", 0.0)
        ),
        "poisson_residual_linf": poisson_residual_linf,
        "poisson_residual_linf_relative": poisson_residual_linf_relative,
    }

    inlet_u = float(thresholds["inlet_u"])
    visual_pass = (
        metrics["max_u"] > 1.5 * inlet_u
        and metrics["centerline_max_u"] > inlet_u
        and metrics["p99_speed"] > inlet_u
    )
    visual_candidate = _gate(
        "pass" if visual_pass else "fail",
        "centerline jet exists" if visual_pass else "centerline jet threshold not met",
    )

    if "mass_imbalance_rel" in missing_metrics:
        mass_quality = _gate("fail", "missing metric: mass_imbalance_rel")
    else:
        mass_abs = abs(metrics["mass_imbalance_rel"])
        if mass_abs < float(thresholds["max_mass_imbalance_rel"]):
            mass_quality = _gate("pass", f"final mass imbalance rel = {mass_abs:.6g}")
        elif mass_abs < 2.0 * float(thresholds["max_mass_imbalance_rel"]):
            mass_quality = _gate("warn", f"final mass imbalance rel = {mass_abs:.6g}")
        else:
            mass_quality = _gate("fail", f"final mass imbalance rel = {mass_abs:.6g}")

    poisson_pass = (
        metrics["poisson_residual_linf"]
        <= float(thresholds["max_poisson_residual_linf_warn"])
        or metrics["poisson_residual_linf_relative"]
        <= float(thresholds["max_poisson_residual_linf_relative"])
    )
    divergence_l2_for_gate = (
        metrics["divergence_l2_excluding_near_solid"]
        if metrics["divergence_l2_excluding_near_solid"] > 0.0
        else metrics["divergence_l2"]
    )
    divergence_linf_for_gate = (
        metrics["divergence_linf_excluding_near_solid"]
        if metrics["divergence_linf_excluding_near_solid"] > 0.0
        else metrics["divergence_linf"]
    )
    incompressibility_pass = (
        divergence_l2_for_gate <= float(thresholds["max_divergence_l2_warn"])
        and divergence_linf_for_gate <= float(thresholds["max_divergence_linf_warn"])
        and poisson_pass
    )
    incompressibility_reasons = [
        f"divergence_l2={metrics['divergence_l2']:.6g}",
        f"divergence_linf={metrics['divergence_linf']:.6g}",
        f"divergence_l2_excluding_near_solid={metrics['divergence_l2_excluding_near_solid']:.6g}",
        f"divergence_linf_excluding_near_solid={metrics['divergence_linf_excluding_near_solid']:.6g}",
        f"poisson_residual_linf={metrics['poisson_residual_linf']:.6g}",
        f"poisson_residual_linf_relative={metrics['poisson_residual_linf_relative']:.6g}",
    ]
    incompressibility_missing = [
        label
        for label in (
            "divergence_l2",
            "divergence_linf",
            "poisson_residual_linf",
            "poisson_residual_linf_relative",
        )
        if label in missing_metrics
    ]
    if incompressibility_missing and not incompressibility_pass:
        # Missing evidence is strictly worse than "measured but marginal":
        # a "warn" implies we have real data to review, so a fully missing
        # required metric must escalate to "fail" instead.
        incompressibility_quality = _gate(
            "fail",
            "; ".join(
                [f"missing metric: {label}" for label in incompressibility_missing]
                + incompressibility_reasons
            ),
        )
    else:
        incompressibility_quality = _gate(
            "pass" if incompressibility_pass else "warn",
            "; ".join(incompressibility_reasons),
        )

    if (
        visual_candidate["status"] == "pass"
        and mass_quality["status"] == "pass"
        and incompressibility_quality["status"] == "pass"
    ):
        overall_status = "candidate_not_parity"
    else:
        overall_status = "diagnostic_only_not_parity"

    return {
        "visual_candidate": visual_candidate,
        "mass_quality": mass_quality,
        "incompressibility_quality": incompressibility_quality,
        "overall_status": overall_status,
        "metrics": metrics,
        "thresholds": thresholds,
        "missing_metrics": missing_metrics,
    }


def _load_numeric_csv(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: _parse_float(value) for key, value in row.items()})
        return rows


def _parse_float(value: str) -> float:
    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _metric(
    primary: dict[str, Any], fallback: dict[str, float], key: str
) -> float:
    return float(primary.get(key, fallback.get(key, 0.0)))


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _gate(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}
