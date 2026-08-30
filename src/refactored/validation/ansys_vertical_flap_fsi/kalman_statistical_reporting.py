"""Deterministic diagnostics and artifact publication for the R24 audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .kalman_statistical_replay import ReplayResult, ReplayRow
from .kalman_statistical_selection import CandidateRanking
from .kalman_statistical_types import _fingerprint


_AXIS_ORDER = ("x", "y", "z")
_LJUNG_BOX_CRITICAL_5PCT = {
    1: 3.841458820694124,
    2: 5.991464547107979,
    3: 7.814727903251179,
}
_SEGMENTS = {
    "steps_1_5": (1, 5),
    "steps_6_15": (6, 15),
    "steps_16_31": (16, 31),
    "steps_32_41": (32, 41),
    "step_42": (42, 42),
    "steps_43_49": (43, 49),
    "step_50": (50, 50),
}


def _finite_values(values: Sequence[float], *, context: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{context} must contain finite samples")
    return result


def _distribution(values: Sequence[float], *, context: str) -> dict[str, float | int]:
    array = _finite_values(values, context=context)
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": median,
        "robust_scale_mad": float(
            1.4826 * np.median(np.abs(array - median))
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag < 1 or lag >= len(values):
        return 0.0
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.dot(centered[:-lag], centered[lag:]) / denominator)


def _ljung_box(values: np.ndarray) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    autocorrelation = {
        f"lag_{lag}": _autocorrelation(values, lag) for lag in range(1, 4)
    }
    count = len(values)
    cumulative = 0.0
    reports: dict[str, dict[str, Any]] = {}
    for lag in range(1, 4):
        rho = autocorrelation[f"lag_{lag}"]
        if count > lag:
            cumulative += rho * rho / (count - lag)
        statistic = float(count * (count + 2) * cumulative)
        critical = _LJUNG_BOX_CRITICAL_5PCT[lag]
        reports[f"lag_{lag}"] = {
            "statistic": statistic,
            "degrees_of_freedom": lag,
            "critical_5pct": critical,
            "reject_white_noise_5pct": statistic > critical,
        }
    return autocorrelation, reports


def _same_sign_runs(values: np.ndarray) -> dict[str, int]:
    signs = np.sign(values)
    longest_positive = 0
    longest_negative = 0
    current_sign = 0.0
    current_length = 0
    run_count = 0
    for sign in signs:
        if sign == 0.0:
            current_sign = 0.0
            current_length = 0
            continue
        if sign != current_sign:
            current_sign = float(sign)
            current_length = 1
            run_count += 1
        else:
            current_length += 1
        if sign > 0:
            longest_positive = max(longest_positive, current_length)
        else:
            longest_negative = max(longest_negative, current_length)
    return {
        "run_count": run_count,
        "longest_positive": longest_positive,
        "longest_negative": longest_negative,
        "longest_any": max(longest_positive, longest_negative),
    }


def _axis_dof(rows: Sequence[ReplayRow], field: str, axis: int) -> np.ndarray:
    values = [
        value
        for row in rows
        for value in getattr(row, field)[axis]
    ]
    return _finite_values(values, context=f"{field}[{axis}]")


def _segment_summary(rows: Sequence[ReplayRow]) -> dict[str, Any]:
    if not rows:
        return {
            "sampled_steps": [],
            "normalized_rmse": 0.0,
            "nis_mean": None,
            "fsi_iterations": 0,
            "cg_iterations": 0,
            "matvec_count": None,
        }
    axis_rmse = np.asarray(
        [row.effective_axis_normalized_rmse for row in rows]
    )
    active = np.asarray(rows[0].active_axes, dtype=bool)
    statistical = [row for row in rows if row.statistical_metrics_available]
    nis_values = [
        value
        for row in statistical
        for axis, axis_values in enumerate(row.nis_dof_by_axis)
        if active[axis]
        for value in axis_values
    ]
    return {
        "sampled_steps": [row.physical_step for row in rows],
        "normalized_rmse": float(np.sqrt(np.mean(np.square(axis_rmse[:, active])))),
        "nis_mean": (
            None
            if not statistical
            else float(np.mean(nis_values))
        ),
        "fsi_iterations": int(sum(row.fsi_iterations for row in rows)),
        "cg_iterations": int(sum(row.cg_iterations for row in rows)),
        "matvec_count": (
            None
            if all(row.matvec_count is None for row in rows)
            else int(sum(row.matvec_count or 0 for row in rows))
        ),
    }


def summarize_candidate(replay: ReplayResult) -> dict[str, Any]:
    """Summarize one replay without discarding recomputable row telemetry."""

    rows = replay.rows
    if not rows:
        raise ValueError("candidate replay contains no rows")
    statistical = any(row.statistical_metrics_available for row in rows)
    active = np.asarray(rows[0].active_axes, dtype=bool)
    axis_payload: dict[str, Any] = {}
    serial_bias_detected = False
    for axis, axis_name in enumerate(_AXIS_ORDER):
        innovations = _axis_dof(rows, "innovation_dof_by_axis", axis)
        innovation_variance = _axis_dof(
            rows, "innovation_variance_dof_by_axis", axis
        )
        nis = _axis_dof(rows, "nis_dof_by_axis", axis)
        gain = _axis_dof(rows, "gain_dof_by_axis", axis)
        time_series = np.asarray(
            [row.innovation_axis_mean_mps[axis] for row in rows],
            dtype=np.float64,
        )
        autocorrelation, ljung_box = _ljung_box(time_series)
        innovation_stats = _distribution(
            innovations, context=f"{axis_name} innovation"
        )
        standardized_bias = abs(float(innovation_stats["mean"])) / max(
            float(innovation_stats["std"]), np.finfo(np.float64).tiny
        )
        persistent = bool(active[axis]) and (
            standardized_bias > 0.25
            or bool(ljung_box["lag_3"]["reject_white_noise_5pct"])
        )
        serial_bias_detected = serial_bias_detected or persistent
        axis_payload[axis_name] = {
            "active": bool(active[axis]),
            "innovation_mps": innovation_stats,
            "innovation_variance": _distribution(
                innovation_variance,
                context=f"{axis_name} innovation variance",
            ),
            "nis": _distribution(nis, context=f"{axis_name} NIS"),
            "nis_95pct_exceedance_fraction": float(
                np.mean(nis > 3.841458820694124)
            ),
            "gain": _distribution(gain, context=f"{axis_name} gain"),
            "standardized_innovation_bias": standardized_bias,
            "lag_autocorrelation": autocorrelation,
            "ljung_box": ljung_box,
            "same_sign_runs": _same_sign_runs(time_series),
            "persistent_bias_or_serial_pattern": persistent,
        }
    segments = {
        name: _segment_summary(
            [
                row
                for row in rows
                if start <= row.physical_step <= stop
            ]
        )
        for name, (start, stop) in _SEGMENTS.items()
    }
    axis_normalized_rmse = np.sqrt(
        np.mean(
            np.square(
                np.asarray(
                    [row.effective_axis_normalized_rmse for row in rows]
                )
            ),
            axis=0,
        )
    )
    covariance_ok = all(
        row.covariance_finite
        and row.covariance_symmetry_error <= 1.0e-10
        and row.covariance_min_eigenvalue >= -1.0e-10
        for row in rows
    )
    nis_axis_means = np.asarray(
        [row.nis_axis_mean for row in rows], dtype=np.float64
    )
    positive_nis = np.maximum(
        np.mean(nis_axis_means[:, active], axis=0), 1.0e-300
    )
    axis_scale_ratio = float(np.max(positive_nis) / np.min(positive_nis))
    return {
        "schema_version": 1,
        "trace_name": replay.trace_name,
        "candidate_id": replay.candidate_id,
        "candidate_fingerprint": replay.candidate_fingerprint,
        "step_count": len(rows),
        "prediction": {
            "normalized_rmse": float(
                np.sqrt(np.mean(np.square(axis_normalized_rmse[active])))
            ),
            "axis_normalized_rmse": {
                axis: float(axis_normalized_rmse[index])
                for index, axis in enumerate(_AXIS_ORDER)
            },
            "maximum_component_error_mps": float(
                max(row.max_component_error_mps for row in rows)
            ),
        },
        "innovation": {
            "available": statistical,
            "axis": axis_payload,
            "persistent_bias_or_serial_pattern": serial_bias_detected,
            "axis_nis_scale_ratio": axis_scale_ratio,
        },
        "covariance": {
            "finite_symmetric_psd": covariance_ok,
            "maximum_symmetry_error": float(
                max(row.covariance_symmetry_error for row in rows)
            ),
            "minimum_eigenvalue": float(
                min(row.covariance_min_eigenvalue for row in rows)
            ),
        },
        "fallback_count": sum(row.fallback_reason is not None for row in rows),
        "reset_count": sum(row.reset_reason is not None for row in rows),
        "segments": segments,
    }


def classify_r24(
    *,
    provenance_ok: bool,
    k0_parity_ok: bool,
    contracts_ok: bool,
    kalman_predictive_value: bool,
    kalman_statistically_valid: bool,
) -> str:
    if not provenance_ok or not k0_parity_ok or not contracts_ok:
        return "FAIL_EVIDENCE_OR_IMPLEMENTATION_CONTRACT"
    if not kalman_predictive_value:
        return "FAIL_NO_KALMAN_PREDICTIVE_VALUE"
    if not kalman_statistically_valid:
        return "FAIL_STATISTICAL_MODEL"
    return "PASS_ADVANCE_TO_R25"


def _fingerprinted(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("artifact_fingerprint", None)
    result["artifact_fingerprint"] = _fingerprint(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _fingerprinted(payload)
    encoded = (
        json.dumps(
            result,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return result


def _write_step_csv(
    path: Path,
    replays_by_split: Mapping[str, Sequence[ReplayResult]],
) -> None:
    fields = (
        "schema_version",
        "split",
        "physical_step",
        "accepted_state_source_step",
        "accepted_state_source_sha256",
        "accepted_measurement_sha256",
        "history_sha256",
        "journal_sha256",
        "layout_id",
        "axis",
        "active_axis",
        "candidate_id",
        "model_name",
        "candidate_fingerprint",
        "raw_prediction_mean_mps",
        "measurement_mean_mps",
        "innovation_mean_mps",
        "innovation_std_mps",
        "innovation_variance_mean",
        "nis_mean",
        "nis_95pct_exceedance_fraction",
        "kalman_gain_mean",
        "p_prior_mean",
        "p_posterior_mean",
        "q_identity",
        "r_identity",
        "p0_identity",
        "fallback_reason",
        "reset_reason",
        "normalized_rmse",
        "max_component_error_mps",
        "representative_error_mps",
        "fsi_iterations",
        "cg_iterations",
        "matvec_count",
        "covariance_finite",
        "covariance_symmetry_error",
        "covariance_min_eigenvalue",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split in sorted(replays_by_split):
            for replay in sorted(
                replays_by_split[split], key=lambda item: item.candidate_id
            ):
                for row in replay.rows:
                    for axis_index, axis in enumerate(_AXIS_ORDER):
                        writer.writerow(
                            {
                                "schema_version": 1,
                                "split": split,
                                "physical_step": row.physical_step,
                                "accepted_state_source_step": (
                                    row.accepted_state_source_step
                                ),
                                "accepted_state_source_sha256": (
                                    row.accepted_state_source_sha256
                                ),
                                "accepted_measurement_sha256": (
                                    row.accepted_measurement_sha256
                                ),
                                "history_sha256": row.history_sha256,
                                "journal_sha256": row.journal_sha256,
                                "layout_id": row.layout_id,
                                "axis": axis,
                                "active_axis": row.active_axes[axis_index],
                                "candidate_id": row.candidate_id,
                                "model_name": row.model,
                                "candidate_fingerprint": (
                                    row.candidate_fingerprint
                                ),
                                "raw_prediction_mean_mps": (
                                    row.raw_prediction_axis_mean_mps[axis_index]
                                ),
                                "measurement_mean_mps": (
                                    row.measurement_axis_mean_mps[axis_index]
                                ),
                                "innovation_mean_mps": (
                                    row.innovation_axis_mean_mps[axis_index]
                                ),
                                "innovation_std_mps": (
                                    row.innovation_axis_std_mps[axis_index]
                                ),
                                "innovation_variance_mean": (
                                    row.innovation_variance_axis_mean[axis_index]
                                ),
                                "nis_mean": row.nis_axis_mean[axis_index],
                                "nis_95pct_exceedance_fraction": (
                                    row.nis_axis_exceedance_fraction[axis_index]
                                ),
                                "kalman_gain_mean": (
                                    row.gain_axis_mean[axis_index]
                                ),
                                "p_prior_mean": (
                                    row.p_prior_axis_mean[axis_index]
                                ),
                                "p_posterior_mean": (
                                    row.p_posterior_axis_mean[axis_index]
                                ),
                                "q_identity": row.q_identity,
                                "r_identity": row.r_identity,
                                "p0_identity": row.p0_identity,
                                "fallback_reason": row.fallback_reason,
                                "reset_reason": row.reset_reason,
                                "normalized_rmse": (
                                    row.effective_axis_normalized_rmse[axis_index]
                                ),
                                "max_component_error_mps": (
                                    row.max_component_error_mps
                                ),
                                "representative_error_mps": (
                                    row.representative_error_mps[axis_index]
                                ),
                                "fsi_iterations": row.fsi_iterations,
                                "cg_iterations": row.cg_iterations,
                                "matvec_count": row.matvec_count,
                                "covariance_finite": row.covariance_finite,
                                "covariance_symmetry_error": (
                                    row.covariance_symmetry_error
                                ),
                                "covariance_min_eigenvalue": (
                                    row.covariance_min_eigenvalue
                                ),
                            }
                        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifact_bundle(
    *,
    output_dir: Path | str,
    split_manifest: Mapping[str, Any],
    replays_by_split: Mapping[str, Sequence[ReplayResult]],
    ranking: CandidateRanking,
    k0_parity: Mapping[str, Any],
    exit_classification: str,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the four deterministic, finite, schema-versioned R24 artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "kalman_data_split_manifest.json"
    audit_path = output / "kalman_innovation_audit.json"
    csv_path = output / "kalman_candidate_step_metrics.csv"
    ranking_path = output / "kalman_candidate_ranking.json"

    split_payload = _write_json(split_path, split_manifest)
    audit_payload = _write_json(
        audit_path,
        {
            "schema_version": 1,
            "k0_parity": dict(k0_parity),
            "splits": {
                split: {
                    replay.candidate_id: {
                        "summary": summarize_candidate(replay),
                        "rows": [row.to_payload() for row in replay.rows],
                    }
                    for replay in sorted(
                        replays, key=lambda item: item.candidate_id
                    )
                }
                for split, replays in sorted(replays_by_split.items())
            },
        },
    )
    _write_step_csv(csv_path, replays_by_split)
    ranking_payload = _write_json(
        ranking_path,
        {
            "schema_version": 1,
            "exit_classification": exit_classification,
            "ranking": ranking.to_payload(),
            "decision": {} if decision is None else dict(decision),
            "artifact_sha256": {
                split_path.name: _sha256_file(split_path),
                audit_path.name: _sha256_file(audit_path),
                csv_path.name: _sha256_file(csv_path),
            },
        },
    )
    return {
        "split_manifest": split_payload,
        "innovation_audit": audit_payload,
        "candidate_ranking": ranking_payload,
        "files": {
            path.name: _sha256_file(path)
            for path in (split_path, audit_path, csv_path, ranking_path)
        },
    }


def run_r24_campaign(**kwargs: Any) -> dict[str, Any]:
    from .kalman_statistical_campaign import run_r24_campaign as run

    return run(**kwargs)


__all__ = [
    "classify_r24",
    "run_r24_campaign",
    "summarize_candidate",
    "write_artifact_bundle",
]
