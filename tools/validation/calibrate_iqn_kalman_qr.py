"""Calibrate IQN--Kalman Q/R from accepted 250-step trial-vector evidence.

This is an offline CPU-only research analyzer.  It does not run a solver,
change a controller, evaluate trial reduction, or claim acceleration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_FRAME_COUNT = 250
_CALIBRATION_END = 100
_EVALUATION_START = 101
_Q_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0, 10.0)
_REQUIRED_FIELDS = (
    "marker_velocity_mps",
    "iqn_trial_guess_mps",
    "iqn_trial_candidate_mps",
    "iqn_trial_residual_mps",
    "iqn_trial_index",
    "iqn_trial_layout_sha256",
    "iqn_trial_step",
    "iqn_trial_time_s",
    "iqn_trial_dt_s",
)
_LAYOUT_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class IqnKalmanStudy:
    """Validated, accepted-only evidence needed for the offline replay."""

    accepted: np.ndarray
    trial_guesses: tuple[np.ndarray, ...]
    trial_candidates: tuple[np.ndarray, ...]
    trial_residuals: tuple[np.ndarray, ...]
    trial_indices: tuple[np.ndarray, ...]
    layout_sha256: str
    dt_s: float
    time_s: np.ndarray
    frame_paths: tuple[Path, ...]


def _as_finite_scalar(value: np.ndarray, *, context: str) -> float:
    if value.size != 1 or value.dtype.kind not in "fiu":
        raise ValueError(f"{context} must be one finite numeric scalar")
    number = float(value.reshape(-1)[0])
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _as_exact_integer(value: np.ndarray, *, context: str) -> int:
    if value.size != 1 or value.dtype.kind not in "iu":
        raise ValueError(f"{context} must be one integer scalar")
    return int(value.reshape(-1)[0])


def _as_layout_sha256(value: np.ndarray, *, context: str) -> str:
    if value.size != 1 or value.dtype.kind not in "US":
        raise ValueError(f"{context} must be one SHA256 string scalar")
    raw = value.reshape(-1)[0]
    layout = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
    if _LAYOUT_SHA256.fullmatch(layout) is None:
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return layout


def _as_finite_array(value: np.ndarray, *, context: str) -> np.ndarray:
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{context} must be numeric")
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} must be finite")
    return np.ascontiguousarray(result)


def _roundoff_close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 64.0 * np.finfo(np.float64).eps * scale


def _frame_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise ValueError(f"step-fields directory not found: {directory}")
    expected = {f"step_{step:04d}.npz" for step in range(1, _FRAME_COUNT + 1)}
    actual = {path.name for path in directory.glob("step_*.npz")}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        detail = ", ".join(
            ([f"missing {missing[0]}"] if missing else [])
            + ([f"unexpected {unexpected[0]}"] if unexpected else [])
        )
        raise ValueError(
            f"step-fields must contain exactly 250 contiguous frames ({detail})"
        )
    return tuple(directory / f"step_{step:04d}.npz" for step in range(1, 251))


def _read_frame(path: Path, *, expected_step: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float, float
]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in _REQUIRED_FIELDS if key not in archive.files]
            if missing:
                raise ValueError(f"{path}: missing required field {missing[0]}")
            accepted = _as_finite_array(
                archive["marker_velocity_mps"], context=f"{path}: marker_velocity_mps"
            )
            guess = _as_finite_array(
                archive["iqn_trial_guess_mps"], context=f"{path}: iqn_trial_guess_mps"
            )
            candidate = _as_finite_array(
                archive["iqn_trial_candidate_mps"],
                context=f"{path}: iqn_trial_candidate_mps",
            )
            residual = _as_finite_array(
                archive["iqn_trial_residual_mps"],
                context=f"{path}: iqn_trial_residual_mps",
            )
            trial_index = np.asarray(archive["iqn_trial_index"])
            layout = _as_layout_sha256(
                np.asarray(archive["iqn_trial_layout_sha256"]),
                context=f"{path}: iqn_trial_layout_sha256",
            )
            step = _as_exact_integer(
                np.asarray(archive["iqn_trial_step"]), context=f"{path}: iqn_trial_step"
            )
            time_s = _as_finite_scalar(
                np.asarray(archive["iqn_trial_time_s"]), context=f"{path}: iqn_trial_time_s"
            )
            dt_s = _as_finite_scalar(
                np.asarray(archive["iqn_trial_dt_s"]), context=f"{path}: iqn_trial_dt_s"
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable IQN trial evidence {path}: {exc}") from exc

    if step != expected_step:
        raise ValueError(f"{path}: iqn_trial_step {step} != filename step {expected_step}")
    if dt_s <= 0.0:
        raise ValueError(f"{path}: iqn_trial_dt_s must be positive")
    if accepted.ndim != 2 or accepted.shape[0] <= 0 or accepted.shape[1] != 3:
        raise ValueError(f"{path}: marker_velocity_mps must have shape (M, 3)")
    expected_trial_shape = (guess.shape[0],) + accepted.shape
    if guess.ndim != 3 or guess.shape != expected_trial_shape:
        raise ValueError(f"{path}: iqn_trial_guess_mps must have shape (T, M, 3)")
    if candidate.shape != expected_trial_shape or residual.shape != expected_trial_shape:
        raise ValueError(f"{path}: IQN trial vector shapes must match (T, M, 3)")
    if guess.shape[0] < 1:
        raise ValueError(f"{path}: at least one trial candidate is required")
    if trial_index.dtype.kind not in "iu" or trial_index.ndim != 1:
        raise ValueError(f"{path}: iqn_trial_index must be a one-dimensional integer array")
    expected_indices = np.arange(guess.shape[0], dtype=trial_index.dtype)
    if not np.array_equal(trial_index, expected_indices):
        raise ValueError(f"{path}: iqn_trial_index must equal contiguous 0-based trial indices")
    if not np.allclose(residual, candidate - guess, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError(f"{path}: iqn_trial_residual_mps does not equal candidate - guess")
    return accepted, guess, candidate, residual, trial_index, layout, time_s, dt_s


def load_iqn_kalman_study(step_fields_directory: Path | str) -> IqnKalmanStudy:
    """Load exactly 250 validated accepted IQN frames, or fail closed."""

    paths = _frame_paths(Path(step_fields_directory))
    accepted_rows: list[np.ndarray] = []
    guesses: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    times: list[float] = []
    expected_shape: tuple[int, int] | None = None
    expected_layout: str | None = None
    expected_dt: float | None = None
    for step, path in enumerate(paths, start=1):
        row = _read_frame(path, expected_step=step)
        accepted, guess, candidate, residual, trial_index, layout, time_s, dt_s = row
        if expected_shape is None:
            expected_shape = accepted.shape
            expected_layout = layout
            expected_dt = dt_s
        elif accepted.shape != expected_shape:
            raise ValueError(f"{path}: marker_velocity_mps shape changed")
        elif layout != expected_layout:
            raise ValueError(f"{path}: IQN trial layout changed")
        elif not _roundoff_close(dt_s, expected_dt):
            raise ValueError(f"{path}: iqn_trial_dt_s changed")
        expected_time = step * float(expected_dt)
        if not _roundoff_close(time_s, expected_time):
            raise ValueError(
                f"{path}: iqn_trial_time_s {time_s!r} is not continuous at step {step}"
            )
        accepted_rows.append(accepted)
        guesses.append(guess)
        candidates.append(candidate)
        residuals.append(residual)
        indices.append(np.ascontiguousarray(trial_index, dtype=np.int64))
        times.append(time_s)
    return IqnKalmanStudy(
        accepted=np.stack(accepted_rows),
        trial_guesses=tuple(guesses),
        trial_candidates=tuple(candidates),
        trial_residuals=tuple(residuals),
        trial_indices=tuple(indices),
        layout_sha256=str(expected_layout),
        dt_s=float(expected_dt),
        time_s=np.asarray(times, dtype=np.float64),
        frame_paths=paths,
    )


def _sample_variance(samples: np.ndarray) -> np.ndarray:
    flattened = samples.reshape(-1, 3)
    if flattened.shape[0] < 2:
        raise ValueError("calibration requires at least two samples per axis")
    return np.var(flattened, axis=0, ddof=1)


def _raise_calibration_trial_count(step: int) -> np.ndarray:
    raise ValueError(
        f"calibration step {step} requires at least two trial candidates"
    )


def _calibrate(study: IqnKalmanStudy) -> tuple[np.ndarray, np.ndarray, list[str]]:
    candidate_deltas = np.concatenate(
        [
            candidate[-1] - candidate[-2]
            if candidate.shape[0] >= 2
            else _raise_calibration_trial_count(step + 1)
            for step, candidate in enumerate(study.trial_candidates[:_CALIBRATION_END])
        ],
        axis=0,
    )
    r_xyz = _sample_variance(candidate_deltas)
    velocity = study.accepted[:_CALIBRATION_END]
    jerk = (velocity[2:] - 2.0 * velocity[1:-1] + velocity[:-2]) / study.dt_s**2
    q0_xyz = _sample_variance(jerk) * study.dt_s
    statuses: list[str] = []
    for q0, measurement in zip(q0_xyz, r_xyz, strict=True):
        if q0 == 0.0 and measurement == 0.0:
            statuses.append("inactive_zero_variance")
        elif measurement == 0.0:
            statuses.append("inactive_zero_measurement_variance")
        elif q0 == 0.0:
            statuses.append("active_zero_process_variance")
        else:
            statuses.append("active")
    return q0_xyz, r_xyz, statuses


def _positive_finite(value: float | None, *, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be explicitly supplied as a positive finite value")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be explicitly supplied as a positive finite value")
    return number


def _replay_kalman(
    accepted: np.ndarray,
    *,
    dt_s: float,
    q_xyz: np.ndarray,
    r_xyz: np.ndarray,
    active_axes: np.ndarray,
    score_start: int,
    score_end: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    marker_count = accepted.shape[1]
    mean = np.zeros((marker_count, 3, 2), dtype=np.float64)
    mean[:, :, 0] = accepted[0]
    covariance = np.zeros((marker_count, 3, 2, 2), dtype=np.float64)
    covariance[:, :, 0, 0] = r_xyz
    covariance[:, :, 1, 1] = r_xyz / dt_s**2
    transition = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=np.float64)
    process_template = np.array(
        [[dt_s**3 / 3.0, dt_s**2 / 2.0], [dt_s**2 / 2.0, dt_s]],
        dtype=np.float64,
    )
    predicted_rows: list[np.ndarray] = []
    innovation_covariances: list[np.ndarray] = []
    for index in range(1, len(accepted)):
        predicted_mean = mean @ transition.T
        predicted_covariance = (
            transition @ covariance @ transition.T
            + q_xyz[None, :, None, None] * process_template[None, None, :, :]
        )
        predicted = predicted_mean[:, :, 0]
        innovation = accepted[index] - predicted
        innovation_variance = predicted_covariance[:, :, 0, 0] + r_xyz[None, :]
        safe_innovation_variance = innovation_variance.copy()
        safe_innovation_variance[:, ~active_axes] = 1.0
        gain = predicted_covariance[:, :, :, 0] / safe_innovation_variance[:, :, None]
        gain[:, ~active_axes, :] = 0.0
        mean = predicted_mean + gain * innovation[:, :, None]
        identity_minus_kh = np.broadcast_to(
            np.eye(2, dtype=np.float64), predicted_covariance.shape
        ).copy()
        identity_minus_kh[:, :, 0, 0] -= gain[:, :, 0]
        identity_minus_kh[:, :, 1, 0] -= gain[:, :, 1]
        covariance = (
            identity_minus_kh @ predicted_covariance @ np.swapaxes(identity_minus_kh, -1, -2)
            + r_xyz[None, :, None, None] * gain[:, :, :, None] * gain[:, :, None, :]
        )
        mean[:, ~active_axes, 0] = accepted[index][:, ~active_axes]
        mean[:, ~active_axes, 1] = 0.0
        covariance[:, ~active_axes] = 0.0
        if score_start <= index < score_end:
            predicted_rows.append(predicted)
            if np.any(active_axes):
                innovation_covariances.append(innovation_variance)
    if not predicted_rows:
        raise ValueError("no replay samples in requested score range")
    covariance_rows = (
        np.stack(innovation_covariances) if innovation_covariances else None
    )
    return np.stack(predicted_rows), covariance_rows


def _carry_predictions(accepted: np.ndarray, *, start: int, end: int) -> np.ndarray:
    return accepted[start - 1 : end - 1].copy()


def _linear_predictions(accepted: np.ndarray, *, start: int, end: int) -> np.ndarray:
    return accepted[start - 1 : end - 1] + (accepted[start - 1 : end - 1] - accepted[start - 2 : end - 2])


def _metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    innovation_covariance: np.ndarray | None,
) -> dict[str, Any]:
    error = predicted - observed
    axis_rmse = np.sqrt(np.mean(error * error, axis=(0, 1)))
    axis_bias = np.mean(error, axis=(0, 1))
    result: dict[str, Any] = {
        "sample_count": int(error.shape[0] * error.shape[1]),
        "axis_rmse_mps": [float(value) for value in axis_rmse],
        "axis_bias_mps": [float(value) for value in axis_bias],
        "global_rmse_mps": float(np.sqrt(np.mean(error * error))),
        "global_bias_mps": float(np.mean(error)),
        "axis_nis": None,
        "global_nis": None,
    }
    if innovation_covariance is not None:
        squared_innovation = (observed - predicted) ** 2
        valid = innovation_covariance > 0.0
        axis_nis: list[float | None] = []
        for axis in range(3):
            axis_valid = valid[:, :, axis]
            if not np.any(axis_valid):
                axis_nis.append(None)
            else:
                axis_nis.append(
                    float(np.mean(squared_innovation[:, :, axis][axis_valid] / innovation_covariance[:, :, axis][axis_valid]))
                )
        all_valid = valid
        result["axis_nis"] = axis_nis
        result["global_nis"] = (
            float(np.mean(squared_innovation[all_valid] / innovation_covariance[all_valid]))
            if np.any(all_valid)
            else None
        )
    return result


def _method_metrics(
    study: IqnKalmanStudy,
    *,
    q_xyz: np.ndarray,
    r_xyz: np.ndarray,
    active_axes: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    observed = study.accepted[start:end]
    carry = _metrics(
        _carry_predictions(study.accepted, start=start, end=end), observed, innovation_covariance=None
    )
    linear = _metrics(
        _linear_predictions(study.accepted, start=start, end=end), observed, innovation_covariance=None
    )
    prediction, covariance = _replay_kalman(
        study.accepted,
        dt_s=study.dt_s,
        q_xyz=q_xyz,
        r_xyz=r_xyz,
        active_axes=active_axes,
        score_start=start,
        score_end=end,
    )
    return {
        "carry_forward": carry,
        "linear_extrapolation": linear,
        "kalman": _metrics(prediction, observed, innovation_covariance=covariance),
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "split", "method", "axis", "sample_count", "rmse_mps", "bias_mps", "nis",
        "layout_sha256", "dt_s", "selected_q_multiplier", "selection_used_frozen_eval",
        "trial_reduction_evaluated", "acceleration_claimed", "measurement_variance_formula",
        "process_variance_formula", "calibration_steps", "frozen_evaluation_steps",
    )
    frozen = result["frozen_evaluation"]
    provenance = result["provenance"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_name, metrics in frozen["methods"].items():
            common = {
                "split": "frozen_evaluation_101_250",
                "method": method_name,
                "sample_count": metrics["sample_count"],
                "layout_sha256": provenance["layout_sha256"],
                "dt_s": provenance["dt_s"],
                "selected_q_multiplier": result["selection"]["selected_q_multiplier"],
                "selection_used_frozen_eval": False,
                "trial_reduction_evaluated": False,
                "acceleration_claimed": False,
                "measurement_variance_formula": result["formula"]["measurement_variance"],
                "process_variance_formula": result["formula"]["process_variance"],
                "calibration_steps": "1-100",
                "frozen_evaluation_steps": "101-250",
            }
            writer.writerow({
                **common, "axis": "global", "rmse_mps": metrics["global_rmse_mps"],
                "bias_mps": metrics["global_bias_mps"], "nis": metrics["global_nis"],
            })
            for axis, (rmse, bias) in enumerate(zip(metrics["axis_rmse_mps"], metrics["axis_bias_mps"], strict=True)):
                nis_values = metrics["axis_nis"]
                writer.writerow({
                    **common, "axis": "xyz"[axis], "rmse_mps": rmse, "bias_mps": bias,
                    "nis": None if nis_values is None else nis_values[axis],
                })


def calibrate_and_evaluate(
    step_fields_directory: Path | str,
    *,
    scalar_q: float | None,
    scalar_r: float | None,
    json_path: Path | str | None = None,
    csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Calibrate on frames 1--100 and causally replay frozen frames 101--250."""

    scalar_q_value = _positive_finite(scalar_q, name="scalar_q")
    scalar_r_value = _positive_finite(scalar_r, name="scalar_r")
    study = load_iqn_kalman_study(step_fields_directory)
    q0_xyz, r_xyz, axis_status = _calibrate(study)
    active_axes = np.asarray(
        [not status.startswith("inactive_") for status in axis_status]
    )

    calibration_scores: list[dict[str, Any]] = []
    for multiplier in _Q_MULTIPLIERS:
        prediction, covariance = _replay_kalman(
            study.accepted,
            dt_s=study.dt_s,
            q_xyz=q0_xyz * multiplier,
            r_xyz=r_xyz,
            active_axes=active_axes,
            score_start=1,
            score_end=_CALIBRATION_END,
        )
        metrics = _metrics(
            prediction,
            study.accepted[1:_CALIBRATION_END],
            innovation_covariance=covariance,
        )
        calibration_scores.append({"q_multiplier": multiplier, **metrics})
    selected = min(calibration_scores, key=lambda row: (row["global_rmse_mps"], row["q_multiplier"]))
    selected_multiplier = float(selected["q_multiplier"])

    start = _EVALUATION_START - 1
    end = _FRAME_COUNT
    frozen_methods = _method_metrics(
        study,
        q_xyz=np.full(3, scalar_q_value),
        r_xyz=np.full(3, scalar_r_value),
        active_axes=np.ones(3, dtype=bool),
        start=start,
        end=end,
    )
    per_axis_prediction, per_axis_covariance = _replay_kalman(
        study.accepted,
        dt_s=study.dt_s,
        q_xyz=q0_xyz * selected_multiplier,
        r_xyz=r_xyz,
        active_axes=active_axes,
        score_start=start,
        score_end=end,
    )
    frozen_methods["scalar_kalman"] = frozen_methods.pop("kalman")
    frozen_methods["per_axis_kalman"] = _metrics(
        per_axis_prediction,
        study.accepted[start:end],
        innovation_covariance=per_axis_covariance,
    )

    result: dict[str, Any] = {
        "provenance": {
            "input_step_fields_directory": str(Path(step_fields_directory)),
            "frame_count": _FRAME_COUNT,
            "frame_name_pattern": "step_XXXX.npz",
            "layout_sha256": study.layout_sha256,
            "marker_count": int(study.accepted.shape[1]),
            "dt_s": study.dt_s,
            "time_range_s": [float(study.time_s[0]), float(study.time_s[-1])],
            "required_fields": list(_REQUIRED_FIELDS),
        },
        "formula": {
            "measurement_variance": "sample_var(last_trial_candidate - penultimate_trial_candidate, ddof=1)",
            "process_variance": "sample_var(jerk(accepted_marker_velocity_mps), ddof=1) * dt_s",
            "jerk": "(v_n - 2*v_n_minus_1 + v_n_minus_2) / dt_s^2",
            "units": {
                "r_xyz": "m^2/s^2",
                "q0_xyz": "m^2/s^5",
                "scalar_r": "m^2/s^2",
                "scalar_q": "m^2/s^5",
            },
        },
        "ranges": {
            "calibration_steps": [1, _CALIBRATION_END],
            "frozen_evaluation_steps": [_EVALUATION_START, _FRAME_COUNT],
        },
        "calibration": {
            "r_xyz_m2_per_s2": [float(value) for value in r_xyz],
            "q0_xyz_m2_per_s5": [float(value) for value in q0_xyz],
            "axis_status": axis_status,
            "scalar_kalman_cli": {"q": scalar_q_value, "r": scalar_r_value},
        },
        "selection": {
            "q_multiplier_candidates": list(_Q_MULTIPLIERS),
            "calibration_scores": calibration_scores,
            "selected_q_multiplier": selected_multiplier,
            "selection_criterion": "minimum causal calibration global RMSE; ties choose lower multiplier",
            "selection_used_frozen_eval": False,
        },
        "frozen_evaluation": {"methods": frozen_methods},
        "trial_reduction_evaluated": False,
        "acceleration_claimed": False,
    }
    if json_path is not None:
        output = Path(json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_path is not None:
        _write_csv(Path(csv_path), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-fields", required=True, type=Path)
    parser.add_argument("--scalar-q", required=True, type=float)
    parser.add_argument("--scalar-r", required=True, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        calibrate_and_evaluate(
            args.step_fields,
            scalar_q=args.scalar_q,
            scalar_r=args.scalar_r,
            json_path=args.output_json,
            csv_path=args.output_csv,
        )
    except ValueError as exc:
        print(f"IQN Kalman calibration failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
