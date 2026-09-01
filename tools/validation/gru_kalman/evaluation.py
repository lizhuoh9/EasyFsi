"""Frozen-holdout prediction, metrics, and exact R25A gate calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .baselines import BaselineResult, evaluate_baseline
from .dataset import ACTIVE_AXES, D1_SCORE_STEPS, DatasetContractError, build_temporal_samples
from .models import ResidualGRU, make_gru_features, to_torch
from .pod import ModalNormalization, PODARModel, PODBasis


class EvaluationContractError(DatasetContractError):
    """A prediction, metric, provenance, or gate input is invalid."""


def _array(values: Any, *, label: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise EvaluationContractError(f"{label} must have rank {ndim}")
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise EvaluationContractError(f"{label} must be non-empty and finite")
    if result.ndim >= 2 and result.shape[-1] == 3 and not np.all(result[..., 0] == 0.0):
        raise EvaluationContractError(f"{label} x axis must be exactly zero")
    return np.array(result, copy=True)


@dataclass(frozen=True)
class Metrics:
    global_active_yz_nrmse: float
    axis_rmse: tuple[float, float, float]
    axis_bias: tuple[float, float, float]
    per_step_rms: tuple[float, ...]
    p95_marker_euclidean: tuple[float, ...]
    max_marker_euclidean: tuple[float, ...]
    rho_per_step: tuple[float, ...]
    rho_median: float
    rho_p95: float
    fraction_rho_lt_1: float
    fraction_rho_lt_01: float
    fraction_rho_gt_2: float
    alpha_parallel: tuple[float, ...]
    r_perp: tuple[float, ...]
    score_steps: tuple[int, ...]
    paired_rho_per_step: tuple[float, ...] = ()
    paired_rho_p95: float | None = None
    global_marker_p95: float = 0.0
    global_marker_max: float = 0.0

    @property
    def nrmse(self) -> float:
        return self.global_active_yz_nrmse

    @property
    def median_step_fraction_beating_carry(self) -> float:
        return float(np.mean(np.asarray(self.rho_per_step) < 1.0))

    @property
    def fraction_beating_paired(self) -> float | None:
        if not self.paired_rho_per_step:
            return None
        return float(np.mean(np.asarray(self.paired_rho_per_step) < 1.0))

    def to_payload(self) -> dict[str, Any]:
        return {
            "global_active_yz_nrmse": self.global_active_yz_nrmse,
            "axis_rmse": list(self.axis_rmse),
            "axis_bias": list(self.axis_bias),
            "per_step_rms": list(self.per_step_rms),
            "p95_marker_euclidean": list(self.p95_marker_euclidean),
            "max_marker_euclidean": list(self.max_marker_euclidean),
            "rho_per_step": list(self.rho_per_step),
            "rho_median": self.rho_median,
            "rho_p95": self.rho_p95,
            "fraction_rho_lt_1": self.fraction_rho_lt_1,
            "fraction_rho_lt_01": self.fraction_rho_lt_01,
            "fraction_rho_gt_2": self.fraction_rho_gt_2,
            "alpha_parallel": list(self.alpha_parallel),
            "r_perp": list(self.r_perp),
            "score_steps": list(self.score_steps),
            "paired_rho_per_step": list(self.paired_rho_per_step),
            "paired_rho_p95": self.paired_rho_p95,
            "global_marker_p95": self.global_marker_p95,
            "global_marker_max": self.global_marker_max,
        }


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    metrics: Metrics

    @property
    def nrmse(self) -> float:
        return self.metrics.global_active_yz_nrmse

    @property
    def fraction_beating_carry(self) -> float:
        return self.metrics.median_step_fraction_beating_carry

    @property
    def p95_rho(self) -> float:
        return self.metrics.rho_p95

    @property
    def fraction_beating_paired(self) -> float | None:
        return self.metrics.fraction_beating_paired


def _score_slice(length: int, start_step: int, end_step: int | None) -> tuple[slice, tuple[int, ...]]:
    if isinstance(start_step, bool) or start_step < 1:
        raise EvaluationContractError("score_start_step must be positive")
    final_step = length if end_step is None else int(end_step)
    if final_step < start_step or final_step > length:
        raise EvaluationContractError("score interval is outside prediction trace")
    return slice(int(start_step) - 1, final_step), tuple(range(int(start_step), final_step + 1))


def compute_metrics(
    prediction: Any,
    truth: Any,
    *,
    carry_prediction: Any,
    d0_train_axis_rms: Any,
    paired_prediction: Any | None = None,
    seed_metrics: Sequence[SeedMetrics] = (),
    score_start_step: int = D1_SCORE_STEPS[0],
    score_end_step: int | None = None,
) -> Metrics:
    """Compute all ranked metrics on the predeclared physical-step interval."""

    pred = _array(prediction, label="prediction", ndim=3)
    target = _array(truth, label="truth", ndim=3)
    carry = _array(carry_prediction, label="carry prediction", ndim=3)
    if pred.shape != target.shape or carry.shape != target.shape or pred.shape[2] != 3:
        raise EvaluationContractError("prediction, truth, and carry shapes must match")
    axis_rms = _array(d0_train_axis_rms, label="D0 train axis RMS", ndim=1)
    if axis_rms.shape != (3,) or np.any(axis_rms[np.asarray(ACTIVE_AXES)] <= 0.0):
        raise EvaluationContractError("active D0 train axis RMS must be positive")
    interval, steps = _score_slice(len(pred), score_start_step, score_end_step)
    error = pred[interval] - target[interval]
    carry_error = carry[interval] - target[interval]
    active = np.asarray(ACTIVE_AXES, dtype=bool)
    active_error = error[..., active]
    active_carry_error = carry_error[..., active]
    global_nrmse = float(np.sqrt(np.mean(np.square(active_error / axis_rms[active]))))
    axis_rmse = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
    axis_bias = np.mean(error, axis=(0, 1))
    per_step = np.sqrt(np.mean(np.square(active_error), axis=(1, 2)))
    marker_error = np.sqrt(np.sum(np.square(active_error), axis=2))
    p95 = np.percentile(marker_error, 95.0, axis=1)
    maximum = np.max(marker_error, axis=1)
    carry_per_step = np.sqrt(np.mean(np.square(active_carry_error), axis=(1, 2)))
    if np.any(carry_per_step <= 0.0) or not np.all(np.isfinite(carry_per_step)):
        raise EvaluationContractError("carry proxy denominator is zero or nonfinite")
    rho = per_step / carry_per_step
    if not np.all(np.isfinite(rho)):
        raise EvaluationContractError("rho proxy is nonfinite")
    oracle_delta = target[interval][..., active] - carry[interval][..., active]
    model_delta = pred[interval][..., active] - carry[interval][..., active]
    denominator = np.sum(np.square(oracle_delta), axis=(1, 2))
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise EvaluationContractError("alpha/r_perp proxy denominator is zero or nonfinite")
    alpha = np.sum(model_delta * oracle_delta, axis=(1, 2)) / denominator
    residual = model_delta - alpha[:, None, None] * oracle_delta
    r_perp = np.sqrt(np.sum(np.square(residual), axis=(1, 2)) / denominator)
    paired_rho: tuple[float, ...] = ()
    paired_p95: float | None = None
    if paired_prediction is not None:
        paired = _array(paired_prediction, label="paired prediction", ndim=3)
        if paired.shape != pred.shape:
            raise EvaluationContractError("paired prediction shape must match prediction")
        paired_error = paired[interval] - target[interval]
        paired_step = np.sqrt(np.mean(np.square(paired_error[..., active]), axis=(1, 2)))
        if np.any(paired_step <= 0.0) or not np.all(np.isfinite(paired_step)):
            raise EvaluationContractError("paired proxy denominator is zero or nonfinite")
        paired_values = per_step / paired_step
        if not np.all(np.isfinite(paired_values)):
            raise EvaluationContractError("paired rho proxy is nonfinite")
        paired_rho = tuple(float(value) for value in paired_values)
        paired_p95 = float(np.percentile(paired_values, 95.0))
    return Metrics(
        global_active_yz_nrmse=global_nrmse,
        axis_rmse=tuple(float(value) for value in axis_rmse),
        axis_bias=tuple(float(value) for value in axis_bias),
        per_step_rms=tuple(float(value) for value in per_step),
        p95_marker_euclidean=tuple(float(value) for value in p95),
        max_marker_euclidean=tuple(float(value) for value in maximum),
        rho_per_step=tuple(float(value) for value in rho),
        rho_median=float(np.median(rho)),
        rho_p95=float(np.percentile(rho, 95.0)),
        fraction_rho_lt_1=float(np.mean(rho < 1.0)),
        fraction_rho_lt_01=float(np.mean(rho < 0.1)),
        fraction_rho_gt_2=float(np.mean(rho > 2.0)),
        alpha_parallel=tuple(float(value) for value in alpha),
        r_perp=tuple(float(value) for value in r_perp),
        score_steps=steps,
        paired_rho_per_step=paired_rho,
        paired_rho_p95=paired_p95,
        global_marker_p95=float(np.percentile(marker_error, 95.0)),
        global_marker_max=float(np.max(marker_error)),
    )


def evaluate_gate_boundaries(
    nrmse_vs_c0_ratio: float,
    nrmse_vs_ar_ratio: float,
    fraction_beating_carry: float,
    worst_seed_nrmse_ratio: float,
) -> bool:
    """Small helper for the inclusive G0 boundaries.

    The final G0 gate uses the worst individual-seed NRMSE versus C0, not a
    carry-relative rho p95 statistic.
    """

    return (
        nrmse_vs_c0_ratio <= 0.95
        and nrmse_vs_ar_ratio <= 0.98
        and fraction_beating_carry >= 0.60
        and worst_seed_nrmse_ratio <= 1.10
    )


def _ordered_r25a_seeds(rows: Sequence[SeedMetrics]) -> bool:
    try:
        return tuple(row.seed for row in rows) == (0, 1, 2)
    except (AttributeError, TypeError):
        return False


def g0_gate(
    seeds: Sequence[SeedMetrics],
    *,
    c0_nrmse: float,
    ar_nrmse: float,
) -> bool:
    if not _ordered_r25a_seeds(seeds):
        return False
    try:
        finite_references = np.isfinite(c0_nrmse) and np.isfinite(ar_nrmse)
    except TypeError:
        finite_references = False
    if not finite_references:
        return False
    nrmse = np.asarray([row.nrmse for row in seeds], dtype=np.float64)
    fraction = np.asarray([row.fraction_beating_carry for row in seeds], dtype=np.float64)
    return bool(
        np.all(np.isfinite(nrmse))
        and np.all(np.isfinite(fraction))
        and np.median(nrmse) <= 0.95 * c0_nrmse
        and np.median(nrmse) <= 0.98 * ar_nrmse
        and np.median(fraction) >= 0.60
        and np.all(nrmse <= 1.10 * c0_nrmse)
    )


def hybrid_gate(
    seeds: Sequence[SeedMetrics],
    paired_g0: Sequence[SeedMetrics],
    *,
    matching_kalman_nrmse: float,
) -> bool:
    """Apply the separately frozen GK0/GK1 gate with inclusive thresholds."""

    if not _ordered_r25a_seeds(seeds) or not _ordered_r25a_seeds(paired_g0):
        return False
    try:
        finite_reference = bool(np.isfinite(matching_kalman_nrmse))
    except TypeError:
        finite_reference = False
    if not finite_reference:
        return False
    nrmse = np.asarray([row.nrmse for row in seeds], dtype=np.float64)
    g0_nrmse = np.asarray([row.nrmse for row in paired_g0], dtype=np.float64)
    fraction_values = [getattr(row, "fraction_beating_paired", None) for row in seeds]
    if any(value is None for value in fraction_values):
        return False
    fractions = np.asarray(fraction_values, dtype=np.float64)
    p95 = np.asarray([row.metrics.rho_p95 for row in seeds], dtype=np.float64)
    if not all(np.all(np.isfinite(value)) for value in (nrmse, g0_nrmse, fractions, p95)):
        return False
    favorable = (nrmse < matching_kalman_nrmse) & (nrmse < g0_nrmse)
    return bool(
        np.median(nrmse) <= 0.95 * matching_kalman_nrmse
        and np.median(nrmse) <= 0.98 * np.median(g0_nrmse)
        and np.median(fractions) >= 0.55
        and np.median(p95)
        <= 1.10 * np.median([row.metrics.rho_p95 for row in paired_g0])
        and int(np.count_nonzero(favorable)) >= 2
    )


def _model_object(model: Any) -> ResidualGRU:
    return model.model if hasattr(model, "model") else model


def predict_gru(
    trace: Any,
    *,
    pod: PODBasis,
    normalization: ModalNormalization,
    model: Any,
    family: str,
    baseline: BaselineResult | None = None,
) -> np.ndarray:
    """Generate a cold-start trace prediction using accepted teacher history."""

    if baseline is None:
        baseline_name = {"gru": "carry", "kalman0_gru": "kalman0", "kalman1_gru": "kalman1"}.get(family)
        if baseline_name is None:
            raise EvaluationContractError(f"unsupported GRU family {family!r}")
        baseline = evaluate_baseline(trace, model=baseline_name)
    if family not in ("gru", "kalman0_gru", "kalman1_gru"):
        raise EvaluationContractError(f"unsupported GRU family {family!r}")
    network = _model_object(model)
    if network.family != family:
        raise EvaluationContractError("model family does not match prediction family")
    output = np.array(baseline.effective_predictions, dtype=np.float64, copy=True)
    sample_count = len(trace.values)
    window = network.architecture.window
    raw_samples = build_temporal_samples(
        trace,
        window=window,
        start_step=window + 1,
        end_step=sample_count,
        baseline_predictions=baseline.effective_predictions,
        innovations=baseline.innovations,
    )
    network.eval()
    with torch.no_grad():
        for sample in raw_samples:
            history_coeff = normalization.normalize(pod.encode(sample.history))
            carry_coeff = normalization.normalize(pod.encode(sample.current_baseline)[None, ...])[0]
            if family == "gru":
                features = make_gru_features(family, history_coeff[None, ...])
            else:
                if sample.innovation_history is None:
                    raise EvaluationContractError("hybrid sample has no innovation history")
                innovation_coeff = pod.encode_residual(sample.innovation_history) / normalization.scale
                features = make_gru_features(
                    family,
                    history_coeff[None, ...],
                    innovations=innovation_coeff[None, ...],
                    current_baseline=carry_coeff[None, ...],
                )
            prediction_coeff = network(to_torch(features), to_torch(carry_coeff[None, ...]))[0].detach().cpu().numpy()
            prediction = pod.decode(normalization.denormalize(prediction_coeff))[...]
            output[sample.target_step - 1] = prediction
    output[..., 0] = 0.0
    if not np.all(np.isfinite(output)):
        raise EvaluationContractError("GRU prediction contains nonfinite values")
    return output


def predict_pod_ar(
    trace: Any,
    *,
    pod: PODBasis,
    normalization: ModalNormalization,
    ar_model: PODARModel,
    baseline: BaselineResult | None = None,
) -> np.ndarray:
    """Evaluate a train-only POD-AR model with accepted teacher history."""

    if baseline is None:
        baseline = evaluate_baseline(trace, model="carry")
    output = np.array(baseline.effective_predictions, dtype=np.float64, copy=True)
    coefficients = normalization.normalize(pod.encode(trace.values))
    for target_step in range(ar_model.window + 1, len(trace.values) + 1):
        history = coefficients[target_step - ar_model.window - 1 : target_step - 1]
        predicted = ar_model.predict(history[None, ...])[0]
        output[target_step - 1] = pod.decode(normalization.denormalize(predicted))[...]
    output[..., 0] = 0.0
    return output


def oracle_predictions(trace: Any) -> np.ndarray:
    """Return Q only as an error lower bound; callers must exclude it from ranking."""

    return np.array(trace.values, dtype=np.float64, copy=True)


__all__ = [
    "EvaluationContractError",
    "Metrics",
    "SeedMetrics",
    "compute_metrics",
    "evaluate_gate_boundaries",
    "g0_gate",
    "hybrid_gate",
    "oracle_predictions",
    "predict_gru",
    "predict_pod_ar",
]
