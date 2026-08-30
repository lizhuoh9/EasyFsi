"""Deterministic candidate replay and production K0 parity adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .kalman_statistical_filter import KalmanTrialEngine
from .kalman_statistical_types import (
    AcceptedTrace,
    CalibrationContractError,
    CandidateSpec,
    KalmanEngineSnapshot,
    SCHEMA_VERSION,
    _AXIS_ORDER,
    _KALMAN_MODELS,
    _NIS_95_ONE_DOF,
    _canonical_json_bytes,
    _fingerprint,
    _readonly,
)


@dataclass(frozen=True)
class ReplayRow:
    physical_step: int
    accepted_state_source_step: int
    accepted_state_source_sha256: str
    accepted_measurement_sha256: str
    history_sha256: str
    journal_sha256: str
    layout_id: str
    axis_order: tuple[str, str, str]
    active_axes: tuple[bool, bool, bool]
    candidate_id: str
    candidate_fingerprint: str
    model: str
    raw_prediction_rms_mps: float
    raw_prediction_axis_mean_mps: tuple[float, float, float]
    measurement_axis_mean_mps: tuple[float, float, float]
    effective_prediction_rms_mps: float
    effective_prediction_bias_mps: float
    effective_axis_rmse_mps: tuple[float, float, float]
    effective_axis_normalized_rmse: tuple[float, float, float]
    max_component_error_mps: float
    representative_error_mps: tuple[float, float, float]
    innovation_axis_mean_mps: tuple[float, float, float]
    innovation_axis_std_mps: tuple[float, float, float]
    innovation_variance_axis_mean: tuple[float, float, float]
    innovation_variance_mean: float
    nis_axis_mean: tuple[float, float, float]
    nis_mean: float
    nis_axis_exceedance_fraction: tuple[float, float, float]
    innovation_dof_by_axis: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    innovation_variance_dof_by_axis: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    nis_dof_by_axis: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    gain_dof_by_axis: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    gain_axis_mean: tuple[float, float, float]
    p_prior_axis_mean: tuple[float, float, float]
    p_posterior_axis_mean: tuple[float, float, float]
    covariance_finite: bool
    covariance_symmetry_error: float
    covariance_min_eigenvalue: float
    q_identity: str
    r_identity: str
    p0_identity: str
    fallback_reason: str | None
    reset_reason: str | None
    statistical_metrics_available: bool
    fsi_iterations: int
    cg_iterations: int
    matvec_count: int | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplaySnapshot:
    candidate_fingerprint: str
    next_index: int
    committed_step: int
    previous_values: np.ndarray
    previous_previous_values: np.ndarray
    engine_snapshot: KalmanEngineSnapshot | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "previous_values", _readonly(self.previous_values))
        object.__setattr__(
            self, "previous_previous_values", _readonly(self.previous_previous_values)
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.candidate_fingerprint.encode())
        digest.update(str(self.next_index).encode())
        digest.update(str(self.committed_step).encode())
        digest.update(np.ascontiguousarray(self.previous_values).tobytes())
        digest.update(np.ascontiguousarray(self.previous_previous_values).tobytes())
        if self.engine_snapshot is not None:
            digest.update(np.ascontiguousarray(self.engine_snapshot.mean).tobytes())
            digest.update(
                np.ascontiguousarray(self.engine_snapshot.covariance).tobytes()
            )
        return digest.hexdigest()


@dataclass(frozen=True)
class ReplayResult:
    trace_name: str
    candidate_id: str
    candidate_fingerprint: str
    rows: tuple[ReplayRow, ...]
    snapshot: ReplaySnapshot

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_name": self.trace_name,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "rows": [row.to_payload() for row in self.rows],
            "snapshot_fingerprint": self.snapshot.fingerprint,
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())


def _covariance_quality(covariance: np.ndarray) -> tuple[bool, float, float]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[:, :, None, None]
    finite = bool(np.all(np.isfinite(matrix)))
    if not finite:
        return False, float(np.finfo(np.float64).max), -float(np.finfo(np.float64).max)
    symmetry = float(
        np.max(np.abs(matrix - np.swapaxes(matrix, -1, -2)))
    )
    minimum = float(np.min(np.linalg.eigvalsh(matrix)))
    return True, symmetry, minimum


def _identity(label: str, values: Sequence[float]) -> str:
    return _fingerprint({"label": label, "xyz": list(values)})


def _accepted_source_sha256(trace: AcceptedTrace, index: int) -> str:
    if index > 0:
        return trace.frame_sha256[index - 1]
    initial = np.zeros_like(trace.values[0], dtype=np.float64)
    return _fingerprint(
        {
            "kind": "r24_initial_accepted_state",
            "layout_id": trace.layout_id,
            "shape": list(initial.shape),
            "dtype": str(initial.dtype),
            "values_sha256": hashlib.sha256(
                np.ascontiguousarray(initial).tobytes()
            ).hexdigest(),
        }
    )


def _make_row(
    *,
    trace: AcceptedTrace,
    index: int,
    candidate: CandidateSpec,
    raw_prediction: np.ndarray,
    effective_prediction: np.ndarray,
    innovation: np.ndarray,
    innovation_variance: np.ndarray,
    nis: np.ndarray,
    value_gain: np.ndarray,
    prior_covariance: np.ndarray,
    posterior_covariance: np.ndarray,
    fallback_reason: str | None,
    statistical_metrics_available: bool,
    covariance_is_normalized: bool = True,
) -> ReplayRow:
    measurement = trace.values[index]
    raw_error = raw_prediction - measurement
    effective_error = effective_prediction - measurement
    scale = np.asarray(candidate.scale_xyz)[None, :]
    axis_rmse = np.sqrt(np.mean(np.square(effective_error), axis=0))
    normalized_rmse = np.sqrt(
        np.mean(np.square(effective_error / scale), axis=0)
    )
    innovation_axis_mean = np.mean(innovation, axis=0)
    innovation_axis_std = np.std(innovation, axis=0)
    s_axis = np.mean(innovation_variance, axis=0)
    nis_axis = np.mean(nis, axis=0)
    exceedance = np.mean(nis > _NIS_95_ONE_DOF, axis=0)
    gain_axis = np.mean(value_gain, axis=0)
    physical_scale_squared = (
        np.square(scale) if covariance_is_normalized else np.ones_like(scale)
    )
    if prior_covariance.ndim == 4:
        prior_value = prior_covariance[:, :, 0, 0] * physical_scale_squared
        posterior_value = (
            posterior_covariance[:, :, 0, 0] * physical_scale_squared
        )
    else:
        prior_value = prior_covariance * physical_scale_squared
        posterior_value = posterior_covariance * physical_scale_squared
    finite, symmetry, minimum = _covariance_quality(posterior_covariance)
    return ReplayRow(
        physical_step=trace.source_steps[index],
        accepted_state_source_step=trace.source_steps[index] - 1,
        accepted_state_source_sha256=_accepted_source_sha256(trace, index),
        accepted_measurement_sha256=trace.frame_sha256[index],
        history_sha256=trace.history_sha256[index],
        journal_sha256=trace.journal_sha256[index],
        layout_id=trace.layout_id,
        axis_order=_AXIS_ORDER,
        active_axes=candidate.active_axes,
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=candidate.fingerprint,
        model=candidate.model,
        raw_prediction_rms_mps=float(np.sqrt(np.mean(np.square(raw_error)))),
        raw_prediction_axis_mean_mps=tuple(
            float(value) for value in np.mean(raw_prediction, axis=0)
        ),
        measurement_axis_mean_mps=tuple(
            float(value) for value in np.mean(measurement, axis=0)
        ),
        effective_prediction_rms_mps=float(
            np.sqrt(np.mean(np.square(effective_error)))
        ),
        effective_prediction_bias_mps=float(np.mean(effective_error)),
        effective_axis_rmse_mps=tuple(float(value) for value in axis_rmse),
        effective_axis_normalized_rmse=tuple(
            float(value) for value in normalized_rmse
        ),
        max_component_error_mps=float(np.max(np.abs(effective_error))),
        representative_error_mps=tuple(float(value) for value in effective_error[0]),
        innovation_axis_mean_mps=tuple(
            float(value) for value in innovation_axis_mean
        ),
        innovation_axis_std_mps=tuple(
            float(value) for value in innovation_axis_std
        ),
        innovation_variance_axis_mean=tuple(float(value) for value in s_axis),
        innovation_variance_mean=float(np.mean(innovation_variance)),
        nis_axis_mean=tuple(float(value) for value in nis_axis),
        nis_mean=float(np.mean(nis)),
        nis_axis_exceedance_fraction=tuple(float(value) for value in exceedance),
        innovation_dof_by_axis=tuple(
            tuple(float(value) for value in innovation[:, axis])
            for axis in range(3)
        ),
        innovation_variance_dof_by_axis=tuple(
            tuple(float(value) for value in innovation_variance[:, axis])
            for axis in range(3)
        ),
        nis_dof_by_axis=tuple(
            tuple(float(value) for value in nis[:, axis])
            for axis in range(3)
        ),
        gain_dof_by_axis=tuple(
            tuple(float(value) for value in value_gain[:, axis])
            for axis in range(3)
        ),
        gain_axis_mean=tuple(float(value) for value in gain_axis),
        p_prior_axis_mean=tuple(
            float(value) for value in np.mean(prior_value, axis=0)
        ),
        p_posterior_axis_mean=tuple(
            float(value) for value in np.mean(posterior_value, axis=0)
        ),
        covariance_finite=finite,
        covariance_symmetry_error=symmetry,
        covariance_min_eigenvalue=minimum,
        q_identity=_identity("q", candidate.q_xyz),
        r_identity=_identity("r", candidate.r_xyz),
        p0_identity=_fingerprint(
            {
                "value": list(candidate.p0_value_xyz),
                "rate": list(candidate.p0_rate_xyz),
            }
        ),
        fallback_reason=fallback_reason,
        reset_reason=None,
        statistical_metrics_available=statistical_metrics_available,
        fsi_iterations=int(trace.fsi_iterations[index]),
        cg_iterations=int(trace.cg_iterations[index]),
        matvec_count=(
            None
            if trace.matvec_count[index] is None
            else int(trace.matvec_count[index])
        ),
    )


def _validate_inactive_axes(trace: AcceptedTrace, candidate: CandidateSpec) -> None:
    active = np.asarray(candidate.active_axes, dtype=bool)
    if np.any(trace.values[:, :, ~active] != 0.0):
        inactive = ",".join(
            axis for axis, enabled in zip(_AXIS_ORDER, active, strict=True) if not enabled
        )
        raise CalibrationContractError(f"inactive axes must remain exactly zero: {inactive}")


def replay_candidate(
    trace: AcceptedTrace,
    candidate: CandidateSpec,
    *,
    snapshot: ReplaySnapshot | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
) -> ReplayResult:
    """Causally replay one frozen candidate on accepted observations."""

    if candidate.layout_id is not None and candidate.layout_id != trace.layout_id:
        raise CalibrationContractError("candidate layout does not match trace layout")
    _validate_inactive_axes(trace, candidate)
    count = len(trace.values)
    stop = count if stop_index is None else int(stop_index)
    if not 0 <= start_index <= stop <= count:
        raise CalibrationContractError("invalid replay index range")
    shape = tuple(trace.values.shape[1:])
    zeros = np.zeros(shape, dtype=np.float64)
    if snapshot is None:
        if start_index != 0:
            raise CalibrationContractError("nonzero start_index requires a checkpoint")
        previous = zeros
        previous_previous = zeros
        committed_step = trace.source_steps[0] - 1
        engine_snapshot = None
    else:
        if snapshot.candidate_fingerprint != candidate.fingerprint:
            raise CalibrationContractError("checkpoint candidate fingerprint mismatch")
        if snapshot.next_index != start_index:
            raise CalibrationContractError("checkpoint next_index mismatch")
        previous = np.array(snapshot.previous_values, copy=True)
        previous_previous = np.array(
            snapshot.previous_previous_values, copy=True
        )
        committed_step = snapshot.committed_step
        engine_snapshot = snapshot.engine_snapshot
    engine = None
    if candidate.model in _KALMAN_MODELS:
        engine = KalmanTrialEngine(
            candidate,
            initial_values=zeros,
            committed_step=committed_step,
            layout_id=trace.layout_id,
            snapshot=engine_snapshot,
        )
    rows: list[ReplayRow] = []
    scale = np.asarray(candidate.scale_xyz)[None, :]
    for index in range(start_index, stop):
        target = trace.source_steps[index]
        source = target - 1
        measurement = trace.values[index]
        if candidate.model == "carry":
            raw = previous.copy()
            effective = raw
            innovation = measurement - raw
            s = np.ones_like(raw)
            nis = np.zeros_like(raw)
            gain = np.zeros_like(raw)
            prior = np.zeros_like(raw)
            posterior = np.zeros_like(raw)
            fallback = None
            statistical = False
        elif candidate.model == "linear":
            raw = previous + candidate.beta * (previous - previous_previous)
            effective = raw
            innovation = measurement - raw
            s = np.ones_like(raw)
            nis = np.zeros_like(raw)
            gain = np.zeros_like(raw)
            prior = np.zeros_like(raw)
            posterior = np.zeros_like(raw)
            fallback = None
            statistical = False
        elif engine is not None:
            prediction = engine.begin_step(
                target_step=target,
                accepted_state_source_step=source,
                dt_s=trace.dt_s,
                layout_id=trace.layout_id,
            )
            raw = prediction.values
            ready = engine.accepted_state_count >= candidate.warmup_accepted_states
            effective = raw if ready else previous
            fallback = None if ready else "kalman_warmup"
            update = engine.assimilate(
                measurement,
                accepted_step=target,
                layout_id=trace.layout_id,
            )
            innovation = update.innovations
            s = update.innovation_variances
            nis = update.nis
            gain = update.value_gain
            prior = prediction.prior_covariance
            posterior = update.posterior_covariance
            statistical = True
            engine.commit_trial()
        else:
            raise CalibrationContractError(
                "production candidates require replay_production_k0"
            )
        rows.append(
            _make_row(
                trace=trace,
                index=index,
                candidate=candidate,
                raw_prediction=raw,
                effective_prediction=effective,
                innovation=innovation,
                innovation_variance=s,
                nis=nis,
                value_gain=gain,
                prior_covariance=prior,
                posterior_covariance=posterior,
                fallback_reason=fallback,
                statistical_metrics_available=statistical,
            )
        )
        previous_previous = previous
        previous = np.array(measurement, copy=True)
    last_step = committed_step if not rows else rows[-1].physical_step
    result_snapshot = ReplaySnapshot(
        candidate_fingerprint=candidate.fingerprint,
        next_index=stop,
        committed_step=last_step,
        previous_values=previous,
        previous_previous_values=previous_previous,
        engine_snapshot=None if engine is None else engine.snapshot(),
    )
    return ReplayResult(
        trace_name=trace.name,
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=candidate.fingerprint,
        rows=tuple(rows),
        snapshot=result_snapshot,
    )

def _load_predictor_module(source: Path) -> Any:
    resolved = source.resolve()
    if not resolved.is_file():
        raise CalibrationContractError(f"K0 predictor source not found: {resolved}")
    module_name = "_r24_k0_" + hashlib.sha256(
        resolved.read_bytes()
    ).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise CalibrationContractError("cannot load K0 predictor source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def replay_production_k0(
    trace: AcceptedTrace,
    config: Mapping[str, Any],
    predictor_source: Path | str,
    *,
    normalization_scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
    active_axes: tuple[bool, bool, bool] = (True, True, True),
) -> ReplayResult:
    """Replay the exact production constant-rate code without package import."""

    module = _load_predictor_module(Path(predictor_source))
    covariance_keys = (
        "rate_process_noise_spectral_density",
        "measurement_variance",
        "initial_value_variance",
        "initial_rate_variance",
    )
    try:
        converted = {
            key: tuple(float(value) for value in config[key])
            for key in covariance_keys
        }
        warmup = int(config["warmup_accepted_states"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationContractError("invalid locked K0 config") from exc
    production_config = module.InterfaceKalmanConfig(
        rate_process_noise_spectral_density=converted[
            "rate_process_noise_spectral_density"
        ],
        measurement_variance=converted["measurement_variance"],
        initial_value_variance=converted["initial_value_variance"],
        initial_rate_variance=converted["initial_rate_variance"],
        warmup_accepted_states=warmup,
    )
    normalization = tuple(float(value) for value in normalization_scale_xyz)
    candidate = CandidateSpec(
        candidate_id="K0",
        model="production",
        axis_order=_AXIS_ORDER,
        active_axes=active_axes,
        scale_xyz=normalization,
        q_xyz=converted["rate_process_noise_spectral_density"],
        r_xyz=converted["measurement_variance"],
        p0_value_xyz=converted["initial_value_variance"],
        p0_rate_xyz=converted["initial_rate_variance"],
        warmup_accepted_states=warmup,
        layout_id=trace.layout_id,
    )
    _validate_inactive_axes(trace, candidate)
    predictor = module.InterfaceKalmanPredictor(production_config)
    previous = np.zeros_like(trace.values[0])
    previous_previous = np.zeros_like(previous)
    predictor.initialize(previous, layout_id=trace.layout_id)
    rows: list[ReplayRow] = []
    for index, measurement in enumerate(trace.values):
        estimate = predictor.predict_trial(dt=trace.dt_s, layout_id=trace.layout_id)
        update = predictor.update_trial(
            measurement, layout_id=trace.layout_id
        )
        ready = index + 1 >= warmup
        effective = estimate.values if ready else previous
        fallback = None if ready else "kalman_warmup"
        prior = estimate.covariances
        posterior = update.estimate.covariances
        s = update.innovation_variances
        gain = prior[:, :, 0, 0] / s
        rows.append(
            _make_row(
                trace=trace,
                index=index,
                candidate=candidate,
                raw_prediction=estimate.values,
                effective_prediction=effective,
                innovation=update.innovations,
                innovation_variance=s,
                nis=update.normalized_innovation_squared,
                value_gain=gain,
                prior_covariance=prior,
                posterior_covariance=posterior,
                fallback_reason=fallback,
                statistical_metrics_available=True,
                covariance_is_normalized=False,
            )
        )
        predictor.commit_trial()
        previous_previous = previous
        previous = np.array(measurement, copy=True)
    snapshot = ReplaySnapshot(
        candidate_fingerprint=candidate.fingerprint,
        next_index=len(trace.values),
        committed_step=trace.source_steps[-1],
        previous_values=previous,
        previous_previous_values=previous_previous,
        engine_snapshot=None,
    )
    return ReplayResult(
        trace_name=trace.name,
        candidate_id="K0",
        candidate_fingerprint=candidate.fingerprint,
        rows=tuple(rows),
        snapshot=snapshot,
    )
