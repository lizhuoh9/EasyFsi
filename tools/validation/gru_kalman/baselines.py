"""Causality-safe C0/C1/K0/K1 baseline adapters for R25A."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_filter import (
    KalmanTrialEngine,
)
from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_types import (
    CandidateSpec,
)

from .dataset import ACTIVE_AXES, AXIS_ORDER, DatasetContractError, EXPECTED_LAYOUT_ID, validate_trace

_PREDICTOR_PATH = Path(__file__).resolve().parents[3] / "simulation_core" / "coupling" / "interface_kalman_predictor.py"
_PREDICTOR_SPEC = importlib.util.spec_from_file_location(
    "_r25a_interface_kalman_predictor", _PREDICTOR_PATH
)
if _PREDICTOR_SPEC is None or _PREDICTOR_SPEC.loader is None:
    raise ImportError(f"cannot load production predictor source {_PREDICTOR_PATH}")
_PREDICTOR_MODULE = importlib.util.module_from_spec(_PREDICTOR_SPEC)
sys.modules[_PREDICTOR_SPEC.name] = _PREDICTOR_MODULE
_PREDICTOR_SPEC.loader.exec_module(_PREDICTOR_MODULE)
InterfaceKalmanConfig = _PREDICTOR_MODULE.InterfaceKalmanConfig
InterfaceKalmanPredictor = _PREDICTOR_MODULE.InterfaceKalmanPredictor

K0_FINGERPRINT = "383f9fc10475449cd88ce4fbc9b0d3b7595b47e62e7ef4aa53a516dd0058e03e"
K1_FINGERPRINT = "603ec011922df847f61a0d8a91216ba2a2e3b2c60eb757092f910df37678d91e"
KALMAN_WARMUP_ACCEPTED_STATES = 6
KALMAN_NORMALIZATION_SCALE = (1.0, 0.012603211359353822, 0.041955797644311746)
K0_CONFIG_PAYLOAD = {
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


def _readonly(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_values(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise DatasetContractError(f"{label} shape or finite-value contract failed")
    if not np.all(result[..., 0] == 0.0):
        raise DatasetContractError(f"{label} x axis must be exactly zero")
    return np.array(result, copy=True)


def exact_k0_config() -> InterfaceKalmanConfig:
    """Return the exact production K0 configuration from the R24 seal."""

    return InterfaceKalmanConfig(
        rate_process_noise_spectral_density=tuple(
            K0_CONFIG_PAYLOAD["rate_process_noise_spectral_density"]
        ),
        measurement_variance=tuple(K0_CONFIG_PAYLOAD["measurement_variance"]),
        initial_value_variance=tuple(K0_CONFIG_PAYLOAD["initial_value_variance"]),
        initial_rate_variance=tuple(K0_CONFIG_PAYLOAD["initial_rate_variance"]),
        warmup_accepted_states=KALMAN_WARMUP_ACCEPTED_STATES,
    )


def exact_k0_candidate(layout_id: str = EXPECTED_LAYOUT_ID) -> CandidateSpec:
    """Build and fingerprint the R24 production K0 candidate identity."""

    candidate = CandidateSpec(
        candidate_id="K0",
        model="production",
        axis_order=AXIS_ORDER,
        active_axes=ACTIVE_AXES,
        scale_xyz=KALMAN_NORMALIZATION_SCALE,
        q_xyz=tuple(K0_CONFIG_PAYLOAD["rate_process_noise_spectral_density"]),
        r_xyz=tuple(K0_CONFIG_PAYLOAD["measurement_variance"]),
        p0_value_xyz=tuple(K0_CONFIG_PAYLOAD["initial_value_variance"]),
        p0_rate_xyz=tuple(K0_CONFIG_PAYLOAD["initial_rate_variance"]),
        warmup_accepted_states=KALMAN_WARMUP_ACCEPTED_STATES,
        layout_id=layout_id,
    )
    if candidate.fingerprint != K0_FINGERPRINT:
        raise DatasetContractError("locked K0 CandidateSpec fingerprint changed")
    return candidate


def exact_k1_candidate(layout_id: str = EXPECTED_LAYOUT_ID) -> CandidateSpec:
    candidate = CandidateSpec(
        candidate_id="K1",
        model="random_walk",
        axis_order=AXIS_ORDER,
        active_axes=ACTIVE_AXES,
        scale_xyz=KALMAN_NORMALIZATION_SCALE,
        q_xyz=(0.0, 0.23754820647609656, 0.14564681242821423),
        r_xyz=(1.0, 0.08414201402530712, 0.018576420357815062),
        p0_value_xyz=(1.0, 0.08414201402530712, 0.018576420357815062),
        p0_rate_xyz=(1.0, 336568.0561012285, 74305.68143126025),
        warmup_accepted_states=KALMAN_WARMUP_ACCEPTED_STATES,
        beta=1.0,
        layout_id=layout_id,
        q_multiplier=1.0,
        r_multiplier=0.3,
    )
    if candidate.fingerprint != K1_FINGERPRINT:
        raise DatasetContractError("locked K1 CandidateSpec fingerprint changed")
    return candidate


@dataclass(frozen=True)
class BaselineRow:
    target_step: int
    accepted_state_source_step: int
    model: str
    raw_prediction: np.ndarray
    effective_prediction: np.ndarray
    innovation: np.ndarray
    ready: bool

    def __post_init__(self) -> None:
        raw = _readonly(self.raw_prediction)
        effective = _readonly(self.effective_prediction)
        innovation = _readonly(self.innovation)
        if raw.ndim != 2 or raw.shape[1] != 3 or effective.shape != raw.shape or innovation.shape != raw.shape:
            raise DatasetContractError("baseline row arrays must have shape (markers, 3)")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(effective)) or not np.all(np.isfinite(innovation)):
            raise DatasetContractError("baseline row arrays must be finite")
        if not np.all(raw[..., 0] == 0.0) or not np.all(effective[..., 0] == 0.0) or not np.all(innovation[..., 0] == 0.0):
            raise DatasetContractError("baseline row x axis must be exactly zero")
        if self.target_step != self.accepted_state_source_step + 1:
            raise DatasetContractError("baseline row source step must precede target")
        object.__setattr__(self, "raw_prediction", raw)
        object.__setattr__(self, "effective_prediction", effective)
        object.__setattr__(self, "innovation", innovation)


@dataclass(frozen=True)
class BaselineResult:
    model: str
    rows: tuple[BaselineRow, ...]

    @property
    def predictions(self) -> np.ndarray:
        return np.stack([row.effective_prediction for row in self.rows], axis=0)

    @property
    def raw_predictions(self) -> np.ndarray:
        return np.stack([row.raw_prediction for row in self.rows], axis=0)

    @property
    def effective_predictions(self) -> np.ndarray:
        return self.predictions

    @property
    def innovations(self) -> np.ndarray:
        return np.stack([row.innovation for row in self.rows], axis=0)

    @property
    def ready(self) -> tuple[bool, ...]:
        return tuple(row.ready for row in self.rows)


class _CarryAdapter:
    def __init__(self, model: str, shape: tuple[int, int], beta: float = 1.0) -> None:
        if model not in ("carry", "linear"):
            raise DatasetContractError("carry adapter model must be carry or linear")
        self.model = model
        self.shape = shape
        self.beta = float(beta)
        if not np.isfinite(self.beta):
            raise DatasetContractError("linear extrapolation beta must be finite")
        self._previous = np.zeros(shape, dtype=np.float64)
        self._previous_previous = np.zeros(shape, dtype=np.float64)
        self._committed_step = 0
        self._pending: BaselineRow | None = None

    def begin_step(self, *, target_step: int, accepted_state_source_step: int, dt_s: float, layout_id: str) -> np.ndarray:
        if self._pending is not None:
            raise DatasetContractError("baseline trial already active")
        if accepted_state_source_step != self._committed_step or target_step != self._committed_step + 1:
            raise DatasetContractError("baseline source/target step is not contiguous")
        if layout_id != EXPECTED_LAYOUT_ID or not np.isclose(dt_s, 0.0005, rtol=0.0, atol=1.0e-15):
            raise DatasetContractError("baseline layout or dt mismatch")
        if self.model == "carry":
            raw = np.array(self._previous, copy=True)
        else:
            raw = self._previous + self.beta * (self._previous - self._previous_previous)
        self._pending = BaselineRow(
            target_step=target_step,
            accepted_state_source_step=accepted_state_source_step,
            model=self.model,
            raw_prediction=raw,
            effective_prediction=raw,
            innovation=np.zeros_like(raw),
            ready=True,
        )
        return _readonly(raw)

    def accept_step(self, accepted_values: Any) -> BaselineRow:
        if self._pending is None:
            raise DatasetContractError("no active baseline trial")
        measurement = _validate_values(accepted_values, self.shape, "accepted values")
        pending = self._pending
        row = BaselineRow(
            target_step=pending.target_step,
            accepted_state_source_step=pending.accepted_state_source_step,
            model=pending.model,
            raw_prediction=pending.raw_prediction,
            effective_prediction=pending.effective_prediction,
            innovation=measurement - pending.raw_prediction,
            ready=True,
        )
        self._previous_previous = self._previous
        self._previous = measurement
        self._committed_step = pending.target_step
        self._pending = None
        return row

    def discard_step(self) -> None:
        if self._pending is None:
            raise DatasetContractError("no active baseline trial")
        self._pending = None


class _K0Adapter:
    def __init__(self, shape: tuple[int, int], layout_id: str) -> None:
        if layout_id != EXPECTED_LAYOUT_ID:
            raise DatasetContractError("K0 layout does not match frozen layout")
        self.shape = shape
        self.model = "kalman0"
        self.layout_id = layout_id
        self.predictor = InterfaceKalmanPredictor(exact_k0_config())
        self.predictor.initialize(np.zeros(shape, dtype=np.float64), layout_id=layout_id)
        self._committed_step = 0
        self._carry = np.zeros(shape, dtype=np.float64)
        self._pending: BaselineRow | None = None

    def begin_step(self, *, target_step: int, accepted_state_source_step: int, dt_s: float, layout_id: str) -> np.ndarray:
        if self._pending is not None:
            raise DatasetContractError("K0 trial already active")
        if accepted_state_source_step != self._committed_step or target_step != self._committed_step + 1:
            raise DatasetContractError("K0 source/target step is not contiguous")
        estimate = self.predictor.predict_trial(dt=dt_s, layout_id=layout_id)
        raw = np.asarray(estimate.values, dtype=np.float64)
        ready = self.predictor.ready
        effective = raw if ready else np.array(self._carry, copy=True)
        self._pending = BaselineRow(
            target_step=target_step,
            accepted_state_source_step=accepted_state_source_step,
            model=self.model,
            raw_prediction=raw,
            effective_prediction=effective,
            innovation=np.zeros_like(raw),
            ready=ready,
        )
        return _readonly(raw)

    def accept_step(self, accepted_values: Any) -> BaselineRow:
        if self._pending is None:
            raise DatasetContractError("no active K0 trial")
        measurement = _validate_values(accepted_values, self.shape, "accepted values")
        pending = self._pending
        update = self.predictor.update_trial(measurement, layout_id=self.layout_id)
        self.predictor.commit_trial()
        row = BaselineRow(
            target_step=pending.target_step,
            accepted_state_source_step=pending.accepted_state_source_step,
            model=self.model,
            raw_prediction=pending.raw_prediction,
            effective_prediction=pending.effective_prediction,
            innovation=np.asarray(update.innovations, dtype=np.float64),
            ready=pending.ready,
        )
        self._carry = measurement
        self._committed_step = pending.target_step
        self._pending = None
        return row

    def discard_step(self) -> None:
        if self._pending is None:
            raise DatasetContractError("no active K0 trial")
        self.predictor.discard_trial()
        self._pending = None


class _K1Adapter:
    def __init__(self, shape: tuple[int, int], layout_id: str) -> None:
        self.shape = shape
        self.model = "kalman1"
        self.layout_id = layout_id
        self.candidate = exact_k1_candidate(layout_id)
        self.engine = KalmanTrialEngine(
            self.candidate,
            initial_values=np.zeros(shape, dtype=np.float64),
            committed_step=0,
            layout_id=layout_id,
        )
        self._committed_step = 0
        self._carry = np.zeros(shape, dtype=np.float64)
        self._pending: BaselineRow | None = None

    def begin_step(self, *, target_step: int, accepted_state_source_step: int, dt_s: float, layout_id: str) -> np.ndarray:
        if self._pending is not None:
            raise DatasetContractError("K1 trial already active")
        if accepted_state_source_step != self._committed_step or target_step != self._committed_step + 1:
            raise DatasetContractError("K1 source/target step is not contiguous")
        prediction = self.engine.begin_step(
            target_step=target_step,
            accepted_state_source_step=accepted_state_source_step,
            dt_s=dt_s,
            layout_id=layout_id,
        )
        raw = np.asarray(prediction.values, dtype=np.float64)
        ready = self.engine.accepted_state_count >= self.candidate.warmup_accepted_states
        effective = raw if ready else np.array(self._carry, copy=True)
        self._pending = BaselineRow(
            target_step=target_step,
            accepted_state_source_step=accepted_state_source_step,
            model=self.model,
            raw_prediction=raw,
            effective_prediction=effective,
            innovation=np.zeros_like(raw),
            ready=ready,
        )
        return _readonly(raw)

    def accept_step(self, accepted_values: Any) -> BaselineRow:
        if self._pending is None:
            raise DatasetContractError("no active K1 trial")
        measurement = _validate_values(accepted_values, self.shape, "accepted values")
        pending = self._pending
        update = self.engine.assimilate(measurement, accepted_step=pending.target_step, layout_id=self.layout_id)
        self.engine.commit_trial()
        row = BaselineRow(
            target_step=pending.target_step,
            accepted_state_source_step=pending.accepted_state_source_step,
            model=self.model,
            raw_prediction=pending.raw_prediction,
            effective_prediction=pending.effective_prediction,
            innovation=np.asarray(update.innovations, dtype=np.float64),
            ready=pending.ready,
        )
        self._carry = measurement
        self._committed_step = pending.target_step
        self._pending = None
        return row

    def discard_step(self) -> None:
        if self._pending is None:
            raise DatasetContractError("no active K1 trial")
        self.engine.discard_trial()
        self._pending = None


def make_baseline_adapter(model: str, shape: tuple[int, int], layout_id: str = EXPECTED_LAYOUT_ID) -> Any:
    if model == "carry":
        return _CarryAdapter(model, shape)
    if model == "linear":
        return _CarryAdapter(model, shape, beta=1.0)
    if model == "kalman0":
        return _K0Adapter(shape, layout_id)
    if model == "kalman1":
        return _K1Adapter(shape, layout_id)
    raise DatasetContractError(f"unsupported baseline model {model!r}")


def evaluate_baseline(trace: Any, *, model: str) -> BaselineResult:
    """Cold-start one-pass baseline evaluation with accepted-only commits."""

    validate_trace(trace, expected_steps=len(trace.values))
    adapter = make_baseline_adapter(model, tuple(trace.values.shape[1:]), trace.layout_id)
    rows: list[BaselineRow] = []
    for index, target_step in enumerate(trace.source_steps):
        adapter.begin_step(
            target_step=target_step,
            accepted_state_source_step=target_step - 1,
            dt_s=trace.dt_s,
            layout_id=trace.layout_id,
        )
        rows.append(adapter.accept_step(trace.values[index]))
    return BaselineResult(model=model, rows=tuple(rows))


__all__ = [
    "K0_CONFIG_PAYLOAD",
    "K0_FINGERPRINT",
    "KALMAN_NORMALIZATION_SCALE",
    "K1_FINGERPRINT",
    "BaselineResult",
    "BaselineRow",
    "evaluate_baseline",
    "exact_k0_config",
    "exact_k0_candidate",
    "exact_k1_candidate",
    "make_baseline_adapter",
]
