"""Accepted-boundary NumPy state machine for experimental K1/K2 filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kalman_statistical_types import (
    CalibrationContractError,
    CandidateSpec,
    KalmanEngineSnapshot,
    KalmanPrediction,
    KalmanUpdate,
    _KALMAN_MODELS,
    _finite_array,
    _positive_float,
    _sha256,
)


@dataclass
class _Trial:
    target_step: int
    mean: np.ndarray
    covariance: np.ndarray
    prediction: KalmanPrediction
    update: KalmanUpdate | None = None


class KalmanTrialEngine:
    """Accepted-boundary experimental K1/K2 state machine."""

    def __init__(
        self,
        candidate: CandidateSpec,
        *,
        initial_values: Any,
        committed_step: int,
        layout_id: str,
        snapshot: KalmanEngineSnapshot | None = None,
    ) -> None:
        if candidate.model not in _KALMAN_MODELS:
            raise CalibrationContractError("KalmanTrialEngine requires K1 or K2")
        self.candidate = candidate
        self.layout_id = _sha256(layout_id, name="layout_id")
        if candidate.layout_id is not None and candidate.layout_id != self.layout_id:
            raise CalibrationContractError("candidate layout does not match trace layout")
        self._scale = np.asarray(candidate.scale_xyz, dtype=np.float64)[None, :]
        self._active = np.asarray(candidate.active_axes, dtype=bool)[None, :]
        self._trial: _Trial | None = None
        if snapshot is not None:
            self._restore(snapshot)
            return
        values = _finite_array(initial_values, name="initial_values")
        if values.ndim != 2 or values.shape[1] != 3:
            raise CalibrationContractError("initial_values must have shape (markers, 3)")
        normalized = values / self._scale
        normalized[:, ~self._active[0]] = 0.0
        marker_count = values.shape[0]
        if candidate.model == "random_walk":
            self._mean = normalized
            self._covariance = np.broadcast_to(
                np.asarray(candidate.p0_value_xyz), (marker_count, 3)
            ).copy()
        else:
            self._mean = np.stack((normalized, np.zeros_like(normalized)), axis=-1)
            self._covariance = np.zeros((marker_count, 3, 2, 2), dtype=np.float64)
            self._covariance[:, :, 0, 0] = candidate.p0_value_xyz
            self._covariance[:, :, 1, 1] = candidate.p0_rate_xyz
        self.committed_step = int(committed_step)
        self.accepted_state_count = self.committed_step + 1

    def _restore(self, snapshot: KalmanEngineSnapshot) -> None:
        if snapshot.candidate_fingerprint != self.candidate.fingerprint:
            raise CalibrationContractError("checkpoint candidate fingerprint mismatch")
        if snapshot.layout_id != self.layout_id or snapshot.model != self.candidate.model:
            raise CalibrationContractError("checkpoint layout/model mismatch")
        self._mean = np.array(snapshot.mean, copy=True)
        self._covariance = np.array(snapshot.covariance, copy=True)
        self.committed_step = snapshot.committed_step
        self.accepted_state_count = snapshot.accepted_state_count

    def begin_step(
        self,
        *,
        target_step: int,
        accepted_state_source_step: int,
        dt_s: float,
        layout_id: str,
    ) -> KalmanPrediction:
        if self._trial is not None:
            raise CalibrationContractError("a Kalman trial is already active")
        if _sha256(layout_id, name="layout_id") != self.layout_id:
            raise CalibrationContractError("layout mismatch")
        if accepted_state_source_step != self.committed_step:
            raise CalibrationContractError(
                "accepted_state_source_step must equal the committed n-1 source_step"
            )
        if target_step != self.committed_step + 1:
            raise CalibrationContractError("target_step must immediately follow source_step")
        dt = _positive_float(dt_s, name="dt_s")
        q = np.asarray(self.candidate.q_xyz, dtype=np.float64)
        if self.candidate.model == "random_walk":
            predicted_mean = np.array(self._mean, copy=True)
            predicted_covariance = self._covariance + q[None, :]
            physical = predicted_mean * self._scale
        else:
            transition = np.asarray([[1.0, dt], [0.0, 1.0]])
            template = np.asarray(
                [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
            )
            predicted_mean = self._mean @ transition.T
            predicted_covariance = (
                transition @ self._covariance @ transition.T
                + q[None, :, None, None] * template[None, None, :, :]
            )
            predicted_covariance = 0.5 * (
                predicted_covariance
                + np.swapaxes(predicted_covariance, -1, -2)
            )
            physical = predicted_mean[:, :, 0] * self._scale
        prediction = KalmanPrediction(physical, predicted_covariance)
        self._trial = _Trial(
            target_step=target_step,
            mean=predicted_mean,
            covariance=predicted_covariance,
            prediction=prediction,
        )
        return prediction

    def assimilate(
        self,
        accepted_values: Any,
        *,
        accepted_step: int,
        layout_id: str,
    ) -> KalmanUpdate:
        if self._trial is None or self._trial.update is not None:
            raise CalibrationContractError("one active, unassimilated trial is required")
        if _sha256(layout_id, name="layout_id") != self.layout_id:
            raise CalibrationContractError("layout mismatch")
        if accepted_step != self._trial.target_step:
            raise CalibrationContractError("accepted_step does not match target_step")
        shape = self._trial.prediction.values.shape
        measurement = _finite_array(
            accepted_values, name="accepted_values", shape=shape
        )
        normalized = measurement / self._scale
        r = np.asarray(self.candidate.r_xyz, dtype=np.float64)[None, :]
        if self.candidate.model == "random_walk":
            innovation = normalized - self._trial.mean
            innovation_variance = self._trial.covariance + r
            gain = self._trial.covariance / innovation_variance
            gain = np.where(self._active, gain, 0.0)
            updated_mean = self._trial.mean + gain * innovation
            updated_covariance = (
                np.square(1.0 - gain) * self._trial.covariance
                + np.square(gain) * r
            )
            value_gain = gain
        else:
            innovation = normalized - self._trial.mean[:, :, 0]
            innovation_variance = self._trial.covariance[:, :, 0, 0] + r
            gain = self._trial.covariance[:, :, :, 0] / innovation_variance[:, :, None]
            gain = np.where(self._active[:, :, None], gain, 0.0)
            updated_mean = self._trial.mean + gain * innovation[:, :, None]
            identity_minus_kh = np.broadcast_to(
                np.eye(2), self._trial.covariance.shape
            ).copy()
            identity_minus_kh[:, :, 0, 0] -= gain[:, :, 0]
            identity_minus_kh[:, :, 1, 0] -= gain[:, :, 1]
            updated_covariance = (
                identity_minus_kh
                @ self._trial.covariance
                @ np.swapaxes(identity_minus_kh, -1, -2)
                + r[:, :, None, None]
                * gain[:, :, :, None]
                * gain[:, :, None, :]
            )
            updated_covariance = 0.5 * (
                updated_covariance + np.swapaxes(updated_covariance, -1, -2)
            )
            value_gain = gain[:, :, 0]
        if np.any(innovation_variance <= 0.0) or not np.all(
            np.isfinite(updated_covariance)
        ):
            raise CalibrationContractError("Kalman covariance is not positive finite")
        nis = np.square(innovation) / innovation_variance
        update = KalmanUpdate(
            innovations=innovation * self._scale,
            innovation_variances=innovation_variance * np.square(self._scale),
            nis=nis,
            value_gain=value_gain,
            posterior_covariance=updated_covariance,
        )
        self._trial.mean = updated_mean
        self._trial.covariance = updated_covariance
        self._trial.update = update
        return update

    def commit_trial(self) -> None:
        if self._trial is None or self._trial.update is None:
            raise CalibrationContractError("cannot commit an unassimilated trial")
        self._mean = self._trial.mean
        self._covariance = self._trial.covariance
        self.committed_step = self._trial.target_step
        self.accepted_state_count += 1
        self._trial = None

    def discard_trial(self) -> None:
        if self._trial is None:
            raise CalibrationContractError("no active trial to discard")
        self._trial = None

    def snapshot(self) -> KalmanEngineSnapshot:
        if self._trial is not None:
            raise CalibrationContractError("cannot snapshot an active trial")
        return KalmanEngineSnapshot(
            candidate_fingerprint=self.candidate.fingerprint,
            model=self.candidate.model,
            layout_id=self.layout_id,
            committed_step=self.committed_step,
            accepted_state_count=self.accepted_state_count,
            mean=self._mean,
            covariance=self._covariance,
        )
