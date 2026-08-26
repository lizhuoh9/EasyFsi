"""Transactional initial-guess policies for one interface marker field.

The controller is deliberately host-only and owns no physical solver state.
It selects the first interface guess for a macro-step and commits only the
formally accepted interface observation.  Consequently, rejected coupling
trials and failed macro-steps cannot advance its history or its Kalman state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np

from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
    InterfaceKalmanPredictor,
    InterfaceKalmanUpdate,
)


INITIAL_GUESS_MODES = frozenset(
    {
        "carry_forward",
        "linear_extrapolation",
        "kalman",
        "oracle_replay",
    }
)


@dataclass(frozen=True)
class _ActiveStep:
    dt_s: float
    layout_id: str
    prediction: np.ndarray
    mode_used: str
    fallback_reason: str | None
    kalman_prediction_used: bool


class InterfaceInitialGuessController:
    """Choose an iteration-0 marker-velocity guess with strict transactions.

    ``begin_step`` may be called once for an FSI macro-step.  ``accept_step``
    commits its final accepted marker velocity; ``discard_step`` leaves the
    accepted history exactly as it was at ``begin_step``.  The caller remains
    responsible for restoring fluid, solid, and interface physical states.

    Q3 is an offline upper bound: each accepted step consumes one supplied
    next-accepted-state replay entry.  It is intentionally not a deployable
    predictor.
    """

    def __init__(
        self,
        mode: str,
        *,
        kalman_config: InterfaceKalmanConfig | None = None,
        oracle_replay: Sequence[Any] | None = None,
    ) -> None:
        if not isinstance(mode, str) or mode not in INITIAL_GUESS_MODES:
            raise ValueError(
                "mode must be one of carry_forward, linear_extrapolation, "
                "kalman, oracle_replay"
            )
        if mode == "kalman":
            if not isinstance(kalman_config, InterfaceKalmanConfig):
                raise TypeError("kalman_config is required for kalman mode")
        elif kalman_config is not None:
            raise ValueError("kalman_config is only valid for kalman mode")
        if mode == "oracle_replay":
            if oracle_replay is None:
                raise ValueError("oracle_replay is required for oracle_replay mode")
            replay = tuple(_finite_values(item, name="oracle_replay entry") for item in oracle_replay)
            if not replay:
                raise ValueError("oracle_replay must be non-empty")
        else:
            if oracle_replay is not None:
                raise ValueError("oracle_replay is only valid for oracle_replay mode")
            replay = ()

        self.mode = mode
        self._kalman = (
            InterfaceKalmanPredictor(kalman_config)
            if kalman_config is not None
            else None
        )
        self._oracle_replay = replay
        self._oracle_cursor = 0
        self._active: _ActiveStep | None = None
        self._layout_id: str | None = None
        self._shape: tuple[int, ...] | None = None
        self._latest_accepted: np.ndarray | None = None
        self._previous_accepted: np.ndarray | None = None
        self._previous_dt_s: float | None = None
        self._accepted_step_count = 0
        self._begin_count = 0
        self._discard_count = 0
        self._last_prediction_rms_mps: float | None = None
        self._last_prediction_bias: float | None = None
        self._last_nis_mean: float | None = None
        self._last_mode_used: str | None = None
        self._last_fallback_reason: str | None = None
        self._last_kalman_prediction_used = False

    @property
    def has_active_step(self) -> bool:
        """Whether one uncommitted macro-step transaction is open."""

        return self._active is not None

    def begin_step(
        self,
        accepted_values: Any,
        *,
        dt_s: float,
        layout_id: str,
    ) -> np.ndarray:
        """Open a macro-step and return a defensive iteration-0 guess."""

        if self._active is not None:
            raise RuntimeError("an initial-guess step is already active")
        current = _finite_values(accepted_values, name="accepted_values")
        dt = _positive_finite_real("dt_s", dt_s)
        layout = _validated_layout_id(layout_id)
        self._validate_or_initialize_layout(current, layout)
        prediction, mode_used, fallback_reason, kalman_prediction_used = (
            self._prediction_for_begin(dt)
        )
        self._active = _ActiveStep(
            dt_s=dt,
            layout_id=layout,
            prediction=_read_only_copy(prediction),
            mode_used=mode_used,
            fallback_reason=fallback_reason,
            kalman_prediction_used=kalman_prediction_used,
        )
        self._last_mode_used = mode_used
        self._last_fallback_reason = fallback_reason
        self._last_kalman_prediction_used = kalman_prediction_used
        self._begin_count += 1
        return np.array(prediction, dtype=np.float64, copy=True)

    def accept_step(self, accepted_values: Any, *, layout_id: str) -> None:
        """Commit the final accepted interface observation for this step."""

        active = self._require_active_step()
        layout = _validated_layout_id(layout_id)
        if layout != active.layout_id:
            raise ValueError("layout_id does not match the active step layout")
        accepted = _finite_values(
            accepted_values,
            name="accepted_values",
            expected_shape=self._require_shape(),
        )
        update: InterfaceKalmanUpdate | None = None
        if self._kalman is not None:
            try:
                update = self._kalman.update_trial(accepted, layout_id=layout)
                self._kalman.commit_trial()
            except Exception:
                if self._kalman.has_active_trial:
                    self._kalman.discard_trial()
                raise

        prediction_error = active.prediction - accepted
        self._last_prediction_rms_mps = _root_mean_square(prediction_error)
        self._last_prediction_bias = float(np.mean(prediction_error))
        self._last_nis_mean = (
            float(np.mean(update.normalized_innovation_squared))
            if update is not None
            else None
        )
        self._previous_accepted = self._latest_accepted
        self._latest_accepted = _read_only_copy(accepted)
        self._previous_dt_s = active.dt_s
        if self.mode == "oracle_replay":
            self._oracle_cursor += 1
        self._accepted_step_count += 1
        self._active = None

    def discard_step(self) -> None:
        """Discard the active transaction without advancing any history."""

        self._require_active_step()
        if self._kalman is not None and self._kalman.has_active_trial:
            self._kalman.discard_trial()
        self._active = None
        self._discard_count += 1

    def report(self) -> dict[str, bool | int | float | str | None]:
        """Return JSON-serializable cumulative controller diagnostics."""

        return {
            "mode": self.mode,
            "offline_oracle": self.mode == "oracle_replay",
            "deployable": self.mode != "oracle_replay",
            "has_active_step": self.has_active_step,
            "begin_count": self._begin_count,
            "accepted_step_count": self._accepted_step_count,
            "discard_count": self._discard_count,
            "oracle_replay_cursor": self._oracle_cursor,
            "kalman_accepted_state_count": (
                self._kalman.accepted_state_count if self._kalman is not None else 0
            ),
            "kalman_ready": self._kalman.ready if self._kalman is not None else False,
            "mode_used": self._last_mode_used,
            "fallback_reason": self._last_fallback_reason,
            "kalman_prediction_used": self._last_kalman_prediction_used,
            "last_prediction_rms_mps": self._last_prediction_rms_mps,
            "last_prediction_bias": self._last_prediction_bias,
            "last_nis_mean": self._last_nis_mean,
        }

    def _validate_or_initialize_layout(
        self,
        values: np.ndarray,
        layout_id: str,
    ) -> None:
        if self._layout_id is None:
            self._layout_id = layout_id
            self._shape = tuple(values.shape)
            self._latest_accepted = _read_only_copy(values)
            if self._kalman is not None:
                self._kalman.initialize(values, layout_id=layout_id)
            return
        if layout_id != self._layout_id:
            raise ValueError("layout_id changed; reset is required before prediction")
        if tuple(values.shape) != self._require_shape():
            raise ValueError(
                "accepted_values shape does not match the accepted layout: "
                f"{tuple(values.shape)} != {self._require_shape()}"
            )

    def _prediction_for_begin(
        self, dt_s: float
    ) -> tuple[np.ndarray, str, str | None, bool]:
        latest = self._require_latest_accepted()
        if self.mode == "carry_forward":
            return latest, "carry_forward", None, False
        if self.mode == "linear_extrapolation":
            if self._previous_accepted is None or self._previous_dt_s is None:
                return (
                    latest,
                    "carry_forward",
                    "insufficient_accepted_history",
                    False,
                )
            ratio = dt_s / self._previous_dt_s
            return (
                latest + ratio * (latest - self._previous_accepted),
                "linear_extrapolation",
                None,
                False,
            )
        if self.mode == "kalman":
            assert self._kalman is not None
            prediction = self._kalman.predict_trial(
                dt=dt_s,
                layout_id=self._require_layout_id(),
            ).values
            if not self._kalman.ready:
                return latest, "carry_forward", "kalman_warmup", False
            return prediction, "kalman", None, True
        assert self.mode == "oracle_replay"
        if self._oracle_cursor >= len(self._oracle_replay):
            raise RuntimeError("oracle_replay is exhausted")
        replay = self._oracle_replay[self._oracle_cursor]
        if tuple(replay.shape) != self._require_shape():
            raise ValueError(
                "oracle_replay entry shape does not match the accepted layout: "
                f"{tuple(replay.shape)} != {self._require_shape()}"
            )
        return replay, "oracle_replay", None, False

    def _require_active_step(self) -> _ActiveStep:
        if self._active is None:
            raise RuntimeError("no active initial-guess step")
        return self._active

    def _require_layout_id(self) -> str:
        if self._layout_id is None:
            raise RuntimeError("initial-guess controller is not initialized")
        return self._layout_id

    def _require_shape(self) -> tuple[int, ...]:
        if self._shape is None:
            raise RuntimeError("initial-guess controller is not initialized")
        return self._shape

    def _require_latest_accepted(self) -> np.ndarray:
        if self._latest_accepted is None:
            raise RuntimeError("initial-guess controller is not initialized")
        return self._latest_accepted


def _finite_values(
    values: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        raise TypeError(f"{name} must not be boolean")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    try:
        converted = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if expected_shape is not None and tuple(converted.shape) != expected_shape:
        raise ValueError(
            f"{name} shape {tuple(converted.shape)} != {expected_shape}"
        )
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_finite_real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _validated_layout_id(layout_id: Any) -> str:
    if not isinstance(layout_id, str):
        raise TypeError("layout_id must be a string")
    if not layout_id.strip():
        raise ValueError("layout_id must be non-empty")
    return layout_id


def _read_only_copy(values: np.ndarray) -> np.ndarray:
    copy = np.array(values, dtype=np.float64, copy=True)
    copy.flags.writeable = False
    return copy


def _root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


__all__ = ["INITIAL_GUESS_MODES", "InterfaceInitialGuessController"]
