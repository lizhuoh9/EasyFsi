"""Transactional initial-guess policies for one interface marker field.

The controller is deliberately host-only and owns no physical solver state.
It selects the first interface guess for a macro-step and commits only the
formally accepted interface observation.  Consequently, rejected coupling
trials and failed macro-steps cannot advance its history or its Kalman state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
    InterfaceKalmanPredictor,
    InterfaceKalmanSnapshot,
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


INTERFACE_INITIAL_GUESS_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _ActiveStep:
    dt_s: float
    layout_id: str
    prediction: np.ndarray
    mode_used: str
    fallback_reason: str | None
    kalman_prediction_used: bool


@dataclass(frozen=True)
class InterfaceInitialGuessSnapshot:
    """Typed accepted-boundary state for one initial-guess controller."""

    schema_version: int
    mode: str
    kalman_config: InterfaceKalmanConfig | None
    oracle_replay: tuple[np.ndarray, ...]
    oracle_replay_hash: str
    oracle_cursor: int
    layout_id: str | None
    shape: tuple[int, ...] | None
    latest_accepted: np.ndarray | None
    previous_accepted: np.ndarray | None
    previous_dt_s: float | None
    accepted_step_count: int
    begin_count: int
    discard_count: int
    last_prediction_rms_mps: float | None
    last_prediction_bias: float | None
    last_nis_mean: float | None
    last_mode_used: str | None
    last_fallback_reason: str | None
    last_kalman_prediction_used: bool
    kalman_snapshot: InterfaceKalmanSnapshot | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, Integral)
        ):
            raise TypeError("snapshot schema_version must be an integer")
        if int(self.schema_version) != INTERFACE_INITIAL_GUESS_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported initial-guess snapshot schema_version")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if not isinstance(self.mode, str) or self.mode not in INITIAL_GUESS_MODES:
            raise ValueError("snapshot mode is invalid")
        if self.mode == "kalman":
            if not isinstance(self.kalman_config, InterfaceKalmanConfig):
                raise TypeError("kalman snapshot requires kalman_config")
        elif self.kalman_config is not None:
            raise ValueError("non-Kalman snapshot must not have kalman_config")
        replay = tuple(
            _finite_values(entry, name="snapshot oracle_replay entry")
            for entry in self.oracle_replay
        )
        if self.mode == "oracle_replay":
            if not replay:
                raise ValueError("oracle snapshot requires replay entries")
        elif replay:
            raise ValueError("non-oracle snapshot must not have replay entries")
        replay_hash = _oracle_replay_hash(replay)
        if self.oracle_replay_hash != replay_hash:
            raise ValueError("snapshot oracle_replay_hash does not match replay")
        cursor = _nonnegative_integer("oracle_cursor", self.oracle_cursor)
        accepted = _nonnegative_integer(
            "accepted_step_count", self.accepted_step_count
        )
        begin = _nonnegative_integer("begin_count", self.begin_count)
        discard = _nonnegative_integer("discard_count", self.discard_count)
        if accepted + discard != begin:
            raise ValueError("snapshot step counters are inconsistent")
        if self.mode == "oracle_replay":
            if cursor != accepted or cursor > len(replay):
                raise ValueError("oracle snapshot cursor is inconsistent")
        elif cursor != 0:
            raise ValueError("non-oracle snapshot must have zero cursor")
        layout = None if self.layout_id is None else _validated_layout_id(self.layout_id)
        shape = _snapshot_shape(self.shape)
        latest = _snapshot_array(self.latest_accepted, "latest_accepted", shape)
        previous = _snapshot_array(
            self.previous_accepted, "previous_accepted", shape
        )
        previous_dt = (
            None
            if self.previous_dt_s is None
            else _positive_finite_real("previous_dt_s", self.previous_dt_s)
        )
        cold = layout is None or shape is None
        if cold:
            if (
                layout is not None
                or shape is not None
                or latest is not None
                or previous is not None
                or previous_dt is not None
                or self.kalman_snapshot is not None
                or accepted
                or begin
                or discard
                or cursor
            ):
                raise ValueError("cold snapshot has accepted payload")
        else:
            if latest is None:
                raise ValueError("initialized snapshot requires latest_accepted")
            if accepted == 0:
                if previous is not None or previous_dt is not None:
                    raise ValueError("unaccepted snapshot has previous history")
            elif previous is None or previous_dt is None:
                raise ValueError("accepted snapshot requires previous history")
        if self.mode == "kalman" and not cold:
            if not isinstance(self.kalman_snapshot, InterfaceKalmanSnapshot):
                raise TypeError("initialized Kalman snapshot requires predictor")
            if self.kalman_snapshot.config != self.kalman_config:
                raise ValueError("Kalman snapshot config does not match")
            if (
                self.kalman_snapshot.layout_id != layout
                or tuple(self.kalman_snapshot.values.shape) != shape
                or self.kalman_snapshot.accepted_state_count != accepted + 1
            ):
                raise ValueError("Kalman snapshot state is inconsistent")
        elif self.kalman_snapshot is not None:
            raise ValueError("unexpected Kalman predictor state")
        for name in (
            "last_prediction_rms_mps",
            "last_prediction_bias",
            "last_nis_mean",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_real(name, value)
        if (
            self.last_mode_used is not None
            and self.last_mode_used not in INITIAL_GUESS_MODES
        ):
            raise ValueError("snapshot last_mode_used is invalid")
        if not isinstance(self.last_kalman_prediction_used, (bool, np.bool_)):
            raise TypeError("last_kalman_prediction_used must be boolean")
        object.__setattr__(
            self, "oracle_replay", tuple(_read_only_copy(entry) for entry in replay)
        )
        object.__setattr__(self, "layout_id", layout)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "latest_accepted", latest)
        object.__setattr__(self, "previous_accepted", previous)
        object.__setattr__(self, "previous_dt_s", previous_dt)
        object.__setattr__(self, "oracle_cursor", cursor)
        object.__setattr__(self, "accepted_step_count", accepted)
        object.__setattr__(self, "begin_count", begin)
        object.__setattr__(self, "discard_count", discard)

    def validate(self) -> None:
        """Revalidate mutable-array payloads before restore."""

        self.__post_init__()


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


    def snapshot(self) -> InterfaceInitialGuessSnapshot:
        """Capture exact host-only accepted state at a transaction boundary."""

        if self._active is not None:
            raise RuntimeError(
                "cannot snapshot while an initial-guess step is active"
            )
        return InterfaceInitialGuessSnapshot(
            schema_version=INTERFACE_INITIAL_GUESS_SNAPSHOT_SCHEMA_VERSION,
            mode=self.mode,
            kalman_config=(
                self._kalman.config if self._kalman is not None else None
            ),
            oracle_replay=self._oracle_replay,
            oracle_replay_hash=_oracle_replay_hash(self._oracle_replay),
            oracle_cursor=self._oracle_cursor,
            layout_id=self._layout_id,
            shape=self._shape,
            latest_accepted=self._latest_accepted,
            previous_accepted=self._previous_accepted,
            previous_dt_s=self._previous_dt_s,
            accepted_step_count=self._accepted_step_count,
            begin_count=self._begin_count,
            discard_count=self._discard_count,
            last_prediction_rms_mps=self._last_prediction_rms_mps,
            last_prediction_bias=self._last_prediction_bias,
            last_nis_mean=self._last_nis_mean,
            last_mode_used=self._last_mode_used,
            last_fallback_reason=self._last_fallback_reason,
            last_kalman_prediction_used=self._last_kalman_prediction_used,
            kalman_snapshot=(
                self._kalman.snapshot()
                if self._kalman is not None and self._layout_id is not None
                else None
            ),
        )

    def restore(self, snapshot: InterfaceInitialGuessSnapshot) -> None:
        """Atomically install a validated accepted-boundary snapshot."""

        if self._active is not None:
            raise RuntimeError(
                "cannot restore while an initial-guess step is active"
            )
        if not isinstance(snapshot, InterfaceInitialGuessSnapshot):
            raise TypeError("snapshot must be an InterfaceInitialGuessSnapshot")
        snapshot.validate()
        if snapshot.mode != self.mode:
            raise ValueError("snapshot mode does not match controller mode")
        config = self._kalman.config if self._kalman is not None else None
        if snapshot.kalman_config != config:
            raise ValueError(
                "snapshot Kalman config does not match controller config"
            )
        if snapshot.oracle_replay_hash != _oracle_replay_hash(self._oracle_replay):
            raise ValueError(
                "snapshot oracle_replay does not match controller replay"
            )
        if self._layout_id is not None and (
            snapshot.layout_id != self._layout_id
            or snapshot.shape != self._shape
        ):
            raise ValueError(
                "snapshot layout or shape does not match controller state"
            )

        restored_kalman: InterfaceKalmanPredictor | None = None
        if config is not None:
            restored_kalman = InterfaceKalmanPredictor(config)
            if snapshot.kalman_snapshot is not None:
                restored_kalman.restore(snapshot.kalman_snapshot)
        latest = _copy_or_none(snapshot.latest_accepted)
        previous = _copy_or_none(snapshot.previous_accepted)

        self._kalman = restored_kalman
        self._oracle_cursor = snapshot.oracle_cursor
        self._layout_id = snapshot.layout_id
        self._shape = snapshot.shape
        self._latest_accepted = latest
        self._previous_accepted = previous
        self._previous_dt_s = snapshot.previous_dt_s
        self._accepted_step_count = snapshot.accepted_step_count
        self._begin_count = snapshot.begin_count
        self._discard_count = snapshot.discard_count
        self._last_prediction_rms_mps = snapshot.last_prediction_rms_mps
        self._last_prediction_bias = snapshot.last_prediction_bias
        self._last_nis_mean = snapshot.last_nis_mean
        self._last_mode_used = snapshot.last_mode_used
        self._last_fallback_reason = snapshot.last_fallback_reason
        self._last_kalman_prediction_used = bool(
            snapshot.last_kalman_prediction_used
        )

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


def _copy_or_none(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else _read_only_copy(values)


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _snapshot_shape(shape: Any) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if not isinstance(shape, tuple):
        raise TypeError("snapshot shape must be a tuple")
    result = tuple(_nonnegative_integer("snapshot shape entry", value) for value in shape)
    if any(value == 0 for value in result):
        raise ValueError("snapshot shape entries must be positive")
    return result


def _snapshot_array(
    values: Any, name: str, shape: tuple[int, ...] | None
) -> np.ndarray | None:
    if values is None:
        return None
    if shape is None:
        raise ValueError(f"{name} requires snapshot shape")
    return _read_only_copy(_finite_values(values, name=name, expected_shape=shape))


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _oracle_replay_hash(replay: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for entry in replay:
        values = np.asarray(entry, dtype=np.float64)
        digest.update(repr(tuple(values.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


__all__ = [
    "INITIAL_GUESS_MODES",
    "INTERFACE_INITIAL_GUESS_SNAPSHOT_SCHEMA_VERSION",
    "InterfaceInitialGuessController",
    "InterfaceInitialGuessSnapshot",
]
