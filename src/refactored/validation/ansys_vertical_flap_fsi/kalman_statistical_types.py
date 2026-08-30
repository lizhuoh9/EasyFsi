"""Immutable data contracts and shared helpers for the R24 Kalman audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 1
_AXIS_ORDER = ("x", "y", "z")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KALMAN_MODELS = frozenset(("random_walk", "constant_rate"))
_MODELS = _KALMAN_MODELS | frozenset(("carry", "linear", "production"))
_NIS_95_ONE_DOF = 3.841458820694124


class CalibrationContractError(ValueError):
    """Fail-closed R24 input, state, or statistical contract violation."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _readonly(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_array(
    values: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationContractError(f"{name} must be real float64 data") from exc
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise CalibrationContractError(f"{name} must be non-empty and finite")
    if shape is not None and tuple(result.shape) != shape:
        raise CalibrationContractError(
            f"{name} shape {tuple(result.shape)} != {shape}"
        )
    return np.array(result, dtype=np.float64, copy=True)


def _positive_float(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationContractError(f"{name} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationContractError(f"{name} must be finite numeric") from exc
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise CalibrationContractError(f"{name} must be finite and {qualifier}")
    return result


def _xyz(
    values: Sequence[float],
    *,
    name: str,
    allow_zero: bool = False,
) -> tuple[float, float, float]:
    if len(values) != 3:
        raise CalibrationContractError(f"{name} must contain xyz values")
    return tuple(
        _positive_float(value, name=f"{name}[{axis}]", allow_zero=allow_zero)
        for axis, value in zip(_AXIS_ORDER, values, strict=True)
    )


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CalibrationContractError(f"{name} must be lowercase SHA256")
    return value


@dataclass(frozen=True)
class AcceptedTrace:
    """Immutable accepted-state observations and their evidence bindings."""

    name: str
    values: np.ndarray
    dt_s: float
    layout_id: str
    axis_order: tuple[str, str, str]
    source_fingerprint: str
    source_steps: tuple[int, ...]
    frame_sha256: tuple[str, ...]
    history_sha256: tuple[str, ...]
    journal_sha256: tuple[str, ...]
    fsi_iterations: tuple[int, ...]
    cg_iterations: tuple[int, ...]
    matvec_count: tuple[int | None, ...]
    canonical_root: str | None = None
    attempt_root: str | None = None
    source_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CalibrationContractError("trace name must be non-empty")
        values = _finite_array(self.values, name="accepted trace values")
        if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] != 3:
            raise CalibrationContractError(
                "accepted trace values must have shape (steps, markers, 3)"
            )
        if tuple(self.axis_order) != _AXIS_ORDER:
            raise CalibrationContractError("axis_order must be exactly ('x', 'y', 'z')")
        dt_s = _positive_float(self.dt_s, name="dt_s")
        layout_id = _sha256(self.layout_id, name="layout_id")
        source_fingerprint = _sha256(
            self.source_fingerprint, name="source_fingerprint"
        )
        count = values.shape[0]
        steps = tuple(int(step) for step in self.source_steps)
        if len(steps) != count or any(
            step <= 0 or (index and step != steps[index - 1] + 1)
            for index, step in enumerate(steps)
        ):
            raise CalibrationContractError(
                "source_steps must be positive, contiguous, and frame-aligned"
            )
        for field_name in ("frame_sha256", "history_sha256", "journal_sha256"):
            hashes = tuple(getattr(self, field_name))
            if len(hashes) != count:
                raise CalibrationContractError(f"{field_name} must align to frames")
            for value in hashes:
                _sha256(value, name=field_name)
            object.__setattr__(self, field_name, hashes)
        for field_name in ("fsi_iterations", "cg_iterations", "matvec_count"):
            values_tuple = tuple(getattr(self, field_name))
            if len(values_tuple) != count:
                raise CalibrationContractError(f"{field_name} must align to frames")
            object.__setattr__(self, field_name, values_tuple)
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "dt_s", dt_s)
        object.__setattr__(self, "layout_id", layout_id)
        object.__setattr__(self, "source_fingerprint", source_fingerprint)
        object.__setattr__(self, "source_steps", steps)
        object.__setattr__(self, "source_sha256", tuple(self.source_sha256))

    @classmethod
    def synthetic(
        cls,
        values: Any,
        *,
        name: str,
        dt_s: float,
        layout_id: str,
        source_fingerprint: str,
        axis_order: tuple[str, str, str] = _AXIS_ORDER,
    ) -> "AcceptedTrace":
        array = _finite_array(values, name="synthetic values")
        if array.ndim != 3:
            raise CalibrationContractError(
                "synthetic values must have shape (steps, markers, 3)"
            )
        count = array.shape[0]
        hashes = tuple(
            hashlib.sha256(
                np.ascontiguousarray(array[index]).tobytes()
                + int(index + 1).to_bytes(8, "big")
            ).hexdigest()
            for index in range(count)
        )
        history = tuple(
            hashlib.sha256(f"history:{value}".encode()).hexdigest()
            for value in hashes
        )
        journal = tuple(
            hashlib.sha256(f"journal:{value}".encode()).hexdigest()
            for value in hashes
        )
        return cls(
            name=name,
            values=array,
            dt_s=dt_s,
            layout_id=layout_id,
            axis_order=axis_order,
            source_fingerprint=source_fingerprint,
            source_steps=tuple(range(1, count + 1)),
            frame_sha256=hashes,
            history_sha256=history,
            journal_sha256=journal,
            fsi_iterations=(0,) * count,
            cg_iterations=(0,) * count,
            matvec_count=(None,) * count,
        )


@dataclass(frozen=True)
class CandidateSpec:
    """Frozen predictor identity in dimensionless axis coordinates."""

    candidate_id: str
    model: str
    axis_order: tuple[str, str, str]
    scale_xyz: tuple[float, float, float]
    q_xyz: tuple[float, float, float]
    r_xyz: tuple[float, float, float]
    p0_value_xyz: tuple[float, float, float]
    p0_rate_xyz: tuple[float, float, float]
    warmup_accepted_states: int
    active_axes: tuple[bool, bool, bool] = (True, True, True)
    beta: float = 1.0
    layout_id: str | None = None
    q_multiplier: float = 1.0
    r_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise CalibrationContractError("candidate_id must be non-empty")
        if self.model not in _MODELS:
            raise CalibrationContractError(f"unsupported model {self.model!r}")
        if tuple(self.axis_order) != _AXIS_ORDER:
            raise CalibrationContractError("candidate axis_order must be xyz")
        active_axes = tuple(self.active_axes)
        if len(active_axes) != 3 or any(
            not isinstance(value, (bool, np.bool_)) for value in active_axes
        ):
            raise CalibrationContractError("active_axes must contain three booleans")
        if not any(active_axes):
            raise CalibrationContractError(
                "at least one statistical axis must be active"
            )
        object.__setattr__(
            self, "active_axes", tuple(bool(value) for value in active_axes)
        )
        object.__setattr__(self, "scale_xyz", _xyz(self.scale_xyz, name="scale_xyz"))
        object.__setattr__(
            self, "q_xyz", _xyz(self.q_xyz, name="q_xyz", allow_zero=True)
        )
        object.__setattr__(self, "r_xyz", _xyz(self.r_xyz, name="r_xyz"))
        object.__setattr__(
            self, "p0_value_xyz", _xyz(self.p0_value_xyz, name="p0_value_xyz")
        )
        object.__setattr__(
            self, "p0_rate_xyz", _xyz(self.p0_rate_xyz, name="p0_rate_xyz")
        )
        if isinstance(self.warmup_accepted_states, bool):
            raise CalibrationContractError("warmup_accepted_states must be integer")
        warmup = int(self.warmup_accepted_states)
        if warmup < 1:
            raise CalibrationContractError(
                "warmup_accepted_states must be at least one"
            )
        object.__setattr__(self, "warmup_accepted_states", warmup)
        object.__setattr__(
            self, "beta", _positive_float(self.beta, name="beta", allow_zero=True)
        )
        object.__setattr__(
            self,
            "q_multiplier",
            _positive_float(self.q_multiplier, name="q_multiplier"),
        )
        object.__setattr__(
            self,
            "r_multiplier",
            _positive_float(self.r_multiplier, name="r_multiplier"),
        )
        if self.layout_id is not None:
            object.__setattr__(
                self, "layout_id", _sha256(self.layout_id, name="candidate layout_id")
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "model": self.model,
            "axis_order": list(self.axis_order),
            "active_axes": list(self.active_axes),
            "scale_xyz": list(self.scale_xyz),
            "q_xyz": list(self.q_xyz),
            "r_xyz": list(self.r_xyz),
            "p0_value_xyz": list(self.p0_value_xyz),
            "p0_rate_xyz": list(self.p0_rate_xyz),
            "warmup_accepted_states": self.warmup_accepted_states,
            "beta": self.beta,
            "layout_id": self.layout_id,
            "q_multiplier": self.q_multiplier,
            "r_multiplier": self.r_multiplier,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_payload())


@dataclass(frozen=True)
class KalmanEngineSnapshot:
    candidate_fingerprint: str
    model: str
    layout_id: str
    committed_step: int
    accepted_state_count: int
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _readonly(self.mean))
        object.__setattr__(self, "covariance", _readonly(self.covariance))


@dataclass(frozen=True)
class KalmanPrediction:
    values: np.ndarray
    prior_covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values))
        object.__setattr__(
            self, "prior_covariance", _readonly(self.prior_covariance)
        )


@dataclass(frozen=True)
class KalmanUpdate:
    innovations: np.ndarray
    innovation_variances: np.ndarray
    nis: np.ndarray
    value_gain: np.ndarray
    posterior_covariance: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "innovations",
            "innovation_variances",
            "nis",
            "value_gain",
            "posterior_covariance",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
