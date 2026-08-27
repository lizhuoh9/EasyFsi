"""Solver-independent temporal Kalman prediction for interface arrays.

The predictor models every scalar entry of an interface array independently
with the constant-rate state ``[value, rate]``.  For a Turek--Hron marker
velocity array, ``value`` is velocity and ``rate`` is acceleration.  A
pressure-Neumann-gradient array, if it is useful for shadow diagnostics, must
use a separate predictor instance because it has different units and noise.
For the current Turek--Hron active path, that derived gradient is recomputed by
the fluid trial and must not be replaced by a Kalman output.

This module deliberately has no Taichi, HIBM, case, or solver imports.  It
never writes physical state.  A future solver adapter owns dtype conversion,
geometry safeguards, residual checks, and the decision to use a prediction as
the first coupling iterate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


INTERFACE_KALMAN_SNAPSHOT_SCHEMA_VERSION = 1

CovarianceSpec = float | tuple[float, float, float]


@dataclass(frozen=True)
class InterfaceKalmanConfig:
    """Numerical configuration for one independently scaled interface field.

    ``rate_process_noise_spectral_density`` is the continuous white-noise
    intensity acting on the derivative of ``rate``.  If ``value`` has unit
    ``U``, its unit is ``U**2 / s**3``.  The resulting discrete covariance is

    ``q * [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]``.

    All remaining covariance parameters are variances, not standard
    deviations.  Their scale must be selected for the field represented by
    this predictor instance.
    """

    rate_process_noise_spectral_density: CovarianceSpec
    measurement_variance: CovarianceSpec
    initial_value_variance: CovarianceSpec
    initial_rate_variance: CovarianceSpec
    warmup_accepted_states: int = 5

    def __post_init__(self) -> None:
        for name in (
            "rate_process_noise_spectral_density",
            "initial_value_variance",
            "initial_rate_variance",
        ):
            value = _covariance_spec(
                name,
                getattr(self, name),
                positive=False,
            )
            object.__setattr__(self, name, value)

        measurement_variance = _covariance_spec(
            "measurement_variance",
            self.measurement_variance,
            positive=True,
        )
        object.__setattr__(
            self, "measurement_variance", measurement_variance
        )

        if isinstance(self.warmup_accepted_states, (bool, np.bool_)):
            raise TypeError("warmup_accepted_states must be an integer")
        if not isinstance(self.warmup_accepted_states, Integral):
            raise TypeError("warmup_accepted_states must be an integer")
        warmup = int(self.warmup_accepted_states)
        if warmup < 1:
            raise ValueError("warmup_accepted_states must be at least 1")
        object.__setattr__(self, "warmup_accepted_states", warmup)


@dataclass(frozen=True)
class InterfaceKalmanEstimate:
    """Immutable, shape-preserving copy of a filter estimate."""

    values: np.ndarray
    rates: np.ndarray
    covariances: np.ndarray
    value_variances: np.ndarray
    rate_variances: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "values",
            "rates",
            "covariances",
            "value_variances",
            "rate_variances",
        ):
            object.__setattr__(self, name, _read_only_copy(getattr(self, name)))


@dataclass(frozen=True)
class InterfaceKalmanUpdate:
    """Accepted-observation diagnostics for one trial update."""

    estimate: InterfaceKalmanEstimate
    innovations: np.ndarray
    innovation_variances: np.ndarray
    normalized_innovation_squared: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "innovations",
            "innovation_variances",
            "normalized_innovation_squared",
        ):
            object.__setattr__(self, name, _read_only_copy(getattr(self, name)))


@dataclass(frozen=True)
class InterfaceKalmanSnapshot:
    """Serializable accepted-boundary state for exact restart continuity."""

    schema_version: int
    config: InterfaceKalmanConfig
    layout_id: str
    accepted_state_count: int
    values: np.ndarray
    rates: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, (bool, np.bool_)) or not isinstance(
            self.schema_version, Integral
        ):
            raise TypeError("schema_version must be an integer")
        schema_version = int(self.schema_version)
        if schema_version != INTERFACE_KALMAN_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported InterfaceKalmanSnapshot schema_version: "
                f"{schema_version}"
            )
        if not isinstance(self.config, InterfaceKalmanConfig):
            raise TypeError("config must be an InterfaceKalmanConfig")
        layout_id = _validated_layout_id(self.layout_id)
        if isinstance(self.accepted_state_count, (bool, np.bool_)) or not isinstance(
            self.accepted_state_count, Integral
        ):
            raise TypeError("accepted_state_count must be an integer")
        accepted_state_count = int(self.accepted_state_count)
        if accepted_state_count < 1:
            raise ValueError("accepted_state_count must be positive")

        values = _finite_array(self.values, name="values")
        rates = _finite_array(
            self.rates,
            name="rates",
            expected_shape=tuple(values.shape),
        )
        covariances = _finite_array(
            self.covariances,
            name="covariances",
            expected_shape=tuple(values.shape) + (2, 2),
        )
        _EstimatorState(
            mean=np.column_stack((values.reshape(-1), rates.reshape(-1))),
            covariance=covariances.reshape(-1, 2, 2),
            accepted_state_count=accepted_state_count,
        )

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "layout_id", layout_id)
        object.__setattr__(
            self, "accepted_state_count", accepted_state_count
        )
        object.__setattr__(self, "values", _read_only_copy(values))
        object.__setattr__(self, "rates", _read_only_copy(rates))
        object.__setattr__(
            self, "covariances", _read_only_copy(covariances)
        )


@dataclass(frozen=True)
class _EstimatorState:
    mean: np.ndarray
    covariance: np.ndarray
    accepted_state_count: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if mean.ndim != 2 or mean.shape[1] != 2:
            raise RuntimeError("Kalman mean must have shape (dof_count, 2)")
        if covariance.shape != (mean.shape[0], 2, 2):
            raise RuntimeError(
                "Kalman covariance must have shape (dof_count, 2, 2)"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise RuntimeError("Kalman state must be finite")
        if self.accepted_state_count < 1:
            raise RuntimeError("accepted_state_count must be positive")

        symmetry_error = np.max(
            np.abs(covariance - np.swapaxes(covariance, -1, -2)),
            axis=(-2, -1),
        )
        covariance_scale = np.maximum(
            np.finfo(np.float64).tiny,
            np.max(np.abs(covariance), axis=(-2, -1)),
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * covariance_scale
        if np.any(symmetry_error > tolerance):
            raise RuntimeError("Kalman covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError("Kalman covariance eigenvalues must be finite")
        if np.any(np.min(eigenvalues, axis=-1) < -tolerance):
            raise RuntimeError("Kalman covariance must be positive semidefinite")

        object.__setattr__(self, "mean", _read_only_copy(mean))
        object.__setattr__(self, "covariance", _read_only_copy(covariance))


@dataclass(frozen=True)
class _TrialState:
    estimator: _EstimatorState
    accepted_observation_assimilated: bool = False


class InterfaceKalmanPredictor:
    """Independent-DOF constant-rate Kalman predictor with trial semantics.

    Initialization counts as the first accepted state.  Consequently a
    ``warmup_accepted_states`` value of five makes the predictor ready after
    five accepted snapshots, before predicting the sixth snapshot.

    A Kalman trial spans one physical time step, not one Picard/Aitken/IQN
    coupling iteration.  Predict once before the inner coupling loop, then
    update and commit only after formal physical-step acceptance; otherwise
    discard the trial.
    """

    def __init__(self, config: InterfaceKalmanConfig) -> None:
        if not isinstance(config, InterfaceKalmanConfig):
            raise TypeError("config must be an InterfaceKalmanConfig")
        self.config = config
        self._committed: _EstimatorState | None = None
        self._trial: _TrialState | None = None
        self._shape: tuple[int, ...] | None = None
        self._layout_id: str | None = None

    @property
    def initialized(self) -> bool:
        return self._committed is not None

    @property
    def has_active_trial(self) -> bool:
        return self._trial is not None

    @property
    def ready(self) -> bool:
        return (
            self._committed is not None
            and self._committed.accepted_state_count
            >= self.config.warmup_accepted_states
        )

    @property
    def accepted_state_count(self) -> int:
        if self._committed is None:
            return 0
        return self._committed.accepted_state_count

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape

    @property
    def layout_id(self) -> str | None:
        return self._layout_id

    def initialize(
        self,
        accepted_values: Any,
        *,
        layout_id: str,
        initial_rates: Any | None = None,
    ) -> None:
        """Initialize from a formally accepted interface snapshot."""

        if self.initialized:
            raise RuntimeError("Kalman predictor is already initialized")
        state, shape, validated_layout = self._initial_state(
            accepted_values,
            initial_rates=initial_rates,
            layout_id=layout_id,
        )
        self._committed = state
        self._shape = shape
        self._layout_id = validated_layout

    def reset(
        self,
        accepted_values: Any,
        *,
        layout_id: str,
        initial_rates: Any | None = None,
    ) -> None:
        """Replace all history after an interface layout or identity change."""

        if self.has_active_trial:
            raise RuntimeError("cannot reset while a Kalman trial is active")
        state, shape, validated_layout = self._initial_state(
            accepted_values,
            initial_rates=initial_rates,
            layout_id=layout_id,
        )
        self._committed = state
        self._shape = shape
        self._layout_id = validated_layout

    def committed_estimate(self) -> InterfaceKalmanEstimate:
        """Return a defensive copy of the accepted estimator state."""

        return self._estimate(self._require_committed())

    def snapshot(self) -> InterfaceKalmanSnapshot:
        """Capture the committed estimator at an accepted-step boundary."""

        committed = self._require_committed()
        if self.has_active_trial:
            raise RuntimeError(
                "cannot snapshot while a Kalman trial is active"
            )
        estimate = self._estimate(committed)
        return InterfaceKalmanSnapshot(
            schema_version=INTERFACE_KALMAN_SNAPSHOT_SCHEMA_VERSION,
            config=self.config,
            layout_id=self._require_layout_id(),
            accepted_state_count=committed.accepted_state_count,
            values=estimate.values,
            rates=estimate.rates,
            covariances=estimate.covariances,
        )

    def restore(self, snapshot: InterfaceKalmanSnapshot) -> None:
        """Atomically restore an exact committed estimator snapshot."""

        if self.has_active_trial:
            raise RuntimeError(
                "cannot restore while a Kalman trial is active"
            )
        if not isinstance(snapshot, InterfaceKalmanSnapshot):
            raise TypeError("snapshot must be an InterfaceKalmanSnapshot")
        if snapshot.config != self.config:
            raise ValueError("snapshot config does not match predictor config")

        shape = tuple(snapshot.values.shape)
        restored = _EstimatorState(
            mean=np.column_stack(
                (snapshot.values.reshape(-1), snapshot.rates.reshape(-1))
            ),
            covariance=snapshot.covariances.reshape(-1, 2, 2),
            accepted_state_count=snapshot.accepted_state_count,
        )
        layout_id = _validated_layout_id(snapshot.layout_id)

        self._committed = restored
        self._shape = shape
        self._layout_id = layout_id

    def trial_estimate(self) -> InterfaceKalmanEstimate:
        """Return a defensive copy of the current predicted/updated trial."""

        if self._trial is None:
            raise RuntimeError("no active Kalman trial")
        return self._estimate(self._trial.estimator)

    def predict_trial(
        self,
        *,
        dt: float,
        layout_id: str,
    ) -> InterfaceKalmanEstimate:
        """Predict one uncommitted time step from the accepted estimator."""

        committed = self._require_committed()
        if self._trial is not None:
            raise RuntimeError("a Kalman trial is already active")
        self._require_layout(layout_id)
        time_step = _finite_real("dt", dt)
        if time_step <= 0.0:
            raise ValueError("dt must be positive")

        transition = np.array(
            [[1.0, time_step], [0.0, 1.0]], dtype=np.float64
        )
        process_noise_template = np.array(
            [
                [time_step**3 / 3.0, time_step**2 / 2.0],
                [time_step**2 / 2.0, time_step],
            ],
            dtype=np.float64,
        )
        process_noise_scale = _covariance_values_for_shape(
            self.config.rate_process_noise_spectral_density,
            self._require_shape(),
            name="rate_process_noise_spectral_density",
        )
        predicted_mean = committed.mean @ transition.T
        predicted_covariance = (
            transition @ committed.covariance @ transition.T
            + process_noise_scale[:, None, None]
            * process_noise_template[None, :, :]
        )
        predicted_covariance = _symmetrized(predicted_covariance)
        predicted = _EstimatorState(
            mean=predicted_mean,
            covariance=predicted_covariance,
            accepted_state_count=committed.accepted_state_count,
        )
        result = self._estimate(predicted)
        self._trial = _TrialState(estimator=predicted)
        return result

    def update_trial(
        self,
        accepted_values: Any,
        *,
        layout_id: str,
    ) -> InterfaceKalmanUpdate:
        """Assimilate one formally accepted snapshot into the active trial."""

        committed = self._require_committed()
        if self._trial is None:
            raise RuntimeError("no active Kalman trial")
        if self._trial.accepted_observation_assimilated:
            raise RuntimeError("the active trial already has an accepted observation")
        self._require_layout(layout_id)
        measurement = _finite_array(
            accepted_values,
            name="accepted_values",
            expected_shape=self._shape,
        ).reshape(-1)

        prior = self._trial.estimator
        with np.errstate(over="ignore", invalid="ignore"):
            innovation = measurement - prior.mean[:, 0]
        if not np.all(np.isfinite(innovation)):
            raise RuntimeError("Kalman innovation must be finite")
        measurement_variance = _covariance_values_for_shape(
            self.config.measurement_variance,
            self._require_shape(),
            name="measurement_variance",
        )
        innovation_variance = (
            prior.covariance[:, 0, 0] + measurement_variance
        )
        if not np.all(np.isfinite(innovation_variance)) or np.any(
            innovation_variance <= 0.0
        ):
            raise RuntimeError("Kalman innovation variance is not positive finite")

        gain = prior.covariance[:, :, 0] / innovation_variance[:, None]
        updated_mean = prior.mean + gain * innovation[:, None]

        identity_minus_kh = np.broadcast_to(
            np.eye(2, dtype=np.float64), prior.covariance.shape
        ).copy()
        identity_minus_kh[:, 0, 0] -= gain[:, 0]
        identity_minus_kh[:, 1, 0] -= gain[:, 1]
        updated_covariance = (
            identity_minus_kh
            @ prior.covariance
            @ np.swapaxes(identity_minus_kh, -1, -2)
            + measurement_variance[:, None, None]
            * gain[:, :, None]
            * gain[:, None, :]
        )
        updated_covariance = _symmetrized(updated_covariance)
        updated = _EstimatorState(
            mean=updated_mean,
            covariance=updated_covariance,
            accepted_state_count=committed.accepted_state_count + 1,
        )
        shape = self._require_shape()
        result = InterfaceKalmanUpdate(
            estimate=self._estimate(updated),
            innovations=innovation.reshape(shape),
            innovation_variances=innovation_variance.reshape(shape),
            normalized_innovation_squared=_finite_normalized_innovation_squared(
                innovation, innovation_variance
            ).reshape(shape),
        )
        self._trial = _TrialState(
            estimator=updated,
            accepted_observation_assimilated=True,
        )
        return result

    def commit_trial(self) -> InterfaceKalmanEstimate:
        """Commit a trial only after an accepted observation was assimilated."""

        if self._trial is None:
            raise RuntimeError("no active Kalman trial")
        if not self._trial.accepted_observation_assimilated:
            raise RuntimeError(
                "cannot commit without an accepted observation"
            )
        committed = self._trial.estimator
        result = self._estimate(committed)
        self._committed = committed
        self._trial = None
        return result

    def discard_trial(self) -> None:
        """Discard a failed/retried physical step without changing history."""

        if self._trial is None:
            raise RuntimeError("no active Kalman trial")
        self._trial = None

    def _initial_state(
        self,
        accepted_values: Any,
        *,
        initial_rates: Any | None,
        layout_id: str,
    ) -> tuple[_EstimatorState, tuple[int, ...], str]:
        values = _finite_array(accepted_values, name="accepted_values")
        shape = tuple(values.shape)
        if initial_rates is None:
            rates = np.zeros(shape, dtype=np.float64)
        else:
            rates = _finite_array(
                initial_rates,
                name="initial_rates",
                expected_shape=shape,
            )
        validated_layout = _validated_layout_id(layout_id)

        mean = np.column_stack((values.reshape(-1), rates.reshape(-1)))
        covariance = np.zeros((values.size, 2, 2), dtype=np.float64)
        covariance[:, 0, 0] = _covariance_values_for_shape(
            self.config.initial_value_variance,
            shape,
            name="initial_value_variance",
        )
        covariance[:, 1, 1] = _covariance_values_for_shape(
            self.config.initial_rate_variance,
            shape,
            name="initial_rate_variance",
        )
        return (
            _EstimatorState(
                mean=mean,
                covariance=covariance,
                accepted_state_count=1,
            ),
            shape,
            validated_layout,
        )

    def _require_committed(self) -> _EstimatorState:
        if self._committed is None:
            raise RuntimeError("Kalman predictor is not initialized")
        return self._committed

    def _require_shape(self) -> tuple[int, ...]:
        if self._shape is None:
            raise RuntimeError("Kalman predictor is not initialized")
        return self._shape

    def _require_layout_id(self) -> str:
        if self._layout_id is None:
            raise RuntimeError("Kalman predictor is not initialized")
        return self._layout_id

    def _require_layout(self, layout_id: str) -> None:
        requested = _validated_layout_id(layout_id)
        if requested != self._layout_id:
            raise ValueError(
                "interface layout mismatch; reset the Kalman predictor "
                "before using a changed or reordered layout"
            )

    def _estimate(self, state: _EstimatorState) -> InterfaceKalmanEstimate:
        shape = self._require_shape()
        covariance_shape = shape + (2, 2)
        return InterfaceKalmanEstimate(
            values=state.mean[:, 0].reshape(shape),
            rates=state.mean[:, 1].reshape(shape),
            covariances=state.covariance.reshape(covariance_shape),
            value_variances=state.covariance[:, 0, 0].reshape(shape),
            rate_variances=state.covariance[:, 1, 1].reshape(shape),
        )


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _covariance_spec(
    name: str,
    value: Any,
    *,
    positive: bool,
) -> CovarianceSpec:
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError(
                f"{name} xyz covariance must contain exactly three values"
            )
        converted: CovarianceSpec = tuple(
            _finite_real(f"{name}[{axis}]", component)
            for axis, component in enumerate(value)
        )
        components = converted
    else:
        converted = _finite_real(name, value)
        components = (converted,)
    if positive:
        if any(component <= 0.0 for component in components):
            raise ValueError(f"{name} must be positive")
    elif any(component < 0.0 for component in components):
        raise ValueError(f"{name} must be non-negative")
    return converted


def _covariance_values_for_shape(
    spec: CovarianceSpec,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray:
    if isinstance(spec, tuple):
        if not shape or shape[-1] != 3:
            raise ValueError(
                f"{name} xyz covariance requires a field with last dimension 3"
            )
        values = np.broadcast_to(np.asarray(spec, dtype=np.float64), shape)
        return np.array(values, dtype=np.float64, copy=True).reshape(-1)
    return np.full(int(np.prod(shape, dtype=np.int64)), spec, dtype=np.float64)


def _finite_array(
    values: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a regular real-valued array") from exc
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must not be a boolean array")
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must contain real values")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to float64") from exc
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if expected_shape is not None and tuple(array.shape) != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: {tuple(array.shape)} != {expected_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, dtype=np.float64, copy=True)


def _validated_layout_id(layout_id: str) -> str:
    if not isinstance(layout_id, str) or not layout_id.strip():
        raise ValueError("layout_id must be a non-empty string")
    return layout_id


def _symmetrized(covariance: np.ndarray) -> np.ndarray:
    return 0.5 * (covariance + np.swapaxes(covariance, -1, -2))


def _finite_normalized_innovation_squared(
    innovation: np.ndarray,
    innovation_variance: np.ndarray,
) -> np.ndarray:
    """Return finite per-DOF NIS, saturating only beyond float64 range."""

    maximum = np.finfo(np.float64).max
    maximum_root = math.sqrt(maximum)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        standardized = np.abs(innovation) / np.sqrt(innovation_variance)
        saturated = standardized >= maximum_root
        clipped = np.minimum(standardized, maximum_root)
        result = clipped * clipped
    return np.where(saturated, maximum, np.minimum(result, maximum))


def _read_only_copy(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "INTERFACE_KALMAN_SNAPSHOT_SCHEMA_VERSION",
    "InterfaceKalmanConfig",
    "InterfaceKalmanEstimate",
    "InterfaceKalmanPredictor",
    "InterfaceKalmanSnapshot",
    "InterfaceKalmanUpdate",
]
