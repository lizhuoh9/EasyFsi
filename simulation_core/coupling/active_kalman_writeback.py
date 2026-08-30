"""Host-only controller for explicitly modified-physics Kalman writeback.

The controller owns filter transaction semantics and NumPy diagnostics only.
The FSI runner remains responsible for Taichi field I/O and for restoring
physical invariants such as fixed particles, plane strain, and derived marker
state after a writeback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
import time
from typing import Any

import numpy as np

from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
    InterfaceKalmanPredictor,
    InterfaceKalmanSnapshot,
)


INTERFACE_MARKER_VELOCITY_OWNER = "interface_marker_velocity"
FLUID_FSI_PRESSURE_FEEDBACK_OWNER = "fluid_fsi_pressure_feedback"
SOLID_PARTICLE_VELOCITY_OWNER = "solid_particle_velocity"

ACTIVE_KALMAN_MODE_OWNERS = {
    "off": (),
    "interface": (INTERFACE_MARKER_VELOCITY_OWNER,),
    "fluid": (FLUID_FSI_PRESSURE_FEEDBACK_OWNER,),
    "solid": (SOLID_PARTICLE_VELOCITY_OWNER,),
    "global": (
        INTERFACE_MARKER_VELOCITY_OWNER,
        FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
        SOLID_PARTICLE_VELOCITY_OWNER,
    ),
}


ACTIVE_KALMAN_WRITEBACK_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActiveKalmanWritebackResult:
    """One owner observation and its optional posterior writeback."""

    owner: str
    writeback_enabled: bool
    writeback_values: np.ndarray | None
    predicted_values: np.ndarray
    posterior_values: np.ndarray
    prediction_rmse: float
    carry_forward_rmse: float
    prediction_bias: float
    posterior_delta_rmse: float
    nis_mean: float
    nis_max: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_values",
            _read_only_float64(self.predicted_values),
        )
        object.__setattr__(
            self,
            "posterior_values",
            _read_only_float64(self.posterior_values),
        )
        if self.writeback_values is not None:
            object.__setattr__(
                self,
                "writeback_values",
                _read_only_float64(self.writeback_values),
            )

    @property
    def report(self) -> dict[str, bool | float]:
        """Return the JSON-serializable diagnostics for this observation."""

        return {
            "writeback_enabled": self.writeback_enabled,
            "prediction_rmse": self.prediction_rmse,
            "carry_forward_rmse": self.carry_forward_rmse,
            "prediction_bias": self.prediction_bias,
            "posterior_delta_rmse": self.posterior_delta_rmse,
            "nis_mean": self.nis_mean,
            "nis_max": self.nis_max,
        }


@dataclass
class _OwnerMetrics:
    trial_count: int = 0
    accepted_update_count: int = 0
    commit_count: int = 0
    writeback_count: int = 0
    rollback_count: int = 0
    prediction_rmse_sum: float = 0.0
    carry_forward_rmse_sum: float = 0.0
    prediction_bias_sum: float = 0.0
    posterior_delta_rmse_sum: float = 0.0
    nis_mean_sum: float = 0.0
    nis_max: float = 0.0
    filter_wall_time_s: float = 0.0


@dataclass(frozen=True)
class ActiveKalmanOwnerMetricsSnapshot:
    trial_count: int
    accepted_update_count: int
    commit_count: int
    writeback_count: int
    rollback_count: int
    prediction_rmse_sum: float
    carry_forward_rmse_sum: float
    prediction_bias_sum: float
    posterior_delta_rmse_sum: float
    nis_mean_sum: float
    nis_max: float
    filter_wall_time_s: float

    def __post_init__(self) -> None:
        for name in (
            "trial_count",
            "accepted_update_count",
            "commit_count",
            "writeback_count",
            "rollback_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Integral
            ):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "prediction_rmse_sum",
            "carry_forward_rmse_sum",
            "prediction_bias_sum",
            "posterior_delta_rmse_sum",
            "nis_mean_sum",
            "nis_max",
            "filter_wall_time_s",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (
            self.writeback_count > self.commit_count
            or self.commit_count > self.accepted_update_count
            or self.accepted_update_count > self.trial_count
        ):
            raise ValueError("owner metric counts are inconsistent")


@dataclass(frozen=True)
class ActiveKalmanWritebackSnapshot:
    """Typed accepted-boundary state for every active Kalman owner."""

    schema_version: int
    mode: str
    enabled_owners: tuple[str, ...]
    configs: tuple[tuple[str, InterfaceKalmanConfig], ...]
    predictor_snapshots: tuple[tuple[str, InterfaceKalmanSnapshot], ...]
    owner_metrics: tuple[tuple[str, ActiveKalmanOwnerMetricsSnapshot], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, Integral)
        ):
            raise TypeError("snapshot schema_version must be an integer")
        if int(self.schema_version) != ACTIVE_KALMAN_WRITEBACK_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported active-Kalman snapshot schema_version")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if not isinstance(self.mode, str) or self.mode not in ACTIVE_KALMAN_MODE_OWNERS:
            raise ValueError("snapshot mode is invalid")
        expected = ACTIVE_KALMAN_MODE_OWNERS[self.mode]
        if self.enabled_owners != expected:
            raise ValueError("snapshot enabled owners do not match mode")
        for name, values, expected_type in (
            ("configs", self.configs, InterfaceKalmanConfig),
            ("predictor_snapshots", self.predictor_snapshots, InterfaceKalmanSnapshot),
            ("owner_metrics", self.owner_metrics, ActiveKalmanOwnerMetricsSnapshot),
        ):
            if tuple(owner for owner, _ in values) != expected:
                raise ValueError(f"snapshot {name} owner keys do not match mode")
            if not all(isinstance(value, expected_type) for _, value in values):
                raise TypeError(f"snapshot {name} have invalid values")
        for (owner, config), (_, predictor), (_, metrics) in zip(
            self.configs, self.predictor_snapshots, self.owner_metrics
        ):
            if predictor.config != config or predictor.layout_id != owner:
                raise ValueError("snapshot predictor identity is inconsistent")
            accepted_state_updates = predictor.accepted_state_count - 1
            if (
                metrics.accepted_update_count != accepted_state_updates
                or metrics.commit_count != accepted_state_updates
            ):
                raise ValueError(
                    "snapshot owner metrics do not match predictor accepted state"
                )
        if self.enabled_owners and len(
            {metrics.commit_count for _, metrics in self.owner_metrics}
        ) != 1:
            raise ValueError("snapshot enabled-owner commit counts are inconsistent")

    def validate(self) -> None:
        """Revalidate nested snapshots before restoration."""

        self.__post_init__()


class ActiveKalmanWritebackController:
    """Coordinate independent owner filters at accepted macro-step boundaries.

    ``off`` mode returns before inspecting ``configs`` or
    ``initial_observations`` and never constructs a predictor.  Active modes
    use the owner name as a stable layout identifier.  Shape changes therefore
    fail closed in the underlying predictor.
    """

    def __init__(
        self,
        mode: str,
        configs: Mapping[str, InterfaceKalmanConfig] | None = None,
        initial_observations: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(mode, str) or mode not in ACTIVE_KALMAN_MODE_OWNERS:
            raise ValueError(
                "mode must be one of off, interface, fluid, solid, global"
            )
        self.mode = mode
        self.enabled_owners = ACTIVE_KALMAN_MODE_OWNERS[mode]
        self._predictors: dict[str, InterfaceKalmanPredictor] = {}
        self._metrics: dict[str, _OwnerMetrics] = {}
        self._step_snapshots: dict[str, InterfaceKalmanSnapshot] | None = None
        self._step_predictions: dict[str, np.ndarray] = {}
        self._step_carry_forward: dict[str, np.ndarray] = {}
        self._step_writeback_ready: dict[str, bool] = {}
        self._step_results: dict[str, ActiveKalmanWritebackResult] = {}

        if not self.enabled_owners:
            return

        _require_exact_owner_keys("configs", configs, self.enabled_owners)
        _require_exact_owner_keys(
            "initial_observations",
            initial_observations,
            self.enabled_owners,
        )
        assert configs is not None
        assert initial_observations is not None

        predictors: dict[str, InterfaceKalmanPredictor] = {}
        metrics: dict[str, _OwnerMetrics] = {}
        for owner in self.enabled_owners:
            config = configs[owner]
            if not isinstance(config, InterfaceKalmanConfig):
                raise TypeError(
                    f"configs[{owner!r}] must be an InterfaceKalmanConfig"
                )
            predictor = InterfaceKalmanPredictor(config)
            started = time.perf_counter()
            predictor.initialize(initial_observations[owner], layout_id=owner)
            metrics[owner] = _OwnerMetrics(
                filter_wall_time_s=time.perf_counter() - started
            )
            predictors[owner] = predictor

        self._predictors = predictors
        self._metrics = metrics

    @property
    def initialized(self) -> bool:
        return not self.enabled_owners or all(
            predictor.initialized for predictor in self._predictors.values()
        )

    @property
    def has_active_step(self) -> bool:
        return self._step_snapshots is not None

    def enabled(self, owner: str) -> bool:
        return owner in self.enabled_owners

    def begin_step(self, dt_s: float) -> dict[str, np.ndarray]:
        """Open one filter trial and return each owner's prior prediction."""

        if not self.enabled_owners:
            return {}
        if self.has_active_step:
            raise RuntimeError("an active Kalman macro-step already exists")

        snapshots: dict[str, InterfaceKalmanSnapshot] = {}
        for owner in self.enabled_owners:
            started = time.perf_counter()
            snapshots[owner] = self._predictors[owner].snapshot()
            self._metrics[owner].filter_wall_time_s += (
                time.perf_counter() - started
            )
        self._step_snapshots = snapshots
        self._step_carry_forward = {
            owner: snapshot.values for owner, snapshot in snapshots.items()
        }
        self._step_writeback_ready = {
            owner: self._predictors[owner].ready for owner in self.enabled_owners
        }

        try:
            for owner in self.enabled_owners:
                started = time.perf_counter()
                prediction = self._predictors[owner].predict_trial(
                    dt=dt_s,
                    layout_id=owner,
                )
                self._metrics[owner].filter_wall_time_s += (
                    time.perf_counter() - started
                )
                self._metrics[owner].trial_count += 1
                self._step_predictions[owner] = prediction.values
        except Exception:
            self._restore_step(count_rollback=True)
            raise

        return dict(self._step_predictions)

    def observe(
        self,
        owner: str,
        raw_observation: Any,
    ) -> ActiveKalmanWritebackResult:
        """Assimilate one raw accepted observation without committing it."""

        if not self.enabled(owner):
            raise ValueError(f"Kalman owner {owner!r} is not enabled")
        if not self.has_active_step:
            raise RuntimeError("no active Kalman macro-step")
        if owner in self._step_results:
            self._restore_step(count_rollback=True)
            raise RuntimeError(f"Kalman owner {owner!r} was already observed")

        started = time.perf_counter()
        try:
            update = self._predictors[owner].update_trial(
                raw_observation,
                layout_id=owner,
            )
            raw = np.array(raw_observation, dtype=np.float64, copy=True)
            prediction = self._step_predictions[owner]
            carry_forward = self._step_carry_forward[owner]
            posterior = update.estimate.values
            writeback_enabled = self._step_writeback_ready[owner]
            result = ActiveKalmanWritebackResult(
                owner=owner,
                writeback_enabled=writeback_enabled,
                writeback_values=posterior if writeback_enabled else None,
                predicted_values=prediction,
                posterior_values=posterior,
                prediction_rmse=_root_mean_square(prediction - raw),
                carry_forward_rmse=_root_mean_square(carry_forward - raw),
                prediction_bias=_stable_mean(prediction - raw),
                posterior_delta_rmse=_root_mean_square(posterior - raw),
                nis_mean=_stable_mean(
                    update.normalized_innovation_squared
                ),
                nis_max=float(
                    np.max(update.normalized_innovation_squared)
                ),
            )
        except Exception:
            self._metrics[owner].filter_wall_time_s += (
                time.perf_counter() - started
            )
            self._restore_step(count_rollback=True)
            raise

        self._metrics[owner].filter_wall_time_s += time.perf_counter() - started
        self._step_results[owner] = result
        return result

    def commit_step(self) -> dict[str, Any]:
        """Atomically commit all owner trials and return the step report."""

        if not self.enabled_owners:
            return self.summary()
        if not self.has_active_step:
            raise RuntimeError("no active Kalman macro-step")
        missing = [
            owner
            for owner in self.enabled_owners
            if owner not in self._step_results
        ]
        if missing:
            self._restore_step(count_rollback=True)
            raise RuntimeError(
                "missing accepted observations for Kalman owners: "
                + ", ".join(missing)
            )

        try:
            for owner in self.enabled_owners:
                started = time.perf_counter()
                self._predictors[owner].commit_trial()
                self._metrics[owner].filter_wall_time_s += (
                    time.perf_counter() - started
                )
        except Exception:
            self._restore_step(count_rollback=True)
            raise

        owner_reports: dict[str, dict[str, Any]] = {}
        for owner in self.enabled_owners:
            result = self._step_results[owner]
            metrics = self._metrics[owner]
            metrics.accepted_update_count += 1
            metrics.commit_count += 1
            metrics.writeback_count += int(result.writeback_enabled)
            metrics.prediction_rmse_sum += result.prediction_rmse
            metrics.carry_forward_rmse_sum += result.carry_forward_rmse
            metrics.prediction_bias_sum += result.prediction_bias
            metrics.posterior_delta_rmse_sum += result.posterior_delta_rmse
            metrics.nis_mean_sum += result.nis_mean
            metrics.nis_max = max(metrics.nis_max, result.nis_max)
            owner_reports[owner] = {
                **result.report,
                "accepted_state_count": self._predictors[
                    owner
                ].accepted_state_count,
            }

        self._clear_step()
        return {
            "mode": self.mode,
            "modified_physics": True,
            "owners": owner_reports,
        }

    def discard_step(self) -> None:
        """Discard every owner trial and restore the begin-step snapshots."""

        if not self.enabled_owners:
            return
        if not self.has_active_step:
            raise RuntimeError("no active Kalman macro-step")
        self._restore_step(count_rollback=True)

    def summary(self) -> dict[str, Any]:
        """Return cumulative JSON-serializable diagnostics."""

        if not self.enabled_owners:
            return {
                "mode": "off",
                "modified_physics": False,
                "owners": {},
            }

        owner_reports: dict[str, dict[str, int | float]] = {}
        for owner in self.enabled_owners:
            metrics = self._metrics[owner]
            accepted = metrics.accepted_update_count
            denominator = float(accepted) if accepted else 1.0
            owner_reports[owner] = {
                "accepted_state_count": self._predictors[
                    owner
                ].accepted_state_count,
                "trial_count": metrics.trial_count,
                "accepted_update_count": accepted,
                "commit_count": metrics.commit_count,
                "writeback_count": metrics.writeback_count,
                "rollback_count": metrics.rollback_count,
                "prediction_rmse_mean": (
                    metrics.prediction_rmse_sum / denominator
                ),
                "carry_forward_rmse_mean": (
                    metrics.carry_forward_rmse_sum / denominator
                ),
                "prediction_bias_mean": (
                    metrics.prediction_bias_sum / denominator
                ),
                "posterior_delta_rmse_mean": (
                    metrics.posterior_delta_rmse_sum / denominator
                ),
                "nis_mean": metrics.nis_mean_sum / denominator,
                "nis_max": metrics.nis_max,
                "filter_wall_time_s": metrics.filter_wall_time_s,
            }
        return {
            "mode": self.mode,
            "modified_physics": True,
            "owners": owner_reports,
        }


    def snapshot(self) -> ActiveKalmanWritebackSnapshot:
        """Capture predictor and metric state at an accepted boundary."""

        if self.has_active_step:
            raise RuntimeError(
                "cannot snapshot while a Kalman macro-step is active"
            )
        return ActiveKalmanWritebackSnapshot(
            schema_version=ACTIVE_KALMAN_WRITEBACK_SNAPSHOT_SCHEMA_VERSION,
            mode=self.mode,
            enabled_owners=self.enabled_owners,
            configs=tuple(
                (owner, self._predictors[owner].config)
                for owner in self.enabled_owners
            ),
            predictor_snapshots=tuple(
                (owner, self._predictors[owner].snapshot())
                for owner in self.enabled_owners
            ),
            owner_metrics=tuple(
                (owner, _metrics_snapshot(self._metrics[owner]))
                for owner in self.enabled_owners
            ),
        )

    def restore(self, snapshot: ActiveKalmanWritebackSnapshot) -> None:
        """Atomically restore predictors and all cumulative metrics."""

        if self.has_active_step:
            raise RuntimeError(
                "cannot restore while a Kalman macro-step is active"
            )
        if not isinstance(snapshot, ActiveKalmanWritebackSnapshot):
            raise TypeError("snapshot must be an ActiveKalmanWritebackSnapshot")
        snapshot.validate()
        if (
            snapshot.mode != self.mode
            or snapshot.enabled_owners != self.enabled_owners
        ):
            raise ValueError(
                "snapshot mode or enabled owners do not match controller"
            )

        restored_predictors: dict[str, InterfaceKalmanPredictor] = {}
        restored_metrics: dict[str, _OwnerMetrics] = {}
        for (owner, config), (_, predictor_snapshot), (
            _,
            metric_snapshot,
        ) in zip(
            snapshot.configs,
            snapshot.predictor_snapshots,
            snapshot.owner_metrics,
        ):
            current = self._predictors[owner]
            if config != current.config:
                raise ValueError(
                    "snapshot config does not match controller config"
                )
            current_snapshot = current.snapshot()
            if tuple(predictor_snapshot.values.shape) != tuple(
                current_snapshot.values.shape
            ):
                raise ValueError(
                    "snapshot predictor shape does not match controller shape"
                )
            predictor = InterfaceKalmanPredictor(config)
            predictor.restore(predictor_snapshot)
            restored_predictors[owner] = predictor
            restored_metrics[owner] = _metrics_from_snapshot(metric_snapshot)

        self._predictors = restored_predictors
        self._metrics = restored_metrics

    def _restore_step(self, *, count_rollback: bool) -> None:
        snapshots = self._step_snapshots
        if snapshots is None:
            return
        for owner in self.enabled_owners:
            predictor = self._predictors[owner]
            started = time.perf_counter()
            if predictor.has_active_trial:
                predictor.discard_trial()
            predictor.restore(snapshots[owner])
            self._metrics[owner].filter_wall_time_s += (
                time.perf_counter() - started
            )
            if count_rollback:
                self._metrics[owner].rollback_count += 1
        self._clear_step()

    def _clear_step(self) -> None:
        self._step_snapshots = None
        self._step_predictions = {}
        self._step_carry_forward = {}
        self._step_writeback_ready = {}
        self._step_results = {}


def _require_exact_owner_keys(
    name: str,
    values: Mapping[str, Any] | None,
    expected: tuple[str, ...],
) -> None:
    if values is None:
        raise ValueError(f"{name} are required for active Kalman mode")
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(values.keys()) != set(expected):
        raise ValueError(
            f"{name} owner keys must exactly match {expected!r}"
        )


def _root_mean_square(values: np.ndarray) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    scale = float(np.max(absolute))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean((absolute / scale) ** 2)))


def _stable_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.mean(array / scale))


def _read_only_float64(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _metrics_snapshot(metrics: _OwnerMetrics) -> ActiveKalmanOwnerMetricsSnapshot:
    return ActiveKalmanOwnerMetricsSnapshot(
        trial_count=metrics.trial_count,
        accepted_update_count=metrics.accepted_update_count,
        commit_count=metrics.commit_count,
        writeback_count=metrics.writeback_count,
        rollback_count=metrics.rollback_count,
        prediction_rmse_sum=metrics.prediction_rmse_sum,
        carry_forward_rmse_sum=metrics.carry_forward_rmse_sum,
        prediction_bias_sum=metrics.prediction_bias_sum,
        posterior_delta_rmse_sum=metrics.posterior_delta_rmse_sum,
        nis_mean_sum=metrics.nis_mean_sum,
        nis_max=metrics.nis_max,
        filter_wall_time_s=metrics.filter_wall_time_s,
    )


def _metrics_from_snapshot(snapshot: ActiveKalmanOwnerMetricsSnapshot) -> _OwnerMetrics:
    return _OwnerMetrics(
        trial_count=snapshot.trial_count,
        accepted_update_count=snapshot.accepted_update_count,
        commit_count=snapshot.commit_count,
        writeback_count=snapshot.writeback_count,
        rollback_count=snapshot.rollback_count,
        prediction_rmse_sum=snapshot.prediction_rmse_sum,
        carry_forward_rmse_sum=snapshot.carry_forward_rmse_sum,
        prediction_bias_sum=snapshot.prediction_bias_sum,
        posterior_delta_rmse_sum=snapshot.posterior_delta_rmse_sum,
        nis_mean_sum=snapshot.nis_mean_sum,
        nis_max=snapshot.nis_max,
        filter_wall_time_s=snapshot.filter_wall_time_s,
    )


__all__ = [
    "ACTIVE_KALMAN_MODE_OWNERS",
    "ACTIVE_KALMAN_WRITEBACK_SNAPSHOT_SCHEMA_VERSION",
    "FLUID_FSI_PRESSURE_FEEDBACK_OWNER",
    "INTERFACE_MARKER_VELOCITY_OWNER",
    "SOLID_PARTICLE_VELOCITY_OWNER",
    "ActiveKalmanOwnerMetricsSnapshot",
    "ActiveKalmanWritebackController",
    "ActiveKalmanWritebackResult",
    "ActiveKalmanWritebackSnapshot",
]
