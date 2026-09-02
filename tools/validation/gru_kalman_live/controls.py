"""Fixed post-hoc G0-M and GDelta-M controls for R25B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from tools.validation.gru_kalman.baselines import BaselineResult, evaluate_baseline
from tools.validation.gru_kalman.dataset import (
    DatasetContractError,
    build_temporal_samples,
    validate_trace,
)
from tools.validation.gru_kalman.models import (
    GRUArchitecture,
    ResidualGRU,
    to_torch,
)
from tools.validation.gru_kalman.pod import (
    ModalNormalization,
    PODBasis,
    fit_normalization,
    fit_pod,
)
from tools.validation.gru_kalman.training import (
    PreparedGRUData,
    SEEDS,
    TrainedGRU,
    fit_gru,
    prepare_gru_data,
)

MATCHED_ARCHITECTURE = GRUArchitecture(8, 4, 16)
MATCHED_SEEDS = SEEDS
MATCHED_CONTROL_IDS = ("g0_matched", "gdelta_matched")
_NETWORK_FAMILY = {
    "g0_matched": "gru",
    "gdelta_matched": "kalman1_gru",
}


class MatchedControlError(DatasetContractError):
    """A matched-control input or frozen training contract is invalid."""


@dataclass(frozen=True)
class MatchedControlModel:
    control_id: str
    seed: int
    trained: TrainedGRU

    def __post_init__(self) -> None:
        if self.control_id not in MATCHED_CONTROL_IDS:
            raise MatchedControlError(f"unsupported matched control {self.control_id!r}")
        if int(self.seed) not in MATCHED_SEEDS:
            raise MatchedControlError("matched-control seed must be 0, 1, or 2")
        if self.trained.architecture != MATCHED_ARCHITECTURE:
            raise MatchedControlError("matched-control architecture must be 8:4:16")
        if self.trained.family != _NETWORK_FAMILY[self.control_id]:
            raise MatchedControlError("matched-control network family mismatch")
        if self.trained.seed != int(self.seed):
            raise MatchedControlError("matched-control seed identity mismatch")


@dataclass(frozen=True)
class MatchedControlTraining:
    pod: PODBasis
    normalization: ModalNormalization
    models: tuple[MatchedControlModel, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (control_id, seed)
            for control_id in MATCHED_CONTROL_IDS
            for seed in MATCHED_SEEDS
        )
        observed = tuple((model.control_id, model.seed) for model in self.models)
        if observed != expected:
            raise MatchedControlError(
                "matched-control training must retain both controls and all three seeds"
            )
        if self.pod.rank != 8 or self.normalization.rank != 8:
            raise MatchedControlError("matched-control POD and normalization rank must be 8")

    def model_for(self, control_id: str, seed: int) -> MatchedControlModel:
        for model in self.models:
            if model.control_id == control_id and model.seed == int(seed):
                return model
        raise MatchedControlError(
            f"matched-control model is missing: {control_id} seed {seed}"
        )


def _validated_control_id(value: object) -> str:
    control_id = str(value)
    if control_id not in MATCHED_CONTROL_IDS:
        raise MatchedControlError(f"unsupported matched control {control_id!r}")
    return control_id


def _target_steps(values: Sequence[int], *, trace_steps: int) -> tuple[int, ...]:
    raw = tuple(values)
    if not raw or any(
        isinstance(step, (bool, np.bool_))
        or not isinstance(step, (int, np.integer))
        for step in raw
    ):
        raise MatchedControlError("target_steps must be non-empty integers")
    result = tuple(int(step) for step in raw)
    if tuple(sorted(set(result))) != result:
        raise MatchedControlError("target_steps must be unique and increasing")
    if result[0] <= MATCHED_ARCHITECTURE.window or result[-1] > trace_steps:
        raise MatchedControlError("target_steps exceed matched-control history bounds")
    return result


def _trace_step_index(trace: Any) -> dict[int, int]:
    return {int(step): index for index, step in enumerate(trace.source_steps)}


def _previous_history_values(
    trace: Any,
    source_steps: tuple[int, ...],
) -> np.ndarray:
    index_by_step = _trace_step_index(trace)
    rows = []
    for source_step in source_steps:
        previous_step = int(source_step) - 1
        if previous_step == 0:
            rows.append(np.zeros_like(trace.values[0], dtype=np.float64))
        elif previous_step in index_by_step:
            rows.append(
                np.asarray(trace.values[index_by_step[previous_step]], dtype=np.float64)
            )
        else:
            raise MatchedControlError(
                "matched-control increment history is not contiguous"
            )
    return np.stack(rows)


def prepare_matched_control_data(
    trace: Any,
    *,
    pod: PODBasis,
    normalization: ModalNormalization,
    carry_baseline: BaselineResult,
    control_id: str,
    target_steps: Sequence[int],
    allowed_history_max_step: int | None = None,
) -> PreparedGRUData:
    """Build fixed state-only or state/increment/carry training features."""

    validate_trace(trace, expected_steps=len(trace.values))
    control = _validated_control_id(control_id)
    if pod.rank != 8 or normalization.rank != 8:
        raise MatchedControlError("matched-control POD and normalization rank must be 8")
    if carry_baseline.model != "carry" or len(carry_baseline.rows) != len(trace.values):
        raise MatchedControlError("matched controls require the complete carry baseline")
    targets = _target_steps(target_steps, trace_steps=len(trace.values))
    if control == "g0_matched":
        return prepare_gru_data(
            trace,
            pod=pod,
            normalization=normalization,
            baseline=carry_baseline,
            family="gru",
            window=MATCHED_ARCHITECTURE.window,
            target_steps=targets,
            allowed_history_max_step=allowed_history_max_step,
        )

    samples = build_temporal_samples(
        trace,
        window=MATCHED_ARCHITECTURE.window,
        start_step=targets[0],
        end_step=targets[-1],
        baseline_predictions=carry_baseline.effective_predictions,
        allowed_history_max_step=allowed_history_max_step,
    )
    selected = {sample.target_step: sample for sample in samples}
    if any(target not in selected for target in targets):
        raise MatchedControlError("requested matched-control samples are missing")
    features = []
    carries = []
    encoded_targets = []
    provenances = []
    for target in targets:
        sample = selected[target]
        if sample.current_baseline is None:
            raise MatchedControlError("matched control requires a causal carry baseline")
        if allowed_history_max_step is not None and max(sample.source_steps) > int(
            allowed_history_max_step
        ):
            raise MatchedControlError("matched-control history exceeds allowed bound")
        state_coefficients = normalization.normalize(pod.encode(sample.history))
        previous_values = _previous_history_values(trace, sample.source_steps)
        previous_coefficients = normalization.normalize(pod.encode(previous_values))
        increments = state_coefficients - previous_coefficients
        carry_coefficient = normalization.normalize(
            pod.encode(sample.current_baseline)[None, ...]
        )[0]
        repeated_carry = np.repeat(
            carry_coefficient[None, :],
            MATCHED_ARCHITECTURE.window,
            axis=0,
        )
        features.append(
            np.concatenate(
                (state_coefficients, increments, repeated_carry),
                axis=-1,
            )
        )
        carries.append(carry_coefficient)
        encoded_targets.append(
            normalization.normalize(pod.encode(sample.target)[None, ...])[0]
        )
        provenances.append(sample.source_steps)
    return PreparedGRUData(
        features=np.stack(features),
        carry=np.stack(carries),
        target=np.stack(encoded_targets),
        target_steps=targets,
        source_steps=tuple(provenances),
    )


def train_matched_controls(d0_trace: Any) -> MatchedControlTraining:
    """Fit only the frozen rank-8/window-4/hidden-16 six-model control set."""

    validate_trace(d0_trace, expected_steps=200)
    fit_steps = tuple(range(1, 101))
    pod = fit_pod(d0_trace.values[:100], rank=8, fit_steps=fit_steps)
    normalization = fit_normalization(
        pod.encode(d0_trace.values[:100]),
        fit_steps=fit_steps,
    )
    carry = evaluate_baseline(d0_trace, model="carry")
    retained = []
    for control_id in MATCHED_CONTROL_IDS:
        train = prepare_matched_control_data(
            d0_trace,
            pod=pod,
            normalization=normalization,
            carry_baseline=carry,
            control_id=control_id,
            target_steps=tuple(range(5, 101)),
            allowed_history_max_step=99,
        )
        selection = prepare_matched_control_data(
            d0_trace,
            pod=pod,
            normalization=normalization,
            carry_baseline=carry,
            control_id=control_id,
            target_steps=tuple(range(101, 201)),
            allowed_history_max_step=199,
        )
        for seed in MATCHED_SEEDS:
            trained = fit_gru(
                _NETWORK_FAMILY[control_id],
                MATCHED_ARCHITECTURE,
                seed=seed,
                train=train,
                selection=selection,
            )
            retained.append(
                MatchedControlModel(
                    control_id=control_id,
                    seed=seed,
                    trained=trained,
                )
            )
    return MatchedControlTraining(
        pod=pod,
        normalization=normalization,
        models=tuple(retained),
    )


def _network(model: MatchedControlModel | TrainedGRU | ResidualGRU) -> ResidualGRU:
    if isinstance(model, MatchedControlModel):
        return model.trained.model
    if isinstance(model, TrainedGRU):
        return model.model
    if isinstance(model, ResidualGRU):
        return model
    raise MatchedControlError("matched-control model type is unsupported")


def predict_matched_control(
    accepted_prefix: Any,
    *,
    pod: PODBasis,
    normalization: ModalNormalization,
    model: MatchedControlModel | TrainedGRU | ResidualGRU,
    control_id: str,
) -> np.ndarray:
    """Predict one next step from accepted prefix values only."""

    control = _validated_control_id(control_id)
    prefix = np.asarray(accepted_prefix)
    if prefix.dtype != np.float64 or prefix.ndim != 3 or prefix.shape[1:] != pod.mean.shape:
        raise MatchedControlError(
            "accepted prefix must be float64 (steps, markers, 3)"
        )
    if prefix.shape[0] < MATCHED_ARCHITECTURE.window:
        raise MatchedControlError("accepted prefix is shorter than the four-step window")
    if not np.all(np.isfinite(prefix)) or not np.all(prefix[..., 0] == 0.0):
        raise MatchedControlError("accepted prefix must be finite with exact-zero x")
    network = _network(model)
    if (
        network.architecture != MATCHED_ARCHITECTURE
        or network.family != _NETWORK_FAMILY[control]
    ):
        raise MatchedControlError("matched-control network identity mismatch")
    history = np.ascontiguousarray(prefix[-MATCHED_ARCHITECTURE.window :])
    state_coefficients = normalization.normalize(pod.encode(history))
    carry = normalization.normalize(pod.encode(prefix[-1][None, ...]))[0]
    if control == "g0_matched":
        features = state_coefficients[None, ...]
    else:
        if prefix.shape[0] == MATCHED_ARCHITECTURE.window:
            previous = np.concatenate(
                (np.zeros_like(prefix[:1]), prefix[:-1]),
                axis=0,
            )
        else:
            previous = prefix[-MATCHED_ARCHITECTURE.window - 1 : -1]
        previous_coefficients = normalization.normalize(pod.encode(previous))
        increments = state_coefficients - previous_coefficients
        repeated = np.repeat(
            carry[None, :], MATCHED_ARCHITECTURE.window, axis=0
        )
        features = np.concatenate(
            (state_coefficients, increments, repeated), axis=-1
        )[None, ...]
    network.eval()
    with torch.no_grad():
        prediction_coefficient = network(
            to_torch(features),
            to_torch(carry[None, ...]),
        )[0].detach().cpu().numpy()
    prediction = pod.decode(normalization.denormalize(prediction_coefficient))
    result = np.ascontiguousarray(prediction, dtype=np.float64)
    result[..., 0] = 0.0
    if not np.all(np.isfinite(result)):
        raise MatchedControlError("matched-control prediction is non-finite")
    return result


def matched_control_training_payload(
    training: MatchedControlTraining,
) -> dict[str, Any]:
    """Return JSON-safe deterministic training evidence without tensor payloads."""

    rows = []
    for model in training.models:
        trained = model.trained
        rows.append(
            {
                "control_id": model.control_id,
                "network_family": trained.family,
                "seed": model.seed,
                "architecture": trained.architecture.to_payload(),
                "best_epoch": trained.best_epoch,
                "best_selection_loss": trained.best_selection_loss,
                "epochs_executed": len(trained.history),
            }
        )
    return {
        "classification": "post_hoc_mechanism_controls",
        "fit_steps": [1, 100],
        "selection_steps": [101, 200],
        "pod_fingerprint": training.pod.fingerprint,
        "normalization_fingerprint": training.normalization.fingerprint,
        "models": rows,
    }


__all__ = [
    "MATCHED_ARCHITECTURE",
    "MATCHED_CONTROL_IDS",
    "MATCHED_SEEDS",
    "MatchedControlError",
    "MatchedControlModel",
    "MatchedControlTraining",
    "matched_control_training_payload",
    "predict_matched_control",
    "prepare_matched_control_data",
    "train_matched_controls",
]
