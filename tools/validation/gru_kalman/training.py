"""Deterministic full-batch CPU training and D0 architecture selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .baselines import BaselineResult
from .dataset import DatasetContractError, build_temporal_samples
from .models import (
    FIXED_ARCHITECTURES,
    MODEL_FAMILIES,
    GRUArchitecture,
    ResidualGRU,
    build_gru,
    make_gru_features,
    to_torch,
)
from .pod import ModalNormalization, PODBasis

LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 500
PATIENCE = 50
MIN_DELTA = 1.0e-8
RIDGE = 1.0e-6
SEEDS = (0, 1, 2)


class TrainingContractError(DatasetContractError):
    """Training input or frozen selection contract failed."""


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise TrainingContractError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise TrainingContractError("weight_decay must be finite and non-negative")
        if not np.isfinite(self.grad_clip) or self.grad_clip <= 0.0:
            raise TrainingContractError("grad_clip must be positive and finite")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (self.max_epochs, self.patience)):
            raise TrainingContractError("epoch and patience values must be integers")
        if self.max_epochs < 1 or self.patience < 1:
            raise TrainingContractError("epoch and patience values must be positive")
        if not np.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise TrainingContractError("min_delta must be finite and non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "min_delta": self.min_delta,
            "optimizer": "AdamW",
            "loss": "MSE",
            "batching": "full_batch",
        }


@dataclass(frozen=True)
class TrainingHistoryRow:
    epoch: int
    train_loss: float
    selection_loss: float
    improved: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "selection_loss": self.selection_loss,
            "improved": self.improved,
        }


@dataclass(frozen=True)
class TrainedGRU:
    family: str
    architecture: GRUArchitecture
    seed: int
    model: ResidualGRU
    state_dict: dict[str, torch.Tensor]
    history: tuple[TrainingHistoryRow, ...]
    best_epoch: int
    best_selection_loss: float

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise TrainingContractError(f"unsupported family {self.family!r}")
        if self.seed not in SEEDS:
            raise TrainingContractError("R25A seeds are exactly 0, 1, and 2")
        if self.model.family != self.family or self.model.architecture != self.architecture:
            raise TrainingContractError("trained model identity mismatch")
        if not self.state_dict:
            raise TrainingContractError("trained state_dict cannot be empty")


@dataclass(frozen=True)
class PreparedGRUData:
    features: np.ndarray
    carry: np.ndarray
    target: np.ndarray
    target_steps: tuple[int, ...]
    source_steps: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        carry = np.asarray(self.carry, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        if features.ndim != 3 or carry.ndim != 2 or target.ndim != 2:
            raise TrainingContractError("prepared data ranks must be 3, 2, and 2")
        if features.shape[0] != carry.shape[0] or target.shape != carry.shape:
            raise TrainingContractError("prepared data batch shapes mismatch")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(carry)) or not np.all(np.isfinite(target)):
            raise TrainingContractError("prepared data must be finite")
        if len(self.target_steps) != features.shape[0] or len(self.source_steps) != features.shape[0]:
            raise TrainingContractError("prepared provenance must align to samples")
        for step, sources in zip(self.target_steps, self.source_steps, strict=True):
            if not sources or max(sources) >= step:
                raise TrainingContractError("prepared feature provenance is noncausal")
        object.__setattr__(self, "features", np.array(features, copy=True))
        object.__setattr__(self, "carry", np.array(carry, copy=True))
        object.__setattr__(self, "target", np.array(target, copy=True))


def prepare_gru_data(
    trace: Any,
    *,
    pod: PODBasis,
    normalization: ModalNormalization,
    baseline: BaselineResult,
    family: str,
    window: int,
    target_steps: Sequence[int],
    allowed_history_max_step: int | None = None,
) -> PreparedGRUData:
    """Encode teacher histories and causal baselines without current leakage."""

    if family not in MODEL_FAMILIES:
        raise TrainingContractError(f"unsupported family {family!r}")
    # The current input is the effective causal baseline (carry during
    # warm-up); innovations remain measurement minus the raw prediction.
    predictions = baseline.effective_predictions
    innovations = baseline.innovations
    samples = build_temporal_samples(
        trace,
        window=window,
        start_step=min(target_steps),
        end_step=max(target_steps),
        baseline_predictions=predictions,
        innovations=innovations,
        allowed_history_max_step=allowed_history_max_step,
    )
    selected = {sample.target_step: sample for sample in samples if sample.target_step in set(target_steps)}
    if len(selected) != len(tuple(target_steps)):
        raise TrainingContractError("requested GRU target steps were not all prepared")
    rows: list[np.ndarray] = []
    carries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    steps: list[int] = []
    source_steps: list[tuple[int, ...]] = []
    for target_step in target_steps:
        sample = selected[int(target_step)]
        history_coeff = normalization.normalize(pod.encode(sample.history))
        target_coeff = normalization.normalize(pod.encode(sample.target))
        if sample.current_baseline is None:
            raise TrainingContractError("baseline prediction is required for GRU data")
        carry_coeff = normalization.normalize(pod.encode(sample.current_baseline)[None, ...])[0]
        if family == "gru":
            feature = make_gru_features(family, history_coeff[None, ...])[0]
        else:
            if sample.innovation_history is None:
                raise TrainingContractError("Kalman-residual GRU requires innovation history")
            innovation_coeff = pod.encode_residual(sample.innovation_history) / normalization.scale
            feature = make_gru_features(
                family,
                history_coeff[None, ...],
                innovations=innovation_coeff[None, ...],
                current_baseline=carry_coeff[None, ...],
            )[0]
        rows.append(feature)
        carries.append(carry_coeff)
        targets.append(target_coeff)
        steps.append(sample.target_step)
        source_steps.append(sample.source_steps)
    return PreparedGRUData(
        features=np.stack(rows),
        carry=np.stack(carries),
        target=np.stack(targets),
        target_steps=tuple(steps),
        source_steps=tuple(source_steps),
    )


def _loss(model: ResidualGRU, data: PreparedGRUData) -> torch.Tensor:
    features = to_torch(data.features)
    carry = to_torch(data.carry)
    target = to_torch(data.target)
    return torch.mean(torch.square(model(features, carry) - target))


def fit_gru(
    family: str,
    architecture: GRUArchitecture,
    *,
    seed: int,
    train: PreparedGRUData,
    selection: PreparedGRUData | None,
    config: TrainingConfig = TrainingConfig(),
) -> TrainedGRU:
    """Fit one architecture/seed and retain its D0-selection best state."""

    if selection is None:
        raise TrainingContractError(
            "D0-selection data is required; training data cannot be used as fallback"
        )
    model = build_gru(family, architecture, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[TrainingHistoryRow] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = _loss(model, train)
        if not torch.isfinite(train_loss):
            raise TrainingContractError("GRU training loss became nonfinite")
        train_loss.backward()
        clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_loss_tensor = _loss(model, selection)
        selection_loss = float(selection_loss_tensor.detach().cpu().item())
        if not np.isfinite(selection_loss):
            raise TrainingContractError("GRU selection loss became nonfinite")
        improved = selection_loss < best_loss - config.min_delta
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append(
            TrainingHistoryRow(
                epoch=epoch,
                train_loss=float(train_loss.detach().cpu().item()),
                selection_loss=selection_loss,
                improved=improved,
            )
        )
        if stale >= config.patience:
            break
    if best_state is None:
        raise TrainingContractError("no finite improved GRU state was retained")
    model.load_state_dict(best_state)
    return TrainedGRU(
        family=family,
        architecture=architecture,
        seed=int(seed),
        model=model,
        state_dict=best_state,
        history=tuple(history),
        best_epoch=best_epoch,
        best_selection_loss=best_loss,
    )


def select_architecture(results: Mapping[str, Mapping[int, float]]) -> str:
    """Select by median active-axis D0-selection score across all three seeds."""

    if set(results) != {f"{r}:{w}:{h}" for r, w, h in FIXED_ARCHITECTURES}:
        raise TrainingContractError("architecture selection must cover exactly the four fixed configs")
    scored: list[tuple[float, str]] = []
    for architecture_id, seed_scores in results.items():
        if set(seed_scores) != set(SEEDS):
            raise TrainingContractError("every architecture must retain all three seed scores")
        values = np.asarray([seed_scores[seed] for seed in SEEDS], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise TrainingContractError("selection scores must be finite")
        scored.append((float(np.median(values)), architecture_id))
    return min(scored, key=lambda item: (item[0], item[1]))[1]


__all__ = [
    "FIXED_ARCHITECTURES",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "MIN_DELTA",
    "MODEL_FAMILIES",
    "PATIENCE",
    "PreparedGRUData",
    "RIDGE",
    "SEEDS",
    "TrainingConfig",
    "TrainingContractError",
    "TrainingHistoryRow",
    "TrainedGRU",
    "WEIGHT_DECAY",
    "fit_gru",
    "prepare_gru_data",
    "select_architecture",
]
