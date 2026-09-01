"""CPU-only one-layer residual GRU models used by the R25A harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import DatasetContractError

FIXED_ARCHITECTURES = (
    (4, 4, 8),
    (8, 4, 16),
    (8, 8, 16),
    (16, 8, 16),
)
MODEL_FAMILIES = ("gru", "kalman0_gru", "kalman1_gru")


class ModelContractError(DatasetContractError):
    """A model violates the frozen R25A CPU architecture contract."""


def configure_deterministic_cpu() -> None:
    """Set the deterministic, single-threaded CPU runtime used by R25A."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch disallows changing inter-op threads after work has started;
        # the already-running process remains single-threaded for intra-op work.
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ModelContractError("seed must be a non-negative integer")
    configure_deterministic_cpu()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


@dataclass(frozen=True)
class GRUArchitecture:
    rank: int
    window: int
    hidden: int

    def __post_init__(self) -> None:
        fields = (self.rank, self.window, self.hidden)
        if any(isinstance(value, bool) or int(value) != value for value in fields):
            raise ModelContractError("GRU architecture values must be integers")
        values = tuple(int(value) for value in fields)
        if values not in FIXED_ARCHITECTURES:
            raise ModelContractError(
                f"architecture {values} is outside the fixed R25A matrix"
            )
        if min(values) < 1:
            raise ModelContractError("GRU architecture values must be positive")
        object.__setattr__(self, "rank", values[0])
        object.__setattr__(self, "window", values[1])
        object.__setattr__(self, "hidden", values[2])

    @property
    def id(self) -> str:
        return f"{self.rank}:{self.window}:{self.hidden}"

    def to_payload(self) -> dict[str, int | str]:
        return {"rank": self.rank, "window": self.window, "hidden": self.hidden, "id": self.id}


def parse_architectures(value: str) -> tuple[GRUArchitecture, ...]:
    """Parse and require the exact frozen four-architecture matrix."""

    if not isinstance(value, str):
        raise ModelContractError("pod-configs must be a comma-separated string")
    tokens = tuple(token.strip() for token in value.split(",") if token.strip())
    parsed: list[GRUArchitecture] = []
    for token in tokens:
        fields = token.split(":")
        if len(fields) != 3:
            raise ModelContractError(f"invalid architecture token {token!r}")
        try:
            parsed.append(GRUArchitecture(*(int(field) for field in fields)))
        except ValueError as exc:
            raise ModelContractError(f"invalid architecture token {token!r}") from exc
    if tuple((item.rank, item.window, item.hidden) for item in parsed) != FIXED_ARCHITECTURES:
        raise ModelContractError("R25A requires the exact four fixed architectures in order")
    return tuple(parsed)


class ResidualGRU(nn.Module):
    """One-layer float64 GRU whose zero head starts at the causal baseline."""

    def __init__(self, family: str, architecture: GRUArchitecture, seed: int) -> None:
        super().__init__()
        if family not in MODEL_FAMILIES:
            raise ModelContractError(f"unsupported neural family {family!r}")
        set_seed(seed)
        self.family = family
        self.architecture = architecture
        self.seed = int(seed)
        self.input_size = architecture.rank if family == "gru" else 3 * architecture.rank
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=architecture.hidden,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float64,
        )
        self.head = nn.Linear(architecture.hidden, architecture.rank, dtype=torch.float64)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or not isinstance(carry, torch.Tensor):
            raise ModelContractError("GRU inputs must be torch tensors")
        if features.ndim != 3 or carry.ndim != 2:
            raise ModelContractError("GRU features/carry must have ranks 3 and 2")
        expected = (features.shape[0], self.architecture.window, self.input_size)
        if tuple(features.shape) != expected:
            raise ModelContractError(f"GRU features shape {tuple(features.shape)} != {expected}")
        if tuple(carry.shape) != (features.shape[0], self.architecture.rank):
            raise ModelContractError("GRU carry shape does not match architecture rank")
        if features.dtype != torch.float64 or carry.dtype != torch.float64:
            raise ModelContractError("R25A GRU tensors must be float64")
        _, hidden = self.gru(features)
        correction = self.head(hidden[-1])
        result = carry + correction
        if not torch.isfinite(result).all():
            raise ModelContractError("GRU produced nonfinite output")
        return result

    @property
    def output_head_zero(self) -> bool:
        return bool(torch.count_nonzero(self.head.weight).item() == 0 and torch.count_nonzero(self.head.bias).item() == 0)


def build_gru(family: str, architecture: GRUArchitecture, *, seed: int) -> ResidualGRU:
    return ResidualGRU(family, architecture, seed)


def make_gru_features(
    family: str,
    states: Any,
    *,
    innovations: Any | None = None,
    current_baseline: Any | None = None,
) -> np.ndarray:
    """Assemble only causal accepted-state/Kalman features for a batch."""

    state_array = np.asarray(states, dtype=np.float64)
    if state_array.ndim != 3 or not np.all(np.isfinite(state_array)):
        raise ModelContractError("state features must be finite (batch, window, rank)")
    if family == "gru":
        if innovations is not None or current_baseline is not None:
            raise ModelContractError("G0 may not receive Kalman/current-baseline feature arrays")
        return np.array(state_array, copy=True)
    if family not in ("kalman0_gru", "kalman1_gru"):
        raise ModelContractError(f"unsupported feature family {family!r}")
    if innovations is None or current_baseline is None:
        raise ModelContractError("GK features require past innovations and current baseline")
    innovation_array = np.asarray(innovations, dtype=np.float64)
    baseline_array = np.asarray(current_baseline, dtype=np.float64)
    if innovation_array.shape != state_array.shape:
        raise ModelContractError("innovation history shape does not match states")
    if baseline_array.shape != (state_array.shape[0], state_array.shape[2]):
        raise ModelContractError("current baseline shape does not match states")
    repeated = np.repeat(baseline_array[:, None, :], state_array.shape[1], axis=1)
    result = np.concatenate((state_array, innovation_array, repeated), axis=-1)
    if not np.all(np.isfinite(result)):
        raise ModelContractError("assembled GRU features are nonfinite")
    return result


def to_torch(values: Any) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ModelContractError("cannot convert nonfinite values to torch")
    return torch.as_tensor(np.array(array, copy=True), dtype=torch.float64)


def model_config_payload(
    families: tuple[str, ...] = MODEL_FAMILIES,
    architectures: tuple[GRUArchitecture, ...] | None = None,
) -> dict[str, Any]:
    selected = FIXED_ARCHITECTURES if architectures is None else tuple(
        (item.rank, item.window, item.hidden) for item in architectures
    )
    return {
        "dtype": "float64",
        "device": "cpu",
        "deterministic_algorithms": True,
        "num_threads": 1,
        "families": list(families),
        "architectures": [
            {"rank": rank, "window": window, "hidden": hidden}
            for rank, window, hidden in selected
        ],
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "residual_output": True,
        "zero_initialized_output_head": True,
    }


__all__ = [
    "FIXED_ARCHITECTURES",
    "MODEL_FAMILIES",
    "GRUArchitecture",
    "ModelContractError",
    "ResidualGRU",
    "build_gru",
    "configure_deterministic_cpu",
    "make_gru_features",
    "model_config_payload",
    "parse_architectures",
    "set_seed",
    "to_torch",
]
