"""Deterministic, train-only POD and modal AR helpers for R25A."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

from .dataset import ACTIVE_AXES, AXIS_ORDER, DatasetContractError, MAX_POD_RANK


def _readonly(values: Any) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _finite_values(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise DatasetContractError(f"{name} must be non-empty and finite")
    return np.array(array, dtype=np.float64, copy=True)


def _validate_velocity_fields(values: Any, *, name: str) -> np.ndarray:
    array = _finite_values(values, name=name)
    if array.ndim < 2 or array.shape[-1] != 3 or array.shape[-2] < 1:
        raise DatasetContractError(f"{name} must end in (markers, 3)")
    if not np.all(array[..., 0] == 0.0):
        raise DatasetContractError(f"{name} x axis must be exactly zero")
    return array


def _fit_steps(values: Sequence[int] | None, count: int) -> tuple[int, ...]:
    if values is None:
        result = tuple(range(1, count + 1))
    else:
        raw = tuple(values)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            for value in raw
        ):
            raise DatasetContractError("fit_steps must contain integer physical steps")
        result = tuple(int(value) for value in raw)
    if (
        len(result) != count
        or not result
        or result[0] != 1
        or any(right != left + 1 for left, right in zip(result, result[1:]))
    ):
        raise DatasetContractError("fit_steps must align to contiguous physical steps")
    if result[-1] > 100:
        raise DatasetContractError("POD and normalization may only fit D0 steps 1-100")
    return result


@dataclass(frozen=True)
class PODBasis:
    """Immutable POD mean and basis with exact zero x reconstruction."""

    mean: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    rank: int
    fit_steps: tuple[int, ...]
    active_axes: tuple[bool, bool, bool] = ACTIVE_AXES

    def __post_init__(self) -> None:
        mean = _validate_velocity_fields(self.mean, name="POD mean")
        basis = _validate_velocity_fields(self.basis, name="POD basis")
        if mean.ndim != 2:
            raise DatasetContractError("POD mean must have shape (markers, 3)")
        if basis.ndim != 3 or basis.shape[1:] != mean.shape:
            raise DatasetContractError("POD basis must have shape (rank, markers, 3)")
        if isinstance(self.rank, bool) or int(self.rank) != basis.shape[0]:
            raise DatasetContractError("POD rank does not match basis")
        if not 1 <= int(self.rank) <= MAX_POD_RANK:
            raise DatasetContractError("POD rank must be within the frozen maximum")
        if tuple(self.active_axes) != ACTIVE_AXES:
            raise DatasetContractError("R25A POD active axes must be y/z")
        singular = _finite_values(self.singular_values, name="POD singular values")
        if singular.shape != (self.rank,):
            raise DatasetContractError("singular_values must align to rank")
        if not np.all(mean[..., 0] == 0.0) or not np.all(basis[..., 0] == 0.0):
            raise DatasetContractError("POD x axis must be exactly zero")
        steps = _fit_steps(self.fit_steps, len(self.fit_steps))
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "basis", _readonly(basis))
        object.__setattr__(self, "singular_values", _readonly(singular))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "fit_steps", steps)

    @property
    def marker_count(self) -> int:
        return self.mean.shape[0]

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.mean).tobytes())
        digest.update(np.ascontiguousarray(self.basis).tobytes())
        digest.update(np.ascontiguousarray(self.singular_values).tobytes())
        digest.update(str(self.fit_steps).encode())
        digest.update(np.asarray(self.active_axes, dtype=np.bool_).tobytes())
        return digest.hexdigest()

    def _check_fields(self, values: Any) -> np.ndarray:
        array = _validate_velocity_fields(values, name="velocity fields")
        if array.shape[-2:] != self.mean.shape:
            raise DatasetContractError(
                f"velocity field shape {array.shape[-2:]} != {self.mean.shape}"
            )
        return array

    def encode(self, values: Any) -> np.ndarray:
        """Encode fields after subtracting the train-only POD mean."""

        array = self._check_fields(values)
        return np.einsum("...mi,rmi->...r", array - self.mean, self.basis)

    def encode_residual(self, values: Any) -> np.ndarray:
        """Project a residual field without incorrectly subtracting the mean."""

        array = self._check_fields(values)
        return np.einsum("...mi,rmi->...r", array, self.basis)

    def decode(self, coefficients: Any) -> np.ndarray:
        coeffs = _finite_values(coefficients, name="modal coefficients")
        if coeffs.ndim < 1 or coeffs.shape[-1] != self.rank:
            raise DatasetContractError("modal coefficients must end in POD rank")
        decoded = self.mean + np.einsum("...r,rmi->...mi", coeffs, self.basis)
        decoded = np.array(decoded, dtype=np.float64, copy=False)
        decoded[..., 0] = 0.0
        if not np.all(np.isfinite(decoded)):
            raise DatasetContractError("POD decode produced nonfinite values")
        return decoded


def fit_pod(
    values: Any,
    *,
    rank: int,
    fit_steps: Sequence[int] | None = None,
    active_axes: tuple[bool, bool, bool] = ACTIVE_AXES,
) -> PODBasis:
    """Fit POD solely on the supplied D0 physical-step fields."""

    array = _validate_velocity_fields(values, name="POD training values")
    if array.ndim != 3:
        raise DatasetContractError("POD training values must have shape (steps, markers, 3)")
    if tuple(active_axes) != ACTIVE_AXES:
        raise DatasetContractError("R25A POD active axes are fixed to y/z")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or not 1 <= int(rank) <= MAX_POD_RANK
    ):
        raise DatasetContractError("POD rank must be an integer from 1 to 16")
    steps = _fit_steps(fit_steps, array.shape[0])
    mean = np.mean(array, axis=0)
    mean[..., 0] = 0.0
    active = np.asarray(active_axes, dtype=bool)
    centered = (array - mean)[..., active]
    matrix = centered.reshape(array.shape[0], -1)
    if int(rank) > min(matrix.shape):
        raise DatasetContractError("POD rank exceeds available training dimensions")
    _, singular_values, right_vectors = np.linalg.svd(
        matrix, full_matrices=False, compute_uv=True
    )
    basis = np.zeros((int(rank), array.shape[1], 3), dtype=np.float64)
    active_count = int(np.count_nonzero(active))
    basis[:, :, active] = right_vectors[: int(rank)].reshape(
        int(rank), array.shape[1], active_count
    )
    # SVD signs are mathematically arbitrary; orient each mode deterministically.
    for mode in range(int(rank)):
        flat = basis[mode, :, active].reshape(-1)
        pivot = int(np.argmax(np.abs(flat)))
        if flat[pivot] < 0.0:
            basis[mode] *= -1.0
    return PODBasis(
        mean=mean,
        basis=basis,
        singular_values=singular_values[: int(rank)],
        rank=int(rank),
        fit_steps=steps,
        active_axes=active_axes,
    )


@dataclass(frozen=True)
class ModalNormalization:
    """Train-only center/scale for POD coefficients."""

    mean: np.ndarray
    scale: np.ndarray
    fit_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        mean = _finite_values(self.mean, name="modal normalization mean")
        scale = _finite_values(self.scale, name="modal normalization scale")
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise DatasetContractError("normalization mean/scale must be one-dimensional")
        if np.any(scale <= 0.0):
            raise DatasetContractError("normalization scales must be positive")
        steps = _fit_steps(self.fit_steps, len(self.fit_steps))
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "scale", _readonly(scale))
        object.__setattr__(self, "fit_steps", steps)

    @property
    def rank(self) -> int:
        return self.mean.shape[0]

    @property
    def fit_max_step(self) -> int:
        return max(self.fit_steps)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(
                np.stack((self.mean, self.scale), axis=0)
            ).tobytes()
            + str(self.fit_steps).encode()
        ).hexdigest()

    def normalize(self, coefficients: Any) -> np.ndarray:
        values = _finite_values(coefficients, name="modal coefficients")
        if values.shape[-1] != self.rank:
            raise DatasetContractError("coefficient rank does not match normalization")
        return (values - self.mean) / self.scale

    def denormalize(self, values: Any) -> np.ndarray:
        normalized = _finite_values(values, name="normalized coefficients")
        if normalized.shape[-1] != self.rank:
            raise DatasetContractError("normalized coefficient rank mismatch")
        return normalized * self.scale + self.mean


def fit_normalization(
    coefficients: Any,
    *,
    fit_steps: Sequence[int] | None = None,
) -> ModalNormalization:
    array = _finite_values(coefficients, name="normalization coefficients")
    if array.ndim != 2:
        raise DatasetContractError("normalization coefficients must have shape (steps, rank)")
    steps = _fit_steps(fit_steps, array.shape[0])
    mean = np.mean(array, axis=0)
    deviation = array - mean
    scale = np.sqrt(np.mean(np.square(deviation), axis=0))
    scale = np.where(scale > 64.0 * np.finfo(np.float64).eps, scale, 1.0)
    return ModalNormalization(mean=mean, scale=scale, fit_steps=steps)


@dataclass(frozen=True)
class PODARModel:
    """Fixed-ridge, multi-output linear autoregressor in modal coordinates."""

    rank: int
    window: int
    ridge: float
    weights: np.ndarray
    bias: np.ndarray
    fit_steps: tuple[int, ...]
    rank_id: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, (int, np.integer))
            or int(self.rank) < 1
        ):
            raise DatasetContractError("AR rank must be positive")
        if (
            isinstance(self.window, bool)
            or not isinstance(self.window, (int, np.integer))
            or int(self.window) < 1
        ):
            raise DatasetContractError("AR window must be positive")
        if not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise DatasetContractError("AR ridge must be positive and finite")
        weights = _finite_values(self.weights, name="AR weights")
        bias = _finite_values(self.bias, name="AR bias")
        if weights.shape != (self.window * self.rank, self.rank) or bias.shape != (self.rank,):
            raise DatasetContractError("AR parameter shapes do not match rank/window")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "window", int(self.window))
        object.__setattr__(self, "weights", _readonly(weights))
        object.__setattr__(self, "bias", _readonly(bias))
        object.__setattr__(
            self, "fit_steps", _fit_steps(self.fit_steps, len(self.fit_steps))
        )

    def predict(self, history: Any) -> np.ndarray:
        array = _finite_values(history, name="AR history")
        if array.ndim < 2 or array.shape[-2:] != (self.window, self.rank):
            raise DatasetContractError("AR history must end in (window, rank)")
        flat = array.reshape(*array.shape[:-2], self.window * self.rank)
        result = flat @ self.weights + self.bias
        if not np.all(np.isfinite(result)):
            raise DatasetContractError("AR prediction is nonfinite")
        return result

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str((self.rank, self.window, self.ridge, self.fit_steps)).encode())
        digest.update(np.ascontiguousarray(self.weights).tobytes())
        digest.update(np.ascontiguousarray(self.bias).tobytes())
        return digest.hexdigest()


def fit_pod_ar(
    coefficients: Any,
    *,
    rank: int,
    window: int,
    ridge: float = 1.0e-6,
    fit_steps: Sequence[int] | None = None,
) -> PODARModel:
    """Fit one fixed-ridge AR on D0 steps 1--100 only."""

    array = _finite_values(coefficients, name="AR training coefficients")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or array.ndim != 2
        or array.shape[1] != int(rank)
    ):
        raise DatasetContractError("AR training coefficients shape mismatch")
    if (
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or int(window) < 1
    ):
        raise DatasetContractError("AR window must be a positive integer")
    steps = _fit_steps(fit_steps, array.shape[0])
    if array.shape[0] <= int(window):
        raise DatasetContractError("AR training trace is shorter than its window")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise DatasetContractError("ridge must be positive and finite")
    x = np.stack(
        [array[index - int(window) : index].reshape(-1) for index in range(int(window), len(array))],
        axis=0,
    )
    y = array[int(window) :]
    augmented = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    regularizer = np.eye(augmented.shape[1], dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    try:
        parameters = np.linalg.solve(
            augmented.T @ augmented + regularizer,
            augmented.T @ y,
        )
    except np.linalg.LinAlgError as exc:
        raise DatasetContractError("AR ridge system is singular") from exc
    return PODARModel(
        rank=int(rank),
        window=int(window),
        ridge=float(ridge),
        weights=parameters[:-1],
        bias=parameters[-1],
        fit_steps=steps,
    )


__all__ = [
    "ModalNormalization",
    "PODARModel",
    "PODBasis",
    "fit_normalization",
    "fit_pod",
    "fit_pod_ar",
]
