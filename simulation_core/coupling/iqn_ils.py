from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IqnIlsConfig:
    history_limit: int = 8
    initial_picard_relaxation: float = 0.5
    svd_relative_cutoff: float = 1.0e-10
    max_condition_number: float = 1.0e10
    max_coefficient_norm: float | None = None
    max_update_ratio: float | None = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.history_limit, bool):
            raise ValueError("history_limit must be a positive integer")
        try:
            history_limit = int(operator.index(self.history_limit))
        except TypeError as exc:
            raise ValueError("history_limit must be a positive integer") from exc
        if history_limit <= 0:
            raise ValueError("history_limit must be a positive integer")
        _finite_in_interval(
            self.initial_picard_relaxation,
            name="initial_picard_relaxation",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        _finite_in_interval(
            self.svd_relative_cutoff,
            name="svd_relative_cutoff",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        condition_limit = _finite_positive(
            self.max_condition_number,
            name="max_condition_number",
        )
        if condition_limit < 1.0:
            raise ValueError("max_condition_number must be at least 1")
        _optional_finite_positive(
            self.max_coefficient_norm,
            name="max_coefficient_norm",
        )
        _optional_finite_positive(
            self.max_update_ratio,
            name="max_update_ratio",
        )

    @property
    def signature(self) -> tuple[int, float, float, float, float | None, float | None]:
        """Stable identity for secants computed with this IQN configuration."""

        return (
            int(self.history_limit),
            float(self.initial_picard_relaxation),
            float(self.svd_relative_cutoff),
            float(self.max_condition_number),
            (
                None
                if self.max_coefficient_norm is None
                else float(self.max_coefficient_norm)
            ),
            None if self.max_update_ratio is None else float(self.max_update_ratio),
        )


@dataclass(frozen=True)
class IqnIlsSecantHistory:
    """Immutable, accepted-step-local IQN secant columns.

    The columns are differences already formed inside one accepted physical
    step.  Raw endpoint vectors are deliberately not retained, so a later step
    cannot accidentally construct a cross-step bridge secant.
    """

    delta_residual: np.ndarray
    delta_candidate: np.ndarray
    source_step: int
    layout_id: str
    dt_s: float
    marker_shape: tuple[int, int]
    config_signature: tuple[int, float, float, float, float | None, float | None]
    terminal_residual_norm: float
    initial_residual_norm: float | None = None

    def __post_init__(self) -> None:
        residual = _secant_columns(self.delta_residual, name="delta_residual")
        candidate = _secant_columns(self.delta_candidate, name="delta_candidate")
        if candidate.shape != residual.shape:
            raise ValueError("delta_candidate shape must match delta_residual shape")
        shape = tuple(self.marker_shape)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != 3:
            raise ValueError("marker_shape must be (marker_count, 3)")
        if residual.shape[0] != int(shape[0]) * 3:
            raise ValueError("secant column shape does not match marker_shape")
        if isinstance(self.source_step, bool) or int(self.source_step) <= 0:
            raise ValueError("source_step must be a positive integer")
        layout_id = str(self.layout_id).strip()
        if not layout_id:
            raise ValueError("layout_id must be non-empty")
        _finite_positive(self.dt_s, name="dt_s")
        _finite_non_negative(
            self.terminal_residual_norm,
            name="terminal_residual_norm",
        )
        initial_residual_norm = (
            self.terminal_residual_norm
            if self.initial_residual_norm is None
            else self.initial_residual_norm
        )
        _finite_non_negative(
            initial_residual_norm,
            name="initial_residual_norm",
        )
        signature = tuple(self.config_signature)
        if len(signature) != 6:
            raise ValueError("config_signature must contain six IQN settings")
        object.__setattr__(self, "delta_residual", _readonly_copy(residual))
        object.__setattr__(self, "delta_candidate", _readonly_copy(candidate))
        object.__setattr__(self, "source_step", int(self.source_step))
        object.__setattr__(self, "layout_id", layout_id)
        object.__setattr__(self, "dt_s", float(self.dt_s))
        object.__setattr__(self, "marker_shape", (int(shape[0]), 3))
        object.__setattr__(self, "config_signature", signature)
        object.__setattr__(
            self,
            "terminal_residual_norm",
            float(self.terminal_residual_norm),
        )
        object.__setattr__(
            self,
            "initial_residual_norm",
            float(initial_residual_norm),
        )

    @property
    def pair_count(self) -> int:
        return int(self.delta_residual.shape[1])


@dataclass(frozen=True)
class IqnIlsUpdate:
    next_guess: np.ndarray
    mode: str
    rank: int
    condition_number: float | None
    coefficient_norm: float | None
    history_pair_count: int
    update_norm: float
    update_limited: bool
    fallback_reason: str | None


class IqnIlsAccelerator:
    """Per-physical-step IQN-ILS update state for marker velocity."""

    def __init__(
        self,
        config: IqnIlsConfig | None = None,
        *,
        retained_history: IqnIlsSecantHistory | None = None,
    ) -> None:
        self.config = config or IqnIlsConfig()
        if retained_history is not None and not isinstance(
            retained_history, IqnIlsSecantHistory
        ):
            raise TypeError("retained_history must be an IqnIlsSecantHistory")
        if (
            retained_history is not None
            and retained_history.config_signature != self.config.signature
        ):
            raise ValueError("retained IQN history configuration does not match")
        self._retained_history = retained_history
        self.reset_step()

    def reset_step(self) -> None:
        self._shape: tuple[int, ...] | None = None
        self._guesses: list[np.ndarray] = []
        self._candidates: list[np.ndarray] = []
        self._residuals: list[np.ndarray] = []
        self._last_matrix_contains_retained = False

    @property
    def has_retained_history(self) -> bool:
        return self._retained_history is not None

    @property
    def last_matrix_contains_retained(self) -> bool:
        return self._last_matrix_contains_retained

    def discard_retained_history(self) -> None:
        """Stop using imported columns while preserving current-step secants."""

        self._retained_history = None

    def update(self, guess: Any, candidate: Any) -> IqnIlsUpdate:
        guess_array = _marker_velocity_array(guess, name="guess")
        candidate_array = _marker_velocity_array(candidate, name="candidate")
        if candidate_array.shape != guess_array.shape:
            raise ValueError(
                "candidate shape must match guess shape: "
                f"{candidate_array.shape} != {guess_array.shape}"
            )
        if self._shape is None:
            self._shape = guess_array.shape
            self._validate_retained_shape(guess_array.shape)
        elif guess_array.shape != self._shape:
            raise ValueError(
                "marker velocity shape changed within one physical step: "
                f"{guess_array.shape} != {self._shape}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            residual_array = candidate_array - guess_array
        if not bool(np.all(np.isfinite(residual_array))):
            raise FloatingPointError("IQN-ILS residual must remain finite")

        self._guesses.append(guess_array.reshape(-1).copy())
        self._candidates.append(candidate_array.reshape(-1).copy())
        self._residuals.append(residual_array.reshape(-1).copy())
        residual_differences, candidate_differences = self._history_columns()
        pair_count = int(residual_differences.shape[1])
        local_pair_count = min(
            int(self.config.history_limit), len(self._residuals) - 1
        )
        matrix_contains_retained = bool(
            self._retained_history is not None and pair_count > local_pair_count
        )
        self._last_matrix_contains_retained = matrix_contains_retained
        first_update_reuse = bool(
            matrix_contains_retained and len(self._residuals) == 1
        )
        if pair_count == 0:
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=0,
            )
        if not bool(
            np.all(np.isfinite(residual_differences))
            and np.all(np.isfinite(candidate_differences))
        ):
            raise FloatingPointError("IQN-ILS history differences must remain finite")

        current_residual = self._residuals[-1]
        try:
            coefficients, _, rank, singular_values = np.linalg.lstsq(
                residual_differences,
                current_residual,
                rcond=float(self.config.svd_relative_cutoff),
            )
        except np.linalg.LinAlgError:
            return self._fallback_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                fallback_reason="least_squares_failure",
                discard_retained=matrix_contains_retained,
            )
        rank = int(rank)
        if rank == 0:
            return self._fallback_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=0,
                fallback_reason="zero_rank_history",
                discard_retained=matrix_contains_retained,
            )
        retained_singular_values = np.asarray(singular_values[:rank], dtype=np.float64)
        condition_number = float(
            retained_singular_values[0] / retained_singular_values[-1]
        )
        if rank < pair_count:
            return self._fallback_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                fallback_reason="rank_deficient_history",
                discard_retained=matrix_contains_retained,
            )
        if (
            not math.isfinite(condition_number)
            or condition_number > float(self.config.max_condition_number)
        ):
            return self._fallback_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                fallback_reason="ill_conditioned_history",
                discard_retained=matrix_contains_retained,
            )
        coefficient_norm = float(np.linalg.norm(coefficients))
        if not math.isfinite(coefficient_norm):
            raise FloatingPointError("IQN-ILS coefficient norm must remain finite")
        coefficient_limit = self.config.max_coefficient_norm
        if (
            coefficient_limit is not None
            and coefficient_norm > float(coefficient_limit)
        ):
            return self._fallback_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                coefficient_norm=coefficient_norm,
                fallback_reason="coefficient_norm_limit",
                discard_retained=matrix_contains_retained,
            )

        with np.errstate(over="ignore", invalid="ignore"):
            proposal = self._candidates[-1] - candidate_differences @ coefficients
        if not bool(np.all(np.isfinite(proposal))):
            raise FloatingPointError("IQN-ILS produced a non-finite marker velocity")
        next_guess = proposal.reshape(guess_array.shape)
        next_guess, update_norm, update_limited = self._limit_update(
            guess_array,
            residual_array,
            next_guess,
        )
        if matrix_contains_retained and update_limited:
            self._retained_history = None
            fallback = self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                coefficient_norm=coefficient_norm,
                fallback_reason="reuse_update_limited",
            )
            return IqnIlsUpdate(
                **{**fallback.__dict__, "update_limited": True}
            )
        return IqnIlsUpdate(
            next_guess=next_guess,
            mode="iqn_ils_reuse" if first_update_reuse else "iqn_ils",
            rank=rank,
            condition_number=condition_number,
            coefficient_norm=coefficient_norm,
            history_pair_count=pair_count,
            update_norm=update_norm,
            update_limited=update_limited,
            fallback_reason=None,
        )

    def _validate_retained_shape(self, shape: tuple[int, ...]) -> None:
        history = self._retained_history
        if history is not None and history.marker_shape != shape:
            raise ValueError("retained IQN history marker shape does not match")

    def _history_columns(self) -> tuple[np.ndarray, np.ndarray]:
        local_residual = [
            self._residuals[index + 1] - self._residuals[index]
            for index in range(len(self._residuals) - 1)
        ]
        local_candidate = [
            self._candidates[index + 1] - self._candidates[index]
            for index in range(len(self._candidates) - 1)
        ]
        local_count = min(int(self.config.history_limit), len(local_residual))
        residual_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        retained = self._retained_history
        retained_count = 0
        if retained is not None:
            retained_count = min(
                int(self.config.history_limit) - local_count,
                retained.pair_count,
            )
            if retained_count:
                residual_parts.append(retained.delta_residual[:, -retained_count:])
                candidate_parts.append(retained.delta_candidate[:, -retained_count:])
        if local_count:
            residual_parts.append(np.column_stack(local_residual[-local_count:]))
            candidate_parts.append(np.column_stack(local_candidate[-local_count:]))
        if not residual_parts:
            dof = int(np.prod(self._shape)) if self._shape is not None else 0
            return np.empty((dof, 0)), np.empty((dof, 0))
        return np.column_stack(residual_parts), np.column_stack(candidate_parts)

    def _fallback_update(
        self,
        guess: np.ndarray,
        residual: np.ndarray,
        *,
        discard_retained: bool,
        **kwargs: Any,
    ) -> IqnIlsUpdate:
        if discard_retained:
            self._retained_history = None
        return self._picard_update(guess, residual, **kwargs)

    def _picard_update(
        self,
        guess: np.ndarray,
        residual: np.ndarray,
        *,
        history_pair_count: int,
        rank: int = 0,
        condition_number: float | None = None,
        coefficient_norm: float | None = None,
        fallback_reason: str | None = None,
    ) -> IqnIlsUpdate:
        with np.errstate(over="ignore", invalid="ignore"):
            next_guess = (
                guess
                + float(self.config.initial_picard_relaxation) * residual
            )
        if not bool(np.all(np.isfinite(next_guess))):
            raise FloatingPointError("Picard fallback produced a non-finite marker velocity")
        return IqnIlsUpdate(
            next_guess=next_guess,
            mode="picard",
            rank=rank,
            condition_number=condition_number,
            coefficient_norm=coefficient_norm,
            history_pair_count=history_pair_count,
            update_norm=float(np.linalg.norm(next_guess - guess)),
            update_limited=False,
            fallback_reason=fallback_reason,
        )

    def _limit_update(
        self,
        guess: np.ndarray,
        residual: np.ndarray,
        proposal: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        update = proposal - guess
        update_norm = float(np.linalg.norm(update))
        ratio = self.config.max_update_ratio
        if ratio is None:
            return proposal.copy(), update_norm, False
        residual_norm = float(np.linalg.norm(residual))
        maximum_norm = float(ratio) * residual_norm
        if update_norm <= maximum_norm or update_norm == 0.0:
            return proposal.copy(), update_norm, False
        if maximum_norm == 0.0:
            return guess.copy(), 0.0, True
        limited = guess + update * (maximum_norm / update_norm)
        if not bool(np.all(np.isfinite(limited))):
            raise FloatingPointError("limited IQN-ILS update must remain finite")
        return limited, maximum_norm, True


def _marker_velocity_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (marker_count, 3)")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _secant_columns(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (marker_dof, pair_count)")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_non_negative(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _optional_finite_positive(value: Any | None, *, name: str) -> None:
    if value is not None:
        _finite_positive(value, name=name)


def _finite_in_interval(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool,
) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    lower_hit = number >= lower if lower_inclusive else number > lower
    if not math.isfinite(number) or not lower_hit or number > upper:
        bracket = "[" if lower_inclusive else "("
        raise ValueError(f"{name} must be in {bracket}{lower}, {upper}]")
