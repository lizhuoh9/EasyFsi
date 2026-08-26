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

    def __init__(self, config: IqnIlsConfig | None = None) -> None:
        self.config = config or IqnIlsConfig()
        self.reset_step()

    def reset_step(self) -> None:
        self._shape: tuple[int, ...] | None = None
        self._guesses: list[np.ndarray] = []
        self._candidates: list[np.ndarray] = []
        self._residuals: list[np.ndarray] = []

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
        if len(self._residuals) < 2:
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=0,
            )

        pair_count = min(
            int(self.config.history_limit),
            len(self._residuals) - 1,
        )
        start = len(self._residuals) - pair_count - 1
        with np.errstate(over="ignore", invalid="ignore"):
            residual_differences = np.column_stack(
                [
                    self._residuals[index + 1] - self._residuals[index]
                    for index in range(start, len(self._residuals) - 1)
                ]
            )
            candidate_differences = np.column_stack(
                [
                    self._candidates[index + 1] - self._candidates[index]
                    for index in range(start, len(self._candidates) - 1)
                ]
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
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                fallback_reason="least_squares_failure",
            )
        rank = int(rank)
        if rank == 0:
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=0,
                fallback_reason="zero_rank_history",
            )
        retained_singular_values = np.asarray(singular_values[:rank], dtype=np.float64)
        condition_number = float(
            retained_singular_values[0] / retained_singular_values[-1]
        )
        if rank < pair_count:
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                fallback_reason="rank_deficient_history",
            )
        if (
            not math.isfinite(condition_number)
            or condition_number > float(self.config.max_condition_number)
        ):
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                fallback_reason="ill_conditioned_history",
            )
        coefficient_norm = float(np.linalg.norm(coefficients))
        if not math.isfinite(coefficient_norm):
            raise FloatingPointError("IQN-ILS coefficient norm must remain finite")
        coefficient_limit = self.config.max_coefficient_norm
        if (
            coefficient_limit is not None
            and coefficient_norm > float(coefficient_limit)
        ):
            return self._picard_update(
                guess_array,
                residual_array,
                history_pair_count=pair_count,
                rank=rank,
                condition_number=condition_number,
                coefficient_norm=coefficient_norm,
                fallback_reason="coefficient_norm_limit",
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
        return IqnIlsUpdate(
            next_guess=next_guess,
            mode="iqn_ils",
            rank=rank,
            condition_number=condition_number,
            coefficient_norm=coefficient_norm,
            history_pair_count=pair_count,
            update_norm=update_norm,
            update_limited=update_limited,
            fallback_reason=None,
        )

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


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
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
