"""Shared R24C oracle-threshold identities and fail-closed errors."""

from __future__ import annotations


THRESHOLD_TARGET_STEPS = (2, 5, 8)
THRESHOLD_OMEGAS = (0.5, 0.75, 1.0)
THRESHOLD_ALPHAS = (
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.996,
    0.9975,
    0.998,
    0.999,
    1.0,
)


class OracleThresholdContractError(RuntimeError):
    """Raised when R24C evidence cannot prove a required invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleThresholdContractError(message)


__all__ = (
    "OracleThresholdContractError",
    "THRESHOLD_ALPHAS",
    "THRESHOLD_OMEGAS",
    "THRESHOLD_TARGET_STEPS",
    "require",
)
