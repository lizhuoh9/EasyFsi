from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


ForceVector = tuple[float, ...]


@dataclass(frozen=True)
class ForceBalanceReport:
    residual_components_n: ForceVector
    residual_norm_n: float
    relative_error: float
    scale_n: float


@dataclass(frozen=True)
class RegionPairInterfaceReactionTarget:
    primary_force_n: tuple[float, float, float]
    secondary_force_n: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "primary_force_n",
            _force_vector3(self.primary_force_n, name="primary_force_n"),
        )
        object.__setattr__(
            self,
            "secondary_force_n",
            _force_vector3(self.secondary_force_n, name="secondary_force_n"),
        )


def action_reaction_balance(
    action_force_n: Sequence[float],
    reaction_force_n: Sequence[float],
) -> ForceBalanceReport:
    """Measure the component-wise balance of an action/reaction pair."""

    action = _force_vector(action_force_n, name="action_force_n")
    reaction = _force_vector(reaction_force_n, name="reaction_force_n")
    if len(action) != len(reaction):
        raise ValueError(
            "action_force_n and reaction_force_n must have the same length"
        )
    residual = _force_vector(
        (
            action_value + reaction_value
            for action_value, reaction_value in zip(action, reaction)
        ),
        name="force balance residual",
    )
    residual_norm = math.hypot(*residual)
    scale = math.hypot(*action) + math.hypot(*reaction)
    if not math.isfinite(scale):
        raise ValueError("force balance scale must be finite")
    scale = max(scale, 1.0e-30)
    return ForceBalanceReport(
        residual_components_n=residual,
        residual_norm_n=residual_norm,
        relative_error=residual_norm / scale,
        scale_n=scale,
    )


def region_pair_interface_reaction_forces(
    *,
    primary_fluid_force_n: Sequence[float],
    secondary_fluid_force_n: Sequence[float],
) -> RegionPairInterfaceReactionTarget:
    """Return equal-and-opposite solid reactions for two 3-D regions."""

    primary = _force_vector3(
        primary_fluid_force_n,
        name="primary_fluid_force_n",
    )
    secondary = _force_vector3(
        secondary_fluid_force_n,
        name="secondary_fluid_force_n",
    )
    return RegionPairInterfaceReactionTarget(
        primary_force_n=tuple(-component for component in primary),
        secondary_force_n=tuple(-component for component in secondary),
    )


def _force_vector(values: Sequence[float], *, name: str) -> ForceVector:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric components") from exc
    if not vector:
        raise ValueError(f"{name} must contain at least one component")
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain finite numeric components")
    return vector


def _force_vector3(
    values: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float]:
    vector = _force_vector(values, name=name)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    return vector[0], vector[1], vector[2]
