from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SolidSubstepPlan:
    substeps: int
    substep_dt_s: float
    velocity_damping: float


def resolve_solid_mpm_substeps(
    *,
    configured_substeps: int,
    dt_s: float,
    stable_dt_s: float,
) -> int:
    substeps = int(configured_substeps)
    if substeps > 0:
        return substeps
    return max(1, math.ceil(float(dt_s) / max(float(stable_dt_s), 1.0e-12)))


def solid_substep_dt_s(*, dt_s: float, substeps: int) -> float:
    return float(dt_s) / float(substeps)


def solid_substep_velocity_damping_factor(
    *,
    step_velocity_damping: float,
    substep_dt_s: float,
    step_dt_s: float,
) -> float:
    return float(step_velocity_damping) ** (
        float(substep_dt_s) / max(float(step_dt_s), 1.0e-12)
    )


def build_solid_substep_plan(
    *,
    configured_substeps: int,
    dt_s: float,
    stable_dt_s: float,
    step_velocity_damping: float,
) -> SolidSubstepPlan:
    substeps = resolve_solid_mpm_substeps(
        configured_substeps=configured_substeps,
        dt_s=dt_s,
        stable_dt_s=stable_dt_s,
    )
    substep_dt = solid_substep_dt_s(dt_s=dt_s, substeps=substeps)
    return SolidSubstepPlan(
        substeps=substeps,
        substep_dt_s=substep_dt,
        velocity_damping=solid_substep_velocity_damping_factor(
            step_velocity_damping=step_velocity_damping,
            substep_dt_s=substep_dt,
            step_dt_s=dt_s,
        ),
    )
