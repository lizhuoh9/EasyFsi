from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class CflSubstepController:
    """Choose substeps from previously computed CFL diagnostics."""

    base_substeps: int = 1
    target_cfl: float = 0.25
    max_substeps: int = 16
    growth_safety: float = 1.25

    def __post_init__(self) -> None:
        for name, value in (
            ("base_substeps", self.base_substeps),
            ("max_substeps", self.max_substeps),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive non-boolean integer")
            object.__setattr__(self, name, int(value))
        if self.max_substeps < self.base_substeps:
            raise ValueError("max_substeps must be >= base_substeps")
        if not math.isfinite(self.target_cfl) or self.target_cfl <= 0.0:
            raise ValueError("target_cfl must be finite and positive")
        if not math.isfinite(self.growth_safety) or self.growth_safety < 1.0:
            raise ValueError("growth_safety must be finite and >= 1")

    def substeps_for_next_step(
        self,
        *,
        previous_cfl: float | None,
        previous_substeps: int | None = None,
    ) -> int:
        if previous_cfl is None:
            return self.base_substeps
        cfl = float(previous_cfl)
        if not math.isfinite(cfl):
            # A NaN/Inf CFL diagnostic means the upstream velocity field has
            # already diverged. The pre-2026-07 behavior lumped this with the
            # "no load" branch below and fell back to the MINIMUM substep
            # count -- the unsafe direction: it feeds a blown-up state
            # forward with the least stabilization. Physics-first: refuse to
            # guess a substep count from a non-physical diagnostic.
            raise ValueError(
                f"previous_cfl is non-finite ({cfl!r}): the upstream fluid "
                "state has diverged; refusing to choose a substep count from "
                "a non-physical CFL diagnostic (the old fallback silently "
                "continued at minimum substeps)"
            )
        if cfl < 0.0:
            raise ValueError("previous_cfl must be non-negative")
        if cfl == 0.0:
            return self.base_substeps
        if previous_substeps is not None and (
            isinstance(previous_substeps, bool)
            or not isinstance(previous_substeps, Integral)
            or previous_substeps <= 0
        ):
            raise ValueError(
                "previous_substeps must be a positive non-boolean integer"
            )
        reference_substeps = max(
            self.base_substeps,
            self.base_substeps
            if previous_substeps is None
            else int(previous_substeps),
        )
        requested = math.ceil(
            reference_substeps * cfl / self.target_cfl * self.growth_safety
        )
        return min(self.max_substeps, max(self.base_substeps, int(requested)))
