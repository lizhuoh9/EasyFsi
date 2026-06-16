from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CflSubstepController:
    """Choose substeps from previously computed CFL diagnostics."""

    base_substeps: int = 1
    target_cfl: float = 0.25
    max_substeps: int = 16
    growth_safety: float = 1.25

    def __post_init__(self) -> None:
        if self.base_substeps <= 0:
            raise ValueError("base_substeps must be positive")
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
        if not math.isfinite(cfl) or cfl <= 0.0:
            return self.base_substeps
        reference_substeps = max(
            self.base_substeps,
            int(previous_substeps or self.base_substeps),
        )
        requested = math.ceil(
            reference_substeps * cfl / self.target_cfl * self.growth_safety
        )
        return min(self.max_substeps, max(self.base_substeps, int(requested)))
