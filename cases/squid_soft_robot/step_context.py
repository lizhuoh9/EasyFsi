from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepTiming:
    step_wall_started_at: float
    fsi_coupling_wall_time_s: float = 0.0
    solid_advance_wall_time_s: float = 0.0
    fluid_advance_wall_time_s: float = 0.0
    sample_wall_time_s: float = 0.0
    surface_diagnostics_wall_time_s: float = 0.0
    checkpoint_wall_time_s: float = 0.0


@dataclass
class AdaptiveStepState:
    previous_step_cfl: float | None
    previous_step_fsi_coupling_residual_norm_n: float | None
    previous_step_fluid_substeps: int
