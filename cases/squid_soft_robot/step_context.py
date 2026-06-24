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


@dataclass(frozen=True)
class StepLoopState:
    first_step: int
    step_count: int
    rows: list[dict[str, object]]
    partial_run_stopped: bool = False
    partial_run_reason: str = ""


@dataclass(frozen=True)
class FsiStepControl:
    requested_iterations: int
    effective_fluid_substeps: int
    fluid_substep_dt_s: float
    adaptive_iterations_triggered: bool = False
    same_step_rerun_triggered: bool = False


@dataclass(frozen=True)
class AcceptedTrialReplayState:
    payload: dict[str, object] | None = None
    reused: bool = False
    readvanced: bool = False


@dataclass(frozen=True)
class StepExecutionResult:
    row: dict[str, object]
    fsi_control: FsiStepControl
    timing: StepTiming
    replay_state: AcceptedTrialReplayState
