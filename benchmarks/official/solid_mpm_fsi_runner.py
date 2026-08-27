from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from simulation_core.coupling.active_kalman_writeback import (
    ACTIVE_KALMAN_MODE_OWNERS,
    FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
    INTERFACE_MARKER_VELOCITY_OWNER,
    SOLID_PARTICLE_VELOCITY_OWNER,
    ActiveKalmanWritebackController,
)
from simulation_core.coupling.interface_initial_guess_controller import (
    INITIAL_GUESS_MODES,
    InterfaceInitialGuessController,
)
from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.fluids.preflow_snapshot import (
    PREFLOW_SNAPSHOT_FIELD_NAMES,
    PreflowSnapshot,
    PreflowSnapshotIdentity,
    load_preflow_snapshot,
    save_preflow_snapshot,
    validate_preflow_snapshot_fields,
)
from simulation_core.coupling.hibm_mpm import (
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmMarkerMacConstraintOperator,
    HibmMpmSurfaceMarkers,
    capture_host_macro_step_state,
    capture_marker_interface_state,
    hibm_mpm_external_force_parts_fresh_for_solid_step,
    marker_layout_identity,
    marker_trial_state,
    restore_host_macro_step_state,
    restore_marker_interface_state,
)
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiCouplingConvergenceError,
    FsiCouplingReport,
    FsiSolverConfig,
    FsiStepContext,
    solve_fsi_step,
    solve_fsi_runtime,
)
from simulation_core.drivers.hibm_mpm_marker_velocity_runtime import (
    HibmMpmMarkerVelocityRuntime,
)
from simulation_core.solids.neo_hookean_mpm import (
    MpmOutOfBoundsError,
    MpmRequiredRegionEmptyError,
    NeoHookeanMpmState,
)
from simulation_core.coupling.pressure_sample_pairs import (
    PressureSamplePairMap,
    RuntimeAnchoredCellPairProvider,
)
from simulation_core.diagnostics.runtime import (
    TaichiRuntimeConfig,
    taichi_runtime_identity,
)
from simulation_core.diagnostics.time_stepping import (
    physical_time_roundoff_tolerance_s,
)

IQN_KALMAN_ORACLE_INTERPOLATION_DEFAULT_ALPHAS = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.975,
    0.99,
    1.0,
)


class PreflowSnapshotValidationError(ValueError):
    """Fail-closed preflow rejection with JSON-safe terminal diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class HibmJointQpConvergenceError(RuntimeError):
    """Fail-closed joint Q/P rejection with JSON-safe cycle diagnostics."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class PreflowStageObserverError(RuntimeError):
    """Host telemetry failure; never reclassify it as a solver failure."""


_HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES = (
    "canonical_ledger_build",
    "canonical_prepare_seal",
    "pressure_reachability_flood",
    "pressure_neumann_assembly",
)


def _empty_hibm_sharp_boundary_stage_wall_times() -> dict[str, float]:
    return {
        stage_name: 0.0
        for stage_name in _HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES
    }


def _require_preflow_ready_for_fsi(
    report: Mapping[str, object],
    *,
    expected_mode: str,
) -> None:
    mode = report.get("preflow_convergence_mode")
    if mode != expected_mode:
        raise RuntimeError(
            "preflow convergence mode does not match the validated config: "
            f"expected={expected_mode!r}, reported={mode!r}"
        )
    if mode == "windowed_stationary" and report.get("preflow_converged") is not True:
        raise RuntimeError(
            "windowed preflow did not reach stationary convergence before FSI: "
            f"status={report.get('preflow_status')!r}, "
            f"steps={report.get('preflow_steps_completed')!r}"
        )


def _require_fresh_external_force_for_solid_step(
    *,
    clear: object,
    scatter: object,
    marker_forces: object,
    stress: object,
    no_slip: object,
    projection: Mapping[str, object],
) -> None:
    if not hibm_mpm_external_force_parts_fresh_for_solid_step(
        clear=clear,
        scatter=scatter,
        marker_forces=marker_forces,
        stress=stress,
        no_slip=no_slip,
        projection=projection,
        pressure_component_overflow=projection.get(
            "pressure_nullspace_component_overflow"
        ),
        pressure_component_labels_converged=projection.get(
            "pressure_nullspace_component_labels_converged"
        ),
    ):
        raise RuntimeError(
            "HIBM-MPM external force transaction is incomplete or invalid; "
            "refusing to advance the solid"
        )


def _synchronize_hibm_sharp_boundary_stage_timing() -> None:
    # Keep Taichi optional at module import time.  Lightweight contract tests
    # use host-only fluid doubles without initializing Taichi; production
    # timing still synchronizes every active CUDA runtime boundary.
    import taichi as ti

    if ti.lang.impl.get_runtime().prog is None:
        return
    ti.sync()


def _measure_taichi_operation_wall_time(
    operation: Callable[[], Any],
    *,
    enabled: bool,
    clock: Callable[[], float] | None = None,
    synchronize: Callable[[], None] | None = None,
) -> tuple[Any, float]:
    if not enabled:
        return operation(), 0.0
    clock_fn = time.perf_counter if clock is None else clock
    synchronize_fn = (
        _synchronize_hibm_sharp_boundary_stage_timing
        if synchronize is None
        else synchronize
    )
    synchronize_fn()
    started_s = float(clock_fn())
    try:
        result = operation()
    except BaseException:
        try:
            synchronize_fn()
        except BaseException:
            # Preserve the primary operation failure.  A closing asynchronous
            # synchronization failure is secondary diagnostic damage.
            pass
        raise
    synchronize_fn()
    elapsed_s = float(clock_fn()) - started_s
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        elapsed_s = 0.0
    return result, elapsed_s


def _capture_solid_positions_for_step(
    solid: Any,
    *,
    profile_wall_time: bool,
) -> tuple[np.ndarray, float]:
    positions, wall_time_s = _measure_taichi_operation_wall_time(
        lambda: solid.x.to_numpy()[: solid.particle_count],
        enabled=profile_wall_time,
    )
    return positions, wall_time_s


_FSI_PROFILE_REQUIRED_STEP_FIELDS = (
    "flow_wall_time_s",
    "snapshot_capture_wall_time_s",
    "step_artifact_export_wall_time_s",
    "hibm_pre_predictor_wall_time_s",
    "hibm_projection_cycle_wall_time_s",
    "hibm_post_solid_observer_wall_time_s",
    "hibm_wall_time_s",
)


def _profile_wall_time_value(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid FSI profile field {field}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid FSI profile field {field}") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"invalid FSI profile field {field}")
    return result


def _fsi_profile_summary(
    history: list[Mapping[str, object]],
) -> dict[str, float]:
    """Sum only explicit per-FSI-step timing measurements."""

    collected = {
        field: [] for field in _FSI_PROFILE_REQUIRED_STEP_FIELDS
    }
    solid_wall_times: list[float] = []
    solid_wall_time_complete = bool(history)
    for step_index, step in enumerate(history, start=1):
        for field in _FSI_PROFILE_REQUIRED_STEP_FIELDS:
            if field not in step:
                raise ValueError(
                    f"FSI profile step {step_index} is missing {field}"
                )
            collected[field].append(
                _profile_wall_time_value(
                    step[field],
                    field=f"step {step_index} {field}",
                )
            )
        if "solid_wall_time_s" not in step:
            solid_wall_time_complete = False
            continue
        solid_wall_times.append(
            _profile_wall_time_value(
                step["solid_wall_time_s"],
                field=f"step {step_index} solid_wall_time_s",
            )
        )
    totals = {
        f"{field}_total": float(math.fsum(values))
        for field, values in collected.items()
    }
    if solid_wall_time_complete:
        totals["solid_wall_time_s_total"] = float(
            math.fsum(solid_wall_times)
        )
    return totals


_FSI_TRIAL_WORK_COUNT_FIELDS = (
    "cg_iterations_total",
    "flow_momentum_advection_substeps_total",
    "flow_sst_transport_substeps_total",
    "solid_substeps_executed_total",
)


def _fsi_trial_work_summary(
    reports: list[Mapping[str, object]],
) -> dict[str, float | int]:
    """Sum attempted work across every rejected and accepted FSI trial."""

    wall_times = {
        "flow_wall_time_s": [],
        "hibm_wall_time_s": [],
        "solid_wall_time_s": [],
    }
    counts = {field: 0 for field in _FSI_TRIAL_WORK_COUNT_FIELDS}
    feedback_consumed_trial_count = 0
    for trial_index, report in enumerate(reports, start=1):
        for field, values in wall_times.items():
            if field not in report:
                raise ValueError(
                    f"FSI trial {trial_index} work report is missing {field}"
                )
            values.append(
                _profile_wall_time_value(
                    report[field],
                    field=f"trial {trial_index} {field}",
                )
            )
        for field in _FSI_TRIAL_WORK_COUNT_FIELDS:
            value = report.get(field)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"invalid FSI trial work field {field}")
            try:
                integer = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid FSI trial work field {field}"
                ) from exc
            if integer < 0 or integer != value:
                raise ValueError(f"invalid FSI trial work field {field}")
            counts[field] += integer
        feedback_consumed = report.get("feedback_consumed")
        if not isinstance(feedback_consumed, (bool, np.bool_)):
            raise ValueError("invalid FSI trial work field feedback_consumed")
        feedback_consumed_trial_count += int(feedback_consumed)

    trial_count = len(reports)
    return {
        "trial_count": trial_count,
        "fluid_solve_count": trial_count,
        "solid_macro_solve_count": trial_count,
        "feedback_consumed_trial_count": feedback_consumed_trial_count,
        "flow_wall_time_s_total": float(math.fsum(wall_times["flow_wall_time_s"])),
        "hibm_wall_time_s_total": float(math.fsum(wall_times["hibm_wall_time_s"])),
        "solid_wall_time_s_total": float(
            math.fsum(wall_times["solid_wall_time_s"])
        ),
        **counts,
    }


def _fsi_coupling_iteration_summary(
    iterations: list[int],
) -> dict[str, float | int]:
    """Summarize accepted macro-step IQN iteration counts."""

    validated: list[int] = []
    for value in iterations:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("coupling iterations must be positive integers")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "coupling iterations must be positive integers"
            ) from exc
        if integer <= 0 or integer != value:
            raise ValueError("coupling iterations must be positive integers")
        validated.append(integer)
    if not validated:
        return {
            "total": 0,
            "minimum": 0,
            "maximum": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }
    return {
        "total": sum(validated),
        "minimum": min(validated),
        "maximum": max(validated),
        "mean": float(np.mean(validated)),
        "median": float(np.median(validated)),
        "p95": float(np.percentile(validated, 95)),
    }


def _load_initial_guess_oracle_replay(
    producer_output: str | Path,
    *,
    expected_steps: int,
) -> tuple[np.ndarray, ...]:
    """Preload an immutable accepted marker-velocity trajectory for Q3."""

    if isinstance(expected_steps, (bool, np.bool_)) or not isinstance(
        expected_steps, Integral
    ):
        raise TypeError("expected_steps must be an integer")
    step_count = int(expected_steps)
    if step_count <= 0:
        raise ValueError("expected_steps must be positive")
    fields_dir = Path(producer_output).expanduser() / "step_fields"
    replay: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for step in range(1, step_count + 1):
        frame_name = f"step_{step:04d}.npz"
        frame_path = fields_dir / frame_name
        if not frame_path.is_file():
            raise ValueError(f"oracle replay is missing {frame_name}")
        try:
            with np.load(frame_path, allow_pickle=False) as frame:
                if "marker_velocity_mps" not in frame.files:
                    raise ValueError("missing marker_velocity_mps")
                values = np.asarray(frame["marker_velocity_mps"])
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"oracle replay frame {frame_name} is unreadable: {exc}"
            ) from exc
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] <= 0:
            raise ValueError(
                f"oracle replay frame {frame_name} must have shape (count, 3)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"oracle replay frame {frame_name} contains non-finite values"
            )
        shape = (int(values.shape[0]), int(values.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"oracle replay frame {frame_name} shape changed: "
                f"{shape} != {expected_shape}"
            )
        immutable = np.ascontiguousarray(values, dtype=np.float64)
        immutable.flags.writeable = False
        replay.append(immutable)
    return tuple(replay)


def _iqn_kalman_oracle_interpolation_config(
    config: Any,
) -> dict[str, object] | None:
    """Validate the explicitly offline, no-commit alpha-sweep request."""

    target = getattr(config, "iqn_kalman_oracle_interpolation_target_step", None)
    oracle_path = getattr(
        config, "iqn_kalman_oracle_interpolation_oracle_path", None
    )
    alphas = tuple(
        getattr(
            config,
            "iqn_kalman_oracle_interpolation_alphas",
            IQN_KALMAN_ORACLE_INTERPOLATION_DEFAULT_ALPHAS,
        )
    )
    if target is None:
        if oracle_path is not None or alphas != IQN_KALMAN_ORACLE_INTERPOLATION_DEFAULT_ALPHAS:
            raise ValueError(
                "Kalman-Oracle interpolation oracle path and alphas require "
                "a target step"
            )
        return None
    if isinstance(target, (bool, np.bool_)) or int(target) != target:
        raise ValueError("Kalman-Oracle interpolation target step must be an integer")
    target_step = int(target)
    if target_step <= 0 or target_step > int(config.step_count):
        raise ValueError("Kalman-Oracle interpolation target step is out of range")
    if str(config.coupling_mode) != "iqn_ils" or str(config.initial_guess_mode) != "kalman":
        raise ValueError(
            "Kalman-Oracle interpolation requires iqn_ils with a Kalman first guess"
        )
    if str(getattr(config, "kalman_writeback_mode", "off")) != "off":
        raise ValueError("Kalman-Oracle interpolation requires kalman_writeback_mode=''off''")
    if bool(getattr(config, "iqn_reuse_previous_step_history", False)):
        raise ValueError(
            "Kalman-Oracle interpolation probe isolates initial-guess "
            "interpolation and requires iqn_reuse_previous_step_history=False"
        )
    if not isinstance(oracle_path, str) or not oracle_path.strip():
        raise ValueError("Kalman-Oracle interpolation requires an oracle path")
    if not alphas:
        raise ValueError("Kalman-Oracle interpolation alphas must be non-empty")
    converted = tuple(float(alpha) for alpha in alphas)
    if not all(math.isfinite(alpha) and 0.0 <= alpha <= 1.0 for alpha in converted):
        raise ValueError("Kalman-Oracle interpolation alphas must be finite in [0, 1]")
    if tuple(sorted(converted)) != converted or len(set(converted)) != len(converted):
        raise ValueError("Kalman-Oracle interpolation alphas must be strictly increasing")
    return {
        "target_step": target_step,
        "oracle_path": oracle_path,
        "alphas": converted,
        "offline_oracle": True,
        "deployable": False,
    }


def _mixed_iqn_kalman_oracle_guess(
    kalman_guess: np.ndarray,
    oracle_velocity: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the fixed alpha blend without mutating either endpoint."""

    kalman = np.asarray(kalman_guess, dtype=np.float64)
    oracle = np.asarray(oracle_velocity, dtype=np.float64)
    if kalman.ndim != 2 or kalman.shape[1] != 3 or kalman.shape != oracle.shape:
        raise ValueError("Kalman and oracle marker velocities must share shape (count, 3)")
    if not np.all(np.isfinite(kalman)) or not np.all(np.isfinite(oracle)):
        raise ValueError("Kalman and oracle marker velocities must be finite")
    weight = float(alpha)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must be finite in [0, 1]")
    return np.ascontiguousarray((1.0 - weight) * kalman + weight * oracle)


def _accepted_iqn_trial_vector_arrays(
    coupling: FsiCouplingReport,
    *,
    context: FsiStepContext,
    layout_sha256: str,
) -> dict[str, np.ndarray]:
    """Return one committed IQN step's opt-in trial-vector artifact payload."""

    traces = (
        coupling.trial_guess_history_mps,
        coupling.trial_candidate_history_mps,
        coupling.trial_residual_history_mps,
    )
    if not coupling.converged or any(trace is None for trace in traces):
        raise RuntimeError("accepted IQN trial-vector trace is unavailable")
    guess, candidate, residual = (
        np.asarray(trace, dtype=np.float64) for trace in traces
    )
    if (
        guess.ndim != 3
        or guess.shape[0] != int(coupling.iterations)
        or guess.shape[1] <= 0
        or guess.shape[2] != 3
        or candidate.shape != guess.shape
        or residual.shape != guess.shape
    ):
        raise RuntimeError(
            "accepted IQN trial vectors must share finite shape "
            "(iterations, marker_count, 3)"
        )
    if not all(np.all(np.isfinite(trace)) for trace in (guess, candidate, residual)):
        raise RuntimeError("accepted IQN trial vectors must be finite")
    if not np.array_equal(residual, candidate - guess):
        raise RuntimeError("accepted IQN trial residual must equal candidate - guess")
    layout = str(layout_sha256)
    if len(layout) != 64 or any(character not in "0123456789abcdef" for character in layout):
        raise RuntimeError("accepted IQN marker layout identity must be lowercase SHA-256")
    return {
        "iqn_trial_guess_mps": np.ascontiguousarray(guess),
        "iqn_trial_candidate_mps": np.ascontiguousarray(candidate),
        "iqn_trial_residual_mps": np.ascontiguousarray(residual),
        "iqn_trial_index": np.arange(int(coupling.iterations), dtype=np.int64),
        "iqn_trial_layout_sha256": np.asarray(layout),
        "iqn_trial_step": np.asarray(int(context.step), dtype=np.int64),
        "iqn_trial_time_s": np.asarray(float(context.time_s), dtype=np.float64),
        "iqn_trial_dt_s": np.asarray(float(context.dt_s), dtype=np.float64),
    }


def _hibm_stage_wall_time_sum(stage_wall_times: object) -> float:
    return float(
        math.fsum(
            _normalized_hibm_sharp_boundary_stage_wall_times(
                stage_wall_times
            ).values()
        )
    )


def _hibm_report_wall_time_s(
    report: Mapping[str, object] | None,
) -> float:
    return _hibm_stage_wall_time_sum(
        None
        if report is None
        else report.get("hibm_sharp_marker_boundary_stage_wall_time_s")
    )


def _fsi_step_hibm_wall_times(
    flow_report: Mapping[str, object],
    post_solid_observer_report: Mapping[str, object] | None,
) -> dict[str, float]:
    pre_predictor_source = flow_report.get("hibm_pre_predictor_wall_time_s")
    if pre_predictor_source is None:
        pre_predictor_source = _hibm_stage_wall_time_sum(
            flow_report.get("hibm_pre_predictor_stage_wall_time_s")
        )
    projection_cycle_source = flow_report.get(
        "hibm_projection_cycle_wall_time_s"
    )
    if projection_cycle_source is None:
        projection_cycle_source = _hibm_stage_wall_time_sum(
            flow_report.get(
                "hibm_sharp_marker_boundary_total_stage_wall_time_s",
                flow_report.get(
                    "hibm_sharp_marker_boundary_stage_wall_time_s"
                ),
            )
        )
    pre_predictor_wall_time_s = _profile_wall_time_value(
        pre_predictor_source,
        field="hibm_pre_predictor_wall_time_s",
    )
    projection_cycle_wall_time_s = _profile_wall_time_value(
        projection_cycle_source,
        field="hibm_projection_cycle_wall_time_s",
    )
    post_solid_observer_wall_time_s = _hibm_report_wall_time_s(
        post_solid_observer_report
    )
    return {
        "hibm_pre_predictor_wall_time_s": pre_predictor_wall_time_s,
        "hibm_projection_cycle_wall_time_s": projection_cycle_wall_time_s,
        "hibm_post_solid_observer_wall_time_s": post_solid_observer_wall_time_s,
        "hibm_wall_time_s": float(
            math.fsum(
                (
                    pre_predictor_wall_time_s,
                    projection_cycle_wall_time_s,
                    post_solid_observer_wall_time_s,
                )
            )
        ),
    }


def _measure_hibm_sharp_boundary_stage(
    stage_wall_times: dict[str, float],
    stage_name: str,
    operation: Callable[[], Any],
    *,
    enabled: bool = True,
    clock: Callable[[], float] | None = None,
    synchronize: Callable[[], None] | None = None,
    excluded_wall_time: Callable[[], float] | None = None,
) -> Any:
    if stage_name not in _HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES:
        raise ValueError(
            "unsupported HIBM sharp-boundary timing stage: "
            f"{stage_name!r}"
        )
    if not enabled:
        return operation()
    clock_fn = time.perf_counter if clock is None else clock
    synchronize_fn = (
        _synchronize_hibm_sharp_boundary_stage_timing
        if synchronize is None
        else synchronize
    )
    synchronize_fn()
    started_s = float(clock_fn())
    excluded_started_s = (
        float(excluded_wall_time()) if excluded_wall_time is not None else 0.0
    )
    try:
        return operation()
    finally:
        # The closing synchronization is deliberately inside ``finally`` so
        # failed kernels cannot leak asynchronous work into the next stage.
        synchronize_fn()
        ended_s = float(clock_fn())
        excluded_ended_s = (
            float(excluded_wall_time()) if excluded_wall_time is not None else 0.0
        )
        excluded_elapsed_s = excluded_ended_s - excluded_started_s
        if not math.isfinite(excluded_elapsed_s) or excluded_elapsed_s < 0.0:
            excluded_elapsed_s = 0.0
        elapsed_s = ended_s - started_s - excluded_elapsed_s
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            elapsed_s = 0.0
        previous_s = float(stage_wall_times.get(stage_name, 0.0))
        if not math.isfinite(previous_s) or previous_s < 0.0:
            previous_s = 0.0
        accumulated_s = previous_s + elapsed_s
        stage_wall_times[stage_name] = (
            accumulated_s if math.isfinite(accumulated_s) else previous_s
        )


def _hibm_sharp_boundary_timing_report_fields(
    stage_wall_times: Mapping[str, object],
) -> dict[str, object]:
    normalized = _normalized_hibm_sharp_boundary_stage_wall_times(
        stage_wall_times
    )
    return {
        "hibm_sharp_marker_boundary_stage_wall_time_s": dict(normalized),
        **{
            f"hibm_sharp_marker_boundary_{stage_name}_wall_time_s": elapsed_s
            for stage_name, elapsed_s in normalized.items()
        },
    }


def _normalized_hibm_sharp_boundary_stage_wall_times(
    stage_wall_times: object,
) -> dict[str, float]:
    source = stage_wall_times if isinstance(stage_wall_times, Mapping) else {}
    normalized: dict[str, float] = {}
    for stage_name in _HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES:
        try:
            value = float(source.get(stage_name, 0.0))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        normalized[stage_name] = (
            value if math.isfinite(value) and value >= 0.0 else 0.0
        )
    return normalized


def _hibm_sharp_boundary_stage_wall_times_from_report(
    report: Mapping[str, object] | None,
) -> dict[str, float]:
    if report is None:
        return _empty_hibm_sharp_boundary_stage_wall_times()
    return _normalized_hibm_sharp_boundary_stage_wall_times(
        report.get("hibm_sharp_marker_boundary_stage_wall_time_s", {})
    )


def _emit_run_progress(
    observer: Callable[[dict[str, object]], None] | None,
    *,
    run_started_s: float,
    phase: str,
    status: str = "running",
    **fields: object,
) -> None:
    if observer is None:
        return
    observer(
        {
            "status": str(status),
            "phase": str(phase),
            "elapsed_s": max(0.0, time.perf_counter() - run_started_s),
            **fields,
        }
    )


PRIMARY_REGION_ID = 101
SECONDARY_REGION_ID = 202
SECONDARY_UNUSED_REGION_ID = SECONDARY_REGION_ID
TIP_CAP_BOUNDARY_REGION_ID = 303
STREAMWISE_AXIS_INDEX = 2
OUT_OF_PLANE_AXIS_INDEX = 0
AXIS_NAMES = ("x", "y", "z")
OUT_OF_PLANE_BOUNDARY_POLICY = "finite_slab_x_faces_no_periodic_or_slip"
STRICT_OUT_OF_PLANE_BOUNDARY_POLICY = "strict_periodic_or_slip"
OUT_OF_PLANE_BOUNDARY_NOTE = (
    "The official case is conceptual 2D. This runner extrudes it into a finite "
    "3D slab and does not yet impose a strict periodic/slip condition on the "
    "out-of-plane x faces, so depth-normalized quantities are diagnostic rather "
    "than a full Fluent parity claim."
)
STRICT_OUT_OF_PLANE_BOUNDARY_NOTE = (
    "The official case is conceptual 2D. This runner extrudes it into a finite "
    "3D slab with exact zero-normal external x-face data and tangential "
    "zero-gradient symmetry, providing a strict slip out-of-plane closure."
)
FLOW_SOLUTION_MODE = "computed_projection"
DEFAULT_SOLID_CFL_TARGET = 0.5
FLOW_DRIVER_PROJECTION_ONLY = "projection_only"
FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC = "reinitialize_inlet_each_step_diagnostic"
FLOW_DRIVER_SUSTAINED_BOUNDARY = "sustained_boundary_inlet"
FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR = "sustained_boundary_predictor"
FLOW_DRIVER_SUSTAINED_SOURCE = "sustained_volume_source_inlet"
FLOW_DRIVER_SUSTAINED_PREDICTOR = "sustained_inlet_predictor"
FLOW_DRIVER_SHARP_REFERENCE = "sharp_hibm_mpm_reference"
FLOW_DRIVER_PHYSICAL_PREDICTOR_MODES = frozenset(
    {FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR, FLOW_DRIVER_SUSTAINED_PREDICTOR}
)
FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS = "cell_obstacle_layers"
FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS = "hibm_sharp_marker_rows"
FLOW_SOLID_BOUNDARY_MODES = {
    FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS,
    FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS,
}
FLOW_INLET_SOURCE_PROFILES = {"constant", "linear_ramp"}
FLOW_INLET_SOURCE_SCHEDULE_SCOPES = {"global", "phase_local"}
FLOW_TURBULENCE_MODELS = {"laminar", "sst_2003"}
FLOW_SST_NEAR_WALL_TREATMENTS = {"resolved", "fluent_correlation"}
FLOW_OUTLET_BALANCE_POLICIES = {"report_only"}
TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES = "dual_physical_faces"
TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE = "single_mid_surface"
TRACTION_MARKER_LAYOUTS = {
    TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES,
    TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE,
}
TRACTION_PRESSURE_TWO_SIDED = "two_sided_pressure_jump"
TRACTION_PRESSURE_ONE_SIDED = "one_sided_surface_pressure"
TRACTION_PRESSURE_SAMPLING_MODES = {
    TRACTION_PRESSURE_TWO_SIDED,
    TRACTION_PRESSURE_ONE_SIDED,
}
TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION = "marker_position"
TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET = "physical_face_offset"
TRACTION_PRESSURE_PROBE_ORIGIN_MODES = {
    TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION,
    TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET,
}
TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL = "current_normal_cell_ladder"
TRACTION_PRESSURE_PROBE_LADDER_MODES = {
    TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL,
}
TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER = "independent_ladder"
TRACTION_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR = "symmetric_cell_pair"
TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR = (
    "baseline_anchored_cell_pair"
)
TRACTION_PRESSURE_PAIR_POLICIES = {
    TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
    TRACTION_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR,
    TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR,
}
TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_DISABLED = "disabled"
TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR = (
    "runtime_anchored_cell_pair"
)
TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDERS = {
    TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_DISABLED,
    TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR,
}
TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED = "disabled"
TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED = "per_face_mirrored"
TRACTION_ONE_SIDED_PRESSURE_POLICIES = {
    TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED,
    TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED,
}
TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX = 4.0
SUPPORTED_FORMAL_FLOW_DRIVER_MODES = {
    FLOW_DRIVER_PROJECTION_ONLY,
    FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC,
    FLOW_DRIVER_SUSTAINED_BOUNDARY,
    FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
    FLOW_DRIVER_SUSTAINED_SOURCE,
    FLOW_DRIVER_SUSTAINED_PREDICTOR,
    FLOW_DRIVER_SHARP_REFERENCE,
}
FLOW_SOURCE_REPORT_KEYS = (
    "source_volume_flux_m3s",
    "positive_source_volume_flux_m3s",
    "abs_source_volume_flux_m3s",
    "zmin_pressure_outlet_flux_m3s",
    "zmin_velocity_outlet_flux_m3s",
    "zmin_pressure_outlet_to_source_ratio",
    "zmin_velocity_outlet_to_source_ratio",
    "zmin_pressure_outlet_to_net_source_ratio",
    "zmin_velocity_outlet_to_net_source_ratio",
    "zmin_pressure_outlet_to_positive_source_ratio",
    "zmin_velocity_outlet_to_positive_source_ratio",
    "zmin_pressure_outlet_to_abs_source_ratio",
    "zmin_velocity_outlet_to_abs_source_ratio",
    "pressure_outlet_flux_ratio",
    "velocity_outlet_flux_ratio",
)
FLOW_OBSTACLE_NORMAL_VELOCITY_POLICIES = {"face_clamp", "cell_zero_only"}
FLOW_PRESSURE_OUTLET_BACKFLOW_POLICIES = {"clamp", "allow"}
FLOW_PROJECTION_REPORT_KEYS = (
    "pressure_solver_requested",
    "pressure_solver",
    "pressure_outlet_backflow_policy",
    "obstacle_normal_velocity_policy",
    "pressure_solver_forced_to_fv_cg",
    "pressure_solver_force_reason",
    "pressure_solve_failed",
    "pressure_solve_failure_action",
    "pressure_projection_physical_failure",
    "pressure_projection_physical_failure_reason",
    "pressure_projection_physical_failure_action",
    "pressure_nullspace_policy",
    "pressure_marker_nullspace_enabled",
    "pressure_marker_nullspace_prepared",
    "pressure_marker_nullspace_active_constraint_count",
    "pressure_marker_nullspace_active_constraint_count_min",
    "pressure_marker_nullspace_active_constraint_count_max",
    "pressure_marker_nullspace_independent_constraint_count",
    "pressure_marker_nullspace_dependent_constraint_count",
    "pressure_marker_nullspace_unactuated_constraint_count",
    "pressure_marker_nullspace_apply_count",
    "pressure_marker_nullspace_pressure_actuation_generation",
    "pressure_marker_nullspace_min_factor_pivot",
    "pressure_marker_nullspace_max_dependent_normalized_pivot",
    "pressure_marker_nullspace_max_input_constraint_mps",
    "pressure_marker_nullspace_max_unactuated_input_constraint_mps",
    "pressure_marker_nullspace_max_constraint_residual_mps",
    "pressure_marker_nullspace_solver_scratch_resource_bytes",
    "pressure_marker_nullspace_marker_operator_resource_bytes",
    "pressure_marker_nullspace_resource_bytes",
    "pressure_marker_nullspace_actuation_invalid_count",
    "pressure_marker_nullspace_correction_invalid_count",
    "pressure_marker_nullspace_operator_apply_count",
    "pressure_marker_nullspace_velocity_correction_apply_count",
    "pressure_marker_nullspace_all_velocity_paths_projected",
    "pressure_marker_nullspace_enabled_all",
    "pressure_marker_nullspace_prepared_all",
    "pressure_marker_nullspace_all_velocity_paths_projected_all",
    "l2",
    "max_abs",
    "pre_projection_l2",
    "pre_projection_max_abs",
    "projection_l2",
    "projection_max_abs",
    "post_boundary_l2",
    "post_boundary_max_abs",
    "cg_project_calls",
    "cg_iterations_max",
    "cg_relative_residual_max",
    "cg_converged_all",
    "cg_breakdown_count",
    "cg_breakdown_code",
    "cg_breakdown",
    "pre_projection_velocity_projector_prepared_all",
    "pre_projection_velocity_projector_converged_all",
    "pre_projection_velocity_projector_committed_all",
    "hibm_joint_qp_measured",
    "hibm_joint_qp_converged",
    "hibm_joint_qp_cycle_budget",
    "hibm_joint_qp_cycles_used",
    "hibm_joint_qp_terminal_no_slip_vector_max_residual_mps",
    "hibm_joint_qp_terminal_divergence_l2_s_inv",
    "hibm_joint_qp_terminal_divergence_max_abs_s_inv",
    "hibm_joint_qp_pressure_exact_relative_residual",
    "hibm_joint_qp_pressure_reintroduced_no_slip_mps",
    "hibm_joint_qp_final_operation",
    "hibm_joint_qp_cycle_trace",
    "flow_symmetry_domain_walls",
    "fsi_pressure_snapshot_updated",
)
FLOW_ADVECTION_SCHEMES = {"euler", "rk2", "muscl_tvd"}
FLOW_PREDICTOR_NO_SLIP_WALL_INDEX = {
    "xmin": 0,
    "xmax": 1,
    "ymin": 2,
    "ymax": 3,
    "zmin": 4,
    "zmax": 5,
}
SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN = "3d_neo_hookean"
SOLID_CONSTITUTIVE_MODEL_PLANE_STRESS_LINEAR = "plane_stress_linear_elastic"
SOLID_CONSTITUTIVE_MODEL_SAINT_VENANT_KIRCHHOFF = "saint_venant_kirchhoff"
SOLID_CONSTITUTIVE_MODEL_SVK = "svk"
SOLID_CONSTITUTIVE_MODELS = {
    SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
    SOLID_CONSTITUTIVE_MODEL_PLANE_STRESS_LINEAR,
    SOLID_CONSTITUTIVE_MODEL_SAINT_VENANT_KIRCHHOFF,
    SOLID_CONSTITUTIVE_MODEL_SVK,
}
PREFLOW_TRACTION_READINESS_FLOW_ONLY = "flow_only"
PREFLOW_TRACTION_READINESS_COUPLING_READY = "coupling_ready"
PREFLOW_TRACTION_READINESS_MODES = {
    PREFLOW_TRACTION_READINESS_FLOW_ONLY,
    PREFLOW_TRACTION_READINESS_COUPLING_READY,
}
PREFLOW_TRACTION_EVALUATED = "evaluated"
PREFLOW_TRACTION_NOT_EVALUATED = "not_evaluated"
PREFLOW_TRACTION_INVALID = "invalid"


def _advance_particle_position_generation(current_generation: int) -> int:
    """Return the next runner-owned generation after one solid position write."""

    return int(current_generation) + 1


class SolidTrialRejectedError(RuntimeError):
    """A retryable numerical solid-trial rejection."""


def _validated_solid_substep_override(config: Any) -> int | None:
    requested_value = getattr(config, "solid_substeps", None)
    if requested_value is None:
        return None
    if isinstance(requested_value, bool) or not isinstance(
        requested_value,
        Integral,
    ):
        raise ValueError(
            "solid_substeps must be None for adaptive mode or a positive integer"
        )
    requested_substeps = int(requested_value)
    if requested_substeps <= 0:
        raise ValueError("solid_substeps must be positive when specified")
    return requested_substeps
def _validated_solid_controller_integer(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field_name} must be {qualifier}")
    return result


def _validated_solid_substep_dt_s(
    requested_macro_dt_s: object,
    solid_substeps: object,
) -> float:
    requested = float(requested_macro_dt_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("solid requested macro dt must be finite and positive")
    substeps = _validated_solid_controller_integer(
        solid_substeps,
        field_name="solid_substeps_selected",
        minimum=1,
    )
    substep_dt_s = requested / float(substeps)
    if not math.isfinite(substep_dt_s) or substep_dt_s <= 0.0:
        raise ValueError(
            "solid substep dt is not representable as a positive finite value: "
            f"requested_macro_dt_s={requested:g}, solid_substeps={substeps}"
        )
    return substep_dt_s




def _macro_time_accounting_report(
    *,
    requested_macro_dt_s: float,
    accepted_time_s: float,
    accepted_substep_count: int,
    rejected_trial_count: int,
    component: str,
) -> dict[str, object]:
    """Fail closed unless accepted physical time closes a macro step."""
    requested = float(requested_macro_dt_s)
    accepted = float(accepted_time_s)
    rejected = int(rejected_trial_count)
    if (
        not math.isfinite(requested)
        or not math.isfinite(accepted)
        or requested <= 0.0
        or accepted < 0.0
        or rejected < 0
    ):
        raise ValueError(f"invalid {component} physical-time accounting")
    tolerance = physical_time_roundoff_tolerance_s(
        requested_time_s=requested,
        accepted_time_s=accepted,
        accepted_substep_count=accepted_substep_count,
    )
    remaining = requested - accepted
    if abs(remaining) > tolerance:
        raise RuntimeError(
            f"{component} did not consume the requested physical macro time: "
            f"requested_macro_dt_s={requested:g}, accepted_time_s={accepted:g}, "
            f"remaining_unadvanced_time_s={remaining:g}"
        )
    prefix = str(component)
    return {
        "requested_macro_dt_s": requested,
        f"{prefix}_accepted_time_s": accepted,
        f"{prefix}_rejected_trial_count": rejected,
        f"{prefix}_remaining_unadvanced_time_s": 0.0,
    }


def _validated_component_accepted_time(
    *,
    requested_time_s: float,
    reported_requested_time_s: float,
    accepted_time_s: float,
    accepted_substep_count: int,
    remaining_unadvanced_time_s: float,
    rejected_trial_count: int,
    component: str,
) -> float:
    """Validate one operator clock without adding it to macro physical time."""

    requested = float(requested_time_s)
    reported_requested = float(reported_requested_time_s)
    accepted = float(accepted_time_s)
    remaining = float(remaining_unadvanced_time_s)
    rejected = int(rejected_trial_count)
    _macro_time_accounting_report(
        requested_macro_dt_s=requested,
        accepted_time_s=reported_requested,
        accepted_substep_count=1,
        rejected_trial_count=0,
        component=f"{component} request",
    )
    _macro_time_accounting_report(
        requested_macro_dt_s=requested,
        accepted_time_s=accepted,
        accepted_substep_count=accepted_substep_count,
        rejected_trial_count=rejected,
        component=component,
    )
    tolerance = physical_time_roundoff_tolerance_s(
        requested_time_s=requested,
        accepted_time_s=accepted,
        accepted_substep_count=accepted_substep_count,
    )
    if not math.isfinite(remaining) or abs(remaining) > tolerance:
        raise RuntimeError(
            f"{component} reported unadvanced physical time after acceptance: "
            f"requested_time_s={requested:g}, accepted_time_s={accepted:g}, "
            f"remaining_unadvanced_time_s={remaining:g}, "
            f"tolerance_s={tolerance:g}"
        )
    return accepted


def _require_healthy_solid_trial_report(report: Any, config: Any) -> None:
    scalar_fields = (
        "particle_spacing_m",
        "grid_spacing_m",
        "total_mass_kg",
        "total_volume_m3",
        "transfer_relative_error",
        "max_speed_mps",
        "max_abs_j",
        "mean_radial_stretch",
        "max_radial_stretch_error",
    )
    vector_fields = (
        "primary_mean_displacement_m",
        "primary_mean_velocity_mps",
        "secondary_mean_displacement_m",
        "secondary_mean_velocity_mps",
        "particle_momentum_kg_mps",
        "grid_momentum_kg_mps",
        "external_force_n",
    )
    for field_name in (*scalar_fields, *vector_fields):
        if not hasattr(report, field_name):
            continue
        raw_value = getattr(report, field_name)
        values = raw_value if isinstance(raw_value, (tuple, list)) else (raw_value,)
        if any(not math.isfinite(float(value)) for value in values):
            raise SolidTrialRejectedError(
                f"solid MPM trial reported non-finite {field_name}"
            )
    if hasattr(report, "max_abs_j") and float(report.max_abs_j) <= 0.0:
        raise SolidTrialRejectedError(
            "solid MPM trial reported non-positive maximum deformation Jacobian"
        )
    clamp_limit = getattr(
        config,
        "solid_max_deformation_clamp_count_per_macro_step",
        None,
    )
    if clamp_limit is not None and hasattr(report, "deformation_clamp_count"):
        limit = _validated_solid_controller_integer(
            clamp_limit,
            field_name="solid_max_deformation_clamp_count_per_macro_step",
            minimum=0,
        )
        if int(report.deformation_clamp_count) > limit:
            raise SolidTrialRejectedError(
                "solid MPM trial exceeded the deformation-clamp limit: "
                f"count={int(report.deformation_clamp_count)}, limit={limit}"
            )


def _select_and_advance_solid_macro_step(
    solid: NeoHookeanMpmState,
    config: Any,
    *,
    mu_pa: float,
    lambda_pa: float,
    retry_prepare: Callable[[], None],
    particle_position_write_observer: Callable[[], None] | None = None,
    profile_wall_time: bool = False,
) -> dict[str, object]:
    latest_selection: dict[str, object] = {}
    selector_evaluation_count = 0

    def select_from_accepted_state() -> int:
        nonlocal latest_selection, selector_evaluation_count
        accepted_speed_mps = float(solid.accepted_particle_max_speed())
        latest_selection = dict(
            solid_substep_cfl_report(
                config,
                max_particle_speed_mps=accepted_speed_mps,
            )
        )
        latest_selection["solid_max_particle_speed_mps"] = accepted_speed_mps
        selector_evaluation_count += 1
        selected = _validated_solid_controller_integer(
            latest_selection["solid_substeps_selected"],
            field_name="solid_substeps_selected",
            minimum=1,
        )
        max_substeps = _validated_solid_controller_integer(
            getattr(config, "solid_max_automatic_substeps", 65536),
            field_name="solid_max_automatic_substeps",
            minimum=1,
        )
        if selected > max_substeps:
            raise RuntimeError(
                "solid selector exceeds solid_max_automatic_substeps: "
                f"selected={selected}, maximum={max_substeps}"
            )
        return selected

    def advance_selected_macro_step() -> tuple[int, dict[str, object]]:
        initial_selected = select_from_accepted_state()
        report = _advance_solid_macro_step_with_retries(
            solid,
            config,
            selected_substeps=initial_selected,
            mu_pa=mu_pa,
            lambda_pa=lambda_pa,
            retry_prepare=retry_prepare,
            retry_selected_substeps=select_from_accepted_state,
            particle_position_write_observer=particle_position_write_observer,
            # The enclosing synchronized boundary covers the initial selector,
            # accepted-state save, every trial/retry, and final commit.
            profile_wall_time=False,
        )
        return initial_selected, report

    if profile_wall_time:
        (
            (initial_selected_substeps, advance_report),
            solid_wall_time_s,
        ) = _measure_taichi_operation_wall_time(
            advance_selected_macro_step,
            enabled=True,
        )
    else:
        macro_started_s = time.perf_counter()
        initial_selected_substeps, advance_report = advance_selected_macro_step()
        solid_wall_time_s = max(0.0, time.perf_counter() - macro_started_s)
    advance_report["solid_wall_time_s"] = solid_wall_time_s
    advance_report["solid_wall_time_synchronized"] = bool(profile_wall_time)
    final_selected_substeps = int(advance_report["solid_substeps_selected"])
    final_substep_dt_s = _validated_solid_substep_dt_s(
        config.dt_s,
        final_selected_substeps,
    )
    result = dict(latest_selection)
    if {
        "solid_elastic_wave_speed_mps",
        "solid_max_particle_speed_mps",
        "solid_min_grid_spacing_m",
    }.issubset(result):
        result["solid_estimated_cfl"] = (
            (
                float(result["solid_elastic_wave_speed_mps"])
                + float(result["solid_max_particle_speed_mps"])
            )
            * final_substep_dt_s
            / float(result["solid_min_grid_spacing_m"])
        )
    result.update(advance_report)
    result["solid_substeps_initial_selected"] = initial_selected_substeps
    result["solid_substeps_selected"] = final_selected_substeps
    result["solid_substep_dt_s"] = final_substep_dt_s
    result["solid_selector_evaluation_count"] = selector_evaluation_count
    result["solid_selector_device_to_host_scalar_read_count"] = (
        selector_evaluation_count
    )
    result["solid_retry_count"] = int(
        result.get("solid_rejected_trial_count", 0)
    )
    return result


def _solid_substep_run_summary(
    step_reports: list[Mapping[str, object]],
) -> dict[str, object]:
    if not step_reports:
        return {
            "solid_substeps_total": 0,
            "solid_substeps_min": 0,
            "solid_substeps_max": 0,
            "solid_substeps_mean": 0.0,
            "solid_step_kernel_launch_count_total": 0,
            "solid_selector_device_to_host_scalar_read_count_total": 0,
            "solid_packed_report_device_to_host_transfer_count_total": 0,
            "solid_guard_batch_count_total": 0,
            "solid_accepted_substeps_total": 0,
            "solid_substeps_selected_min": 0,
            "solid_substeps_selected_max": 0,
            "solid_substeps_selected_mean": 0.0,
            "solid_retry_count_total": 0,
            "solid_rejected_trial_count_total": 0,
            "solid_wall_time_s": 0.0,
        }
    executed = [
        int(
            report.get(
                "solid_substeps_executed_total",
                report["solid_accepted_substep_count"],
            )
        )
        for report in step_reports
    ]
    accepted = [
        int(report["solid_accepted_substep_count"])
        for report in step_reports
    ]
    retries = [
        int(report.get("solid_retry_count", 0)) for report in step_reports
    ]
    rejected = [
        int(report.get("solid_rejected_trial_count", 0))
        for report in step_reports
    ]
    step_kernel_launches = [
        int(
            report.get(
                "solid_step_kernel_launch_count",
                report.get(
                    "solid_substeps_executed_total",
                    report["solid_accepted_substep_count"],
                ),
            )
        )
        for report in step_reports
    ]
    selector_host_reads = [
        int(
            report.get(
                "solid_selector_device_to_host_scalar_read_count",
                report.get("solid_selector_evaluation_count", 0),
            )
        )
        for report in step_reports
    ]
    packed_report_transfers = [
        int(report["solid_packed_report_device_to_host_transfer_count"])
        for report in step_reports
    ]
    guard_batches = [
        int(
            report.get(
                "solid_guard_batch_count",
                int(report.get("solid_rejected_trial_count", 0)) + 1,
            )
        )
        for report in step_reports
    ]
    return {
        "solid_substeps_total": sum(executed),
        "solid_substeps_min": min(executed),
        "solid_substeps_max": max(executed),
        "solid_substeps_mean": math.fsum(executed) / len(executed),
        "solid_step_kernel_launch_count_total": sum(step_kernel_launches),
        "solid_selector_device_to_host_scalar_read_count_total": sum(
            selector_host_reads
        ),
        "solid_packed_report_device_to_host_transfer_count_total": sum(
            packed_report_transfers
        ),
        "solid_guard_batch_count_total": sum(guard_batches),
        "solid_accepted_substeps_total": sum(accepted),
        "solid_substeps_selected_min": min(accepted),
        "solid_substeps_selected_max": max(accepted),
        "solid_substeps_selected_mean": math.fsum(accepted) / len(accepted),
        "solid_retry_count_total": sum(retries),
        "solid_rejected_trial_count_total": sum(rejected),
        "solid_wall_time_s": math.fsum(
            float(report.get("solid_wall_time_s", 0.0))
            for report in step_reports
        ),
    }


def _advance_solid_macro_step_with_retries(
    solid: NeoHookeanMpmState,
    config: Any,
    *,
    selected_substeps: int,
    mu_pa: float,
    lambda_pa: float,
    retry_prepare: Callable[[], None],
    retry_selected_substeps: Callable[[], int] | None = None,
    particle_position_write_observer: Callable[[], None] | None = None,
    profile_wall_time: bool = False,
) -> dict[str, object]:
    """Commit one full solid macro step or restore and retry a typed trial fault."""
    substeps = _validated_solid_controller_integer(
        selected_substeps,
        field_name="selected_substeps",
        minimum=1,
    )
    max_retries = _validated_solid_controller_integer(
        getattr(config, "solid_max_substep_retries", 3),
        field_name="solid_max_substep_retries",
        minimum=0,
    )
    max_substeps = _validated_solid_controller_integer(
        getattr(config, "solid_max_automatic_substeps", 65536),
        field_name="solid_max_automatic_substeps",
        minimum=1,
    )
    if substeps > max_substeps:
        raise RuntimeError(
            "initial solid selector exceeds solid_max_automatic_substeps"
        )
    requested_dt_s = float(config.dt_s)
    _validated_solid_substep_dt_s(requested_dt_s, substeps)
    solid.save_state()
    rejected = 0
    attempted_substeps = [0]
    packed_report_transfer_attempts = [0]
    if profile_wall_time:
        _synchronize_hibm_sharp_boundary_stage_timing()
    macro_started_s = time.perf_counter()
    while True:
        substep_dt_s = _validated_solid_substep_dt_s(
            requested_dt_s,
            substeps,
        )
        try:
            report = _advance_solid_substeps_batched(
                solid, config, solid_substeps=substeps,
                solid_substep_dt_s=substep_dt_s, mu_pa=mu_pa,
                lambda_pa=lambda_pa,
                solid_substep_velocity_damping=_solid_substep_velocity_damping(
                    config, solid_substeps=substeps
                ),
                particle_position_write_observer=particle_position_write_observer,
                solid_substep_attempt_counter=attempted_substeps,
                solid_packed_report_transfer_counter=(
                    packed_report_transfer_attempts
                ),
            )
            _require_healthy_solid_trial_report(report, config)
        except (
            SolidTrialRejectedError,
            MpmOutOfBoundsError,
            MpmRequiredRegionEmptyError,
            FloatingPointError,
        ) as error:
            solid.restore_state()
            if particle_position_write_observer is not None:
                particle_position_write_observer()
            rejected += 1
            if rejected > max_retries:
                raise RuntimeError(
                    "solid MPM macro step could not commit after rollback retries"
                ) from error
            retry_prepare()
            reselected = (
                substeps
                if retry_selected_substeps is None
                else retry_selected_substeps()
            )
            validated_reselected = _validated_solid_controller_integer(
                reselected,
                field_name="restored-state solid selector result",
                minimum=1,
            )
            candidate = max(2 * substeps, validated_reselected)
            if candidate > max_substeps:
                raise RuntimeError(
                    "solid MPM retry exceeds solid_max_automatic_substeps"
                ) from error
            substeps = candidate
            continue
        except Exception:
            # A non-retryable error may occur after a batch has advanced the
            # particle fields. Preserve its original type and traceback, but
            # restore the last accepted snapshot before it leaves this scope.
            solid.restore_state()
            if particle_position_write_observer is not None:
                particle_position_write_observer()
            raise
        if profile_wall_time:
            _synchronize_hibm_sharp_boundary_stage_timing()
        solid_wall_time_s = max(
            0.0, time.perf_counter() - macro_started_s
        )
        time_report = _macro_time_accounting_report(
            requested_macro_dt_s=requested_dt_s,
            accepted_time_s=float(substeps) * substep_dt_s,
            accepted_substep_count=substeps,
            rejected_trial_count=rejected,
            component="solid",
        )
        return {
            "solid_report": report,
            "solid_substeps_selected": substeps,
            "solid_accepted_substep_count": substeps,
            "solid_substeps_executed_total": int(attempted_substeps[0]),
            "solid_step_kernel_launch_count": int(attempted_substeps[0]),
            "solid_guard_batch_count": rejected + 1,
            "solid_packed_report_device_to_host_transfer_count": int(
                packed_report_transfer_attempts[0]
            ),
            "solid_substep_dt_s": substep_dt_s,
            "solid_wall_time_s": solid_wall_time_s,
            "solid_wall_time_synchronized": bool(profile_wall_time),
            **time_report,
        }


def _advance_solid_substeps_batched(
    solid: NeoHookeanMpmState,
    config: Any,
    *,
    solid_substeps: int,
    solid_substep_dt_s: float,
    mu_pa: float,
    lambda_pa: float,
    solid_substep_velocity_damping: float,
    particle_position_write_observer: Callable[[], None] | None = None,
    solid_substep_attempt_counter: list[int] | None = None,
    solid_packed_report_transfer_counter: list[int] | None = None,
) -> Any:
    """Advance one FSI solid step with one fail-closed host guard read.

    The device-side guard retains the largest out-of-bounds particle count
    observed across every substep.  Ending the batch performs the single final
    packed-report read before marker feedback or any later coupling work can
    consume the advanced solid state.
    """

    if solid_substep_attempt_counter is not None and len(
        solid_substep_attempt_counter
    ) != 1:
        raise ValueError("solid_substep_attempt_counter must contain one integer")
    if solid_packed_report_transfer_counter is not None and len(
        solid_packed_report_transfer_counter
    ) != 1:
        raise ValueError("solid_packed_report_transfer_counter must contain one integer")
    solid.begin_out_of_bounds_guard_batch()
    try:
        for _solid_substep in range(solid_substeps):
            if solid_substep_attempt_counter is not None:
                solid_substep_attempt_counter[0] += 1
            solid.step(
                dt_s=solid_substep_dt_s,
                mu_pa=mu_pa,
                lambda_pa=lambda_pa,
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_REGION_ID,
                velocity_damping=solid_substep_velocity_damping,
                fixed_node_lock_policy=str(
                    getattr(config, "fixed_node_lock_policy", "any_fixed_particle")
                ),
                constitutive_model=str(
                    getattr(
                        config,
                        "solid_constitutive_model",
                        SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
                    )
                ),
                velocity_transfer_flip_blend=float(
                    getattr(config, "solid_velocity_transfer_flip_blend", 0.0)
                ),
                read_report=False,
            )
            if particle_position_write_observer is not None:
                particle_position_write_observer()
            if config.enforce_plane_strain_x:
                solid.enforce_rest_x_plane()
                if particle_position_write_observer is not None:
                    particle_position_write_observer()
        if solid_packed_report_transfer_counter is not None:
            solid_packed_report_transfer_counter[0] += 1
        return solid.end_out_of_bounds_guard_batch()
    except BaseException:
        # end_out_of_bounds_guard_batch() closes its host lifecycle before its
        # fail-closed report check.  abort() is intentionally idempotent, so it
        # also safely clears the lifecycle when either a substep or that final
        # report raises.  State rollback remains the caller's responsibility.
        solid.abort_out_of_bounds_guard_batch()
        raise


def run_hibm_mpm_fsi(
    *,
    case_id: str,
    case_metadata: Mapping[str, Any],
    boundary_conditions: Mapping[str, Any],
    reference_results: Mapping[str, Any],
    config: Any,
    step_observer: Callable[
        [int, float, dict[str, object], dict[str, np.ndarray]], None
    ]
    | None = None,
    progress_observer: Callable[[dict[str, object]], None] | None = None,
    profile_wall_time: bool = False,
) -> dict[str, object]:
    """Run the canonical Cartesian rectangular-solid HIBM-MPM pipeline."""
    run_started_s = time.perf_counter()
    _validate_rectangular_solid_config(config)
    flow_driver_mode = (
        _require_fsi_physical_flow_driver_mode(config)
        if int(config.step_count) > 0
        else _effective_flow_driver_mode(config, flow_phase="fsi")
    )
    _emit_run_progress(
        progress_observer,
        run_started_s=run_started_s,
        phase="initialization_fluid_build",
    )
    particle_position_generation = 0

    def record_particle_position_write() -> None:
        nonlocal particle_position_generation
        particle_position_generation = _advance_particle_position_generation(
            particle_position_generation
        )

    runtime = TaichiRuntimeConfig(arch="cuda", strict_arch=True)
    fluid = _build_fluid(config, runtime)
    runtime_identity = taichi_runtime_identity()
    _initialize_computed_flow(fluid, config)
    markers = _build_markers(config, runtime)
    anchor_install_report = _install_selected_pressure_pair_anchor_markers(
        markers,
        config,
    )
    pressure_pair_anchor_pair_map = dict(
        anchor_install_report.pop("pressure_pair_anchor_pair_map", {})
    )
    pressure_pair_anchor_runtime_refresh_count = 0

    def refresh_runtime_pressure_pair_anchors() -> None:
        nonlocal anchor_install_report
        nonlocal pressure_pair_anchor_pair_map
        nonlocal pressure_pair_anchor_runtime_refresh_count
        next_refresh_count = pressure_pair_anchor_runtime_refresh_count + 1
        refreshed = _refresh_runtime_pressure_pair_anchor_markers(
            markers,
            fluid,
            config,
            refresh_count=next_refresh_count,
        )
        if refreshed is None:
            return
        anchor_install_report, pair_map = refreshed
        pressure_pair_anchor_pair_map = dict(pair_map.as_diagnostics())
        pressure_pair_anchor_runtime_refresh_count = next_refresh_count

    solid = _build_solid(config, runtime)
    record_particle_position_write()
    # Install the physical MPM volume before fixed-solid preflow.  In sharp
    # mode it is stored in a dedicated layer; the first HIBM assembly then
    # combines it with the static geometry and carves only external row owners.
    if bool(getattr(config, "flow_hibm_dynamic_solid_volume_enabled", False)):
        latest_dynamic_obstacle_report = _update_fluid_obstacle_from_solid(
            fluid,
            solid,
            config,
        )
    else:
        latest_dynamic_obstacle_report = _fluid_obstacle_update_disabled_report()
    refresh_runtime_pressure_pair_anchors()
    fixed_mask, tip_mask = _solid_masks(solid, config)
    # cache the constant rest positions once so the per-step displacement report
    # does not re-fetch the whole rest array from the device every step
    rest_positions_m = solid.rest_x.to_numpy()[: solid.particle_count]
    mu_pa, lambda_pa = _lame_parameters(config)
    solid_substep_cfl: dict[str, object] = {}
    solid_seeding = _enforce_solid_seeding_limit(config)
    preflow_report = _run_or_restore_fixed_solid_preflow(
        markers=markers,
        fluid=fluid,
        solid=solid,
        config=config,
        progress_observer=progress_observer,
        run_started_s=run_started_s,
        profile_wall_time=profile_wall_time,
        particle_position_generation=particle_position_generation,
    )
    preflow_history = preflow_report["preflow_history"]
    if int(config.step_count) > 0:
        _require_preflow_ready_for_fsi(
            preflow_report,
            expected_mode=str(config.preflow_convergence_mode),
        )
    else:
        preflow_accepted_speed_mps = float(solid.accepted_particle_max_speed())
        solid_substep_cfl = dict(
            solid_substep_cfl_report(
                config,
                max_particle_speed_mps=preflow_accepted_speed_mps,
            )
        )
        solid_substep_cfl["solid_max_particle_speed_mps"] = preflow_accepted_speed_mps
    # A restored snapshot or the final fixed-solid HIBM assembly may replace
    # the obstacle view.  Seal anchors against that actual pre-FSI state.
    refresh_runtime_pressure_pair_anchors()
    kalman_writeback_mode = _modified_physics_kalman_mode(config)
    kalman_controller = (
        _initialize_modified_physics_kalman_controller(
            config,
            fluid=fluid,
            solid=solid,
            markers=markers,
        )
        if int(config.step_count) > 0
        else None
    )

    latest_stress_report = None
    latest_force_report = None
    latest_scatter_report = None
    latest_solid_report = None
    latest_solid_step_report = None
    latest_feedback_report = None
    latest_flow_report = None
    latest_feedback_constraint_report = None
    fluid_projection_count = 0
    fluid_projection_after_feedback_count = 0
    fluid_projection_consumed_feedback_count = 0
    fluid_projection_consumed_feedback_trial_count = 0
    feedback_available_for_projection = False
    history: list[dict[str, object]] = []
    solid_step_execution_reports: list[dict[str, object]] = []
    solid_trial_execution_reports: list[dict[str, object]] = []
    coupling_step_reports: list[dict[str, object]] = []
    coupling_trial_work_reports: list[dict[str, object]] = []
    research_probe_trial_work_reports: list[dict[str, object]] = []
    research_probe_solid_trial_reports: list[dict[str, object]] = []
    research_probe_active = False
    final_flow_field_snapshot: dict[str, np.ndarray] = {}
    apply_feedback = bool(getattr(config, "apply_marker_feedback_to_fluid", True))
    sharp_boundary_cache: dict[str, object] = {}
    coupling_mode = str(
        getattr(config, "coupling_mode", "direct_explicit")
    ).strip().lower()
    initial_guess_mode = str(
        getattr(config, "initial_guess_mode", "carry_forward")
    ).strip().lower()
    record_iqn_trial_vectors = bool(
        getattr(step_observer, "record_iqn_trial_vectors", False)
    )
    if record_iqn_trial_vectors and coupling_mode != "iqn_ils":
        raise ValueError("IQN trial-vector export requires coupling_mode='iqn_ils'")
    initial_guess_controller: InterfaceInitialGuessController | None = None
    prior_iqn_secant_history = None
    research_probe_config = _iqn_kalman_oracle_interpolation_config(config)
    research_probe_oracle = (
        _load_initial_guess_oracle_replay(
            str(research_probe_config["oracle_path"]),
            expected_steps=int(config.step_count),
        )
        if research_probe_config is not None
        else None
    )
    marker_reference_positions_m = None
    if coupling_mode == "iqn_ils":
        marker_reference_positions_m = np.asarray(
            capture_marker_interface_state(markers)["x_gamma_m"],
            dtype=np.float32,
        ).copy()
        oracle_replay = (
            _load_initial_guess_oracle_replay(
                str(config.initial_guess_oracle_path),
                expected_steps=int(config.step_count),
            )
            if initial_guess_mode == "oracle_replay"
            and int(config.step_count) > 0
            else None
        )
        initial_guess_controller = InterfaceInitialGuessController(
            initial_guess_mode,
            kalman_config=(
                config.initial_guess_kalman_config
                if initial_guess_mode == "kalman"
                else None
            ),
            oracle_replay=oracle_replay,
        )

    def current_marker_pressure_neumann_gradient_field() -> Any | None:
        cache_entry = sharp_boundary_cache.get("hibm_sharp_marker_boundary")
        if not isinstance(cache_entry, dict):
            return None
        ib_boundary = cache_entry.get("ib_boundary")
        if ib_boundary is None:
            return None
        return getattr(
            ib_boundary,
            "marker_pressure_neumann_gradient_field",
            None,
        )

    if coupling_mode == "iqn_ils" and int(config.step_count) > 0:
        iqn_base_topology_report = _apply_hibm_sharp_marker_boundary_to_fluid(
            markers,
            fluid,
            config,
            update_pressure_gradient=True,
            boundary_cache=sharp_boundary_cache,
            topology_only=False,
            measure_wall_times=profile_wall_time,
        )
        _require_hibm_velocity_dirichlet_health(
            iqn_base_topology_report,
            context="IQN-ILS accepted pre-step base assembly",
        )
    export_final_flow_snapshot = bool(
        getattr(config, "export_final_flow_snapshot", False)
    )
    immutable_flow_geometry = (
        _immutable_flow_geometry_snapshot(
            fluid,
            include_full_geometry=export_final_flow_snapshot,
        )
        if _flow_geometry_snapshot_cache_required(
            step_count=int(config.step_count),
            has_step_observer=step_observer is not None,
            export_final_flow_snapshot=export_final_flow_snapshot,
        )
        else None
    )

    for step_index in range(config.step_count):
        accepted_iqn_trial_vectors: dict[str, np.ndarray] | None = None
        kalman_raw_writeback_targets: dict[str, np.ndarray] = {}
        kalman_adapter_wall_time_s = 0.0
        kalman_filter_wall_time_before_s = (
            _kalman_controller_filter_wall_time_s(kalman_controller)
        )
        kalman_step_report = _empty_modified_physics_kalman_step_report(
            kalman_writeback_mode
        )
        feedback_available_before_projection = False
        flow_wall_time_s = 0.0
        kalman_fluid_feedback_pressure_raw_min_pa = 0.0
        kalman_fluid_feedback_pressure_raw_max_pa = 0.0
        kalman_fluid_feedback_pressure_min_pa = 0.0
        kalman_fluid_feedback_pressure_max_pa = 0.0
        observer_flow_snapshot = None
        snapshot_capture_wall_time_s = 0.0
        kalman_solid_integrator_raw_max_speed_mps = 0.0
        kalman_solid_accepted_max_speed_mps = 0.0
        kalman_interface_raw_max_speed_mps = 0.0
        kalman_interface_accepted_max_speed_mps = 0.0
        latest_observer_topology_report: dict[str, object] = {}
        trial_work_start_index = len(coupling_trial_work_reports)
        initial_guess_step_report: dict[str, object] = {
            "mode": initial_guess_mode,
            "mode_used": (
                "direct_explicit"
                if coupling_mode == "direct_explicit"
                else None
            ),
            "fallback_reason": None,
            "offline_oracle": False,
            "deployable": True,
        }

        def _run_hibm_mpm_coupling_trial() -> None:
            nonlocal feedback_available_before_projection
            nonlocal final_flow_field_snapshot
            nonlocal flow_wall_time_s
            nonlocal fluid_projection_after_feedback_count
            nonlocal fluid_projection_consumed_feedback_trial_count
            nonlocal fluid_projection_count
            nonlocal kalman_adapter_wall_time_s
            nonlocal kalman_fluid_feedback_pressure_max_pa
            nonlocal kalman_fluid_feedback_pressure_min_pa
            nonlocal kalman_fluid_feedback_pressure_raw_max_pa
            nonlocal kalman_fluid_feedback_pressure_raw_min_pa
            nonlocal kalman_interface_accepted_max_speed_mps
            nonlocal kalman_interface_raw_max_speed_mps
            nonlocal kalman_solid_accepted_max_speed_mps
            nonlocal kalman_solid_integrator_raw_max_speed_mps
            nonlocal latest_dynamic_obstacle_report
            nonlocal latest_feedback_constraint_report
            nonlocal latest_feedback_report
            nonlocal latest_flow_report
            nonlocal latest_force_report
            nonlocal latest_observer_topology_report
            nonlocal latest_scatter_report
            nonlocal latest_solid_report
            nonlocal latest_solid_step_report
            nonlocal latest_stress_report
            nonlocal observer_flow_snapshot
            nonlocal snapshot_capture_wall_time_s
            nonlocal solid_substep_cfl

            if _flow_driver_requires_full_field_reinitialize(flow_driver_mode):
                _initialize_computed_flow(fluid, config)
            feedback_available_before_projection = (
                feedback_available_for_projection and apply_feedback
            )
            latest_feedback_constraint_report = _apply_marker_feedback_to_fluid(
                markers,
                fluid,
                config,
                feedback_available=feedback_available_before_projection,
            )
            latest_flow_report, flow_wall_time_s = (
                _measure_taichi_operation_wall_time(
                    lambda: _flow_advance_current_step(
                        fluid,
                        config,
                        markers=markers,
                        sharp_boundary_cache=sharp_boundary_cache,
                        flow_phase="fsi",
                        step_index_local=step_index,
                        step_index_global=len(preflow_history) + step_index,
                        preflow_history=preflow_history,
                        reset_pressure=(
                            bool(
                                getattr(
                                    config,
                                    "flow_reset_pressure_each_step",
                                    False,
                                )
                            )
                            or (step_index == 0 and not preflow_history)
                        ),
                        measure_wall_times=profile_wall_time,
                    ),
                    enabled=profile_wall_time,
                )
            )
            if kalman_controller is not None:
                kalman_hook_started_s = time.perf_counter()
                kalman_controller.begin_step(dt_s=float(config.dt_s))
                kalman_adapter_wall_time_s += (
                    time.perf_counter() - kalman_hook_started_s
                )
            kalman_fluid_feedback_pressure_raw_min_pa = float(
                latest_flow_report["pressure_min_pa"]
            )
            kalman_fluid_feedback_pressure_raw_max_pa = float(
                latest_flow_report["pressure_max_pa"]
            )
            kalman_fluid_feedback_pressure_min_pa = (
                kalman_fluid_feedback_pressure_raw_min_pa
            )
            kalman_fluid_feedback_pressure_max_pa = (
                kalman_fluid_feedback_pressure_raw_max_pa
            )
            if (
                kalman_controller is not None
                and kalman_controller.enabled(FLUID_FSI_PRESSURE_FEEDBACK_OWNER)
            ):
                kalman_hook_started_s = time.perf_counter()
                try:
                    raw_pressure_pa = _kalman_fluid_observation(fluid)
                    fluid_kalman_result = kalman_controller.observe(
                        FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
                        raw_pressure_pa,
                    )
                    actual_pressure_pa = raw_pressure_pa
                    if fluid_kalman_result.writeback_values is not None:
                        kalman_raw_writeback_targets[
                            FLUID_FSI_PRESSURE_FEEDBACK_OWNER
                        ] = np.ascontiguousarray(raw_pressure_pa, dtype=np.float64)
                        actual_pressure_pa = _apply_kalman_fluid_writeback(
                            fluid,
                            fluid_kalman_result.writeback_values,
                        )
                    kalman_fluid_feedback_pressure_raw_min_pa = float(
                        np.min(raw_pressure_pa)
                    )
                    kalman_fluid_feedback_pressure_raw_max_pa = float(
                        np.max(raw_pressure_pa)
                    )
                    kalman_fluid_feedback_pressure_min_pa = float(
                        np.min(actual_pressure_pa)
                    )
                    kalman_fluid_feedback_pressure_max_pa = float(
                        np.max(actual_pressure_pa)
                    )
                except Exception:
                    _discard_modified_physics_kalman_step(
                        kalman_controller,
                        kalman_raw_writeback_targets,
                        fluid=fluid,
                        solid=solid,
                        markers=markers,
                    )
                    raise
                finally:
                    kalman_adapter_wall_time_s += (
                        time.perf_counter() - kalman_hook_started_s
                    )
            try:
                snapshot_capture_wall_time_s = 0.0
                observer_flow_snapshot = None
                if step_observer is not None and not research_probe_active:
                    (
                        observer_flow_snapshot,
                        observer_snapshot_wall_time_s,
                    ) = _measure_taichi_operation_wall_time(
                        lambda: _synchronized_flow_boundary_snapshot(
                            _flow_parity_snapshot(
                                fluid,
                                immutable_geometry=immutable_flow_geometry,
                            ),
                            stage="pre_solid_projection",
                        ),
                        enabled=profile_wall_time,
                    )
                    snapshot_capture_wall_time_s = float(
                        observer_snapshot_wall_time_s
                    )
                if not research_probe_active and export_final_flow_snapshot and (
                    step_index + 1 == int(config.step_count)
                ):
                    (
                        final_flow_field_snapshot,
                        final_snapshot_wall_time_s,
                    ) = _measure_taichi_operation_wall_time(
                        lambda: _synchronized_flow_boundary_snapshot(
                            _flow_field_snapshot(
                                fluid,
                                immutable_geometry=immutable_flow_geometry,
                            ),
                            stage="pre_solid_projection",
                        ),
                        enabled=profile_wall_time,
                    )
                    snapshot_capture_wall_time_s = float(
                        math.fsum(
                            (
                                snapshot_capture_wall_time_s,
                                final_snapshot_wall_time_s,
                            )
                        )
                    )
            except Exception:
                _discard_modified_physics_kalman_step(
                    kalman_controller,
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise
            latest_feedback_constraint_report[
                "no_slip_projected_residual_after_projection_mps"
            ] = float(latest_flow_report["hibm_no_slip_max_residual_mps"])
            if not research_probe_active:
                fluid_projection_count += 1
                if feedback_available_before_projection:
                    fluid_projection_after_feedback_count += 1
                if latest_feedback_constraint_report["fluid_projection_consumed_feedback"]:
                    fluid_projection_consumed_feedback_trial_count += 1
            try:
                latest_stress_report = _sample_stress_to_marker_forces(
                    markers,
                    fluid,
                    config,
                )
                latest_force_report = markers.aggregate_region_forces(
                    primary_region_id=PRIMARY_REGION_ID,
                    secondary_region_id=SECONDARY_REGION_ID,
                )
            except Exception:
                _discard_modified_physics_kalman_step(
                    kalman_controller,
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise

            def prepare_solid_external_force() -> tuple[Any, Any]:
                clear_report = markers.clear_mpm_external_forces(
                    solid.external_force_n,
                    particle_count=solid.particle_count,
                )
                scatter_report = markers.scatter_marker_forces_to_mpm_particles(
                    solid.external_force_n,
                    solid.x,
                    particle_count=solid.particle_count,
                    support_radius_m=config.mpm_support_radius_m,
                    particle_position_generation=particle_position_generation,
                )
                _require_fresh_external_force_for_solid_step(
                    clear=clear_report,
                    scatter=scatter_report,
                    marker_forces=latest_force_report,
                    stress=latest_stress_report,
                    no_slip=latest_flow_report.get("hibm_no_slip_report"),
                    projection=latest_flow_report["projection_report"],
                )
                return clear_report, scatter_report

            try:
                latest_clear_report, latest_scatter_report = (
                    prepare_solid_external_force()
                )
            except Exception:
                _discard_modified_physics_kalman_step(
                    kalman_controller,
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise

            def retry_prepare_solid_external_force() -> None:
                nonlocal latest_clear_report, latest_scatter_report
                (
                    latest_clear_report,
                    latest_scatter_report,
                ) = prepare_solid_external_force()

            try:
                latest_solid_step_report = _select_and_advance_solid_macro_step(
                    solid,
                    config,
                    mu_pa=mu_pa,
                    lambda_pa=lambda_pa,
                    retry_prepare=retry_prepare_solid_external_force,
                    particle_position_write_observer=record_particle_position_write,
                    profile_wall_time=profile_wall_time,
                )
            except Exception:
                _discard_modified_physics_kalman_step(
                    kalman_controller,
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise
            latest_solid_report = latest_solid_step_report["solid_report"]
            solid_substep_cfl = {
                key: value
                for key, value in latest_solid_step_report.items()
                if key != "solid_report"
            }
            if research_probe_active:
                research_probe_solid_trial_reports.append(dict(solid_substep_cfl))
            else:
                solid_trial_execution_reports.append(dict(solid_substep_cfl))
            kalman_solid_integrator_raw_max_speed_mps = float(
                latest_solid_report.max_speed_mps
            )
            kalman_solid_accepted_max_speed_mps = (
                kalman_solid_integrator_raw_max_speed_mps
            )
            if (
                kalman_controller is not None
                and kalman_controller.enabled(SOLID_PARTICLE_VELOCITY_OWNER)
            ):
                kalman_hook_started_s = time.perf_counter()
                try:
                    raw_solid_velocity_mps = _kalman_solid_observation(solid)
                    solid_kalman_result = kalman_controller.observe(
                        SOLID_PARTICLE_VELOCITY_OWNER,
                        raw_solid_velocity_mps,
                    )
                    actual_solid_velocity_mps = raw_solid_velocity_mps
                    if solid_kalman_result.writeback_values is not None:
                        kalman_raw_writeback_targets[
                            SOLID_PARTICLE_VELOCITY_OWNER
                        ] = np.ascontiguousarray(
                            solid.v.to_numpy(),
                            dtype=np.float32,
                        )
                        actual_solid_velocity_mps = _apply_kalman_solid_writeback(
                            solid,
                            solid_kalman_result.writeback_values,
                            fixed_mask=fixed_mask,
                            enforce_plane_strain_x=bool(config.enforce_plane_strain_x),
                        )
                    kalman_solid_accepted_max_speed_mps = float(
                        np.max(np.linalg.norm(actual_solid_velocity_mps, axis=1))
                    )
                except Exception:
                    _discard_modified_physics_kalman_step(
                        kalman_controller,
                        kalman_raw_writeback_targets,
                        fluid=fluid,
                        solid=solid,
                        markers=markers,
                    )
                    raise
                finally:
                    kalman_adapter_wall_time_s += (
                        time.perf_counter() - kalman_hook_started_s
                    )
            try:
                latest_feedback_report = (
                    markers.update_surface_feedback_from_mpm_surface_particles(
                        solid.x,
                        solid.v,
                        solid.surface_normal,
                        solid.area_weight_m2,
                        particle_count=solid.particle_count,
                        support_radius_m=config.mpm_support_radius_m,
                        dt_s=config.dt_s,
                        preserve_marker_area=bool(
                            getattr(
                                config,
                                "preserve_marker_area_during_surface_feedback",
                                False,
                            )
                        ),
                        particle_position_generation=particle_position_generation,
                    )
                )
            except Exception:
                _discard_modified_physics_kalman_step(
                    kalman_controller,
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise
            raw_marker_velocity_mps = None
            kalman_interface_raw_max_speed_mps = float(
                latest_feedback_report.max_marker_speed_mps
            )
            kalman_interface_accepted_max_speed_mps = (
                kalman_interface_raw_max_speed_mps
            )
            if (
                kalman_controller is not None
                and kalman_controller.enabled(INTERFACE_MARKER_VELOCITY_OWNER)
            ):
                kalman_hook_started_s = time.perf_counter()
                try:
                    raw_marker_velocity_mps = _kalman_interface_observation(markers)
                    interface_kalman_result = kalman_controller.observe(
                        INTERFACE_MARKER_VELOCITY_OWNER,
                        raw_marker_velocity_mps,
                    )
                    actual_marker_velocity_mps = raw_marker_velocity_mps
                    if interface_kalman_result.writeback_values is not None:
                        kalman_raw_writeback_targets[
                            INTERFACE_MARKER_VELOCITY_OWNER
                        ] = np.ascontiguousarray(
                            markers.v_gamma_mps.to_numpy(),
                            dtype=np.float32,
                        )
                        actual_marker_velocity_mps = (
                            _apply_kalman_interface_writeback(
                                markers,
                                interface_kalman_result.writeback_values,
                            )
                        )
                    kalman_interface_raw_max_speed_mps = float(
                        np.max(np.linalg.norm(raw_marker_velocity_mps, axis=1))
                    )
                    kalman_interface_accepted_max_speed_mps = float(
                        np.max(np.linalg.norm(actual_marker_velocity_mps, axis=1))
                    )
                except Exception:
                    _discard_modified_physics_kalman_step(
                        kalman_controller,
                        kalman_raw_writeback_targets,
                        fluid=fluid,
                        solid=solid,
                        markers=markers,
                    )
                    raise
                finally:
                    kalman_adapter_wall_time_s += (
                        time.perf_counter() - kalman_hook_started_s
                    )
            try:
                latest_dynamic_obstacle_report = _update_fluid_obstacle_from_solid(
                    fluid,
                    solid,
                    config,
                )
                latest_observer_topology_report = (
                    _apply_hibm_sharp_marker_boundary_to_fluid(
                        markers,
                        fluid,
                        config,
                        update_pressure_gradient=False,
                        boundary_cache=sharp_boundary_cache,
                        # Rebuild rows as well as topology so the per-step and final
                        # snapshots never combine the post-solid obstacle with the
                        # previous fluid stage's boundary mask.  The resulting cache
                        # also lets the next predictor reuse search and cleanup.
                        topology_only=False,
                        measure_wall_times=profile_wall_time,
                    )
                )
                _require_hibm_velocity_dirichlet_health(
                    latest_observer_topology_report,
                    context=f"FSI step {step_index + 1} post-solid observer assembly",
                )
                projection_report = latest_flow_report.get("projection_report")
                if not isinstance(projection_report, Mapping):
                    raise RuntimeError(
                        "FSI trial flow report omitted projection_report"
                    )
                trial_hibm_wall_times = _fsi_step_hibm_wall_times(
                    latest_flow_report,
                    latest_observer_topology_report,
                )
                (
                    research_probe_trial_work_reports
                    if research_probe_active
                    else coupling_trial_work_reports
                ).append(
                    {
                        "flow_wall_time_s": float(flow_wall_time_s),
                        "hibm_wall_time_s": float(
                            trial_hibm_wall_times["hibm_wall_time_s"]
                        ),
                        "solid_wall_time_s": float(
                            solid_substep_cfl["solid_wall_time_s"]
                        ),
                        "cg_iterations_total": int(
                            projection_report.get("cg_iterations_total", 0)
                        ),
                        "flow_momentum_advection_substeps_total": int(
                            latest_flow_report.get(
                                "flow_momentum_advection_substeps_total", 0
                            )
                        ),
                        "flow_sst_transport_substeps_total": int(
                            latest_flow_report.get(
                                "flow_sst_transport_substeps_total", 0
                            )
                        ),
                        "solid_substeps_executed_total": int(
                            solid_substep_cfl["solid_substeps_executed_total"]
                        ),
                        "feedback_consumed": bool(
                            latest_feedback_constraint_report[
                                "fluid_projection_consumed_feedback"
                            ]
                        ),
                    }
                )
            except Exception:
                _discard_modified_physics_kalman_step(
                    kalman_controller,
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise

        if coupling_mode == "direct_explicit":
            _run_hibm_mpm_coupling_trial()
            coupling_step_report = {
                "hibm_coupling_scheme": "explicit_loose",
                "hibm_fsi_coupling_iterations_used": 1,
                "hibm_fsi_coupling_converged": False,
                "hibm_fsi_coupling_explicit_single_pass": True,
                "hibm_fsi_coupling_rejected_trial_count": 0,
                "hibm_fsi_coupling_residual_source": "unmeasured_single_pass",
                "hibm_fsi_coupling_residual_norm_mps": None,
                "hibm_fsi_coupling_tolerance_mps": None,
                "hibm_fsi_coupling_residual_history_mps": [],
                "hibm_fsi_coupling_tolerance_history_mps": [],
                "hibm_fsi_coupling_relaxation_history": [],
                "hibm_fsi_coupling_update_mode_history": [],
                "hibm_fsi_coupling_iqn_rank_history": [],
                "hibm_fsi_coupling_iqn_condition_number_history": [],
                "hibm_fsi_coupling_iqn_fallback_count": 0,
                "hibm_fsi_coupling_base_assembly_count": 0,
            }
        else:
            if initial_guess_controller is None:
                raise RuntimeError(
                    "IQN-ILS initial-guess controller was not initialized"
                )
            marker_pressure_neumann_gradient_field = (
                current_marker_pressure_neumann_gradient_field()
            )
            if marker_pressure_neumann_gradient_field is None:
                raise RuntimeError(
                    "IQN-ILS requires an initialized marker pressure-Neumann "
                    "gradient field"
                )

            def capture_iqn_step_state():
                return capture_host_macro_step_state(
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                    accepted_step_index=step_index,
                    accepted_time_s=float(config.dt_s) * float(step_index),
                    feedback_available_for_projection=(
                        feedback_available_for_projection
                    ),
                    marker_pressure_neumann_gradient_field=(
                        current_marker_pressure_neumann_gradient_field()
                    ),
                )

            def restore_iqn_step_state(state, _context) -> None:
                nonlocal feedback_available_for_projection
                gradient_field = (
                    current_marker_pressure_neumann_gradient_field()
                    if state.marker_pressure_neumann_gradient is not None
                    else None
                )
                restore_host_macro_step_state(
                    state,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                    marker_pressure_neumann_gradient_field=gradient_field,
                    record_particle_position_write=(
                        record_particle_position_write
                    ),
                )
                feedback_available_for_projection = bool(
                    state.feedback_available_for_projection
                )
                _invalidate_hibm_sharp_boundary_derived_cache(
                    sharp_boundary_cache
                )

            def apply_iqn_marker_velocity_guess(
                marker_base: Mapping[str, Any],
                guess: np.ndarray,
            ) -> None:
                restore_marker_interface_state(
                    markers,
                    marker_trial_state(marker_base, guess),
                )
                # Restoring the accepted marker state advances its geometry
                # revision, which intentionally retires pressure-pair anchors.
                # Reseal them before this trial samples interface traction.
                refresh_runtime_pressure_pair_anchors()
                _invalidate_hibm_sharp_boundary_derived_cache(
                    sharp_boundary_cache
                )

            def commit_iqn_case_step(_context, _trial, _coupling):
                nonlocal feedback_available_for_projection
                nonlocal accepted_iqn_trial_vectors
                refresh_runtime_pressure_pair_anchors()
                feedback_available_for_projection = True
                if record_iqn_trial_vectors:
                    accepted_iqn_trial_vectors = _accepted_iqn_trial_vector_arrays(
                        _coupling,
                        context=_context,
                        layout_sha256=marker_layout_identity(
                            markers,
                            reference_positions_m=np.asarray(
                                marker_reference_positions_m,
                                dtype=np.float32,
                            ),
                            namespace=f"{case_id}:marker_velocity",
                        ),
                    )
                return {}

            iqn_runtime = HibmMpmMarkerVelocityRuntime(
                capture_step_state=capture_iqn_step_state,
                restore_step_state=restore_iqn_step_state,
                prepare_step=lambda _context: None,
                capture_marker_state=lambda: capture_marker_interface_state(
                    markers
                ),
                apply_marker_velocity_guess=apply_iqn_marker_velocity_guess,
                advance_trial=lambda _context, _trial_index: (
                    _run_hibm_mpm_coupling_trial()
                    or {
                        "flow_report": latest_flow_report,
                        "solid_step_report": latest_solid_step_report,
                        "feedback_report": latest_feedback_report,
                    }
                ),
                commit_case_step=commit_iqn_case_step,
                finalize_case_run=lambda: {},
                layout_identity=lambda: marker_layout_identity(
                    markers,
                    reference_positions_m=np.asarray(
                        marker_reference_positions_m,
                        dtype=np.float32,
                    ),
                    namespace=f"{case_id}:marker_velocity",
                ),
                begin_initial_guess_step=(
                    lambda context, carry_forward, layout_id: (
                        initial_guess_controller.begin_step(
                            carry_forward,
                            dt_s=float(context.dt_s),
                            layout_id=layout_id,
                        )
                    )
                ),
                accept_initial_guess_step=(
                    lambda _context, accepted, layout_id: (
                        initial_guess_controller.accept_step(
                            accepted,
                            layout_id=layout_id,
                        )
                    )
                ),
                discard_initial_guess_step=(
                    initial_guess_controller.discard_step
                ),
            )
            if (
                research_probe_config is not None
                and int(research_probe_config["target_step"]) == step_index + 1
            ):
                assert research_probe_oracle is not None
                assert initial_guess_controller is not None
                context = FsiStepContext(
                    step=step_index + 1,
                    step_index=step_index,
                    time_s=float(step_index + 1) * float(config.dt_s),
                    dt_s=float(config.dt_s),
                )
                accepted_base = capture_iqn_step_state()
                marker_base = capture_marker_interface_state(markers)
                layout_id = marker_layout_identity(
                    markers,
                    reference_positions_m=np.asarray(marker_reference_positions_m, dtype=np.float32),
                    namespace=f"{case_id}:marker_velocity",
                )
                preview_controller = deepcopy(initial_guess_controller)
                kalman_guess = preview_controller.begin_step(
                    marker_base["v_gamma_mps"],
                    dt_s=float(config.dt_s),
                    layout_id=layout_id,
                )
                preview_controller.discard_step()
                oracle_velocity = research_probe_oracle[step_index]
                probe_rows: list[dict[str, object]] = []
                probe_anchor_refresh_before = pressure_pair_anchor_runtime_refresh_count
                probe_started_s = time.perf_counter()
                try:
                    for alpha in research_probe_config["alphas"]:
                        restore_iqn_step_state(accepted_base, context)
                        probe_work_start = len(research_probe_trial_work_reports)
                        probe_solid_start = len(research_probe_solid_trial_reports)
                        mixed_guess = _mixed_iqn_kalman_oracle_guess(
                            kalman_guess, oracle_velocity, float(alpha)
                        )
                        probe_runtime = deepcopy(iqn_runtime)
                        probe_runtime._begin_initial_guess_step = (
                            lambda _context, _carry, _layout, guess=mixed_guess: guess.copy()
                        )
                        probe_runtime._accept_initial_guess_step = (
                            lambda _context, _accepted, _layout: None
                        )
                        probe_runtime._discard_initial_guess_step = lambda: None
                        research_probe_active = True
                        try:
                            _trial, coupling = solve_fsi_step(
                                probe_runtime,
                                context,
                                FsiCouplingConfig(
                                    max_iterations=int(config.fsi_coupling_max_iterations),
                                    relative_tolerance=float(config.fsi_coupling_relative_tolerance),
                                    absolute_tolerance_mps=float(config.fsi_coupling_absolute_tolerance_mps),
                                    initial_relaxation=float(config.iqn_initial_picard_relaxation),
                                    history_limit=int(config.iqn_history_limit),
                                    iqn_svd_relative_cutoff=float(config.iqn_svd_relative_cutoff),
                                ),
                            )
                            probe_rows.append({"alpha": float(alpha), "converged": bool(coupling.converged), "iterations": int(coupling.iterations), "relative_residual_history": list(coupling.relative_residual_history), "absolute_residual_history_mps": list(coupling.absolute_residual_history_mps), "candidate_velocity_rms_history_mps": list(coupling.candidate_velocity_rms_history_mps), "max_marker_residual_history_mps": list(coupling.max_marker_residual_history_mps), "effective_tolerance_history_mps": list(coupling.effective_tolerance_history_mps), "residual_to_effective_tolerance_history": list(coupling.residual_to_effective_tolerance_history), "update_mode_history": list(coupling.update_modes), "iqn_rank_history": list(coupling.iqn_rank_history), "iqn_condition_number_history": list(coupling.iqn_condition_number_history), "iqn_fallback_reasons": list(coupling.iqn_fallback_reasons), "iqn_fallback_count": int(coupling.iqn_fallback_count), "iqn_update_limited_history": list(coupling.iqn_update_limited_history), "trial_work": _fsi_trial_work_summary(research_probe_trial_work_reports[probe_work_start:]), "solid_trial_reports": research_probe_solid_trial_reports[probe_solid_start:]})
                        except FsiCouplingConvergenceError as error:
                            probe_rows.append({"alpha": float(alpha), "converged": False, "iterations": len(research_probe_trial_work_reports[probe_work_start:]), "error": repr(error), "trial_work": _fsi_trial_work_summary(research_probe_trial_work_reports[probe_work_start:]), "solid_trial_reports": research_probe_solid_trial_reports[probe_solid_start:]})
                        finally:
                            research_probe_active = False
                            probe_runtime.rollback_step(context)
                    restore_iqn_step_state(accepted_base, context)
                finally:
                    restore_iqn_step_state(accepted_base, context)
                return {
                    "case": case_id,
                    "status": "research_probe_terminal",
                    "config": asdict(config),
                    "history": history,
                    "research_probe_terminal": True,
                    "offline_oracle": True,
                    "deployable": False,
                    "accepted_step_count": step_index,
                    "accepted_time_s": float(step_index) * float(config.dt_s),
                    "research_probe_wall_time_s": time.perf_counter() - probe_started_s,
                    "research_probe_rows": probe_rows,
                    "research_probe_trial_work": research_probe_trial_work_reports,
                    "computed_result_sources": {
                        "research_probe_rows": (
                            "same accepted HostMacroStepState; uncommitted "
                            "solve_fsi_step Kalman-Oracle alpha sweep"
                        ),
                    },
                    "research_probe_anchor_refresh_delta": (
                        pressure_pair_anchor_runtime_refresh_count
                        - probe_anchor_refresh_before
                    ),
                }
            generic_run = solve_fsi_runtime(
                iqn_runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=float(config.dt_s),
                    completed_step_offset=step_index,
                    coupling=FsiCouplingConfig(
                        max_iterations=int(
                            config.fsi_coupling_max_iterations
                        ),
                        relative_tolerance=float(
                            config.fsi_coupling_relative_tolerance
                        ),
                        absolute_tolerance_mps=float(
                            config.fsi_coupling_absolute_tolerance_mps
                        ),
                        initial_relaxation=float(
                            config.iqn_initial_picard_relaxation
                        ),
                        history_limit=int(config.iqn_history_limit),
                        iqn_svd_relative_cutoff=float(
                            config.iqn_svd_relative_cutoff
                        ),
                        record_trial_vectors=record_iqn_trial_vectors,
                        iqn_reuse_previous_step_history=bool(
                            getattr(
                                config,
                                "iqn_reuse_previous_step_history",
                                False,
                            )
                        ),
                    ),
                ),
                prior_iqn_secant_history=prior_iqn_secant_history,
            )
            if bool(
                getattr(config, "iqn_reuse_previous_step_history", False)
            ):
                prior_iqn_secant_history = generic_run.next_iqn_secant_history
            if len(generic_run.history) != 1:
                raise RuntimeError(
                    "one ANSYS FSI physical step must commit exactly one "
                    "generic runtime history row"
                )
            generic_row = dict(generic_run.history[0])
            initial_guess_step_report = dict(
                initial_guess_controller.report()
            )
            coupling_step_report = {
                "hibm_coupling_scheme": "iterative_marker_velocity_iqn_ils",
                "hibm_iqn_reuse": {
                    "enabled": bool(generic_row["fsi_iqn_reuse_enabled"]),
                    "used": bool(generic_row["fsi_iqn_reuse_used"]),
                    "reset_reason": generic_row["fsi_iqn_reuse_reset_reason"],
                    "source_step": generic_row["fsi_iqn_reuse_source_step"],
                    "imported_pair_count": int(
                        generic_row["fsi_iqn_reuse_imported_pair_count"]
                    ),
                    "local_pair_count": int(
                        generic_row["fsi_iqn_reuse_local_pair_count"]
                    ),
                    "retained_pair_count": int(
                        generic_row["fsi_iqn_reuse_retained_pair_count"]
                    ),
                    "first_update_mode": generic_row[
                        "fsi_iqn_reuse_first_update_mode"
                    ],
                    "prior_initial_residual_norm": generic_row[
                        "fsi_iqn_reuse_prior_initial_residual_norm"
                    ],
                    "first_residual_norm": generic_row[
                        "fsi_iqn_reuse_first_residual_norm"
                    ],
                },
                "hibm_fsi_coupling_iterations_used": int(
                    generic_row["fsi_coupling_iterations"]
                ),
                "hibm_fsi_coupling_converged": bool(
                    generic_row["fsi_coupling_converged"]
                ),
                "hibm_fsi_coupling_explicit_single_pass": False,
                "hibm_fsi_coupling_rejected_trial_count": max(
                    int(generic_row["fsi_coupling_iterations"]) - 1,
                    0,
                ),
                "hibm_fsi_coupling_residual_source": (
                    "generic_marker_velocity_rms"
                ),
                "hibm_fsi_coupling_residual_norm_mps": float(
                    generic_row["fsi_coupling_absolute_residual_mps"]
                ),
                "hibm_fsi_coupling_tolerance_mps": float(
                    config.fsi_coupling_absolute_tolerance_mps
                ),
                "hibm_fsi_coupling_residual_history_mps": list(
                    generic_row[
                        "fsi_coupling_absolute_residual_history_mps"
                    ]
                ),
                "hibm_fsi_coupling_relative_residual_history": list(
                    generic_row["fsi_coupling_relative_residual_history"]
                ),
                "hibm_fsi_coupling_absolute_residual_history_mps": list(
                    generic_row[
                        "fsi_coupling_absolute_residual_history_mps"
                    ]
                ),
                "hibm_fsi_coupling_candidate_velocity_rms_history_mps": list(
                    generic_row[
                        "fsi_coupling_candidate_velocity_rms_history_mps"
                    ]
                ),
                "hibm_fsi_coupling_max_marker_residual_history_mps": list(
                    generic_row[
                        "fsi_coupling_max_marker_residual_history_mps"
                    ]
                ),
                "hibm_fsi_coupling_relative_tolerance_equivalent_history_mps": list(
                    generic_row[
                        "fsi_coupling_relative_tolerance_equivalent_history_mps"
                    ]
                ),
                "hibm_fsi_coupling_effective_tolerance_history_mps": list(
                    generic_row[
                        "fsi_coupling_effective_tolerance_history_mps"
                    ]
                ),
                "hibm_fsi_coupling_residual_to_effective_tolerance_history": list(
                    generic_row[
                        "fsi_coupling_residual_to_effective_tolerance_history"
                    ]
                ),
                "hibm_fsi_coupling_tolerance_history_mps": [],
                "hibm_fsi_coupling_relaxation_history": [],
                "hibm_fsi_coupling_update_mode_history": list(
                    generic_row["fsi_coupling_update_modes"]
                ),
                "hibm_fsi_coupling_iqn_rank_history": list(
                    generic_row["fsi_iqn_rank_history"]
                ),
                "hibm_fsi_coupling_iqn_condition_number_history": list(
                    generic_row["fsi_iqn_condition_number_history"]
                ),
                "hibm_fsi_coupling_iqn_fallback_count": int(
                    generic_row["fsi_iqn_fallback_count"]
                ),
                "hibm_fsi_coupling_first_relative_residual": float(
                    generic_row["fsi_coupling_first_relative_residual"]
                ),
                "hibm_fsi_coupling_first_absolute_residual_mps": float(
                    generic_row[
                        "fsi_coupling_first_absolute_residual_mps"
                    ]
                ),
                "hibm_fsi_coupling_base_assembly_count": (
                    1 if step_index == 0 else 0
                ),
            }
        step_trial_work_summary = _fsi_trial_work_summary(
            coupling_trial_work_reports[trial_work_start_index:]
        )
        expected_trial_count = int(
            coupling_step_report["hibm_fsi_coupling_iterations_used"]
        )
        if any(
            int(step_trial_work_summary[field]) != expected_trial_count
            for field in (
                "trial_count",
                "fluid_solve_count",
                "solid_macro_solve_count",
            )
        ):
            raise RuntimeError(
                "FSI trial-work ledger does not match coupling iterations: "
                f"expected={expected_trial_count}, "
                f"work={step_trial_work_summary}"
            )
        if bool(
            latest_feedback_constraint_report[
                "fluid_projection_consumed_feedback"
            ]
        ):
            fluid_projection_consumed_feedback_count += 1
        solid_step_execution_reports.append(dict(solid_substep_cfl))
        coupling_step_reports.append(dict(coupling_step_report))
        try:
            step_hibm_wall_times = _fsi_step_hibm_wall_times(
                latest_flow_report,
                latest_observer_topology_report,
            )
            step_artifact_export_wall_time_s = 0.0
            if coupling_mode == "direct_explicit":
                refresh_runtime_pressure_pair_anchors()
        except Exception:
            _discard_modified_physics_kalman_step(
                kalman_controller,
                kalman_raw_writeback_targets,
                fluid=fluid,
                solid=solid,
                markers=markers,
            )
            raise
        if kalman_controller is not None:
            kalman_hook_started_s = time.perf_counter()
            try:
                kalman_step_report = dict(kalman_controller.commit_step())
            except Exception:
                _restore_modified_physics_kalman_targets(
                    kalman_raw_writeback_targets,
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                )
                raise
            finally:
                kalman_adapter_wall_time_s += (
                    time.perf_counter() - kalman_hook_started_s
                )
            kalman_filter_wall_time_s = float(
                _kalman_controller_filter_wall_time_s(kalman_controller)
                - kalman_filter_wall_time_before_s
            )
            kalman_step_report["filter_wall_time_s"] = (
                kalman_filter_wall_time_s
            )
            kalman_step_report["state_transfer_wall_time_s"] = float(
                max(0.0, kalman_adapter_wall_time_s - kalman_filter_wall_time_s)
            )
            kalman_step_report["total_overhead_s"] = float(
                kalman_adapter_wall_time_s
            )
        feedback_available_for_projection = True
        (
            step_solid_positions_m,
            solid_position_snapshot_wall_time_s,
        ) = _capture_solid_positions_for_step(
            solid,
            profile_wall_time=profile_wall_time,
        )
        snapshot_capture_wall_time_s = float(
            math.fsum(
                (
                    snapshot_capture_wall_time_s,
                    solid_position_snapshot_wall_time_s,
                )
            )
        )
        step_displacement = _solid_displacement_report(
            solid,
            fixed_mask,
            tip_mask,
            rest=rest_positions_m,
            positions=step_solid_positions_m,
        )
        history.append(
            {
                "step": step_index + 1,
                **coupling_step_report,
                "initial_guess_report": dict(initial_guess_step_report),
                "initial_guess_mode_requested": initial_guess_mode,
                "initial_guess_mode_used": initial_guess_step_report.get(
                    "mode_used"
                ),
                "initial_guess_fallback_reason": initial_guess_step_report.get(
                    "fallback_reason"
                ),
                "initial_guess_prediction_rms_mps": (
                    initial_guess_step_report.get(
                        "last_prediction_rms_mps"
                    )
                ),
                "initial_guess_prediction_bias_mps": (
                    initial_guess_step_report.get("last_prediction_bias")
                ),
                "initial_guess_kalman_nis_mean": (
                    initial_guess_step_report.get("last_nis_mean")
                ),
                "hibm_fsi_trial_work_report": dict(
                    step_trial_work_summary
                ),
                "hibm_fsi_trial_cg_iterations_total": int(
                    step_trial_work_summary["cg_iterations_total"]
                ),
                "hibm_fsi_trial_flow_momentum_advection_substeps_total": int(
                    step_trial_work_summary[
                        "flow_momentum_advection_substeps_total"
                    ]
                ),
                "hibm_fsi_trial_flow_sst_transport_substeps_total": int(
                    step_trial_work_summary[
                        "flow_sst_transport_substeps_total"
                    ]
                ),
                "hibm_fsi_trial_solid_substeps_executed_total": int(
                    step_trial_work_summary[
                        "solid_substeps_executed_total"
                    ]
                ),
                "kalman_writeback_mode": kalman_writeback_mode,
                "kalman_modified_physics": bool(kalman_controller is not None),
                "kalman_step_report": kalman_step_report,
                "kalman_filter_overhead_s": float(
                    kalman_step_report.get("filter_wall_time_s", 0.0)
                ),
                "kalman_state_transfer_overhead_s": float(
                    kalman_step_report.get("state_transfer_wall_time_s", 0.0)
                ),
                "kalman_total_overhead_s": float(
                    kalman_step_report.get("total_overhead_s", 0.0)
                ),
                "kalman_projection_residual_state": (
                    "raw_pre_kalman_projection"
                ),
                "kalman_fluid_feedback_pressure_raw_min_pa": (
                    kalman_fluid_feedback_pressure_raw_min_pa
                ),
                "kalman_fluid_feedback_pressure_raw_max_pa": (
                    kalman_fluid_feedback_pressure_raw_max_pa
                ),
                "kalman_fluid_feedback_pressure_min_pa": (
                    kalman_fluid_feedback_pressure_min_pa
                ),
                "kalman_fluid_feedback_pressure_max_pa": (
                    kalman_fluid_feedback_pressure_max_pa
                ),
                "kalman_solid_integrator_raw_max_speed_mps": (
                    kalman_solid_integrator_raw_max_speed_mps
                ),
                "kalman_solid_accepted_max_speed_mps": (
                    kalman_solid_accepted_max_speed_mps
                ),
                "kalman_interface_raw_max_speed_mps": (
                    kalman_interface_raw_max_speed_mps
                ),
                "kalman_interface_accepted_max_speed_mps": (
                    kalman_interface_accepted_max_speed_mps
                ),
                "flow_wall_time_s": float(flow_wall_time_s),
                "snapshot_capture_wall_time_s": float(
                    snapshot_capture_wall_time_s
                ),
                "step_artifact_export_wall_time_s": (
                    step_artifact_export_wall_time_s
                ),
                **step_hibm_wall_times,
                "apply_marker_feedback_to_fluid": apply_feedback,
                "flow_driver_mode": latest_flow_report["flow_driver_mode"],
                "flow_driver_diagnostic_only": latest_flow_report[
                    "flow_driver_diagnostic_only"
                ],
                "flow_driver_uses_full_velocity_reset": latest_flow_report[
                    "flow_driver_uses_full_velocity_reset"
                ],
                "flow_full_field_reinitialized": latest_flow_report[
                    "flow_full_field_reinitialized"
                ],
                "flow_inlet_boundary_reapplied": latest_flow_report[
                    "flow_inlet_boundary_reapplied"
                ],
                "flow_volume_source_applied": latest_flow_report[
                    "flow_volume_source_applied"
                ],
                "flow_inlet_source_strength": float(
                    getattr(config, "flow_inlet_source_strength", 1.0)
                ),
                "flow_inlet_source_profile": str(
                    getattr(config, "flow_inlet_source_profile", "constant")
                ),
                "flow_inlet_source_ramp_steps": int(
                    getattr(config, "flow_inlet_source_ramp_steps", 0)
                ),
                "flow_inlet_source_schedule_scope": str(
                    getattr(config, "flow_inlet_source_schedule_scope", "global")
                ),
                "flow_inlet_source_factor": latest_flow_report[
                    "flow_inlet_source_factor"
                ],
                "flow_inlet_source_normal_velocity_mps": latest_flow_report[
                    "flow_inlet_source_normal_velocity_mps"
                ],
                "flow_pressure_outlet_enabled": bool(
                    getattr(config, "flow_pressure_outlet_enabled", True)
                ),
                "flow_outlet_balance_policy": str(
                    getattr(config, "flow_outlet_balance_policy", "report_only")
                ),
                "flow_predictor_applied": latest_flow_report[
                    "flow_predictor_applied"
                ],
                "flow_predictor_note": latest_flow_report["flow_predictor_note"],
                "flow_predictor_kinematic_viscosity_m2_s": latest_flow_report[
                    "flow_predictor_kinematic_viscosity_m2_s"
                ],
                "flow_predictor_no_slip_domain_walls": latest_flow_report[
                    "flow_predictor_no_slip_domain_walls"
                ],
                "flow_sst_transport_wall_time_s": float(
                    latest_flow_report.get(
                        "flow_sst_transport_wall_time_s", 0.0
                    )
                ),
                "flow_momentum_predictor_wall_time_s": float(
                    latest_flow_report.get(
                        "flow_momentum_predictor_wall_time_s", 0.0
                    )
                ),
                "flow_solid_boundary_mode": latest_flow_report[
                    "flow_solid_boundary_mode"
                ],
                "flow_pressure_outlet_backflow_policy": latest_flow_report[
                    "flow_pressure_outlet_backflow_policy"
                ],
                "hibm_sharp_marker_boundary_enabled": latest_flow_report[
                    "hibm_sharp_marker_boundary_enabled"
                ],
                "hibm_sharp_marker_boundary_search_reused": latest_flow_report[
                    "hibm_sharp_marker_boundary_search_reused"
                ],
                "hibm_sharp_marker_boundary_topology_reused": latest_flow_report[
                    "hibm_sharp_marker_boundary_topology_reused"
                ],
                "hibm_preassembly_overflow_singleton_cleanup_cell_count": (
                    latest_flow_report[
                        "hibm_preassembly_overflow_singleton_cleanup_cell_count"
                    ]
                ),
                "hibm_preassembly_overflow_singleton_cleanup_component_count": (
                    latest_flow_report[
                        "hibm_preassembly_overflow_singleton_cleanup_component_count"
                    ]
                ),
                "hibm_preassembly_tiny_unreached_cleanup_cell_count": (
                    latest_flow_report[
                        "hibm_preassembly_tiny_unreached_cleanup_cell_count"
                    ]
                ),
                "hibm_preassembly_tiny_unreached_cleanup_component_count": (
                    latest_flow_report[
                        "hibm_preassembly_tiny_unreached_cleanup_component_count"
                    ]
                ),
                "hibm_preassembly_tiny_unreached_cleanup_pass_count": (
                    latest_flow_report[
                        "hibm_preassembly_tiny_unreached_cleanup_pass_count"
                    ]
                ),
                "hibm_preassembly_remaining_unreached_cell_count": (
                    latest_flow_report[
                        "hibm_preassembly_remaining_unreached_cell_count"
                    ]
                ),
                "hibm_preassembly_cleanup_reused": latest_flow_report[
                    "hibm_preassembly_cleanup_reused"
                ],
                "hibm_preassembly_topology_mutated": latest_flow_report[
                    "hibm_preassembly_topology_mutated"
                ],
                "hibm_sharp_marker_boundary_near_node_count": latest_flow_report[
                    "hibm_sharp_marker_boundary_near_node_count"
                ],
                "hibm_sharp_marker_boundary_external_node_count": latest_flow_report[
                    "hibm_sharp_marker_boundary_external_node_count"
                ],
                "hibm_sharp_marker_boundary_internal_node_count": latest_flow_report[
                    "hibm_sharp_marker_boundary_internal_node_count"
                ],
                "hibm_sharp_marker_boundary_internal_obstacle_cell_count": (
                    latest_flow_report[
                        "hibm_sharp_marker_boundary_internal_obstacle_cell_count"
                    ]
                ),
                "hibm_sharp_marker_boundary_no_slip_rows": latest_flow_report[
                    "hibm_sharp_marker_boundary_no_slip_rows"
                ],
                **_hibm_velocity_dirichlet_mapping_fields(latest_flow_report),
                **_hibm_velocity_dirichlet_mapping_fields(
                    latest_observer_topology_report,
                    stage="observer",
                ),
                "hibm_sharp_marker_boundary_pressure_neumann_rows": (
                    latest_flow_report[
                        "hibm_sharp_marker_boundary_pressure_neumann_rows"
                    ]
                ),
                "hibm_sharp_marker_boundary_pressure_gradient_updated": (
                    latest_flow_report[
                        "hibm_sharp_marker_boundary_pressure_gradient_updated"
                    ]
                ),
                "hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_skipped_velocity_dirichlet_count"
                    ]
                ),
                "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count"
                    ]
                ),
                "hibm_pressure_neumann_skipped_obstacle_owner_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_skipped_obstacle_owner_count"
                    ]
                ),
                "hibm_pressure_neumann_relocated_obstacle_owner_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_relocated_obstacle_owner_count"
                    ]
                ),
                "hibm_pressure_neumann_duplicate_owner_count": latest_flow_report[
                    "hibm_pressure_neumann_duplicate_owner_count"
                ],
                "hibm_pressure_neumann_invalid_reconstruction_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_invalid_reconstruction_count"
                    ]
                ),
                "hibm_pressure_neumann_invalid_unreconstructable_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_invalid_unreconstructable_count"
                    ]
                ),
                "hibm_pressure_neumann_invalid_bad_marker_count": latest_flow_report[
                    "hibm_pressure_neumann_invalid_bad_marker_count"
                ],
                "hibm_pressure_neumann_invalid_nonpositive_volume_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_invalid_nonpositive_volume_count"
                    ]
                ),
                "flow_inlet_boundary_active_cell_count": latest_flow_report[
                    "flow_inlet_boundary_active_cell_count"
                ],
                "flow_inlet_boundary_obstacle_cell_count": latest_flow_report[
                    "flow_inlet_boundary_obstacle_cell_count"
                ],
                "flow_phase": latest_flow_report["flow_phase"],
                "flow_step_index_local": latest_flow_report[
                    "flow_step_index_local"
                ],
                "flow_step_index_global": latest_flow_report[
                    "flow_step_index_global"
                ],
                "flow_source_schedule_step_index": latest_flow_report[
                    "flow_source_schedule_step_index"
                ],
                "flow_source_schedule_scope": latest_flow_report[
                    "flow_source_schedule_scope"
                ],
                "flow_source_ramp_restarted_after_preflow": latest_flow_report[
                    "flow_source_ramp_restarted_after_preflow"
                ],
                "flow_reset_pressure_each_step": bool(
                    getattr(config, "flow_reset_pressure_each_step", False)
                ),
                "flow_pressure_reset_applied": latest_flow_report[
                    "flow_pressure_reset_applied"
                ],
                "flow_reinitialize_inlet_each_step": bool(
                    getattr(config, "flow_reinitialize_inlet_each_step", False)
                ),
                "fluid_recomputed": True,
                "fluid_recomputed_after_feedback": (
                    feedback_available_before_projection
                ),
                "feedback_available_before_projection": (
                    feedback_available_before_projection
                ),
                "fluid_projection_consumed_feedback": (
                    latest_feedback_constraint_report[
                        "fluid_projection_consumed_feedback"
                    ]
                ),
                "fluid_feedback_constraint_marker_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_marker_count"
                    ]
                ),
                "fluid_feedback_constraint_active_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_active_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_cleared_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_cleared_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_obstacle_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_obstacle_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_non_obstacle_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_non_obstacle_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_projection_participating_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_projection_participating_cell_count"
                    ]
                ),
                "fluid_marker_velocity_constraints_enabled": (
                    latest_feedback_constraint_report[
                        "fluid_marker_velocity_constraints_enabled"
                    ]
                ),
                "fluid_marker_velocity_constraint_active_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_marker_velocity_constraint_active_cell_count"
                    ]
                ),
                "fluid_marker_feedback_enforcement_mode": (
                    latest_feedback_constraint_report[
                        "fluid_marker_feedback_enforcement_mode"
                    ]
                ),
                **latest_dynamic_obstacle_report,
                "hibm_observer_topology_refreshed": bool(
                    latest_observer_topology_report.get(
                        "hibm_sharp_marker_boundary_enabled",
                        False,
                    )
                ),
                "hibm_observer_topology_near_node_count": int(
                    latest_observer_topology_report.get(
                        "hibm_sharp_marker_boundary_near_node_count",
                        0,
                    )
                ),
                "hibm_observer_topology_external_node_count": int(
                    latest_observer_topology_report.get(
                        "hibm_sharp_marker_boundary_external_node_count",
                        0,
                    )
                ),
                "hibm_observer_topology_internal_node_count": int(
                    latest_observer_topology_report.get(
                        "hibm_sharp_marker_boundary_internal_node_count",
                        0,
                    )
                ),
                "no_slip_residual_before_mps": latest_feedback_constraint_report[
                    "no_slip_residual_before_mps"
                ],
                "no_slip_residual_after_mps": latest_feedback_constraint_report[
                    "no_slip_residual_after_mps"
                ],
                "no_slip_target_residual_after_assembly_mps": (
                    latest_feedback_constraint_report[
                        "no_slip_target_residual_after_assembly_mps"
                    ]
                ),
                "no_slip_projected_residual_after_projection_mps": (
                    latest_feedback_constraint_report[
                        "no_slip_projected_residual_after_projection_mps"
                    ]
                ),
                "hibm_no_slip_valid_marker_count": latest_flow_report[
                    "hibm_no_slip_valid_marker_count"
                ],
                "hibm_no_slip_invalid_marker_count": latest_flow_report[
                    "hibm_no_slip_invalid_marker_count"
                ],
                "hibm_no_slip_max_residual_mps": latest_flow_report[
                    "hibm_no_slip_max_residual_mps"
                ],
                "hibm_no_slip_l2_residual_mps": latest_flow_report[
                    "hibm_no_slip_l2_residual_mps"
                ],
                "hibm_no_slip_direct_sample_marker_count": latest_flow_report[
                    "hibm_no_slip_direct_sample_marker_count"
                ],
                "hibm_no_slip_normal_walk_sample_marker_count": latest_flow_report[
                    "hibm_no_slip_normal_walk_sample_marker_count"
                ],
                "hibm_no_slip_nearest_fluid_sample_marker_count": latest_flow_report[
                    "hibm_no_slip_nearest_fluid_sample_marker_count"
                ],
                "hibm_no_slip_no_fluid_sample_marker_count": latest_flow_report[
                    "hibm_no_slip_no_fluid_sample_marker_count"
                ],
                "hibm_post_dirichlet_consistency_projection_count": (
                    latest_flow_report[
                        "hibm_post_dirichlet_consistency_projection_count"
                    ]
                ),
                "hibm_post_dirichlet_consistency_projection_applied": (
                    latest_flow_report[
                        "hibm_post_dirichlet_consistency_projection_applied"
                    ]
                ),
                "local_velocity_peak_mps": latest_flow_report[
                    "local_velocity_peak_mps"
                ],
                "fluid_speed_p99_mps": latest_flow_report["fluid_speed_p99_mps"],
                "fluid_speed_p999_mps": latest_flow_report["fluid_speed_p999_mps"],
                "pressure_min_pa": kalman_fluid_feedback_pressure_min_pa,
                "pressure_max_pa": kalman_fluid_feedback_pressure_max_pa,
                "projection_raw_pressure_min_pa": latest_flow_report[
                    "pressure_min_pa"
                ],
                "projection_raw_pressure_max_pa": latest_flow_report[
                    "pressure_max_pa"
                ],
                "flow_projection_report": latest_flow_report["projection_report"],
                **_flow_projection_report_fields(latest_flow_report),
                **_flow_source_report_fields(latest_flow_report),
                **_flow_transport_report_fields(latest_flow_report),
                "solid_substep_cfl_report": dict(solid_substep_cfl),
                "solid_substeps_selected": int(
                    latest_solid_step_report["solid_substeps_selected"]
                ),
                "solid_substep_dt_s": float(
                    latest_solid_step_report["solid_substep_dt_s"]
                ),
                "solid_estimated_cfl": float(
                    latest_solid_step_report["solid_estimated_cfl"]
                ),
                "solid_elastic_wave_speed_mps": float(
                    latest_solid_step_report["solid_elastic_wave_speed_mps"]
                ),
                "solid_max_particle_speed_mps": float(
                    latest_solid_step_report["solid_max_particle_speed_mps"]
                ),
                "solid_accepted_time_s": float(
                    latest_solid_step_report["solid_accepted_time_s"]
                ),
                "solid_remaining_unadvanced_time_s": float(
                    latest_solid_step_report["solid_remaining_unadvanced_time_s"]
                ),
                "solid_rejected_trial_count": int(
                    latest_solid_step_report["solid_rejected_trial_count"]
                ),
                "solid_retry_count": int(latest_solid_step_report["solid_retry_count"]),
                "solid_accepted_substep_count": int(
                    latest_solid_step_report["solid_accepted_substep_count"]
                ),
                "solid_substeps_executed_total": int(
                    latest_solid_step_report["solid_substeps_executed_total"]
                ),
                "solid_step_kernel_launch_count": int(
                    latest_solid_step_report["solid_step_kernel_launch_count"]
                ),
                "solid_selector_evaluation_count": int(
                    latest_solid_step_report[
                        "solid_selector_evaluation_count"
                    ]
                ),
                "solid_selector_device_to_host_scalar_read_count": int(
                    latest_solid_step_report[
                        "solid_selector_device_to_host_scalar_read_count"
                    ]
                ),
                "solid_packed_report_device_to_host_transfer_count": int(
                    latest_solid_step_report[
                        "solid_packed_report_device_to_host_transfer_count"
                    ]
                ),
                "solid_guard_batch_count": int(
                    latest_solid_step_report["solid_guard_batch_count"]
                ),
                "solid_wall_time_s": float(
                    latest_solid_step_report["solid_wall_time_s"]
                ),
                "solid_wall_time_synchronized": bool(
                    latest_solid_step_report["solid_wall_time_synchronized"]
                ),
                "solid_constitutive_model": str(
                    getattr(
                        config,
                        "solid_constitutive_model",
                        SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
                    )
                ),
                "solid_fixed_node_lock_policy": str(
                    getattr(config, "fixed_node_lock_policy", "any_fixed_particle")
                ),
                "solid_velocity_transfer_flip_blend": float(
                    getattr(config, "solid_velocity_transfer_flip_blend", 0.0)
                ),
                "stress_valid_marker_count": latest_stress_report.valid_marker_count,
                "stress_invalid_marker_count": (
                    latest_stress_report.invalid_marker_count
                ),
                **_marker_projection_boundary_report_fields(
                    markers,
                    traction_tip_cap_pressure_enabled=(
                        _traction_tip_cap_pressure_enabled(config)
                    ),
                    canonical_velocity_dirichlet_report=latest_flow_report.get(
                        "canonical_velocity_dirichlet_report"
                    ),
                ),
                "scatter_invalid_marker_count": (
                    latest_scatter_report.invalid_marker_count
                ),
                "feedback_invalid_marker_count": (
                    latest_feedback_report.invalid_marker_count
                ),
                "surface_feedback_preserve_marker_area": bool(
                    getattr(
                        config,
                        "preserve_marker_area_during_surface_feedback",
                        False,
                    )
                ),
                "surface_feedback_geometry_updated_marker_count": (
                    latest_feedback_report.geometry_updated_marker_count
                ),
                "surface_feedback_max_area_change_m2": (
                    latest_feedback_report.max_marker_area_change_m2
                ),
                "total_marker_force_n": latest_force_report.total_marker_force_n,
                **_marker_force_report_fields(latest_force_report),
                **_stress_sampling_report_fields(latest_stress_report),
                **_marker_traction_report_fields(
                    markers, include_face_diagnostics=False
                ),
                **anchor_install_report,
                **_scatter_report_fields(latest_scatter_report),
                "mpm_external_force_n": latest_solid_report.external_force_n,
                "mpm_primary_mean_velocity_mps": (
                    latest_solid_report.primary_mean_velocity_mps
                ),
                "mpm_secondary_mean_velocity_mps": (
                    latest_solid_report.secondary_mean_velocity_mps
                ),
                "mpm_primary_mean_displacement_m": (
                    latest_solid_report.primary_mean_displacement_m
                ),
                "mpm_secondary_mean_displacement_m": (
                    latest_solid_report.secondary_mean_displacement_m
                ),
                "mpm_active_grid_nodes": latest_solid_report.active_grid_nodes,
                "mpm_grid_out_of_bounds_particle_count": (
                    latest_solid_report.grid_out_of_bounds_particle_count
                ),
                "mpm_max_speed_mps": kalman_solid_accepted_max_speed_mps,
                "mpm_deformation_clamp_count": (
                    latest_solid_report.deformation_clamp_count
                ),
                "max_displacement_m": step_displacement["max_displacement_m"],
                "root_max_displacement_m": step_displacement[
                    "root_max_displacement_m"
                ],
                "tip_mean_displacement_m": step_displacement[
                    "tip_mean_displacement_m"
                ],
            }
        )
        if step_observer is not None:
            if observer_flow_snapshot is None:
                raise RuntimeError("step observer flow snapshot was not captured")
            (
                step_observer_snapshot,
                direct_snapshot_wall_time_s,
            ) = _measure_taichi_operation_wall_time(
                lambda: _direct_step_observer_snapshot(
                    observer_flow_snapshot,
                    solid,
                    markers,
                    solid_positions_m=step_solid_positions_m,
                    solid_rest_positions_m=rest_positions_m,
                    fixed_mask=fixed_mask,
                    tip_mask=tip_mask,
                ),
                enabled=profile_wall_time,
            )
            if record_iqn_trial_vectors:
                if accepted_iqn_trial_vectors is None:
                    raise RuntimeError(
                        "accepted IQN step did not produce a trial-vector trace"
                    )
                step_observer_snapshot = {
                    **step_observer_snapshot,
                    **accepted_iqn_trial_vectors,
                }
            snapshot_capture_wall_time_s = float(
                math.fsum(
                    (
                        snapshot_capture_wall_time_s,
                        direct_snapshot_wall_time_s,
                    )
                )
            )
            history[-1]["snapshot_capture_wall_time_s"] = (
                snapshot_capture_wall_time_s
            )
            observer_started_s = (
                time.perf_counter() if profile_wall_time else None
            )
            try:
                step_observer(
                    step_index + 1,
                    float(config.dt_s) * float(step_index + 1),
                    dict(history[-1]),
                    step_observer_snapshot,
                )
            finally:
                if observer_started_s is not None:
                    observer_elapsed_s = max(
                        0.0,
                        time.perf_counter() - observer_started_s,
                    )
                    history[-1]["step_artifact_export_wall_time_s"] = (
                        observer_elapsed_s
                    )

        _emit_run_progress(
            progress_observer,
            run_started_s=run_started_s,
            phase="fsi_step",
            step_completed=step_index + 1,
            time_s=float(config.dt_s) * float(step_index + 1),
            max_displacement_m=float(step_displacement["max_displacement_m"]),
        )
    coupling_iterations = [
        int(report["hibm_fsi_coupling_iterations_used"])
        for report in coupling_step_reports
    ]
    coupling_rejected_trial_total = sum(
        int(report["hibm_fsi_coupling_rejected_trial_count"])
        for report in coupling_step_reports
    )
    trial_work_run_summary = _fsi_trial_work_summary(
        coupling_trial_work_reports
    )
    coupling_iteration_summary = _fsi_coupling_iteration_summary(
        coupling_iterations
    )
    coupling_iterations_total = int(coupling_iteration_summary["total"])
    if any(
        int(trial_work_run_summary[field]) != coupling_iterations_total
        for field in (
            "trial_count",
            "fluid_solve_count",
            "solid_macro_solve_count",
        )
    ):
        raise RuntimeError(
            "run-level FSI trial-work ledger does not match coupling "
            f"iterations: iterations={coupling_iterations_total}, "
            f"work={trial_work_run_summary}"
        )
    coupling_run_summary = {
        "hibm_coupling_scheme": (
            "explicit_loose"
            if coupling_mode == "direct_explicit"
            else "iterative_marker_velocity_iqn_ils"
        ),
        "hibm_fsi_accepted_macro_step_count": len(coupling_step_reports),
        "hibm_fsi_coupling_iterations_total": coupling_iterations_total,
        "hibm_fsi_coupling_iterations_min": int(
            coupling_iteration_summary["minimum"]
        ),
        "hibm_fsi_coupling_iterations_max": int(
            coupling_iteration_summary["maximum"]
        ),
        "hibm_fsi_coupling_iterations_mean": float(
            coupling_iteration_summary["mean"]
        ),
        "hibm_fsi_coupling_iterations_median": float(
            coupling_iteration_summary["median"]
        ),
        "hibm_fsi_coupling_iterations_p95": float(
            coupling_iteration_summary["p95"]
        ),
        "hibm_fsi_coupling_rejected_trial_count_total": (
            coupling_rejected_trial_total
        ),
        "hibm_fsi_coupling_fluid_solve_count": int(
            trial_work_run_summary["fluid_solve_count"]
        ),
        "hibm_fsi_coupling_solid_macro_solve_count": int(
            trial_work_run_summary["solid_macro_solve_count"]
        ),
        "hibm_fsi_coupling_converged_step_count": sum(
            bool(report["hibm_fsi_coupling_converged"])
            for report in coupling_step_reports
        ),
        "hibm_fsi_coupling_explicit_single_pass_step_count": sum(
            bool(report["hibm_fsi_coupling_explicit_single_pass"])
            for report in coupling_step_reports
        ),
        "solid_trial_substeps_executed_total": sum(
            int(report.get("solid_substeps_executed_total", 0))
            for report in solid_trial_execution_reports
        ),
        "hibm_fsi_trial_work_report": dict(trial_work_run_summary),
        "hibm_fsi_trial_cg_iterations_total": int(
            trial_work_run_summary["cg_iterations_total"]
        ),
        "hibm_fsi_trial_flow_momentum_advection_substeps_total": int(
            trial_work_run_summary[
                "flow_momentum_advection_substeps_total"
            ]
        ),
        "hibm_fsi_trial_flow_sst_transport_substeps_total": int(
            trial_work_run_summary["flow_sst_transport_substeps_total"]
        ),
        "hibm_fsi_trial_solid_substeps_executed_total": int(
            trial_work_run_summary["solid_substeps_executed_total"]
        ),
        "hibm_fsi_trial_flow_wall_time_s_total": float(
            trial_work_run_summary["flow_wall_time_s_total"]
        ),
        "hibm_fsi_trial_hibm_wall_time_s_total": float(
            trial_work_run_summary["hibm_wall_time_s_total"]
        ),
        "hibm_fsi_trial_solid_wall_time_s_total": float(
            trial_work_run_summary["solid_wall_time_s_total"]
        ),
    }
    solid_substep_summary = _solid_substep_run_summary(
        solid_step_execution_reports
    )
    if config.step_count == 0 and preflow_history:
        return _preflow_only_report(
            case_id=case_id,
            case_metadata=case_metadata,
            boundary_conditions=boundary_conditions,
            reference_results=reference_results,
            config=config,
            markers=markers,
            solid=solid,
            fixed_mask=fixed_mask,
            tip_mask=tip_mask,
            solid_substep_cfl=solid_substep_cfl,
            solid_substep_summary=solid_substep_summary,
            preflow_report=preflow_report,
            runtime_identity=runtime_identity,
            profile_wall_time=profile_wall_time,
        )

    if (
        latest_stress_report is None
        or latest_force_report is None
        or latest_scatter_report is None
        or latest_solid_report is None
        or latest_feedback_report is None
        or latest_flow_report is None
        or latest_feedback_constraint_report is None
    ):
        raise RuntimeError("rectangular solid marker-MPM FSI smoke did not advance")

    displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
    reference_displacement = float(reference_results["max_displacement_m"])
    reference_velocity_peak = float(reference_results["local_velocity_peak_mps"])
    max_displacement = float(displacement["max_displacement_m"])
    local_velocity_peak_mps = float(latest_flow_report["local_velocity_peak_mps"])
    displacement_relative_error = (
        abs(max_displacement - reference_displacement) / reference_displacement
    )
    velocity_relative_error = (
        abs(local_velocity_peak_mps - reference_velocity_peak) / reference_velocity_peak
    )
    pressure_force_source = (
        "total_marker_force_n_pressure_only"
        if not _traction_include_viscous(config)
        else "total_marker_force_n_pressure_plus_viscous"
    )
    slab_diagnostics = slab_equivalence_diagnostics(
        config,
        interface_force_total_n=latest_force_report.total_marker_force_n,
        pressure_force_total_n=latest_force_report.total_marker_force_n,
        marker_total_area_m2=_marker_total_area_m2(markers),
        solid_mass_total_kg=latest_solid_report.total_mass_kg,
        max_displacement_m=max_displacement,
        pressure_force_source=pressure_force_source,
    )

    return {
        "case": case_id,
        "case_metadata": dict(case_metadata),
        "config": asdict(config),
        "taichi_runtime_identity": dict(runtime_identity),
        "profile_wall_time_enabled": bool(profile_wall_time),
        **coupling_run_summary,
        "initial_guess_mode": initial_guess_mode,
        "initial_guess_summary": (
            dict(initial_guess_controller.report())
            if initial_guess_controller is not None
            else {
                "mode": initial_guess_mode,
                "mode_used": "direct_explicit",
                "offline_oracle": False,
                "deployable": True,
                "begin_count": 0,
                "accepted_step_count": 0,
                "discard_count": 0,
            }
        ),
        "kalman_writeback_mode": kalman_writeback_mode,
        "kalman_modified_physics": bool(kalman_controller is not None),
        "kalman_summary": (
            kalman_controller.summary()
            if kalman_controller is not None
            else {
                "mode": "off",
                "modified_physics": False,
                "owners": {},
            }
        ),
        "kalman_filter_overhead_s_total": float(
            math.fsum(
                float(row.get("kalman_filter_overhead_s", 0.0))
                for row in history
            )
        ),
        "kalman_state_transfer_overhead_s_total": float(
            math.fsum(
                float(row.get("kalman_state_transfer_overhead_s", 0.0))
                for row in history
            )
        ),
        "kalman_total_overhead_s_total": float(
            math.fsum(
                float(row.get("kalman_total_overhead_s", 0.0))
                for row in history
            )
        ),
        "flow_solution_mode": FLOW_SOLUTION_MODE,
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        **slab_diagnostics,
        **preflow_report,
        "apply_marker_feedback_to_fluid": apply_feedback,
        "flow_driver_mode": flow_driver_mode,
        "flow_driver_diagnostic_only": (
            flow_driver_mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
        ),
        "flow_inlet_source_strength": float(
            getattr(config, "flow_inlet_source_strength", 1.0)
        ),
        "flow_inlet_source_profile": str(
            getattr(config, "flow_inlet_source_profile", "constant")
        ),
        "flow_inlet_source_ramp_steps": int(
            getattr(config, "flow_inlet_source_ramp_steps", 0)
        ),
        "flow_inlet_source_schedule_scope": str(
            getattr(config, "flow_inlet_source_schedule_scope", "global")
        ),
        "flow_pressure_outlet_enabled": bool(
            getattr(config, "flow_pressure_outlet_enabled", True)
        ),
        "flow_outlet_balance_policy": str(
            getattr(config, "flow_outlet_balance_policy", "report_only")
        ),
        "flow_reset_pressure_each_step": bool(
            getattr(config, "flow_reset_pressure_each_step", False)
        ),
        "flow_pressure_reset_applied": latest_flow_report[
            "flow_pressure_reset_applied"
        ],
        "flow_reinitialize_inlet_each_step": bool(
            getattr(config, "flow_reinitialize_inlet_each_step", False)
        ),
        "official_half_domain": _is_official_half_domain(case_metadata),
        "full_domain_two_flap": False,
        "flap_count_modeled": 1,
        "flap_count_displayed_after_symmetry_mirror": (
            2 if _is_official_half_domain(case_metadata) else 1
        ),
        "modeled_grid_nodes": list(config.grid_nodes),
        "display_grid_after_symmetry_mirror": _display_grid_after_symmetry_mirror(
            config,
            case_metadata,
        ),
        "flap_box_m": {
            "min": list(_solid_box(config)[0]),
            "max": list(_solid_box(config)[1]),
        },
        "marker_face_count": _traction_marker_face_count(config),
        "marker_count_per_face": int(config.marker_count),
        "marker_count_actual": int(markers.marker_count),
        "marker_projection_mode": (
            "segments" if int(markers.projection_segment_count) > 0 else "points"
        ),
        "marker_projection_segment_count": int(markers.projection_segment_count),
        **_marker_projection_boundary_report_fields(
            markers,
            traction_tip_cap_pressure_enabled=(
                _traction_tip_cap_pressure_enabled(config)
            ),
            canonical_velocity_dirichlet_report=latest_flow_report.get(
                "canonical_velocity_dirichlet_report"
            ),
        ),
        "flow_projection_iterations_actual": int(config.flow_projection_iterations),
        "solid_seeding_report": solid_seeding,
        "solid_substep_cfl_report": solid_substep_cfl,
        "solid_substeps_requested": solid_substep_cfl["solid_substeps_requested"],
        "solid_substeps_selected": solid_substep_cfl["solid_substeps_selected"],
        "solid_substeps_cfl_minimum": solid_substep_cfl[
            "solid_substeps_cfl_minimum"
        ],
        "solid_estimated_cfl": solid_substep_cfl["solid_estimated_cfl"],
        "solid_elastic_wave_speed_mps": solid_substep_cfl[
            "solid_elastic_wave_speed_mps"
        ],
        "solid_min_grid_spacing_m": solid_substep_cfl["solid_min_grid_spacing_m"],
        "solid_cfl_target": solid_substep_cfl["solid_cfl_target"],
        "solid_substep_dt_s": solid_substep_cfl["solid_substep_dt_s"],
        "solid_max_particle_speed_mps": solid_substep_cfl[
            "solid_max_particle_speed_mps"
        ],
        "solid_accepted_time_s": solid_substep_cfl["solid_accepted_time_s"],
        "solid_remaining_unadvanced_time_s": solid_substep_cfl[
            "solid_remaining_unadvanced_time_s"
        ],
        "solid_rejected_trial_count": solid_substep_cfl[
            "solid_rejected_trial_count"
        ],
        "solid_retry_count": solid_substep_cfl["solid_retry_count"],
        "solid_accepted_substep_count": solid_substep_cfl[
            "solid_accepted_substep_count"
        ],
        "solid_substeps_executed_last_step": solid_substep_cfl[
            "solid_substeps_executed_total"
        ],
        **solid_substep_summary,
        "computed_result_sources": {
            "pressure_pa": (
                "filtered fluid.fsi_pressure feedback state"
                if kalman_controller is not None
                and kalman_controller.enabled(
                    FLUID_FSI_PRESSURE_FEEDBACK_OWNER
                )
                else "fluid.fsi_pressure"
            ),
            "local_velocity_peak_mps": "max(norm(fluid.velocity))",
            "fluid_interface_force_n": "HIBM marker traction integral",
            "max_displacement_m": "solid.x-rest_x",
        },
        "boundary_conditions": dict(boundary_conditions),
        "reference_results": dict(reference_results),
        "flow_projection_report": latest_flow_report["projection_report"],
        "flow_phase": latest_flow_report["flow_phase"],
        "flow_step_index_local": latest_flow_report["flow_step_index_local"],
        "flow_step_index_global": latest_flow_report["flow_step_index_global"],
        "flow_source_schedule_step_index": latest_flow_report[
            "flow_source_schedule_step_index"
        ],
        "flow_source_schedule_scope": latest_flow_report["flow_source_schedule_scope"],
        "flow_source_ramp_restarted_after_preflow": latest_flow_report[
            "flow_source_ramp_restarted_after_preflow"
        ],
        **_flow_source_report_fields(latest_flow_report),
        **_flow_transport_report_fields(latest_flow_report),
        "flow_obstacle_cell_count": latest_flow_report["obstacle_cell_count"],
        "flow_fluid_cell_count": latest_flow_report["fluid_cell_count"],
        "computed_pressure_min_pa": kalman_fluid_feedback_pressure_min_pa,
        "computed_pressure_max_pa": kalman_fluid_feedback_pressure_max_pa,
        "projection_raw_pressure_min_pa": latest_flow_report["pressure_min_pa"],
        "projection_raw_pressure_max_pa": latest_flow_report["pressure_max_pa"],
        "pressure_sign_convention": latest_flow_report["pressure_sign_convention"],
        "local_velocity_peak_mps": local_velocity_peak_mps,
        "fluid_speed_p99_mps": latest_flow_report["fluid_speed_p99_mps"],
        "fluid_speed_p999_mps": latest_flow_report["fluid_speed_p999_mps"],
        "local_velocity_peak_relative_error": velocity_relative_error,
        "velocity_peak_tolerance": config.velocity_peak_tolerance,
        "fluid_recomputed_after_feedback": (
            fluid_projection_after_feedback_count > 0
        ),
        "feedback_closure_status": (
            "CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK"
            if fluid_projection_after_feedback_count > 0
            else "OPEN_LOOP_OR_PREFEEDBACK_ONLY"
        ),
        "fluid_recompute_count": fluid_projection_count,
        "fluid_projection_count": fluid_projection_count,
        "fluid_projection_after_feedback_count": (
            fluid_projection_after_feedback_count
        ),
        "fluid_projection_consumed_feedback_count": (
            fluid_projection_consumed_feedback_count
        ),
        "fluid_projection_consumed_feedback_trial_count": (
            fluid_projection_consumed_feedback_trial_count
        ),
        "fluid_projection_consumed_feedback": latest_feedback_constraint_report[
            "fluid_projection_consumed_feedback"
        ],
        "fluid_feedback_constraint_marker_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_marker_count"
            ]
        ),
        "fluid_feedback_constraint_active_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_active_cell_count"
            ]
        ),
        "fluid_feedback_constraint_cleared_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_cleared_cell_count"
            ]
        ),
        "fluid_feedback_constraint_obstacle_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_obstacle_cell_count"
            ]
        ),
        "fluid_feedback_constraint_non_obstacle_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_non_obstacle_cell_count"
            ]
        ),
        "fluid_feedback_constraint_projection_participating_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_projection_participating_cell_count"
            ]
        ),
        "fluid_marker_velocity_constraints_enabled": (
            latest_feedback_constraint_report[
                "fluid_marker_velocity_constraints_enabled"
            ]
        ),
        "fluid_marker_velocity_constraint_active_cell_count": (
            latest_feedback_constraint_report[
                "fluid_marker_velocity_constraint_active_cell_count"
            ]
        ),
        "no_slip_residual_before_mps": latest_feedback_constraint_report[
            "no_slip_residual_before_mps"
        ],
        "no_slip_residual_after_mps": latest_feedback_constraint_report[
            "no_slip_residual_after_mps"
        ],
        "no_slip_target_residual_after_assembly_mps": (
            latest_feedback_constraint_report[
                "no_slip_target_residual_after_assembly_mps"
            ]
        ),
        "no_slip_projected_residual_after_projection_mps": (
            latest_feedback_constraint_report[
                "no_slip_projected_residual_after_projection_mps"
            ]
        ),
        "stress_valid_marker_count": latest_stress_report.valid_marker_count,
        "stress_invalid_marker_count": latest_stress_report.invalid_marker_count,
        "two_sided_pressure_marker_count": (
            latest_stress_report.two_sided_pressure_marker_count
        ),
        "max_abs_traction_pa": latest_stress_report.max_abs_traction_pa,
        "total_marker_force_n": latest_force_report.total_marker_force_n,
        **_marker_force_report_fields(latest_force_report),
        **_stress_sampling_report_fields(latest_stress_report),
        **_marker_traction_report_fields(markers, include_face_diagnostics=True),
        **anchor_install_report,
        "scatter_invalid_marker_count": latest_scatter_report.invalid_marker_count,
        "scatter_active_marker_count": latest_scatter_report.active_marker_count,
        "scatter_active_particle_count": latest_scatter_report.active_pair_count,
        **_scatter_report_fields(latest_scatter_report),
        "mpm_external_force_n": latest_solid_report.external_force_n,
        "surface_feedback_updated_marker_count": (
            latest_feedback_report.updated_marker_count
        ),
        "surface_feedback_invalid_marker_count": (
            latest_feedback_report.invalid_marker_count
        ),
        "surface_feedback_max_marker_displacement_m": (
            latest_feedback_report.max_marker_displacement_m
        ),
        "final_stress_marker_diagnostics": markers.stress_marker_diagnostics(),
        "final_stress_face_diagnostics": markers.stress_face_diagnostics(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
            streamwise_axis_index=STREAMWISE_AXIS_INDEX,
            include_face_diagnostics=True,
        ),
        "pressure_pair_anchor_pair_map": pressure_pair_anchor_pair_map,
        **_fsi_profile_summary(history),
        "history": history,
        "max_displacement_m": max_displacement,
        "reference_max_displacement_m": reference_displacement,
        "max_displacement_relative_error": displacement_relative_error,
        "displacement_tolerance": config.displacement_tolerance,
        "final_flow_field_snapshot": (
            final_flow_field_snapshot
            if history and bool(getattr(config, "export_final_flow_snapshot", False))
            else {}
        ),
        **displacement,
    }


def _preflow_only_report(
    *,
    case_id: str,
    case_metadata: Mapping[str, Any],
    boundary_conditions: Mapping[str, Any],
    reference_results: Mapping[str, Any],
    config: Any,
    markers: HibmMpmSurfaceMarkers,
    solid: NeoHookeanMpmState,
    fixed_mask: np.ndarray,
    tip_mask: np.ndarray,
    solid_substep_cfl: Mapping[str, object],
    solid_substep_summary: Mapping[str, object],
    preflow_report: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    profile_wall_time: bool,
) -> dict[str, object]:
    preflow_history = list(preflow_report["preflow_history"])
    latest_preflow = dict(preflow_history[-1])
    projection_report = latest_preflow["flow_projection_report"]
    displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
    marker_force = tuple(latest_preflow["total_marker_force_n"])
    reference_velocity_peak = float(reference_results["local_velocity_peak_mps"])
    local_velocity_peak_mps = float(latest_preflow["local_velocity_peak_mps"])
    velocity_relative_error = (
        abs(local_velocity_peak_mps - reference_velocity_peak) / reference_velocity_peak
    )
    flow_driver_mode = _effective_flow_driver_mode(config)
    slab_diagnostics = slab_equivalence_diagnostics(
        config,
        interface_force_total_n=marker_force,
        pressure_force_total_n=marker_force,
        marker_total_area_m2=_marker_total_area_m2(markers),
        max_displacement_m=displacement["max_displacement_m"],
        pressure_force_source="preflow_total_marker_force_n_pressure_only",
    )
    return {
        "case": case_id,
        "case_metadata": dict(case_metadata),
        "config": asdict(config),
        "taichi_runtime_identity": dict(runtime_identity),
        "profile_wall_time_enabled": bool(profile_wall_time),
        "flow_solution_mode": FLOW_SOLUTION_MODE,
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        **slab_diagnostics,
        **preflow_report,
        "apply_marker_feedback_to_fluid": bool(
            getattr(config, "apply_marker_feedback_to_fluid", True)
        ),
        "flow_driver_mode": flow_driver_mode,
        "flow_driver_diagnostic_only": (
            flow_driver_mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
        ),
        "flow_inlet_source_strength": float(
            getattr(config, "flow_inlet_source_strength", 1.0)
        ),
        "flow_inlet_source_profile": str(
            getattr(config, "flow_inlet_source_profile", "constant")
        ),
        "flow_inlet_source_ramp_steps": int(
            getattr(config, "flow_inlet_source_ramp_steps", 0)
        ),
        "flow_inlet_source_schedule_scope": str(
            getattr(config, "flow_inlet_source_schedule_scope", "global")
        ),
        "flow_pressure_outlet_enabled": bool(
            getattr(config, "flow_pressure_outlet_enabled", True)
        ),
        "flow_outlet_balance_policy": str(
            getattr(config, "flow_outlet_balance_policy", "report_only")
        ),
        "flow_reset_pressure_each_step": bool(
            getattr(config, "flow_reset_pressure_each_step", False)
        ),
        "flow_pressure_reset_applied": latest_preflow["flow_pressure_reset_applied"],
        "flow_reinitialize_inlet_each_step": bool(
            getattr(config, "flow_reinitialize_inlet_each_step", False)
        ),
        "official_half_domain": _is_official_half_domain(case_metadata),
        "full_domain_two_flap": False,
        "flap_count_modeled": 1,
        "flap_count_displayed_after_symmetry_mirror": (
            2 if _is_official_half_domain(case_metadata) else 1
        ),
        "modeled_grid_nodes": list(config.grid_nodes),
        "display_grid_after_symmetry_mirror": _display_grid_after_symmetry_mirror(
            config,
            case_metadata,
        ),
        "flap_box_m": {
            "min": list(_solid_box(config)[0]),
            "max": list(_solid_box(config)[1]),
        },
        "marker_face_count": _traction_marker_face_count(config),
        "marker_count_per_face": int(config.marker_count),
        "marker_count_actual": int(markers.marker_count),
        "marker_projection_mode": (
            "segments" if int(markers.projection_segment_count) > 0 else "points"
        ),
        "marker_projection_segment_count": int(markers.projection_segment_count),
        **_marker_projection_boundary_report_fields(
            markers,
            traction_tip_cap_pressure_enabled=(
                _traction_tip_cap_pressure_enabled(config)
            ),
            canonical_velocity_dirichlet_report=latest_preflow.get(
                "canonical_velocity_dirichlet_report"
            ),
        ),
        "flow_projection_iterations_actual": int(config.flow_projection_iterations),
        "solid_substep_cfl_report": dict(solid_substep_cfl),
        "solid_substeps_requested": solid_substep_cfl["solid_substeps_requested"],
        "solid_substeps_selected": solid_substep_cfl["solid_substeps_selected"],
        "solid_substeps_cfl_minimum": solid_substep_cfl[
            "solid_substeps_cfl_minimum"
        ],
        "solid_estimated_cfl": solid_substep_cfl["solid_estimated_cfl"],
        "solid_elastic_wave_speed_mps": solid_substep_cfl[
            "solid_elastic_wave_speed_mps"
        ],
        "solid_min_grid_spacing_m": solid_substep_cfl["solid_min_grid_spacing_m"],
        "solid_cfl_target": solid_substep_cfl["solid_cfl_target"],
        **dict(solid_substep_summary),
        "computed_result_sources": {
            "pressure_pa": "fluid.fsi_pressure",
            "local_velocity_peak_mps": "max(norm(fluid.velocity))",
            "fluid_interface_force_n": "HIBM marker traction integral",
            "max_displacement_m": "solid.x-rest_x",
        },
        "boundary_conditions": dict(boundary_conditions),
        "reference_results": dict(reference_results),
        "flow_projection_report": projection_report,
        "flow_phase": latest_preflow["flow_phase"],
        "flow_step_index_local": latest_preflow["flow_step_index_local"],
        "flow_step_index_global": latest_preflow["flow_step_index_global"],
        "flow_source_schedule_step_index": latest_preflow[
            "flow_source_schedule_step_index"
        ],
        "flow_source_schedule_scope": latest_preflow["flow_source_schedule_scope"],
        "flow_source_ramp_restarted_after_preflow": latest_preflow[
            "flow_source_ramp_restarted_after_preflow"
        ],
        **_flow_source_report_fields(latest_preflow),
        **_flow_transport_report_fields(latest_preflow),
        "computed_pressure_min_pa": latest_preflow["pressure_min_pa"],
        "computed_pressure_max_pa": latest_preflow["pressure_max_pa"],
        "pressure_sign_convention": "fluid.fsi_pressure feedback field is sampled for reports and traction",
        "local_velocity_peak_mps": local_velocity_peak_mps,
        "fluid_speed_p99_mps": latest_preflow["fluid_speed_p99_mps"],
        "fluid_speed_p999_mps": latest_preflow["fluid_speed_p999_mps"],
        "local_velocity_peak_relative_error": velocity_relative_error,
        "velocity_peak_tolerance": config.velocity_peak_tolerance,
        "fluid_recomputed_after_feedback": False,
        "feedback_closure_status": "PREFLOW_ONLY_FIXED_SOLID",
        "fluid_recompute_count": int(preflow_report["preflow_steps_completed"]),
        "fluid_projection_count": int(preflow_report["preflow_steps_completed"]),
        "fluid_projection_after_feedback_count": 0,
        "fluid_projection_consumed_feedback_count": int(
            bool(latest_preflow["fluid_marker_velocity_constraints_enabled"])
        ),
        "fluid_projection_consumed_feedback_trial_count": 0,
        "fluid_projection_consumed_feedback": bool(
            latest_preflow["fluid_marker_velocity_constraints_enabled"]
        ),
        "fluid_feedback_constraint_marker_count": latest_preflow[
            "fluid_feedback_constraint_marker_count"
        ],
        "fluid_feedback_constraint_active_cell_count": latest_preflow[
            "fluid_feedback_constraint_active_cell_count"
        ],
        "fluid_feedback_constraint_cleared_cell_count": latest_preflow[
            "fluid_feedback_constraint_cleared_cell_count"
        ],
        "fluid_feedback_constraint_obstacle_cell_count": latest_preflow[
            "fluid_feedback_constraint_obstacle_cell_count"
        ],
        "fluid_feedback_constraint_non_obstacle_cell_count": latest_preflow[
            "fluid_feedback_constraint_non_obstacle_cell_count"
        ],
        "fluid_feedback_constraint_projection_participating_cell_count": latest_preflow[
            "fluid_feedback_constraint_projection_participating_cell_count"
        ],
        "fluid_marker_velocity_constraints_enabled": latest_preflow[
            "fluid_marker_velocity_constraints_enabled"
        ],
        "fluid_marker_velocity_constraint_active_cell_count": latest_preflow[
            "fluid_marker_velocity_constraint_active_cell_count"
        ],
        "no_slip_residual_before_mps": latest_preflow[
            "no_slip_residual_before_mps"
        ],
        "no_slip_residual_after_mps": latest_preflow["no_slip_residual_after_mps"],
        "no_slip_target_residual_after_assembly_mps": latest_preflow[
            "no_slip_target_residual_after_assembly_mps"
        ],
        "no_slip_projected_residual_after_projection_mps": latest_preflow[
            "no_slip_projected_residual_after_projection_mps"
        ],
        "stress_valid_marker_count": latest_preflow["stress_valid_marker_count"],
        "stress_invalid_marker_count": latest_preflow["stress_invalid_marker_count"],
        "two_sided_pressure_marker_count": latest_preflow[
            "two_sided_pressure_marker_count"
        ],
        "max_abs_traction_pa": latest_preflow.get("max_abs_traction_pa", ""),
        "one_sided_pressure_marker_count": latest_preflow.get(
            "one_sided_pressure_marker_count",
            "",
        ),
        "total_marker_force_n": marker_force,
        "fluid_reaction_force_n": tuple(latest_preflow["fluid_reaction_force_n"]),
        "fluid_reaction_force_z_N": latest_preflow["fluid_reaction_force_z_N"],
        "marker_force_z_N": latest_preflow["marker_force_z_N"],
        "marker_action_reaction_residual_n": latest_preflow[
            "marker_action_reaction_residual_n"
        ],
        "marker_action_reaction_residual_N": latest_preflow[
            "marker_action_reaction_residual_N"
        ],
        "primary_face_force_n": tuple(latest_preflow["primary_face_force_n"]),
        "secondary_face_force_n": tuple(latest_preflow["secondary_face_force_n"]),
        "primary_face_force_z_N": latest_preflow["primary_face_force_z_N"],
        "secondary_face_force_z_N": latest_preflow["secondary_face_force_z_N"],
        "primary_face_marker_count": latest_preflow["primary_face_marker_count"],
        "secondary_face_marker_count": latest_preflow["secondary_face_marker_count"],
        "primary_face_valid_marker_count": latest_preflow[
            "primary_face_valid_marker_count"
        ],
        "secondary_face_valid_marker_count": latest_preflow[
            "secondary_face_valid_marker_count"
        ],
        "primary_face_invalid_marker_count": latest_preflow[
            "primary_face_invalid_marker_count"
        ],
        "secondary_face_invalid_marker_count": latest_preflow[
            "secondary_face_invalid_marker_count"
        ],
        "scatter_invalid_marker_count": latest_preflow["scatter_invalid_marker_count"],
        "scatter_active_marker_count": latest_preflow["scatter_active_marker_count"],
        "scatter_active_particle_count": latest_preflow[
            "scatter_active_particle_count"
        ],
        "scatter_action_reaction_residual_n": latest_preflow[
            "scatter_action_reaction_residual_n"
        ],
        "scatter_action_reaction_residual_N": latest_preflow[
            "scatter_action_reaction_residual_N"
        ],
        "mpm_external_force_n": tuple(latest_preflow["mpm_external_force_n"]),
        "surface_feedback_updated_marker_count": 0,
        "surface_feedback_invalid_marker_count": 0,
        "surface_feedback_max_marker_displacement_m": 0.0,
        "history": [],
        "max_displacement_m": displacement["max_displacement_m"],
        "reference_max_displacement_m": float(reference_results["max_displacement_m"]),
        "max_displacement_relative_error": 1.0,
        "displacement_tolerance": config.displacement_tolerance,
        **displacement,
    }


def _traction_marker_layout(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_marker_layout",
            TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES,
        )
    )


def _traction_pressure_sampling_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_sampling_mode",
            TRACTION_PRESSURE_TWO_SIDED,
        )
    )


def _traction_marker_face_offset_cells(config: Any) -> float:
    return float(getattr(config, "traction_marker_face_offset_cells", 0.51))


def _traction_pressure_probe_origin_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_probe_origin_mode",
            TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION,
        )
    )


def _traction_pressure_probe_origin_offset_cells(config: Any) -> float | None:
    value = getattr(config, "traction_pressure_probe_origin_offset_cells", None)
    if value is None:
        return None
    return float(value)


def _traction_pressure_probe_start_offset_cells(config: Any) -> float | None:
    value = getattr(config, "traction_pressure_probe_start_offset_cells", None)
    if value is None:
        return None
    return float(value)


def _traction_pressure_probe_ladder_spacing_cells(config: Any) -> float:
    return float(getattr(config, "traction_pressure_probe_ladder_spacing_cells", 0.5))


def _traction_pressure_probe_ladder_rung_count(config: Any) -> int:
    return int(getattr(config, "traction_pressure_probe_ladder_rung_count", 5))


def _traction_pressure_probe_ladder_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_probe_ladder_mode",
            TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL,
        )
    )


def _traction_pressure_pair_policy(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_pair_policy",
            TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
        )
    )


def _traction_pressure_pair_max_cell_delta(config: Any) -> int:
    return int(getattr(config, "traction_pressure_pair_max_cell_delta", 1))


def _traction_pressure_pair_require_opposite_sides(config: Any) -> bool:
    return bool(getattr(config, "traction_pressure_pair_require_opposite_sides", True))


def _traction_one_sided_pressure_policy(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_one_sided_pressure_policy",
            TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED,
        )
    )


def _traction_one_sided_primary_fluid_side_normal_sign(config: Any) -> float | None:
    value = getattr(config, "traction_one_sided_primary_fluid_side_normal_sign", None)
    if value is None:
        return None
    return float(value)


def _traction_one_sided_secondary_fluid_side_normal_sign(config: Any) -> float | None:
    value = getattr(config, "traction_one_sided_secondary_fluid_side_normal_sign", None)
    if value is None:
        return None
    return float(value)


def _traction_one_sided_primary_reference_pressure_pa(config: Any) -> float:
    return float(getattr(config, "traction_one_sided_primary_reference_pressure_pa", 0.0))


def _traction_one_sided_secondary_reference_pressure_pa(config: Any) -> float:
    return float(
        getattr(config, "traction_one_sided_secondary_reference_pressure_pa", 0.0)
    )


def _traction_one_sided_pressure_pair_policy(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_one_sided_pressure_pair_policy",
            TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR,
        )
    )


def _traction_pressure_pair_anchor_markers_json(config: Any) -> str | None:
    value = getattr(config, "traction_pressure_pair_anchor_markers_json", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _traction_pressure_pair_runtime_provider_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_pair_runtime_provider_mode",
            TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_DISABLED,
        )
    )


def _traction_marker_face_count(config: Any) -> int:
    if _traction_marker_layout(config) == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE:
        return 1
    return 2


def _traction_include_viscous(config: Any) -> bool:
    return bool(getattr(config, "traction_include_viscous", False))


def _traction_tip_cap_pressure_enabled(config: Any) -> bool:
    return bool(getattr(config, "traction_tip_cap_pressure_enabled", False))


def _is_default_traction_formulation(config: Any) -> bool:
    return (
        _traction_marker_layout(config) == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and _traction_pressure_sampling_mode(config) == TRACTION_PRESSURE_TWO_SIDED
        and math.isclose(_traction_marker_face_offset_cells(config), 0.51)
        and not _traction_include_viscous(config)
        and _traction_pressure_probe_origin_mode(config)
        == TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION
        and _traction_pressure_probe_origin_offset_cells(config) is None
        and _traction_pressure_probe_start_offset_cells(config) is None
        and _traction_pressure_probe_ladder_mode(config)
        == TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL
        and _traction_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER
        and _traction_pressure_pair_max_cell_delta(config) == 1
        and _traction_pressure_pair_require_opposite_sides(config)
        and _traction_one_sided_pressure_policy(config)
        == TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED
        and _traction_one_sided_primary_fluid_side_normal_sign(config) is None
        and _traction_one_sided_secondary_fluid_side_normal_sign(config) is None
        and _traction_one_sided_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
        and _traction_pressure_pair_anchor_markers_json(config) is None
    )


def _is_selected_traction_formulation_coupled_smoke(config: Any) -> bool:
    if not bool(getattr(config, "allow_selected_traction_formulation_coupled_smoke", False)):
        return False
    max_selected_step_count = (
        250
        if bool(
            getattr(
                config,
                "allow_selected_traction_formulation_coupled_research_250",
                False,
            )
        )
        else (
            50
            if bool(
                getattr(
                    config,
                    "allow_selected_traction_formulation_coupled_long_validation",
                    False,
                )
            )
            else 10
        )
    )
    return (
        0 < int(getattr(config, "step_count", 0)) <= max_selected_step_count
        and _traction_marker_layout(config)
        == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and _traction_pressure_sampling_mode(config) == TRACTION_PRESSURE_ONE_SIDED
        and math.isclose(_traction_marker_face_offset_cells(config), 0.0)
        and not _traction_include_viscous(config)
        and _traction_pressure_probe_origin_mode(config)
        == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
        and math.isclose(
            float(_traction_pressure_probe_origin_offset_cells(config) or -1.0),
            0.51,
        )
        and _traction_pressure_probe_start_offset_cells(config) is None
        and _traction_pressure_probe_ladder_mode(config)
        == TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL
        and _traction_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
        and _traction_pressure_pair_max_cell_delta(config) == 1
        and _traction_pressure_pair_require_opposite_sides(config)
        and _traction_one_sided_pressure_policy(config)
        == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED
        and _traction_one_sided_primary_fluid_side_normal_sign(config) == 1.0
        and _traction_one_sided_secondary_fluid_side_normal_sign(config) == 1.0
        and _traction_one_sided_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
    )


def _traction_viscosity_pa_s(config: Any) -> float:
    if not _traction_include_viscous(config):
        return 0.0
    configured = float(
        getattr(
            config,
            "traction_viscosity_pa_s",
            getattr(config, "air_viscosity_pa_s", 0.0),
        )
    )
    if configured == 0.0:
        return float(getattr(config, "air_viscosity_pa_s", 0.0))
    return configured


def traction_formulation_supported(config: Any) -> tuple[bool, str]:
    marker_layout = _traction_marker_layout(config)
    pressure_sampling_mode = _traction_pressure_sampling_mode(config)
    if (
        marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
    ):
        if (
            _traction_one_sided_pressure_policy(config)
            == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED
        ):
            return True, "supported"
        return (
            False,
            "dual-face one-sided pressure requires "
            "traction_one_sided_pressure_policy='per_face_mirrored'",
        )
    if (
        marker_layout == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE
        and pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
    ):
        return (
            False,
            "single-mid one-sided pressure has ambiguous fluid side without "
            "explicit one_sided_fluid_side_normal_sign",
        )
    return True, "supported"


def _preflow_snapshot_paths(config: Any) -> tuple[object | None, object | None]:
    input_path = getattr(config, "preflow_snapshot_input_path", None)
    output_path = getattr(config, "preflow_snapshot_output_path", None)
    if input_path is not None and output_path is not None:
        raise ValueError(
            "preflow_snapshot_input_path and preflow_snapshot_output_path "
            "cannot both be set"
        )
    return input_path, output_path


def _validate_rectangular_solid_config(config: Any) -> None:
    coupling_mode = str(
        getattr(config, "coupling_mode", "direct_explicit")
    ).strip().lower()
    if coupling_mode not in {"direct_explicit", "iqn_ils"}:
        raise ValueError(f"unsupported coupling_mode: {coupling_mode!r}")
    fsi_max_iterations_raw = getattr(config, "fsi_coupling_max_iterations", 16)
    if isinstance(fsi_max_iterations_raw, (bool, np.bool_)):
        raise ValueError("fsi_max_iterations must be an integer")
    fsi_max_iterations = int(fsi_max_iterations_raw)
    if fsi_max_iterations != fsi_max_iterations_raw:
        raise ValueError("fsi_max_iterations must be an integer")
    if fsi_max_iterations <= 0:
        raise ValueError("fsi_coupling_max_iterations must be positive")
    initial_guess_mode = str(
        getattr(config, "initial_guess_mode", "carry_forward")
    ).strip().lower()
    if initial_guess_mode not in INITIAL_GUESS_MODES:
        raise ValueError(
            "initial_guess_mode must be carry_forward, linear_extrapolation, "
            "kalman, or oracle_replay"
        )
    absolute_tolerance = float(config.fsi_coupling_absolute_tolerance_mps)
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError(
            "fsi_coupling_absolute_tolerance_mps must be finite and non-negative"
        )
    relative_tolerance = float(config.fsi_coupling_relative_tolerance)
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError(
            "fsi_coupling_relative_tolerance must be finite and positive"
        )
    initial_relaxation = float(config.iqn_initial_picard_relaxation)
    if (
        not math.isfinite(initial_relaxation)
        or not 0.0 < initial_relaxation <= 1.0
    ):
        raise ValueError(
            "iqn_initial_picard_relaxation must be finite and in (0, 1]"
        )
    iqn_history_limit = getattr(config, "iqn_history_limit")
    if isinstance(iqn_history_limit, (bool, np.bool_)) or int(
        iqn_history_limit
    ) != iqn_history_limit or int(iqn_history_limit) <= 0:
        raise ValueError("iqn_history_limit must be a positive integer")
    svd_cutoff = float(config.iqn_svd_relative_cutoff)
    if not math.isfinite(svd_cutoff) or not 0.0 < svd_cutoff <= 1.0:
        raise ValueError("iqn_svd_relative_cutoff must be in (0, 1]")
    if coupling_mode == "direct_explicit":
        if initial_guess_mode != "carry_forward":
            raise ValueError(
                "direct_explicit requires initial_guess_mode='carry_forward'"
            )
    else:
        if fsi_max_iterations < 2:
            raise ValueError(
                "iqn_ils requires fsi_coupling_max_iterations >= 2"
            )
        if str(getattr(config, "kalman_writeback_mode", "off")) != "off":
            raise ValueError(
                "iqn_ils requires modified-physics Kalman writeback off"
            )
    initial_guess_kalman_config = getattr(
        config,
        "initial_guess_kalman_config",
        None,
    )
    initial_guess_oracle_path = getattr(
        config,
        "initial_guess_oracle_path",
        None,
    )
    if initial_guess_mode == "kalman":
        if initial_guess_kalman_config is None:
            raise ValueError(
                "initial_guess_mode='kalman' requires "
                "initial_guess_kalman_config"
            )
    elif initial_guess_kalman_config is not None:
        raise ValueError(
            "initial_guess_kalman_config is only valid for kalman mode"
        )
    if initial_guess_mode == "oracle_replay":
        if not isinstance(initial_guess_oracle_path, str) or not (
            initial_guess_oracle_path.strip()
        ):
            raise ValueError(
                "initial_guess_mode='oracle_replay' requires "
                "initial_guess_oracle_path"
            )
    elif initial_guess_oracle_path is not None:
        raise ValueError(
            "initial_guess_oracle_path is only valid for oracle_replay mode"
        )
    _iqn_kalman_oracle_interpolation_config(config)
    _modified_physics_kalman_configs(config)
    boundary_mode = str(
        getattr(
            config,
            "flow_solid_boundary_mode",
            FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS,
        )
    )
    if boundary_mode != FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS:
        raise ValueError(
            "flow_solid_boundary_mode must be "
            f"{FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS!r}; got {boundary_mode!r}"
        )

    _preflow_snapshot_paths(config)
    _preflow_traction_readiness_mode(config)
    solid_boundary_mode = _flow_solid_boundary_mode(config)
    if solid_boundary_mode not in FLOW_SOLID_BOUNDARY_MODES:
        raise ValueError(f"unsupported flow_solid_boundary_mode: {solid_boundary_mode!r}")
    dynamic_solid_volume_enabled = bool(
        getattr(config, "flow_hibm_dynamic_solid_volume_enabled", False)
    )
    if dynamic_solid_volume_enabled:
        if solid_boundary_mode != FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS:
            raise ValueError(
                "HIBM dynamic solid volume requires hibm_sharp_marker_rows"
            )
        if not bool(getattr(config, "update_fluid_obstacle_from_solid", False)):
            raise ValueError(
                "HIBM dynamic solid volume requires "
                "update_fluid_obstacle_from_solid=True"
            )
    tiny_unreached_cleanup_cells = int(
        getattr(
            config,
            "flow_hibm_tiny_unreached_cleanup_component_cells",
            0,
        )
    )
    if tiny_unreached_cleanup_cells < 0:
        raise ValueError(
            "flow_hibm_tiny_unreached_cleanup_component_cells must be non-negative"
        )
    for mode_field_name in ("flow_driver_mode", "preflow_flow_driver_mode"):
        flow_driver_mode = getattr(config, mode_field_name, None)
        if flow_driver_mode is None and mode_field_name == "preflow_flow_driver_mode":
            continue
        flow_driver_mode = str(
            FLOW_DRIVER_PROJECTION_ONLY
            if flow_driver_mode is None
            else flow_driver_mode
        )
        if not flow_driver_mode and mode_field_name == "preflow_flow_driver_mode":
            continue
        if flow_driver_mode not in SUPPORTED_FORMAL_FLOW_DRIVER_MODES:
            raise ValueError(f"unsupported {mode_field_name}: {flow_driver_mode!r}")
        if flow_driver_mode == FLOW_DRIVER_SHARP_REFERENCE:
            raise ValueError(
                "sharp_hibm_mpm_reference is reserved for a later sharp-path runner"
            )
    source_strength = float(getattr(config, "flow_inlet_source_strength", 1.0))
    if not math.isfinite(source_strength) or source_strength < 0.0:
        raise ValueError("flow_inlet_source_strength must be finite and non-negative")
    source_profile = str(getattr(config, "flow_inlet_source_profile", "constant"))
    if source_profile not in FLOW_INLET_SOURCE_PROFILES:
        raise ValueError(f"unsupported flow_inlet_source_profile: {source_profile!r}")
    source_scope = str(getattr(config, "flow_inlet_source_schedule_scope", "global"))
    if source_scope not in FLOW_INLET_SOURCE_SCHEDULE_SCOPES:
        raise ValueError(
            f"unsupported flow_inlet_source_schedule_scope: {source_scope!r}"
        )
    advection_scheme = str(getattr(config, "flow_advection_scheme", "euler")).lower()
    if advection_scheme not in FLOW_ADVECTION_SCHEMES:
        raise ValueError(f"unsupported flow_advection_scheme: {advection_scheme!r}")
    predictor_substeps = int(getattr(config, "flow_predictor_substeps", 1))
    if predictor_substeps <= 0:
        raise ValueError("flow_predictor_substeps must be positive")
    ymin_no_slip_rows = int(getattr(config, "flow_ymin_no_slip_rows", 0))
    if ymin_no_slip_rows < 0:
        raise ValueError("flow_ymin_no_slip_rows must be non-negative")
    obstacle_no_slip_layers = int(getattr(config, "flow_obstacle_no_slip_layers", 0))
    if obstacle_no_slip_layers < 0:
        raise ValueError("flow_obstacle_no_slip_layers must be non-negative")
    obstacle_no_slip_weight = float(getattr(config, "flow_obstacle_no_slip_weight", 1.0))
    if not math.isfinite(obstacle_no_slip_weight) or not 0.0 <= obstacle_no_slip_weight <= 1.0:
        raise ValueError("flow_obstacle_no_slip_weight must be in [0, 1]")
    obstacle_cap_no_slip_weight = getattr(
        config,
        "flow_obstacle_cap_no_slip_weight",
        None,
    )
    if obstacle_cap_no_slip_weight is not None:
        obstacle_cap_no_slip_weight = float(obstacle_cap_no_slip_weight)
        if (
            not math.isfinite(obstacle_cap_no_slip_weight)
            or not 0.0 <= obstacle_cap_no_slip_weight <= 1.0
        ):
            raise ValueError("flow_obstacle_cap_no_slip_weight must be in [0, 1]")
    obstacle_wake_no_slip_layers = int(
        getattr(config, "flow_obstacle_wake_no_slip_layers", 0)
    )
    if obstacle_wake_no_slip_layers < 0:
        raise ValueError("flow_obstacle_wake_no_slip_layers must be non-negative")
    obstacle_wake_no_slip_weight = float(
        getattr(config, "flow_obstacle_wake_no_slip_weight", 0.5)
    )
    if (
        not math.isfinite(obstacle_wake_no_slip_weight)
        or not 0.0 <= obstacle_wake_no_slip_weight <= 1.0
    ):
        raise ValueError("flow_obstacle_wake_no_slip_weight must be in [0, 1]")
    viscosity_multiplier = float(
        getattr(config, "flow_predictor_kinematic_viscosity_multiplier", 1.0)
    )
    if not math.isfinite(viscosity_multiplier) or viscosity_multiplier < 0.0:
        raise ValueError(
            "flow_predictor_kinematic_viscosity_multiplier must be finite and non-negative"
        )
    turbulence_model = str(getattr(config, "flow_turbulence_model", "laminar"))
    if turbulence_model not in FLOW_TURBULENCE_MODELS:
        raise ValueError(f"unsupported flow_turbulence_model: {turbulence_model!r}")
    if turbulence_model == "sst_2003":
        for field_name in (
            "flow_turbulence_intensity",
            "flow_backflow_turbulence_intensity",
        ):
            value = float(getattr(config, field_name, 0.05))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
        for field_name in (
            "flow_turbulent_viscosity_ratio",
            "flow_backflow_turbulent_viscosity_ratio",
        ):
            value = float(getattr(config, field_name, 10.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite")
        if int(getattr(config, "flow_sst_max_automatic_substeps", 4096)) <= 0:
            raise ValueError("flow_sst_max_automatic_substeps must be positive")
        near_wall_treatment = str(
            getattr(config, "flow_sst_near_wall_treatment", "resolved")
        ).lower()
        if near_wall_treatment not in FLOW_SST_NEAR_WALL_TREATMENTS:
            raise ValueError(
                "unsupported flow_sst_near_wall_treatment: "
                f"{near_wall_treatment!r}"
            )
        inlet_face = str(
            getattr(config, "flow_turbulence_inlet_face", "zmax")
        ).lower()
        outlet_face = str(
            getattr(config, "flow_turbulence_outlet_face", "zmin")
        ).lower()
        if inlet_face not in FLOW_PREDICTOR_NO_SLIP_WALL_INDEX:
            raise ValueError("flow_turbulence_inlet_face must name a physical face")
        if outlet_face not in FLOW_PREDICTOR_NO_SLIP_WALL_INDEX:
            raise ValueError("flow_turbulence_outlet_face must name a physical face")
        if inlet_face == outlet_face:
            raise ValueError("flow_turbulence_inlet_face and outlet face must differ")
    _flow_predictor_no_slip_domain_walls(config)
    _flow_symmetry_domain_walls(config)
    constraint_blend = float(getattr(config, "marker_velocity_constraint_blend", 1.0))
    if not math.isfinite(constraint_blend) or not 0.0 <= constraint_blend <= 1.0:
        raise ValueError("marker_velocity_constraint_blend must be in [0, 1]")
    constraint_mobility_ratio = float(
        getattr(config, "marker_velocity_constraint_solid_mobility_ratio", 0.0)
    )
    if not math.isfinite(constraint_mobility_ratio) or constraint_mobility_ratio < 0.0:
        raise ValueError(
            "marker_velocity_constraint_solid_mobility_ratio must be finite and non-negative"
        )
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if ramp_steps < 0:
        raise ValueError("flow_inlet_source_ramp_steps must be non-negative")
    outlet_policy = str(getattr(config, "flow_outlet_balance_policy", "report_only"))
    if outlet_policy not in FLOW_OUTLET_BALANCE_POLICIES:
        raise ValueError(f"unsupported flow_outlet_balance_policy: {outlet_policy!r}")
    pressure_outlet_backflow_policy = str(
        getattr(config, "flow_pressure_outlet_backflow_policy", "clamp")
    )
    if pressure_outlet_backflow_policy not in FLOW_PRESSURE_OUTLET_BACKFLOW_POLICIES:
        raise ValueError(
            "unsupported flow_pressure_outlet_backflow_policy: "
            f"{pressure_outlet_backflow_policy!r}"
        )
    obstacle_normal_velocity_policy = str(
        getattr(config, "flow_obstacle_normal_velocity_policy", "face_clamp")
    )
    if obstacle_normal_velocity_policy not in FLOW_OBSTACLE_NORMAL_VELOCITY_POLICIES:
        raise ValueError(
            "unsupported flow_obstacle_normal_velocity_policy: "
            f"{obstacle_normal_velocity_policy!r}"
        )
    marker_layout = _traction_marker_layout(config)
    if marker_layout not in TRACTION_MARKER_LAYOUTS:
        raise ValueError(f"unsupported traction_marker_layout: {marker_layout!r}")
    tip_cap_enabled = _traction_tip_cap_pressure_enabled(config)
    if marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
        if tip_cap_enabled:
            raise ValueError(
                "direct HIBM-MPM traction requires "
                "traction_tip_cap_pressure_enabled=False"
            )
    pressure_sampling_mode = _traction_pressure_sampling_mode(config)
    if pressure_sampling_mode not in TRACTION_PRESSURE_SAMPLING_MODES:
        raise ValueError(
            f"unsupported traction_pressure_sampling_mode: {pressure_sampling_mode!r}"
        )
    formulation_supported, formulation_reason = traction_formulation_supported(config)
    if not formulation_supported:
        raise ValueError(f"unsupported traction formulation: {formulation_reason}")
    marker_face_offset_cells = _traction_marker_face_offset_cells(config)
    if not math.isfinite(marker_face_offset_cells) or marker_face_offset_cells < 0.0:
        raise ValueError(
            "traction_marker_face_offset_cells must be finite and non-negative"
        )
    if marker_face_offset_cells > TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX:
        raise ValueError(
            "traction_marker_face_offset_cells is outside the fixed-solid "
            "diagnostic range"
        )
    probe_origin_mode = _traction_pressure_probe_origin_mode(config)
    if probe_origin_mode not in TRACTION_PRESSURE_PROBE_ORIGIN_MODES:
        raise ValueError(
            f"unsupported traction_pressure_probe_origin_mode: {probe_origin_mode!r}"
        )
    probe_origin_offset_cells = _traction_pressure_probe_origin_offset_cells(config)
    if probe_origin_offset_cells is not None:
        if (
            not math.isfinite(probe_origin_offset_cells)
            or probe_origin_offset_cells < 0.0
        ):
            raise ValueError(
                "traction_pressure_probe_origin_offset_cells must be finite "
                "and non-negative"
            )
        if (
            probe_origin_offset_cells
            > TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX
        ):
            raise ValueError(
                "traction_pressure_probe_origin_offset_cells is outside the "
                "diagnostic range"
            )
    if (
        probe_origin_mode == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
        and probe_origin_offset_cells is None
    ):
        raise ValueError(
            "traction_pressure_probe_origin_offset_cells is required for "
            "physical_face_offset probe origins"
        )
    probe_start_offset_cells = _traction_pressure_probe_start_offset_cells(config)
    if probe_start_offset_cells is not None:
        if (
            not math.isfinite(probe_start_offset_cells)
            or probe_start_offset_cells < 0.0
        ):
            raise ValueError(
                "traction_pressure_probe_start_offset_cells must be finite "
                "and non-negative"
            )
        if (
            probe_start_offset_cells
            > TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX
        ):
            raise ValueError(
                "traction_pressure_probe_start_offset_cells is outside the "
                "diagnostic range"
            )
    probe_ladder_spacing_cells = _traction_pressure_probe_ladder_spacing_cells(config)
    if (
        not math.isfinite(probe_ladder_spacing_cells)
        or probe_ladder_spacing_cells <= 0.0
    ):
        raise ValueError(
            "traction_pressure_probe_ladder_spacing_cells must be finite and positive"
        )
    probe_ladder_rung_count = _traction_pressure_probe_ladder_rung_count(config)
    if probe_ladder_rung_count <= 0:
        raise ValueError("traction_pressure_probe_ladder_rung_count must be positive")
    probe_ladder_mode = _traction_pressure_probe_ladder_mode(config)
    if probe_ladder_mode not in TRACTION_PRESSURE_PROBE_LADDER_MODES:
        raise ValueError(
            f"unsupported traction_pressure_probe_ladder_mode: {probe_ladder_mode!r}"
        )
    pressure_pair_policy = _traction_pressure_pair_policy(config)
    if pressure_pair_policy not in TRACTION_PRESSURE_PAIR_POLICIES:
        raise ValueError(
            f"unsupported traction_pressure_pair_policy: {pressure_pair_policy!r}"
        )
    runtime_pair_provider = _traction_pressure_pair_runtime_provider_mode(config)
    if runtime_pair_provider not in TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDERS:
        raise ValueError(
            "unsupported traction_pressure_pair_runtime_provider_mode: "
            f"{runtime_pair_provider!r}"
        )
    pressure_pair_max_cell_delta = _traction_pressure_pair_max_cell_delta(config)
    if pressure_pair_max_cell_delta < 0:
        raise ValueError("traction_pressure_pair_max_cell_delta must be non-negative")
    one_sided_policy = _traction_one_sided_pressure_policy(config)
    if one_sided_policy not in TRACTION_ONE_SIDED_PRESSURE_POLICIES:
        raise ValueError(
            f"unsupported traction_one_sided_pressure_policy: {one_sided_policy!r}"
        )
    one_sided_pair_policy = _traction_one_sided_pressure_pair_policy(config)
    if one_sided_pair_policy not in TRACTION_PRESSURE_PAIR_POLICIES:
        raise ValueError(
            "unsupported traction_one_sided_pressure_pair_policy: "
            f"{one_sided_pair_policy!r}"
        )
    primary_side_sign = _traction_one_sided_primary_fluid_side_normal_sign(config)
    secondary_side_sign = _traction_one_sided_secondary_fluid_side_normal_sign(config)
    primary_reference_pressure = _traction_one_sided_primary_reference_pressure_pa(config)
    secondary_reference_pressure = (
        _traction_one_sided_secondary_reference_pressure_pa(config)
    )
    if not math.isfinite(primary_reference_pressure):
        raise ValueError(
            "traction_one_sided_primary_reference_pressure_pa must be finite"
        )
    if not math.isfinite(secondary_reference_pressure):
        raise ValueError(
            "traction_one_sided_secondary_reference_pressure_pa must be finite"
        )
    if one_sided_policy == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED:
        if marker_layout != TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
            raise ValueError(
                "per_face_mirrored one-sided pressure requires dual_physical_faces"
            )
        if pressure_sampling_mode != TRACTION_PRESSURE_ONE_SIDED:
            raise ValueError(
                "per_face_mirrored one-sided pressure requires "
                "traction_pressure_sampling_mode='one_sided_surface_pressure'"
            )
        if primary_side_sign not in (-1.0, 1.0):
            raise ValueError(
                "traction_one_sided_primary_fluid_side_normal_sign must be -1.0 or 1.0"
            )
        if secondary_side_sign not in (-1.0, 1.0):
            raise ValueError(
                "traction_one_sided_secondary_fluid_side_normal_sign must be -1.0 or 1.0"
            )
        if one_sided_pair_policy != pressure_pair_policy:
            raise ValueError(
                "traction_one_sided_pressure_pair_policy must match "
                "traction_pressure_pair_policy for per-face diagnostics"
            )
    elif pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED:
        if marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
            raise ValueError(
                "dual-face one-sided pressure requires "
                "traction_one_sided_pressure_policy='per_face_mirrored'"
            )
    traction_viscosity = _traction_viscosity_pa_s(config)
    if not math.isfinite(traction_viscosity) or traction_viscosity < 0.0:
        raise ValueError("traction viscosity must be finite and non-negative")
    if config.step_count < 0:
        raise ValueError("step_count must be non-negative")
    if (
        config.step_count > 0
        and not _is_default_traction_formulation(config)
        and not _is_selected_traction_formulation_coupled_smoke(config)
    ):
        raise ValueError(
            "non-default traction formulations are fixed-solid diagnostics only"
        )
    anchor_markers_json = _traction_pressure_pair_anchor_markers_json(config)
    if (
        _is_selected_traction_formulation_coupled_smoke(config)
        and pressure_pair_policy == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
        and anchor_markers_json is None
        and runtime_pair_provider
        != TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR
    ):
        raise ValueError(
            "selected coupled smoke requires "
            "traction_pressure_pair_anchor_markers_json"
        )
    if (
        anchor_markers_json is not None
        and not _is_selected_traction_formulation_coupled_smoke(config)
    ):
        raise ValueError(
            "traction_pressure_pair_anchor_markers_json is selected coupled smoke only"
        )
    if config.step_count == 0 and int(getattr(config, "preflow_steps", 0)) <= 0:
        raise ValueError("step_count=0 is only valid for preflow-only diagnostics")
    if min(config.grid_nodes) < 4:
        raise ValueError("grid_nodes must be at least 4 in each direction")
    if min(config.solid_particle_counts) <= 0:
        raise ValueError("solid_particle_counts must be positive")
    if config.marker_count <= 0:
        raise ValueError("marker_count must be positive")
    if (
        marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and int(config.marker_count) < 2
    ):
        raise ValueError(
            "dual physical-face tip-cap projection requires at least two "
            "markers per face"
        )
    if config.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    _validated_solid_substep_override(config)
    _validated_solid_controller_integer(
        getattr(config, "solid_max_substep_retries", 3),
        field_name="solid_max_substep_retries",
        minimum=0,
    )
    _validated_solid_controller_integer(
        getattr(config, "solid_max_automatic_substeps", 65536),
        field_name="solid_max_automatic_substeps",
        minimum=1,
    )
    solid_clamp_limit = getattr(
        config, "solid_max_deformation_clamp_count_per_macro_step", None
    )
    if solid_clamp_limit is not None:
        _validated_solid_controller_integer(
            solid_clamp_limit,
            field_name="solid_max_deformation_clamp_count_per_macro_step",
            minimum=0,
        )
    solid_velocity_transfer_flip_blend = float(
        getattr(config, "solid_velocity_transfer_flip_blend", 0.0)
    )
    if not 0.0 <= solid_velocity_transfer_flip_blend <= 1.0:
        raise ValueError("solid_velocity_transfer_flip_blend must be in [0, 1]")
    if int(getattr(config, "preflow_steps", 0)) < 0:
        raise ValueError("preflow_steps must be non-negative")
    if float(getattr(config, "preflow_convergence_tolerance", 0.0)) < 0.0:
        raise ValueError("preflow_convergence_tolerance must be non-negative")
    preflow_convergence_mode = str(
        getattr(config, "preflow_convergence_mode", "single_step_legacy")
    )
    if preflow_convergence_mode not in {
        "single_step_legacy",
        "windowed_stationary",
    }:
        raise ValueError(
            f"unsupported preflow_convergence_mode: {preflow_convergence_mode!r}"
        )
    if int(getattr(config, "preflow_stationary_min_steps", 20)) < 0:
        raise ValueError("preflow_stationary_min_steps must be non-negative")
    if int(getattr(config, "preflow_stationary_window_steps", 10)) <= 0:
        raise ValueError("preflow_stationary_window_steps must be positive")
    if int(getattr(config, "preflow_stationary_consecutive_windows", 3)) <= 0:
        raise ValueError("preflow_stationary_consecutive_windows must be positive")
    for field_name in (
        "preflow_stationary_tolerance",
        "preflow_stationary_divergence_tolerance",
        "preflow_stationary_no_slip_tolerance_fraction",
    ):
        field_value = float(getattr(config, field_name, 0.05))
        if not 0.0 < field_value < 1.0:
            raise ValueError(f"{field_name} must be in (0, 1)")
    if float(getattr(config, "solid_cfl_target", DEFAULT_SOLID_CFL_TARGET)) <= 0.0:
        raise ValueError("solid_cfl_target must be positive")
    flap_streamwise_min_m = getattr(config, "flap_streamwise_min_m", None)
    flap_streamwise_max_m = getattr(config, "flap_streamwise_max_m", None)
    if (flap_streamwise_min_m is None) != (flap_streamwise_max_m is None):
        raise ValueError("flap streamwise bounds must be configured as a pair")
    if flap_streamwise_min_m is not None:
        if (
            float(flap_streamwise_min_m) < 0.0
            or float(flap_streamwise_max_m) > float(config.duct_length_m)
            or float(flap_streamwise_min_m) >= float(flap_streamwise_max_m)
        ):
            raise ValueError("flap streamwise bounds must lie inside the duct")
    if config.flow_projection_iterations <= 0:
        raise ValueError("flow_projection_iterations must be positive")
    if config.flow_cg_tolerance < 0.0:
        raise ValueError("flow_cg_tolerance must be non-negative")
    consistency_projection_iterations = int(
        getattr(
            config,
            "flow_post_dirichlet_consistency_projection_iterations",
            0,
        )
    )
    if consistency_projection_iterations < 0:
        raise ValueError(
            "flow_post_dirichlet_consistency_projection_iterations must be "
            "non-negative"
        )
    reprojection_iterations = getattr(config, "flow_reprojection_iterations", None)
    if reprojection_iterations is not None and int(reprojection_iterations) <= 0:
        raise ValueError("flow_reprojection_iterations must be positive when set")
    reprojection_tolerance = getattr(
        config,
        "flow_reprojection_cg_tolerance",
        None,
    )
    if reprojection_tolerance is not None and float(reprojection_tolerance) <= 0.0:
        raise ValueError(
            "flow_reprojection_cg_tolerance must be positive when set"
        )
    flow_cg_preconditioner = str(getattr(config, "flow_cg_preconditioner", "auto"))
    if flow_cg_preconditioner not in {
        "auto",
        "jacobi",
        "fv_multigrid",
        "fv_multigrid_light",
    }:
        raise ValueError(
            f"unsupported flow_cg_preconditioner: {flow_cg_preconditioner!r}"
        )
    pressure_solve_failure_policy = str(
        getattr(config, "flow_pressure_solve_failure_policy", "report")
    )
    if pressure_solve_failure_policy not in {"raise", "report"}:
        raise ValueError(
            "flow_pressure_solve_failure_policy must be 'raise' or 'report'; "
            f"got {pressure_solve_failure_policy!r}"
        )
    if config.flow_divergence_cleanup_iterations < 0:
        raise ValueError("flow_divergence_cleanup_iterations must be non-negative")
    if config.displacement_tolerance <= 0.0:
        raise ValueError("displacement_tolerance must be positive")
    if config.velocity_peak_tolerance <= 0.0:
        raise ValueError("velocity_peak_tolerance must be positive")
    if not (0.0 < config.poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in (0, 0.5)")
    solid_model = str(
        getattr(
            config,
            "solid_constitutive_model",
            SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
        )
    )
    if solid_model not in SOLID_CONSTITUTIVE_MODELS:
        raise ValueError(
            f"solid_constitutive_model must be one of "
            f"{sorted(SOLID_CONSTITUTIVE_MODELS)!r}; got {solid_model!r}"
        )


def _flow_solid_boundary_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "flow_solid_boundary_mode",
            FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS,
        )
    )


def _flow_pressure_outlet_backflow_policy(config: Any) -> str:
    return str(getattr(config, "flow_pressure_outlet_backflow_policy", "clamp"))


def _flow_obstacle_normal_velocity_policy(config: Any) -> str:
    return str(getattr(config, "flow_obstacle_normal_velocity_policy", "face_clamp"))


def _use_hibm_sharp_marker_boundary(config: Any) -> bool:
    return (
        _flow_solid_boundary_mode(config)
        == FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS
    )


def _domain_bounds(config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (0.0, 0.0, 0.0),
        (config.span_m, 0.5 * config.duct_height_m, config.duct_length_m),
    )


def _official_streamwise_to_solver_z(config: Any, streamwise_m: float) -> float:
    return float(config.duct_length_m) - float(streamwise_m)


def _solid_box(config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center_z = 0.5 * config.duct_length_m
    z_min = getattr(config, "flap_streamwise_min_m", None)
    z_max = getattr(config, "flap_streamwise_max_m", None)
    if z_min is None or z_max is None:
        z_min = center_z - 0.5 * config.flap_thickness_m
        z_max = center_z + 0.5 * config.flap_thickness_m
    else:
        solver_z_min = _official_streamwise_to_solver_z(config, z_max)
        solver_z_max = _official_streamwise_to_solver_z(config, z_min)
        z_min = min(solver_z_min, solver_z_max)
        z_max = max(solver_z_min, solver_z_max)
    root_y = 0.0
    return (
        (
            0.0,
            root_y,
            float(z_min),
        ),
        (
            config.span_m,
            root_y + config.flap_height_m,
            float(z_max),
        ),
    )


def _solid_mpm_bounds(
    config: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    domain_min, domain_max = _domain_bounds(config)
    solid_min, solid_max = _solid_box(config)
    base_dy = (float(domain_max[1]) - float(domain_min[1])) / float(
        config.grid_nodes[1]
    )
    pad_y = 3.0 * base_dy
    return (
        (
            float(domain_min[0]),
            min(float(domain_min[1]), float(solid_min[1]) - pad_y),
            float(domain_min[2]),
        ),
        (
            float(domain_max[0]),
            max(float(domain_max[1]), float(solid_max[1]) + pad_y),
            float(domain_max[2]),
        ),
    )


def _solid_mpm_grid_spacing_m(config: Any) -> tuple[float, float, float]:
    bounds_min, bounds_max = _solid_mpm_bounds(config)
    grid_nodes = tuple(int(value) for value in config.grid_nodes)
    return tuple(
        (float(max_value) - float(min_value)) / float(node_count)
        for min_value, max_value, node_count in zip(
            bounds_min,
            bounds_max,
            grid_nodes,
            strict=True,
        )
    )


def _lame_parameters(config: Any) -> tuple[float, float]:
    young = float(config.young_modulus_pa)
    nu = float(config.poisson_ratio)
    mu = young / (2.0 * (1.0 + nu))
    solid_model = str(
        getattr(
            config,
            "solid_constitutive_model",
            SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
        )
    )
    if solid_model == SOLID_CONSTITUTIVE_MODEL_PLANE_STRESS_LINEAR:
        lam = young * nu / (1.0 - nu * nu)
    else:
        lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def solid_substep_cfl_report(
    config: Any,
    *,
    max_particle_speed_mps: float = 0.0,
) -> dict[str, object]:
    """Select a full-macro-step solid substep count from accepted state."""
    requested_dt_s = float(config.dt_s)
    cfl_target = float(
        getattr(config, "solid_cfl_target", DEFAULT_SOLID_CFL_TARGET)
    )
    accepted_speed_mps = float(max_particle_speed_mps)
    if (
        not math.isfinite(requested_dt_s)
        or requested_dt_s <= 0.0
        or not math.isfinite(cfl_target)
        or cfl_target <= 0.0
        or not math.isfinite(accepted_speed_mps)
        or accepted_speed_mps < 0.0
    ):
        raise ValueError(
            "dt_s and solid_cfl_target must be positive finite and "
            "max_particle_speed_mps must be finite and non-negative"
        )
    try:
        mu, lam = _lame_parameters(config)
        density_kgm3 = float(config.solid_density_kgm3)
        wave_speed_mps = math.sqrt((lam + 2.0 * mu) / density_kgm3)
        min_spacing_m = min(_solid_mpm_grid_spacing_m(config))
    except (ArithmeticError, ValueError, OverflowError) as error:
        raise ValueError("invalid solid material or MPM spacing") from error
    if (
        not math.isfinite(wave_speed_mps)
        or wave_speed_mps <= 0.0
        or not math.isfinite(min_spacing_m)
        or min_spacing_m <= 0.0
    ):
        raise ValueError(
            "solid material wave speed and MPM spacing must be positive finite"
        )
    requested_substeps = _validated_solid_substep_override(config)
    cfl_minimum = max(
        1,
        int(
            math.ceil(
                (wave_speed_mps + accepted_speed_mps)
                * requested_dt_s
                / (cfl_target * min_spacing_m)
            )
        ),
    )
    selected_substeps = (
        cfl_minimum
        if requested_substeps is None
        else max(requested_substeps, cfl_minimum)
    )
    substep_dt_s = _validated_solid_substep_dt_s(
        requested_dt_s,
        selected_substeps,
    )
    estimated_cfl = (
        (wave_speed_mps + accepted_speed_mps)
        * substep_dt_s
        / min_spacing_m
    )
    return {
        "solid_substeps_requested": requested_substeps,
        "solid_substeps_mode": (
            "adaptive" if requested_substeps is None else "fixed_override"
        ),
        "solid_substeps_cfl_minimum": cfl_minimum,
        "solid_substeps_selected": selected_substeps,
        "solid_substeps_auto_applied": (
            requested_substeps is None or selected_substeps != requested_substeps
        ),
        "solid_elastic_wave_speed_mps": wave_speed_mps,
        "solid_accepted_max_particle_speed_mps": accepted_speed_mps,
        "solid_min_grid_spacing_m": min_spacing_m,
        "solid_cfl_target": cfl_target,
        "solid_estimated_cfl": estimated_cfl,
        "solid_substep_dt_s": substep_dt_s,
    }


def solid_seeding_report(config: Any) -> dict[str, object]:
    """Per-axis MPM particle spacing relative to the solid background grid.

    Explicit MPM loses inter-particle grid connectivity when particle
    spacing exceeds roughly one background cell: the quadratic B-spline
    stencils of adjacent particle layers stop sharing well-supported nodes,
    the body numerically fractures, and a fixed-root structure free-falls
    instead of bending (observed on the 2026-07-02 fine flap campaign:
    grid 4x256x320 with solid_particle_counts (1, 64, 12) put ~2 cells
    between wall-normal particle layers and ejected particles by step 30,
    while (1, 256, 20) on the same grid rings stably about the
    Euler-Bernoulli static deflection).

    The span axis (x) is excluded from the guard: the vertical-flap slab is
    a 2D-equivalent extrusion with an x-uniform solution, where a single
    particle column is intentional and empirically sound.
    """
    solid_min, solid_max = _solid_box(config)
    grid_spacing_m = _solid_mpm_grid_spacing_m(config)
    particle_counts = tuple(int(value) for value in config.solid_particle_counts)
    particle_spacing_m = tuple(
        (float(solid_max[axis]) - float(solid_min[axis]))
        / float(max(particle_counts[axis], 1))
        for axis in range(3)
    )
    spacing_cells = tuple(
        particle_spacing_m[axis] / grid_spacing_m[axis] for axis in range(3)
    )
    max_spacing_cells = float(
        getattr(config, "solid_seeding_max_spacing_cells", 1.5)
    )
    guard_enabled = bool(getattr(config, "enforce_solid_seeding_limit", False))
    guarded_axes = (1, 2)  # wall-normal (y) and streamwise (z); x is span
    worst_guarded_spacing_cells = max(
        spacing_cells[axis] for axis in guarded_axes
    )
    return {
        "solid_particle_spacing_m": particle_spacing_m,
        "solid_grid_spacing_m": grid_spacing_m,
        "solid_particle_spacing_cells": spacing_cells,
        "solid_seeding_guarded_axes": guarded_axes,
        "solid_seeding_worst_guarded_spacing_cells": worst_guarded_spacing_cells,
        "solid_seeding_max_spacing_cells": max_spacing_cells,
        "solid_seeding_guard_enabled": guard_enabled,
        "solid_seeding_guard_satisfied": (
            worst_guarded_spacing_cells <= max_spacing_cells
        ),
    }


def _enforce_solid_seeding_limit(config: Any) -> dict[str, object]:
    report = solid_seeding_report(config)
    if report["solid_seeding_guard_enabled"] and not report[
        "solid_seeding_guard_satisfied"
    ]:
        spacing_cells = report["solid_particle_spacing_cells"]
        raise ValueError(
            "solid particle seeding is too sparse for the MPM background "
            f"grid: particle spacing per cell (x, y, z) = "
            f"({spacing_cells[0]:.2f}, {spacing_cells[1]:.2f}, "
            f"{spacing_cells[2]:.2f}) exceeds "
            f"{report['solid_seeding_max_spacing_cells']:.2f} on a guarded "
            "axis (y/z); increase solid_particle_counts so adjacent particle "
            "layers stay within ~1 background cell, or disable "
            "enforce_solid_seeding_limit"
        )
    return report


def _solid_substep_velocity_damping(config: Any, *, solid_substeps: int) -> float:
    substeps = int(solid_substeps)
    if substeps <= 0:
        raise ValueError("solid_substeps must be positive")
    damping = float(getattr(config, "velocity_damping", 1.0))
    if damping < 0.0:
        raise ValueError("velocity_damping must be non-negative")
    if damping == 0.0 or substeps == 1:
        return damping
    return damping ** (1.0 / float(substeps))


def _grid_spacing_m(config: Any) -> tuple[float, float, float]:
    bounds_min, bounds_max = _domain_bounds(config)
    return tuple(
        (float(bounds_max[axis]) - float(bounds_min[axis]))
        / float(config.grid_nodes[axis])
        for axis in range(3)
    )


def _positive_finite(value: object, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _finite_vector3(
    value: tuple[float, float, float] | list[float] | np.ndarray | None,
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


def _divide_vector3(
    value: tuple[float, float, float],
    denominator: float,
) -> tuple[float, float, float]:
    return tuple(float(component) / denominator for component in value)


def _marker_total_area_m2(markers: HibmMpmSurfaceMarkers) -> float:
    marker_count = int(getattr(markers, "marker_count", 0))
    if marker_count <= 0:
        return 0.0
    return float(np.sum(markers.A_gamma_m2.to_numpy()[:marker_count]))


def _out_of_plane_boundary_policy(config: Any) -> str:
    symmetry_flags = _flow_symmetry_domain_walls(config)
    if symmetry_flags[0] and symmetry_flags[1]:
        return STRICT_OUT_OF_PLANE_BOUNDARY_POLICY
    return OUT_OF_PLANE_BOUNDARY_POLICY


def slab_equivalence_diagnostics(
    config: Any,
    *,
    interface_force_total_n: tuple[float, float, float]
    | list[float]
    | np.ndarray
    | None = None,
    pressure_force_total_n: tuple[float, float, float]
    | list[float]
    | np.ndarray
    | None = None,
    marker_total_area_m2: float | None = None,
    solid_mass_total_kg: float | None = None,
    max_displacement_m: float | None = None,
    flap_count: int = 1,
    marker_face_count: int | None = None,
    conceptual_coordinate_model: str = "cartesian-2d",
    runtime_discretization_model: str = "cartesian-3d-half-domain",
    out_of_plane_boundary_policy: str | None = None,
    pressure_force_source: str = "marker_traction_pressure_integral",
) -> dict[str, object]:
    resolved_out_of_plane_boundary_policy = (
        _out_of_plane_boundary_policy(config)
        if out_of_plane_boundary_policy is None
        else str(out_of_plane_boundary_policy)
    )
    out_of_plane_boundary_note = (
        STRICT_OUT_OF_PLANE_BOUNDARY_NOTE
        if resolved_out_of_plane_boundary_policy
        == STRICT_OUT_OF_PLANE_BOUNDARY_POLICY
        else OUT_OF_PLANE_BOUNDARY_NOTE
    )
    extrusion_depth_m = _positive_finite(
        getattr(config, "span_m"),
        field_name="span_m/extrusion_depth_m",
    )
    solid_min, solid_max = _solid_box(config)
    flap_height_m = _positive_finite(
        float(solid_max[1]) - float(solid_min[1]),
        field_name="flap_height_m",
    )
    flap_streamwise_thickness_m = _positive_finite(
        float(solid_max[2]) - float(solid_min[2]),
        field_name="flap_streamwise_thickness_m",
    )
    resolved_flap_count = int(flap_count)
    if resolved_flap_count <= 0:
        raise ValueError("flap_count must be positive")
    resolved_marker_face_count = (
        _traction_marker_face_count(config)
        if marker_face_count is None
        else int(marker_face_count)
    )
    if resolved_marker_face_count <= 0:
        raise ValueError("marker_face_count must be positive")
    expected_marker_total_area_m2 = (
        float(resolved_marker_face_count) * flap_height_m * extrusion_depth_m
    )
    resolved_marker_total_area_m2 = (
        expected_marker_total_area_m2
        if marker_total_area_m2 is None
        else float(marker_total_area_m2)
    )
    if (
        not math.isfinite(resolved_marker_total_area_m2)
        or resolved_marker_total_area_m2 < 0.0
    ):
        raise ValueError("marker_total_area_m2 must be finite and non-negative")

    expected_solid_volume_m3 = (
        float(resolved_flap_count)
        * extrusion_depth_m
        * flap_height_m
        * flap_streamwise_thickness_m
    )
    expected_solid_mass_kg = (
        expected_solid_volume_m3 * float(config.solid_density_kgm3)
    )
    resolved_solid_mass_kg = (
        expected_solid_mass_kg
        if solid_mass_total_kg is None
        else float(solid_mass_total_kg)
    )
    if not math.isfinite(resolved_solid_mass_kg) or resolved_solid_mass_kg < 0.0:
        raise ValueError("solid_mass_total_kg must be finite and non-negative")

    interface_force = _finite_vector3(
        interface_force_total_n,
        field_name="interface_force_total_n",
    )
    pressure_force = _finite_vector3(
        pressure_force_total_n,
        field_name="pressure_force_total_n",
    )
    interface_force_per_depth = _divide_vector3(interface_force, extrusion_depth_m)
    pressure_force_per_depth = _divide_vector3(pressure_force, extrusion_depth_m)

    displacement_value: float | str = (
        "" if max_displacement_m is None else float(max_displacement_m)
    )
    if displacement_value != "" and not math.isfinite(float(displacement_value)):
        raise ValueError("max_displacement_m must be finite")

    return {
        "conceptual_coordinate_model": str(conceptual_coordinate_model),
        "runtime_discretization_model": str(runtime_discretization_model),
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        "extrusion_depth_m": extrusion_depth_m,
        "extrusion_depth_source": "VerticalFlapFsiConfig.span_m",
        "span_m": extrusion_depth_m,
        "span_is_extrusion_depth": True,
        "flap_streamwise_thickness_m": flap_streamwise_thickness_m,
        "flap_streamwise_thickness_source": (
            "VerticalFlapFsiConfig.flap_thickness_m"
        ),
        "flap_thickness_is_streamwise_not_extrusion": True,
        "flap_count": resolved_flap_count,
        "marker_face_count": int(resolved_marker_face_count),
        "marker_total_area_m2": resolved_marker_total_area_m2,
        "marker_expected_total_area_m2": expected_marker_total_area_m2,
        "marker_total_area_per_depth_m": (
            resolved_marker_total_area_m2 / extrusion_depth_m
        ),
        "solid_volume_total_m3": expected_solid_volume_m3,
        "solid_volume_per_depth_m2": expected_solid_volume_m3 / extrusion_depth_m,
        "solid_mass_total_kg": resolved_solid_mass_kg,
        "solid_expected_mass_total_kg": expected_solid_mass_kg,
        "solid_mass_per_depth_kgpm": resolved_solid_mass_kg / extrusion_depth_m,
        "interface_force_total_n": interface_force,
        "interface_force_z_N": interface_force[2],
        "interface_force_per_depth_npm": interface_force_per_depth,
        "interface_force_z_per_depth_N_per_m": interface_force_per_depth[2],
        "pressure_force_total_n": pressure_force,
        "pressure_force_z_N": pressure_force[2],
        "pressure_force_per_depth_npm": pressure_force_per_depth,
        "pressure_force_z_per_depth_N_per_m": pressure_force_per_depth[2],
        "pressure_force_source": str(pressure_force_source),
        "max_displacement_m": displacement_value,
        "displacement_depth_scaling_expectation": (
            "depth_invariant_when_force_and_mass_scale_together"
        ),
        "out_of_plane_boundary_policy": resolved_out_of_plane_boundary_policy,
        "out_of_plane_boundary_residual_modeling_error": (
            resolved_out_of_plane_boundary_policy
            != STRICT_OUT_OF_PLANE_BOUNDARY_POLICY
        ),
        "out_of_plane_boundary_note": out_of_plane_boundary_note,
        "fluent_parity_claimed": False,
    }


def _is_official_half_domain(case_metadata: Mapping[str, Any]) -> bool:
    geometry = case_metadata.get("geometry", {})
    if not isinstance(geometry, Mapping):
        return False
    return geometry.get("modeled_domain") == "lower-symmetry-half"


def _display_grid_after_symmetry_mirror(
    config: Any,
    case_metadata: Mapping[str, Any],
) -> list[int]:
    grid = list(config.grid_nodes)
    if _is_official_half_domain(case_metadata):
        grid[1] *= 2
    return grid


def _materialize_symmetry_external_normal_faces(
    fluid: CartesianFluidSolver,
    config: Any,
) -> None:
    """Register exact zero-normal data on configured symmetry faces.

    The compact backward-MAC field has no storage row for a maximum-side
    physical face.  Post-projection symmetry copying therefore cannot by
    itself close that face for MUSCL transport: without a directed external
    normal, the transport ledger would reuse the last *internal* compact face.
    Keep the tangential components unprescribed so the existing symmetry
    kernels retain their zero-gradient/free-slip behavior.
    """

    symmetry_flags = _flow_symmetry_domain_walls(config)
    for boundary_face_index, active in enumerate(symmetry_flags):
        if not active:
            continue
        axis_index = boundary_face_index // 2
        side_index = boundary_face_index % 2
        fluid.refresh_external_velocity_boundary_face_uniform(
            axis_index=axis_index,
            side_index=side_index,
            target_velocity_mps=(0.0, 0.0, 0.0),
            active_component_mask=1 << axis_index,
        )


def _build_fluid(config: Any, runtime: TaichiRuntimeConfig) -> CartesianFluidSolver:
    bounds_min, bounds_max = _domain_bounds(config)
    fluid = CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            grid_nodes=config.grid_nodes,
            density_kgm3=config.air_density_kgm3,
            viscosity_pa_s=config.air_viscosity_pa_s,
            dt_s=config.dt_s,
        ),
        runtime=runtime,
    )
    if _use_hibm_sharp_marker_boundary(config):
        fluid.set_velocity_dirichlet_boundary_authority("canonical")
    fluid.obstacle.from_numpy(_initial_fluid_obstacle(config))
    if _flow_turbulence_model(config) == "sst_2003":
        fluid.configure_sst_2003(
            inlet_velocity_mps=float(config.inlet_velocity_mps),
            turbulence_intensity=float(
                getattr(config, "flow_turbulence_intensity", 0.05)
            ),
            turbulent_viscosity_ratio=float(
                getattr(config, "flow_turbulent_viscosity_ratio", 10.0)
            ),
            backflow_turbulence_intensity=float(
                getattr(config, "flow_backflow_turbulence_intensity", 0.05)
            ),
            backflow_turbulent_viscosity_ratio=float(
                getattr(
                    config,
                    "flow_backflow_turbulent_viscosity_ratio",
                    10.0,
                )
            ),
            inlet_face=str(
                getattr(config, "flow_turbulence_inlet_face", "zmax")
            ),
            outlet_face=str(
                getattr(config, "flow_turbulence_outlet_face", "zmin")
            ),
            no_slip_domain_walls=_flow_predictor_no_slip_domain_walls(config),
            near_wall_treatment=str(
                getattr(config, "flow_sst_near_wall_treatment", "resolved")
            ),
            max_automatic_substeps=int(
                getattr(config, "flow_sst_max_automatic_substeps", 4096)
            ),
            defer_wall_distance=_use_hibm_sharp_marker_boundary(config),
        )
    _materialize_symmetry_external_normal_faces(fluid, config)
    return fluid


def _cell_interval_overlaps(
    cell_min: float,
    cell_max: float,
    box_min: float,
    box_max: float,
) -> bool:
    return cell_min < box_max and cell_max > box_min


def _solid_obstacle(config: Any) -> np.ndarray:
    nx, ny, nz = config.grid_nodes
    bounds_min, bounds_max = _domain_bounds(config)
    solid_min, solid_max = _solid_box(config)
    dx = (bounds_max[0] - bounds_min[0]) / nx
    dy = (bounds_max[1] - bounds_min[1]) / ny
    dz = (bounds_max[2] - bounds_min[2]) / nz
    obstacle = np.zeros((nx, ny, nz), dtype=np.int32)
    for i in range(nx):
        x_min = bounds_min[0] + i * dx
        x_max = x_min + dx
        x_overlaps = _cell_interval_overlaps(x_min, x_max, solid_min[0], solid_max[0])
        for j in range(ny):
            y_min = bounds_min[1] + j * dy
            y_max = y_min + dy
            y_overlaps = _cell_interval_overlaps(y_min, y_max, solid_min[1], solid_max[1])
            for k in range(nz):
                z_min = bounds_min[2] + k * dz
                z_max = z_min + dz
                if (
                    x_overlaps
                    and y_overlaps
                    and _cell_interval_overlaps(z_min, z_max, solid_min[2], solid_max[2])
                ):
                    obstacle[i, j, k] = 1
    return obstacle


def _initial_fluid_obstacle(config: Any) -> np.ndarray:
    if _use_hibm_sharp_marker_boundary(config):
        return np.zeros(tuple(int(value) for value in config.grid_nodes), dtype=np.int32)
    return _solid_obstacle(config)


def _fluid_obstacle_update_disabled_report() -> dict[str, object]:
    return {
        "fluid_dynamic_obstacle_update_enabled": False,
        "fluid_dynamic_obstacle_is_hibm_solid_volume": False,
        "fluid_dynamic_obstacle_cell_count": "",
        "fluid_dynamic_obstacle_added_cell_count": "",
        "fluid_dynamic_obstacle_removed_cell_count": "",
    }


def _update_fluid_obstacle_from_solid(
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    config: Any,
) -> dict[str, object]:
    sharp_dynamic_volume = bool(
        _use_hibm_sharp_marker_boundary(config)
        and getattr(config, "flow_hibm_dynamic_solid_volume_enabled", False)
    )
    if not bool(getattr(config, "update_fluid_obstacle_from_solid", False)):
        return _fluid_obstacle_update_disabled_report()
    if _use_hibm_sharp_marker_boundary(config) and not sharp_dynamic_volume:
        return _fluid_obstacle_update_disabled_report()

    device_update = getattr(fluid, "update_dynamic_solid_obstacle_from_particles", None)
    if device_update is not None:
        solid_min, solid_max = _solid_box(config)
        particle_counts = tuple(int(value) for value in config.solid_particle_counts)
        particle_support_size_m = tuple(
            (float(solid_max[axis]) - float(solid_min[axis]))
            / float(particle_counts[axis])
            for axis in range(3)
        )
        report = device_update(
            solid.x,
            particle_count=solid.particle_count,
            particle_support_size_m=particle_support_size_m,
            particle_deformation_gradient=getattr(solid, "F", None),
            store_as_hibm_dynamic_solid_volume=sharp_dynamic_volume,
        )
        return {
            "fluid_dynamic_obstacle_update_enabled": True,
            "fluid_dynamic_obstacle_is_hibm_solid_volume": sharp_dynamic_volume,
            **report,
        }

    previous_obstacle = fluid.obstacle.to_numpy()
    obstacle = _solid_obstacle_from_mpm_particles(solid, config)
    if sharp_dynamic_volume:
        report = fluid.set_hibm_dynamic_solid_volume_obstacle_from_numpy(obstacle)
        return {
            "fluid_dynamic_obstacle_update_enabled": True,
            "fluid_dynamic_obstacle_is_hibm_solid_volume": True,
            **report,
        }
    velocity = fluid.velocity.to_numpy()
    velocity_prev = fluid.velocity_prev.to_numpy()
    solid_cells = obstacle != 0
    velocity[solid_cells] = 0.0
    velocity_prev[solid_cells] = 0.0
    fluid.obstacle.from_numpy(obstacle)
    fluid.velocity.from_numpy(velocity)
    fluid.velocity_prev.from_numpy(velocity_prev)
    return {
        "fluid_dynamic_obstacle_update_enabled": True,
        "fluid_dynamic_obstacle_is_hibm_solid_volume": False,
        "fluid_dynamic_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "fluid_dynamic_obstacle_added_cell_count": int(
            np.count_nonzero((obstacle != 0) & (previous_obstacle == 0))
        ),
        "fluid_dynamic_obstacle_removed_cell_count": int(
            np.count_nonzero((obstacle == 0) & (previous_obstacle != 0))
        ),
    }


def _solid_obstacle_from_mpm_particles(
    solid: NeoHookeanMpmState,
    config: Any,
) -> np.ndarray:
    nx, ny, nz = config.grid_nodes
    bounds_min, bounds_max = _domain_bounds(config)
    dx = (bounds_max[0] - bounds_min[0]) / nx
    dy = (bounds_max[1] - bounds_min[1]) / ny
    dz = (bounds_max[2] - bounds_min[2]) / nz
    positions = solid.x.to_numpy()[: solid.particle_count]
    rest = solid.rest_x.to_numpy()[: solid.particle_count]
    obstacle = np.zeros((nx, ny, nz), dtype=np.int32)
    if positions.size == 0:
        return obstacle

    solid_min, solid_max = _solid_box(config)
    row_height = float(config.flap_height_m) / float(config.solid_particle_counts[1])
    x_min = float(solid_min[0])
    x_max = float(solid_max[0])
    # Group by rest-y row so the deformed flap remains a continuous thin wall
    # instead of a cloud of isolated obstacle cells on coarse grids.
    row_count = int(config.solid_particle_counts[1])
    row_indices = np.clip(
        np.floor((rest[:, 1] - solid_min[1]) / max(row_height, 1.0e-12)).astype(int),
        0,
        row_count - 1,
    )

    row_particle_count = np.bincount(row_indices, minlength=row_count).astype(np.int32)
    active_rows = row_particle_count > 0
    y_sum = np.bincount(row_indices, weights=positions[:, 1], minlength=row_count)
    y_center = np.zeros(row_count, dtype=np.float64)
    y_center[active_rows] = y_sum[active_rows] / row_particle_count[active_rows]
    y_min = y_center - 0.5 * row_height
    y_max = y_center + 0.5 * row_height

    z_min = np.full(row_count, np.inf, dtype=np.float64)
    z_max = np.full(row_count, -np.inf, dtype=np.float64)
    np.minimum.at(z_min, row_indices, positions[:, 2])
    np.maximum.at(z_max, row_indices, positions[:, 2])
    # Keep at least the physical thickness represented even when all
    # particles in a row compress into the same streamwise cell.
    too_thin = active_rows & (
        (z_max - z_min) < 0.25 * float(config.flap_thickness_m)
    )
    z_mid = 0.5 * (z_min[too_thin] + z_max[too_thin])
    half_thickness = 0.5 * float(config.flap_thickness_m)
    z_min[too_thin] = z_mid - half_thickness
    z_max[too_thin] = z_mid + half_thickness

    x_cell_min = bounds_min[0] + np.arange(nx, dtype=np.float64) * dx
    y_cell_min = bounds_min[1] + np.arange(ny, dtype=np.float64) * dy
    z_cell_min = bounds_min[2] + np.arange(nz, dtype=np.float64) * dz
    x_overlap = (x_cell_min < x_max) & ((x_cell_min + dx) > x_min)
    y_overlap = (
        active_rows[:, None]
        & (y_cell_min[None, :] < y_max[:, None])
        & ((y_cell_min[None, :] + dy) > y_min[:, None])
    )
    z_overlap = (
        active_rows[:, None]
        & (z_cell_min[None, :] < z_max[:, None])
        & ((z_cell_min[None, :] + dz) > z_min[:, None])
    )
    yz_overlap = np.any(y_overlap[:, :, None] & z_overlap[:, None, :], axis=0)
    obstacle[x_overlap, :, :] = yz_overlap.astype(np.int32)
    return obstacle


def _initialize_inlet_flow(
    fluid: CartesianFluidSolver,
    config: Any,
) -> np.ndarray:
    nx, ny, nz = config.grid_nodes
    canonical_authority = fluid.velocity_dirichlet_boundary_authority == "canonical"
    obstacle = fluid.obstacle.to_numpy()
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    velocity[:, :, :, STREAMWISE_AXIS_INDEX] = -float(config.inlet_velocity_mps)
    velocity[obstacle != 0] = 0.0
    fluid.velocity.from_numpy(velocity)
    fluid.velocity_prev.from_numpy(velocity)

    active = np.zeros((nx, ny, nz), dtype=np.int32)
    values = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    weights = np.zeros((nx, ny, nz), dtype=np.float32)
    enforcement_weights = np.zeros((nx, ny, nz), dtype=np.float32)
    marker_regions = np.full((nx, ny, nz), -1, dtype=np.int32)
    hard_masks = np.zeros((nx, ny, nz), dtype=np.int32)
    external_exact_masks = np.zeros((nx, ny, nz), dtype=np.int32)
    owned_rows = np.zeros((nx, ny, nz), dtype=np.int32)
    _apply_ymin_no_slip_rows(
        active,
        values,
        weights,
        marker_regions,
        hard_masks,
        external_exact_masks,
        owned_rows,
        obstacle,
        config,
    )
    if not _use_hibm_sharp_marker_boundary(config):
        _apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )
    if not canonical_authority:
        # Legacy storage aliases the plus-side physical face onto the final
        # compact row.  Canonical storage has a separate directed zmax face;
        # its final compact row remains the backward internal MAC face.
        active[:, :, nz - 1] = 1
        values[:, :, nz - 1, STREAMWISE_AXIS_INDEX] = -float(
            config.inlet_velocity_mps
        )
        weights[:, :, nz - 1] = 1.0
        marker_regions[:, :, nz - 1] = -1
        hard_masks[:, :, nz - 1] = 0b111
        external_exact_masks[:, :, nz - 1] |= np.int32(0b100)
        owned_rows[:, :, nz - 1] = 0
    active[obstacle != 0] = 0
    values[obstacle != 0] = 0.0
    weights[obstacle != 0] = 0.0
    marker_regions[obstacle != 0] = -1
    hard_masks[obstacle != 0] = 0
    external_exact_masks[obstacle != 0] = 0
    owned_rows[obstacle != 0] = 0
    # Every row assembled here is direct/non-owned.  Later HIBM assembly
    # replaces its owned rows with the split pressure-alpha/enforcement ledger.
    enforcement_weights[...] = weights
    if canonical_authority:
        fluid._invalidate_velocity_dirichlet_component_ledger()
        # These scalar fields are diagnostics derived from the component
        # ledger under canonical authority. Reset them with the new boundary
        # state so a reinitialization cannot retain weights from an earlier
        # HIBM assembly or rejected FSI trial.
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        fluid.velocity_dirichlet_boundary_enforcement_weight.from_numpy(
            enforcement_weights
        )
        active_component_masks = np.where(active != 0, 0b111, 0).astype(np.int32)
        pressure_mobility = np.ones((nx, ny, nz, 3), dtype=np.float32)
        component_enforcement_weights = np.repeat(
            enforcement_weights[:, :, :, None],
            3,
            axis=3,
        ).astype(np.float32)
        component_regions = np.repeat(
            marker_regions[:, :, :, None],
            3,
            axis=3,
        ).astype(np.int32)
        owned_component_masks = np.where(
            owned_rows != 0,
            active_component_masks,
            0,
        ).astype(np.int32)
        for axis in range(3):
            axis_bit = np.int32(1 << axis)
            axis_active = (active_component_masks & axis_bit) != 0
            axis_hard = (hard_masks & axis_bit) != 0
            pressure_mobility[:, :, :, axis] = np.where(
                axis_active,
                np.where(axis_hard, 0.0, weights),
                1.0,
            ).astype(np.float32)
        fluid.velocity_dirichlet_boundary_active_component_mask.from_numpy(
            active_component_masks
        )
        # The value/hard/external fields above are three of the canonical eight
        # fields and intentionally remain the single shared storage authority.
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
            pressure_mobility
        )
        fluid.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
            component_enforcement_weights
        )
        fluid.velocity_dirichlet_boundary_component_region_id.from_numpy(
            component_regions
        )
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            hard_masks
        )
        fluid.velocity_dirichlet_boundary_external_exact_component_mask.from_numpy(
            external_exact_masks
        )
        fluid.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
            owned_component_masks
        )
        fluid.refresh_zmax_inlet_boundary_canonical(
            inlet_velocity_mps=float(config.inlet_velocity_mps),
            streamwise_axis_index=STREAMWISE_AXIS_INDEX,
        )
        _prepare_and_seal_canonical_velocity_dirichlet_component_ledger(fluid)
    else:
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        fluid.velocity_dirichlet_boundary_enforcement_weight.from_numpy(
            enforcement_weights
        )
        fluid.velocity_dirichlet_boundary_marker_region_id.from_numpy(marker_regions)
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            hard_masks
        )
        fluid.velocity_dirichlet_boundary_external_exact_component_mask.from_numpy(
            external_exact_masks
        )
        fluid.velocity_dirichlet_boundary_owned_row.from_numpy(owned_rows)
    zero_pressure = np.zeros((nx, ny, nz), dtype=np.float32)
    fluid.pressure.from_numpy(zero_pressure)
    fluid.fsi_pressure.from_numpy(zero_pressure)
    return obstacle


def _initialize_computed_flow(
    fluid: CartesianFluidSolver,
    config: Any,
) -> np.ndarray:
    return _initialize_inlet_flow(fluid, config)


def _project_current_flow(
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    reset_pressure: bool,
    pressure_solve_context: Mapping[str, object] | None = None,
    projection_iterations: int | None = None,
    cg_tolerance: float | None = None,
    accumulate_pressure_into_previous: bool = False,
    homogenize_pressure_interface_rhs_for_increment: bool = False,
    preserve_velocity_constraints: bool | None = None,
    velocity_dirichlet_soft_rows_already_applied: bool = False,
    pre_projection_velocity_projector: object | None = None,
    pressure_velocity_nullspace_projector: object | None = None,
) -> dict[str, object]:
    configured_velocity_inlet_zmax = getattr(
        config,
        "flow_projection_velocity_inlet_zmax",
        None,
    )
    if configured_velocity_inlet_zmax is not None and not isinstance(
        configured_velocity_inlet_zmax,
        (bool, np.bool_),
    ):
        raise ValueError(
            "flow_projection_velocity_inlet_zmax must be bool or None"
        )
    velocity_inlet_zmax = (
        None
        if configured_velocity_inlet_zmax is None
        else bool(configured_velocity_inlet_zmax)
    )
    effective_preserve_velocity_constraints = (
        bool(getattr(config, "preserve_marker_velocity_constraints", True))
        if preserve_velocity_constraints is None
        else bool(preserve_velocity_constraints)
    )
    project_tiny_unreached_cleanup_cells = int(
        getattr(
            config,
            "flow_hibm_tiny_unreached_cleanup_component_cells",
            0,
        )
    )
    pressure_outlet_enabled = bool(
        getattr(config, "flow_pressure_outlet_enabled", True)
    )
    sharp_reachability_prepared = bool(
        _use_hibm_sharp_marker_boundary(config) and pressure_outlet_enabled
    )
    if sharp_reachability_prepared:
        # The sharp path stabilizes topology before it assembles the current
        # pressure-interface matrix.  Repeating cleanup inside project() would
        # be both redundant and too late to rebuild those rows safely.
        project_tiny_unreached_cleanup_cells = 0
    projection_report = dict(
        fluid.project(
            iterations=(
                int(config.flow_projection_iterations)
                if projection_iterations is None
                else int(projection_iterations)
            ),
            pressure_outlet_zmin=pressure_outlet_enabled,
            pressure_outlet_backflow_policy=str(
                getattr(config, "flow_pressure_outlet_backflow_policy", "clamp")
            ),
            obstacle_normal_velocity_policy=str(
                getattr(config, "flow_obstacle_normal_velocity_policy", "face_clamp")
            ),
            preserve_velocity_constraints=effective_preserve_velocity_constraints,
            velocity_constraint_blend=float(
                getattr(config, "marker_velocity_constraint_blend", 1.0)
            ),
            velocity_constraint_solid_mobility_ratio=float(
                getattr(
                    config,
                    "marker_velocity_constraint_solid_mobility_ratio",
                    0.0,
                )
            ),
            reset_pressure=reset_pressure,
            accumulate_pressure_into_previous=bool(
                accumulate_pressure_into_previous
            ),
            homogenize_pressure_interface_rhs_for_increment=bool(
                homogenize_pressure_interface_rhs_for_increment
            ),
            pressure_solver=config.flow_pressure_solver,
            cg_tolerance=(
                float(config.flow_cg_tolerance)
                if cg_tolerance is None
                else float(cg_tolerance)
            ),
            cg_preconditioner=str(
                getattr(config, "flow_cg_preconditioner", "auto")
            ),
            pressure_solve_failure_policy=str(
                getattr(config, "flow_pressure_solve_failure_policy", "report")
            ),
            pressure_solve_context=pressure_solve_context,
            divergence_cleanup_iterations=config.flow_divergence_cleanup_iterations,
            hibm_tiny_unreached_cleanup_component_cells=(
                project_tiny_unreached_cleanup_cells
            ),
            hibm_pressure_reachability_prepared=sharp_reachability_prepared,
            velocity_dirichlet_soft_rows_already_applied=bool(
                velocity_dirichlet_soft_rows_already_applied
            ),
            pre_projection_velocity_projector=(
                pre_projection_velocity_projector
            ),
            pressure_velocity_nullspace_projector=(
                pressure_velocity_nullspace_projector
            ),
            velocity_inlet_zmax=velocity_inlet_zmax,
            dt_s=float(config.dt_s),
        )
    )
    symmetry_domain_walls = _flow_symmetry_domain_walls(config)
    if any(symmetry_domain_walls):
        fluid.apply_symmetry_domain_walls(symmetry_domain_walls)
    projection_report["flow_symmetry_domain_walls"] = [
        bool(flag) for flag in symmetry_domain_walls
    ]
    projection_report["velocity_dirichlet_soft_rows_already_applied"] = bool(
        velocity_dirichlet_soft_rows_already_applied
    )
    projection_report.update(
        fluid.pressure_outlet_fv_flux_report(dt_s=float(config.dt_s))
    )
    projection_report["fsi_pressure_snapshot_updated"] = bool(
        fluid.snapshot_pressure(preserve_if_current_is_zero=True)
    )
    return _flow_state_report(
        fluid,
        projection_report,
        include_percentiles=bool(getattr(config, "flow_report_include_percentiles", False)),
    )


def _combine_flow_projection_reports(
    projection_reports: list[dict[str, object]],
) -> dict[str, object]:
    """Combine a main projection and its pressure-increment corrections."""

    if not projection_reports:
        return {}
    combined = dict(projection_reports[-1])
    sum_keys = (
        "cg_project_calls",
        "cg_iterations_total",
        "cg_host_residual_checks",
        "cg_mean_host_reads",
        "cg_mean_projection_count",
        "cg_componentwise_mean_projection_count",
        "cg_unreached_set_mean_projection_count",
        "cg_breakdown_count",
        "cg_restart_count",
        "cg_multigrid_apply_count",
        "cg_multigrid_to_jacobi_fallback_count",
        "cg_exact_residual_confirmation_count",
        "velocity_dirichlet_boundary_apply_calls",
        "velocity_dirichlet_boundary_active_cells_total",
        "hibm_projection_overflow_singleton_cleanup_cell_count",
        "hibm_projection_overflow_singleton_cleanup_component_count",
        "hibm_projection_tiny_unreached_cleanup_cell_count",
        "hibm_projection_tiny_unreached_cleanup_component_count",
        "pressure_marker_nullspace_apply_count",
        "pressure_marker_nullspace_actuation_invalid_count",
        "pressure_marker_nullspace_correction_invalid_count",
        "pressure_marker_nullspace_operator_apply_count",
        "pressure_marker_nullspace_velocity_correction_apply_count",
    )
    for key in sum_keys:
        if any(key in report for report in projection_reports):
            combined[key] = sum(int(report.get(key, 0)) for report in projection_reports)
    max_keys = (
        "cg_iterations_max",
        "cg_initial_relative_residual_max",
        "cg_relative_residual_max",
        "cg_exact_relative_residual_max",
        "velocity_dirichlet_boundary_active_cells_max",
        "velocity_dirichlet_boundary_max_delta_mps",
        "hibm_unreached_component_rhs_mean_max_abs",
        "hibm_unreached_component_rhs_integral_max_abs",
        "pressure_nullspace_component_rhs_mean_max_abs",
        "pressure_nullspace_component_rhs_integral_max_abs",
        "pressure_marker_nullspace_max_input_constraint_mps",
        "pressure_marker_nullspace_max_unactuated_input_constraint_mps",
        "pressure_marker_nullspace_max_dependent_normalized_pivot",
        "pressure_marker_nullspace_max_constraint_residual_mps",
    )
    for key in max_keys:
        if any(key in report for report in projection_reports):
            combined[key] = max(
                float(report.get(key, 0.0)) for report in projection_reports
            )
    max_int_keys = (
        "hibm_unreached_incompatible_component_count",
        "cg_unreached_component_count",
        "cg_unreached_component_raw_count",
        "cg_unreached_component_largest_cell_count",
        "cg_unreached_component_singleton_count",
        "cg_unreached_component_small_count",
        "cg_unreached_component_small_cell_count",
        "pressure_nullspace_incompatible_component_count",
        "pressure_marker_nullspace_active_constraint_count",
        "pressure_marker_nullspace_independent_constraint_count",
        "pressure_marker_nullspace_dependent_constraint_count",
        "pressure_marker_nullspace_unactuated_constraint_count",
        "pressure_marker_nullspace_pressure_actuation_generation",
        "pressure_marker_nullspace_solver_scratch_resource_bytes",
        "pressure_marker_nullspace_marker_operator_resource_bytes",
        "pressure_marker_nullspace_resource_bytes",
    )
    for key in max_int_keys:
        if any(key in report for report in projection_reports):
            combined[key] = max(
                int(report.get(key, 0)) for report in projection_reports
            )
    active_cells_key = "velocity_dirichlet_boundary_active_cells_total"
    mean_delta_key = "velocity_dirichlet_boundary_mean_delta_mps"
    if any(mean_delta_key in report for report in projection_reports):
        active_cells_total = sum(
            int(report.get(active_cells_key, 0)) for report in projection_reports
        )
        combined[mean_delta_key] = (
            sum(
                float(report.get(mean_delta_key, 0.0))
                * int(report.get(active_cells_key, 0))
                for report in projection_reports
            )
            / float(active_cells_total)
            if active_cells_total > 0
            else 0.0
        )
    momentum_key = "velocity_dirichlet_boundary_momentum_delta_n_s"
    if any(momentum_key in report for report in projection_reports):
        combined[momentum_key] = tuple(
            sum(
                float(report.get(momentum_key, (0.0, 0.0, 0.0))[axis])
                for report in projection_reports
            )
            for axis in range(3)
        )
    if any("cg_converged_all" in report for report in projection_reports):
        combined["cg_converged_all"] = all(
            bool(report.get("cg_converged_all", True))
            for report in projection_reports
        )
    for report_key, aggregate_key in (
        (
            "pre_projection_velocity_projector_prepared",
            "pre_projection_velocity_projector_prepared_all",
        ),
        (
            "pre_projection_velocity_projector_converged",
            "pre_projection_velocity_projector_converged_all",
        ),
        (
            "pre_projection_velocity_projector_committed",
            "pre_projection_velocity_projector_committed_all",
        ),
        (
            "pressure_marker_nullspace_enabled",
            "pressure_marker_nullspace_enabled_all",
        ),
        (
            "pressure_marker_nullspace_prepared",
            "pressure_marker_nullspace_prepared_all",
        ),
        (
            "pressure_marker_nullspace_all_velocity_paths_projected",
            "pressure_marker_nullspace_all_velocity_paths_projected_all",
        ),
    ):
        combined[aggregate_key] = all(
            bool(
                report.get(
                    aggregate_key,
                    report.get(report_key, False),
                )
            )
            for report in projection_reports
        )
    pressure_marker_active_key = (
        "pressure_marker_nullspace_active_constraint_count"
    )
    pressure_marker_independent_key = (
        "pressure_marker_nullspace_independent_constraint_count"
    )
    pressure_marker_dependent_key = (
        "pressure_marker_nullspace_dependent_constraint_count"
    )
    pressure_marker_unactuated_key = (
        "pressure_marker_nullspace_unactuated_constraint_count"
    )
    pressure_marker_pivot_key = "pressure_marker_nullspace_min_factor_pivot"
    pressure_marker_reports = [
        report
        for report in projection_reports
        if pressure_marker_active_key in report or pressure_marker_pivot_key in report
    ]
    if pressure_marker_reports:
        active_counts = [
            int(report.get(pressure_marker_active_key, 0))
            for report in pressure_marker_reports
        ]
        combined[f"{pressure_marker_active_key}_min"] = min(active_counts)
        combined[f"{pressure_marker_active_key}_max"] = max(active_counts)
        active_pivots: list[float] = []
        rank_partitions: list[tuple[int, int, int, int]] = []
        for report, active_count in zip(
            pressure_marker_reports,
            active_counts,
            strict=True,
        ):
            pivot = float(report.get(pressure_marker_pivot_key, 0.0))
            independent_count = int(
                report.get(pressure_marker_independent_key, active_count)
            )
            dependent_count = int(
                report.get(pressure_marker_dependent_key, 0)
            )
            unactuated_count = int(
                report.get(pressure_marker_unactuated_key, 0)
            )
            if (
                active_count < 0
                or independent_count < 0
                or dependent_count < 0
                or unactuated_count < 0
                or independent_count + dependent_count + unactuated_count
                != active_count
            ):
                raise RuntimeError(
                    "pressure marker-nullspace rank partition is inconsistent: "
                    f"active={active_count}, independent={independent_count}, "
                    f"dependent={dependent_count}, unactuated={unactuated_count}"
                )
            if (
                independent_count + dependent_count > 0
                and independent_count == 0
            ):
                raise RuntimeError(
                    "pressure marker-nullspace has pressure-actuated rows but "
                    "zero rank"
                )
            rank_partitions.append(
                (
                    active_count,
                    independent_count,
                    dependent_count,
                    unactuated_count,
                )
            )
            for diagnostic_key in (
                "pressure_marker_nullspace_max_dependent_normalized_pivot",
                "pressure_marker_nullspace_max_input_constraint_mps",
                "pressure_marker_nullspace_max_unactuated_input_constraint_mps",
                "pressure_marker_nullspace_max_constraint_residual_mps",
            ):
                if diagnostic_key in report:
                    diagnostic = float(report[diagnostic_key])
                    if not math.isfinite(diagnostic) or diagnostic < 0.0:
                        raise RuntimeError(
                            "pressure marker-nullspace diagnostic must be "
                            "finite and non-negative: "
                            f"{diagnostic_key}={diagnostic}"
                        )
            if independent_count > 0:
                if not math.isfinite(pivot) or pivot <= 0.0:
                    raise RuntimeError(
                        "invalid active pressure marker-nullspace factor pivot: "
                        f"{pivot}"
                    )
                active_pivots.append(pivot)
        if any(
            partition != rank_partitions[0]
            for partition in rank_partitions[1:]
        ):
            raise RuntimeError(
                "pressure marker-nullspace rank partition changed across "
                f"projection cycles: {rank_partitions}"
            )
        combined[pressure_marker_pivot_key] = (
            min(active_pivots) if active_pivots else 0.0
        )
    for failure_key, action_key in (
        ("pressure_solve_failed", "pressure_solve_failure_action"),
        (
            "pressure_projection_physical_failure",
            "pressure_projection_physical_failure_action",
        ),
    ):
        if any(failure_key in report for report in projection_reports):
            failed = any(bool(report.get(failure_key, False)) for report in projection_reports)
            combined[failure_key] = failed
            if failed:
                actions = [
                    str(report.get(action_key, "reported"))
                    for report in projection_reports
                    if bool(report.get(failure_key, False))
                ]
                combined[action_key] = ",".join(dict.fromkeys(actions))
                reason_key = f"{failure_key}_reason"
                reasons = [
                    str(report.get(reason_key, ""))
                    for report in projection_reports
                    if bool(report.get(failure_key, False))
                    and str(report.get(reason_key, ""))
                ]
                if reasons:
                    combined[reason_key] = ",".join(dict.fromkeys(reasons))
            else:
                combined[action_key] = "none"
                combined[f"{failure_key}_reason"] = ""
    consistency_count = max(0, len(projection_reports) - 1)
    combined["hibm_post_dirichlet_consistency_projection_count"] = consistency_count
    combined["hibm_post_dirichlet_consistency_projection_applied"] = bool(
        consistency_count
    )
    return combined


def _finite_report_float(
    report: Mapping[str, object],
    key: str,
) -> float | None:
    try:
        value = float(report[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _report_int(
    report: Mapping[str, object],
    key: str,
    *,
    default: int,
) -> int:
    try:
        return int(report.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _finite_report_float_vector(
    report: Mapping[str, object],
    key: str,
) -> tuple[float | None, ...]:
    raw_values = report.get(key, ())
    if isinstance(raw_values, (str, bytes, Mapping)):
        return ()
    try:
        values = tuple(raw_values)
    except (TypeError, ValueError, OverflowError):
        return ()
    normalized: list[float | None] = []
    for raw_value in values:
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            normalized.append(None)
            continue
        normalized.append(value if math.isfinite(value) else None)
    return tuple(normalized)


def _json_safe_diagnostic_value(value: object) -> object:
    """Return a strict-JSON-safe copy without masking diagnostic failure."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        finite_value = float(value)
        return finite_value if math.isfinite(finite_value) else None
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if isinstance(raw_key, str):
                key = raw_key
            elif isinstance(raw_key, (bool, int, float, np.integer, np.floating)):
                key = str(_json_safe_diagnostic_value(raw_key))
            else:
                key = f"<non-json-key:{type(raw_key).__name__}>"
            normalized_mapping[key] = _json_safe_diagnostic_value(raw_value)
        return normalized_mapping
    if isinstance(value, tuple):
        return tuple(_json_safe_diagnostic_value(item) for item in value)
    if isinstance(value, list):
        return [_json_safe_diagnostic_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_diagnostic_value(value.tolist())
    return f"<non-json:{type(value).__name__}>"


def _strict_report_bool(
    report: Mapping[str, object],
    key: str,
) -> bool | None:
    value = report.get(key)
    if not isinstance(value, (bool, np.bool_)):
        return None
    return bool(value)


def _hibm_joint_qp_no_slip_diagnostics(
    *,
    no_slip_report: Mapping[str, object],
    no_slip_absolute_tolerance_mps: float,
) -> dict[str, object]:
    no_slip_tolerance = float(no_slip_absolute_tolerance_mps)
    if not math.isfinite(no_slip_tolerance) or no_slip_tolerance <= 0.0:
        raise ValueError("joint Q/P no-slip tolerance must be finite and positive")
    try:
        valid_marker_count = int(no_slip_report["hibm_no_slip_valid_marker_count"])
        invalid_marker_count = int(
            no_slip_report["hibm_no_slip_invalid_marker_count"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        valid_marker_count = -1
        invalid_marker_count = -1
    terminal_no_slip = _finite_report_float(
        no_slip_report,
        "hibm_no_slip_max_residual_mps",
    )
    no_slip_measured = bool(
        valid_marker_count > 0
        and invalid_marker_count == 0
        and terminal_no_slip is not None
    )
    no_slip_converged = bool(
        no_slip_measured and terminal_no_slip <= no_slip_tolerance
    )
    return {
        "no_slip_measured": bool(no_slip_measured),
        "no_slip_converged": bool(no_slip_converged),
        "no_slip_valid_marker_count": int(valid_marker_count),
        "no_slip_invalid_marker_count": int(invalid_marker_count),
        "no_slip_vector_max_residual_mps": terminal_no_slip,
        "no_slip_absolute_tolerance_mps": float(no_slip_tolerance),
    }


def _hibm_joint_qp_pressure_diagnostics(
    *,
    pressure_report: Mapping[str, object],
    pressure_cg_tolerance: float,
) -> dict[str, object]:
    pressure_tolerance = float(pressure_cg_tolerance)
    if not math.isfinite(pressure_tolerance) or pressure_tolerance < 0.0:
        raise ValueError(
            "joint Q/P pressure CG tolerance must be finite and non-negative"
        )

    cg_converged = _strict_report_bool(pressure_report, "cg_converged_all")
    pressure_solve_failed = _strict_report_bool(
        pressure_report,
        "pressure_solve_failed",
    )
    physical_failure = _strict_report_bool(
        pressure_report,
        "pressure_projection_physical_failure",
    )
    divergence_l2 = _finite_report_float(pressure_report, "l2")
    divergence_max_abs = _finite_report_float(pressure_report, "max_abs")
    pressure_exact_residual = _finite_report_float(
        pressure_report,
        "cg_exact_relative_residual_max",
    )
    pressure_measured = bool(
        cg_converged is not None
        and pressure_solve_failed is not None
        and physical_failure is not None
        and divergence_l2 is not None
        and divergence_max_abs is not None
        and pressure_exact_residual is not None
    )
    pressure_converged = bool(
        pressure_measured
        and cg_converged
        and not pressure_solve_failed
        and not physical_failure
        and pressure_exact_residual <= pressure_tolerance
    )
    return {
        "pressure_measured": bool(pressure_measured),
        "pressure_converged": bool(pressure_converged),
        "pressure_cg_converged_all": cg_converged,
        "pressure_solve_failed": pressure_solve_failed,
        "pressure_projection_physical_failure": physical_failure,
        "pressure_divergence_l2_s_inv": divergence_l2,
        "pressure_divergence_max_abs_s_inv": divergence_max_abs,
        "pressure_exact_relative_residual": pressure_exact_residual,
        "pressure_effective_cg_tolerance": float(pressure_tolerance),
    }


def _hibm_joint_qp_cycle_diagnostics(
    *,
    cycle_index: int,
    projection_stage: str,
    no_slip_report: Mapping[str, object],
    pressure_report: Mapping[str, object],
    no_slip_absolute_tolerance_mps: float,
    pressure_cg_tolerance: float,
    sharp_boundary_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate one terminal P state against both shared Q and P contracts."""

    if int(cycle_index) <= 0:
        raise ValueError("joint Q/P cycle_index must be positive")
    no_slip = _hibm_joint_qp_no_slip_diagnostics(
        no_slip_report=no_slip_report,
        no_slip_absolute_tolerance_mps=no_slip_absolute_tolerance_mps,
    )
    pressure = _hibm_joint_qp_pressure_diagnostics(
        pressure_report=pressure_report,
        pressure_cg_tolerance=pressure_cg_tolerance,
    )

    post_q_component_max = _finite_report_float(
        pressure_report,
        "max_residual_mps",
    )
    terminal_no_slip = no_slip["no_slip_vector_max_residual_mps"]
    post_q_vector_upper_bound = (
        math.sqrt(3.0) * post_q_component_max
        if post_q_component_max is not None
        else None
    )
    pressure_reintroduced_no_slip = (
        max(0.0, terminal_no_slip - post_q_vector_upper_bound)
        if isinstance(terminal_no_slip, float)
        and post_q_vector_upper_bound is not None
        else None
    )
    no_slip_sampling_fields = {
        "no_slip_direct_sample_marker_count": _report_int(
            no_slip_report,
            "hibm_no_slip_direct_sample_marker_count",
            default=-1,
        ),
        "no_slip_normal_walk_sample_marker_count": _report_int(
            no_slip_report,
            "hibm_no_slip_normal_walk_sample_marker_count",
            default=-1,
        ),
        "no_slip_nearest_fluid_sample_marker_count": _report_int(
            no_slip_report,
            "hibm_no_slip_nearest_fluid_sample_marker_count",
            default=-1,
        ),
        "no_slip_no_fluid_sample_marker_count": _report_int(
            no_slip_report,
            "hibm_no_slip_no_fluid_sample_marker_count",
            default=-1,
        ),
        "no_slip_sampling_identity_generation": _report_int(
            no_slip_report,
            "hibm_no_slip_sampling_identity_generation",
            default=0,
        ),
        "no_slip_topology_generation": _report_int(
            no_slip_report,
            "hibm_no_slip_topology_generation",
            default=0,
        ),
        "no_slip_component_face_valid_mask_generation": _report_int(
            no_slip_report,
            "hibm_no_slip_component_face_valid_mask_generation",
            default=0,
        ),
        "no_slip_argmax_marker_index": _report_int(
            no_slip_report,
            "hibm_no_slip_residual_argmax_marker_index",
            default=-1,
        ),
        "no_slip_argmax_marker_region_id": _report_int(
            no_slip_report,
            "hibm_no_slip_residual_argmax_marker_region_id",
            default=-1,
        ),
        "no_slip_argmax_residual_vector_mps": _finite_report_float_vector(
            no_slip_report,
            "hibm_no_slip_residual_argmax_residual_vector_mps",
        ),
        "no_slip_argmax_sample_source": str(
            no_slip_report.get(
                "hibm_no_slip_residual_argmax_sample_source",
                "none",
            )
        ),
        "no_slip_argmax_sample_position_m": _finite_report_float_vector(
            no_slip_report,
            "hibm_no_slip_residual_argmax_sample_position_m",
        ),
        "no_slip_argmax_fluid_velocity_mps": _finite_report_float_vector(
            no_slip_report,
            "hibm_no_slip_residual_argmax_fluid_velocity_mps",
        ),
    }
    return {
        "cycle_index": int(cycle_index),
        "projection_stage": str(projection_stage),
        **no_slip,
        **no_slip_sampling_fields,
        "post_q_component_max_residual_mps": post_q_component_max,
        "post_q_vector_residual_upper_bound_mps": post_q_vector_upper_bound,
        "pre_projection_velocity_iterations": _report_int(
            pressure_report,
            "iterations",
            default=-1,
        ),
        "pre_projection_velocity_max_residual_mps": post_q_component_max,
        **pressure,
        "pressure_reintroduced_no_slip_mps": pressure_reintroduced_no_slip,
        "pressure_reintroduced_no_slip_definition": (
            "lower_bound_from_terminal_vector_minus_sqrt3_post_q_component_max"
        ),
        "hibm_sharp_marker_boundary_stage_wall_time_s": (
            _hibm_sharp_boundary_stage_wall_times_from_report(
                sharp_boundary_report
            )
        ),
        "converged": bool(
            no_slip["no_slip_converged"] and pressure["pressure_converged"]
        ),
        "final_operation": "pressure_projection",
    }


def _hibm_joint_qp_terminal_diagnostics(
    *,
    cycle_budget: int,
    cycle_trace: list[dict[str, object]],
) -> dict[str, object]:
    budget = int(cycle_budget)
    if budget <= 0:
        raise ValueError("joint Q/P cycle budget must be positive")
    if not cycle_trace or len(cycle_trace) > budget:
        raise ValueError("joint Q/P cycle trace must be non-empty and within budget")
    trace = []
    for cycle in cycle_trace:
        normalized_cycle = _json_safe_diagnostic_value(cycle)
        trace.append(
            dict(normalized_cycle)
            if isinstance(normalized_cycle, Mapping)
            else {}
        )
    terminal = trace[-1]
    terminal_stage_wall_times = (
        _normalized_hibm_sharp_boundary_stage_wall_times(
            terminal.get("hibm_sharp_marker_boundary_stage_wall_time_s", {})
        )
    )
    total_stage_wall_times = _empty_hibm_sharp_boundary_stage_wall_times()
    for cycle in trace:
        cycle_stage_wall_times = (
            _normalized_hibm_sharp_boundary_stage_wall_times(
                cycle.get("hibm_sharp_marker_boundary_stage_wall_time_s", {})
            )
        )
        for stage_name in _HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES:
            accumulated = (
                total_stage_wall_times[stage_name]
                + cycle_stage_wall_times[stage_name]
            )
            if math.isfinite(accumulated):
                total_stage_wall_times[stage_name] = accumulated
    return {
        "hibm_joint_qp_measured": bool(
            terminal["no_slip_measured"] and terminal["pressure_measured"]
        ),
        "hibm_joint_qp_converged": bool(terminal["converged"]),
        "hibm_joint_qp_cycle_budget": int(budget),
        "hibm_joint_qp_cycles_used": int(len(trace)),
        "hibm_joint_qp_terminal_no_slip_vector_max_residual_mps": terminal[
            "no_slip_vector_max_residual_mps"
        ],
        "hibm_joint_qp_terminal_divergence_l2_s_inv": terminal[
            "pressure_divergence_l2_s_inv"
        ],
        "hibm_joint_qp_terminal_divergence_max_abs_s_inv": terminal[
            "pressure_divergence_max_abs_s_inv"
        ],
        "hibm_joint_qp_pressure_exact_relative_residual": terminal[
            "pressure_exact_relative_residual"
        ],
        "hibm_joint_qp_pressure_reintroduced_no_slip_mps": terminal[
            "pressure_reintroduced_no_slip_mps"
        ],
        "hibm_joint_qp_final_operation": str(terminal["final_operation"]),
        "hibm_joint_qp_cycle_trace": trace,
        "hibm_sharp_marker_boundary_terminal_stage_wall_time_s": (
            terminal_stage_wall_times
        ),
        "hibm_sharp_marker_boundary_total_stage_wall_time_s": (
            total_stage_wall_times
        ),
    }


def _require_hibm_joint_qp_convergence(
    diagnostics: Mapping[str, object],
    *,
    context: str,
) -> None:
    if bool(diagnostics.get("hibm_joint_qp_converged", False)):
        return
    normalized_diagnostics = _json_safe_diagnostic_value(diagnostics)
    safe_diagnostics = (
        dict(normalized_diagnostics)
        if isinstance(normalized_diagnostics, Mapping)
        else {}
    )
    serialized = json.dumps(
        safe_diagnostics,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    raise HibmJointQpConvergenceError(
        f"HIBM joint Q/P terminal convergence budget exhausted ({context}): "
        f"{serialized}",
        diagnostics=safe_diagnostics,
    )


def _sample_hibm_no_slip_report(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    *,
    pre_projection_velocity_projector: (
        _HibmPreProjectionVelocityProjector | None
    ) = None,
) -> dict[str, object]:
    prepared_sampling_identity = None
    topology_generation = None
    component_face_valid_mask_generation = None
    obstacle_field = fluid.obstacle
    if pre_projection_velocity_projector is None:
        component_face_valid_mask = (
            fluid.build_hibm_no_slip_component_face_valid_mask()
        )
    else:
        if pre_projection_velocity_projector.markers_owner is not markers:
            raise RuntimeError(
                "terminal no-slip sampler marker owner does not match Q"
            )
        prepared_sampling_identity = (
            pre_projection_velocity_projector.last_prepared_sampling_identity
        )
        component_face_valid_mask = (
            pre_projection_velocity_projector.last_component_face_valid_mask
        )
        obstacle_field = pre_projection_velocity_projector.last_sampling_obstacle
        if (
            prepared_sampling_identity is None
            or component_face_valid_mask is None
            or obstacle_field is None
        ):
            raise RuntimeError(
                "terminal no-slip sampler has no committed shared Q identity"
            )
        topology_generation = int(fluid.hibm_reachability_revision)
        component_face_valid_mask_generation = int(
            fluid.velocity_dirichlet_component_ledger_generation
        )
    report = markers.sample_no_slip_residual(
        fluid.velocity,
        obstacle_field,
        component_face_valid_mask,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.grid.grid_nodes,
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
        prepared_sampling_identity=prepared_sampling_identity,
        topology_generation=topology_generation,
        component_face_valid_mask_generation=(
            component_face_valid_mask_generation
        ),
    )
    return {
        "hibm_no_slip_report": asdict(report),
        "hibm_no_slip_valid_marker_count": int(report.valid_marker_count),
        "hibm_no_slip_invalid_marker_count": int(report.invalid_marker_count),
        "hibm_no_slip_max_residual_mps": float(report.max_no_slip_residual_mps),
        "hibm_no_slip_l2_residual_mps": float(report.l2_no_slip_residual_mps),
        "hibm_no_slip_residual_argmax_marker_index": int(
            report.argmax_marker_index
        ),
        "hibm_no_slip_residual_argmax_marker_region_id": int(
            report.argmax_marker_region_id
        ),
        "hibm_no_slip_residual_argmax_residual_vector_mps": tuple(
            float(value) for value in report.argmax_residual_vector_mps
        ),
        "hibm_no_slip_residual_argmax_sample_source": str(
            report.argmax_sample_source
        ),
        "hibm_no_slip_residual_argmax_sample_position_m": tuple(
            float(value) for value in report.argmax_sample_position_m
        ),
        "hibm_no_slip_residual_argmax_fluid_velocity_mps": tuple(
            float(value) for value in report.argmax_fluid_velocity_mps
        ),
        "hibm_no_slip_direct_sample_marker_count": int(
            report.direct_sample_marker_count
        ),
        "hibm_no_slip_normal_walk_sample_marker_count": int(
            report.normal_walk_sample_marker_count
        ),
        "hibm_no_slip_nearest_fluid_sample_marker_count": int(
            report.nearest_fluid_sample_marker_count
        ),
        "hibm_no_slip_no_fluid_sample_marker_count": int(
            report.no_fluid_sample_marker_count
        ),
        "hibm_no_slip_sampling_identity_generation": int(
            prepared_sampling_identity.generation
            if prepared_sampling_identity is not None
            else 0
        ),
        "hibm_no_slip_topology_generation": int(
            topology_generation if topology_generation is not None else 0
        ),
        "hibm_no_slip_component_face_valid_mask_generation": int(
            component_face_valid_mask_generation
            if component_face_valid_mask_generation is not None
            else 0
        ),
    }


CANONICAL_HIBM_VELOCITY_DIRICHLET_NUMERIC_DEVICE_REPORT_KEYS = (
    "schema_version",
    "authority",
    "new_owned_claim_component_count",
    "duplicate_claim_component_count",
    "direct_geometry_reconstructed_component_count",
    "direct_geometry_one_sided_component_count",
    "segment_supported_pair_route_fallback_count",
    "max_compatible_direct_target_spread_mps",
    "final_active_component_count",
    "final_owned_component_count",
    "final_external_exact_component_count",
    "final_hard_component_count",
    "final_soft_component_count",
    "final_active_storage_row_count",
    "final_active_x_component_count",
    "final_active_y_component_count",
    "final_active_z_component_count",
    "primary_region_active_component_count",
    "secondary_region_active_component_count",
    "other_region_active_component_count",
    "unassigned_region_active_component_count",
    "mixed_region_storage_row_count",
    "active_on_obstacle_storage_component_count",
    "legal_obstacle_interface_storage_component_count",
    "illegal_active_on_obstacle_storage_component_count",
    "max_abs_claim_target_mps",
    "max_abs_committed_target_mps",
    "min_active_pressure_mobility",
    "max_active_pressure_mobility",
    "min_active_enforcement_weight",
    "max_active_enforcement_weight",
    "invalid_mask_bits_count",
    "mask_subset_violation_count",
    "external_owned_overlap_count",
    "external_not_hard_count",
    "active_provenance_missing_count",
    "inactive_neutral_violation_count",
    "nonfinite_active_value_count",
    "nonfinite_active_mobility_count",
    "nonfinite_active_enforcement_count",
    "active_mobility_range_violation_count",
    "active_enforcement_range_violation_count",
    "hard_mobility_contract_violation_count",
    "hard_enforcement_contract_violation_count",
    "claim_conflict_count",
    "target_conflict_count",
    "region_conflict_count",
    "alpha_conflict_count",
    "nonfinite_claim_target_count",
    "nonfinite_geometry_count",
    "degenerate_geometry_count",
    "external_claim_collision_count",
    "missing_actual_sample_count",
    "actual_sample_evaluation_count",
    "actual_geometry_claim_count",
    "nominal_direct_claim_count",
    "relocated_claim_count",
    "relocation_merged_count",
    "relocation_blocked_count",
    "relocation_unavailable_count",
    "projection_only_region_seam_merged_count",
)
CANONICAL_HIBM_VELOCITY_DIRICHLET_MARKER_TARGET_CLOSURE_REPORT_KEYS = (
    "enabled",
    "constraint_count",
    "adjustable_constraint_count",
    "immutable_constraint_count",
    "solver",
    "solve_count",
    "initial_max_residual_mps",
    "final_max_residual_mps",
    "final_max_adjustable_residual_mps",
    "final_max_immutable_residual_mps",
    "absolute_tolerance_mps",
    "closure_tolerance_mps",
    "density_kgm3",
    "projection_only_marker_count",
    "projection_only_evaluated_axis_count",
    "projection_only_invalid_axis_count",
    "projection_only_constraint_count",
    "projection_only_max_residual_mps",
)
CANONICAL_HIBM_VELOCITY_DIRICHLET_DEVICE_REPORT_KEYS = (
    *CANONICAL_HIBM_VELOCITY_DIRICHLET_NUMERIC_DEVICE_REPORT_KEYS,
    "marker_target_closure",
)

CANONICAL_HIBM_VELOCITY_DIRICHLET_SEGMENT_RUNNER_REPORT_KEYS = (
    "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count",
    "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count",
    "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio",
)
CANONICAL_HIBM_VELOCITY_DIRICHLET_RUNNER_REPORT_KEYS = (
    "hibm_velocity_dirichlet_authority",
    "hibm_velocity_dirichlet_ledger_generation",
    "hibm_velocity_dirichlet_authority_registered",
    "hibm_velocity_dirichlet_authority_sealed",
    *CANONICAL_HIBM_VELOCITY_DIRICHLET_SEGMENT_RUNNER_REPORT_KEYS,
    "canonical_velocity_dirichlet_report",
)

CANONICAL_HIBM_VELOCITY_DIRICHLET_ZERO_INVARIANT_KEYS = (
    "illegal_active_on_obstacle_storage_component_count",
    "invalid_mask_bits_count",
    "mask_subset_violation_count",
    "external_owned_overlap_count",
    "external_not_hard_count",
    "active_provenance_missing_count",
    "inactive_neutral_violation_count",
    "nonfinite_active_value_count",
    "nonfinite_active_mobility_count",
    "nonfinite_active_enforcement_count",
    "active_mobility_range_violation_count",
    "active_enforcement_range_violation_count",
    "hard_mobility_contract_violation_count",
    "hard_enforcement_contract_violation_count",
    "claim_conflict_count",
    "target_conflict_count",
    "region_conflict_count",
    "alpha_conflict_count",
    "nonfinite_claim_target_count",
    "nonfinite_geometry_count",
    "degenerate_geometry_count",
    "external_claim_collision_count",
    "missing_actual_sample_count",
    "relocation_blocked_count",
    "relocation_unavailable_count",
    "direct_geometry_one_sided_component_count",
)


def _canonical_hibm_velocity_dirichlet_report_fields(
    builder_result: Mapping[str, object],
    *,
    fluid: CartesianFluidSolver,
) -> dict[str, object]:
    """Attach the real fluid lifecycle to the device-measured report.

    The boundary builder owns only the device field reduction.  Generation,
    consumer registration and sealing are fluid-solver state and are sampled
    only after the full prepare/seal sequence has completed.
    """

    device_report = builder_result.get("canonical_velocity_dirichlet_report")
    if not isinstance(device_report, Mapping):
        raise RuntimeError(
            "canonical velocity Dirichlet builder omitted its device report"
        )
    generation_error_groups = (
        fluid._velocity_dirichlet_component_ledger_generation_errors()
    )
    authority_registered = not any(
        bool(group) for group in generation_error_groups
    )
    return {
        "hibm_velocity_dirichlet_authority": str(
            fluid.velocity_dirichlet_boundary_authority
        ),
        "hibm_velocity_dirichlet_ledger_generation": int(
            fluid.velocity_dirichlet_component_ledger_generation
        ),
        "hibm_velocity_dirichlet_authority_registered": bool(
            authority_registered
        ),
        "hibm_velocity_dirichlet_authority_sealed": bool(
            fluid.velocity_dirichlet_component_ledger_sealed
        ),
        "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": (
            builder_result[
                "segment_identical_provenance_merged_component_count"
            ]
        ),
        "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": (
            builder_result["segment_endpoint_clamped_component_count"]
        ),
        "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": (
            builder_result[
                "max_segment_endpoint_clamp_overrun_support_ratio"
            ]
        ),
        "canonical_velocity_dirichlet_report": dict(device_report),
    }


def _hibm_velocity_dirichlet_mapping_fields(
    report: Mapping[str, object],
    *,
    stage: str | None = None,
) -> dict[str, object]:
    fields = {
        key: report[key]
        for key in CANONICAL_HIBM_VELOCITY_DIRICHLET_RUNNER_REPORT_KEYS
    }
    if stage is None:
        return fields
    stage_name = str(stage).strip()
    if not stage_name or not stage_name.replace("_", "").isalnum():
        raise ValueError("velocity Dirichlet diagnostic stage must be an identifier")
    staged_fields: dict[str, object] = {}
    for key, value in fields.items():
        if key == "canonical_velocity_dirichlet_report":
            staged_key = f"hibm_{stage_name}_{key}"
        else:
            staged_key = key.replace(
                "hibm_velocity_dirichlet_",
                f"hibm_{stage_name}_velocity_dirichlet_",
                1,
            )
        staged_fields[staged_key] = value
    return staged_fields


def _canonical_marker_target_closure_health_failure(
    closure_report: object,
) -> str | None:
    if not isinstance(closure_report, Mapping):
        return "canonical marker-target closure report is missing or invalid"
    expected_keys = set(
        CANONICAL_HIBM_VELOCITY_DIRICHLET_MARKER_TARGET_CLOSURE_REPORT_KEYS
    )
    actual_keys = set(closure_report)
    missing_keys = tuple(sorted(expected_keys - actual_keys))
    if missing_keys:
        return (
            "canonical marker-target closure report is missing required key: "
            f"{missing_keys[0]}"
        )
    unexpected_keys = tuple(sorted(actual_keys - expected_keys))
    if unexpected_keys:
        return (
            "canonical marker-target closure report has an unexpected key: "
            f"{unexpected_keys[0]}"
        )
    if closure_report.get("enabled") is not True:
        return "canonical marker-target closure is not enabled"
    solver = closure_report.get("solver")
    if solver != "serialized_kaczmarz":
        return (
            "canonical marker-target closure solver is invalid: "
            f"{solver!r}"
        )

    count_keys = (
        "constraint_count",
        "adjustable_constraint_count",
        "immutable_constraint_count",
        "solve_count",
        "projection_only_marker_count",
        "projection_only_evaluated_axis_count",
        "projection_only_invalid_axis_count",
        "projection_only_constraint_count",
    )
    counts: dict[str, int] = {}
    for key in count_keys:
        value = closure_report[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            return (
                "canonical marker-target closure count must be an integral "
                f"non-bool value: {key}={value!r}"
            )
        counts[key] = int(value)
    negative_counts = tuple(key for key, value in counts.items() if value < 0)
    if negative_counts:
        return (
            "canonical marker-target closure count is negative: "
            f"{negative_counts[0]}"
        )
    max_solve_count = max(4, counts["adjustable_constraint_count"])
    if counts["solve_count"] > max_solve_count:
        return (
            "canonical marker-target closure solve count is invalid: "
            f"{counts['solve_count']} > {max_solve_count}"
        )
    if (
        counts["adjustable_constraint_count"]
        + counts["immutable_constraint_count"]
        != counts["constraint_count"]
    ):
        return (
            "canonical marker-target closure constraint partition is inconsistent: "
            f"constraints={counts['constraint_count']}, "
            f"adjustable={counts['adjustable_constraint_count']}, "
            f"immutable={counts['immutable_constraint_count']}"
        )
    expected_projection_axis_count = 3 * counts["projection_only_marker_count"]
    if (
        counts["projection_only_evaluated_axis_count"]
        != expected_projection_axis_count
    ):
        return (
            "canonical marker-target closure projection-only evaluation is "
            "incomplete: "
            f"evaluated={counts['projection_only_evaluated_axis_count']}, "
            f"expected={expected_projection_axis_count}"
        )
    if counts["projection_only_invalid_axis_count"] != 0:
        return (
            "canonical marker-target closure projection-only axis is invalid: "
            f"count={counts['projection_only_invalid_axis_count']}"
        )
    if (
        counts["projection_only_constraint_count"]
        > counts["projection_only_evaluated_axis_count"]
    ):
        return (
            "canonical marker-target closure projection-only constraint count "
            "exceeds evaluated axes"
        )

    scalar_keys = (
        "initial_max_residual_mps",
        "final_max_residual_mps",
        "final_max_adjustable_residual_mps",
        "final_max_immutable_residual_mps",
        "absolute_tolerance_mps",
        "closure_tolerance_mps",
        "density_kgm3",
        "projection_only_max_residual_mps",
    )
    scalars: dict[str, float] = {}
    for key in scalar_keys:
        value = closure_report[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            return (
                "canonical marker-target closure scalar must be numeric and "
                f"non-bool: {key}={value!r}"
            )
        scalars[key] = float(value)
    if any(not math.isfinite(value) for value in scalars.values()):
        return f"canonical marker-target closure scalars are non-finite: {scalars}"
    residual_keys = (
        "initial_max_residual_mps",
        "final_max_residual_mps",
        "final_max_adjustable_residual_mps",
        "final_max_immutable_residual_mps",
        "projection_only_max_residual_mps",
    )
    if any(scalars[key] < 0.0 for key in residual_keys):
        return f"canonical marker-target closure residual is negative: {scalars}"
    absolute_tolerance = scalars["absolute_tolerance_mps"]
    closure_tolerance = scalars["closure_tolerance_mps"]
    if not 0.0 < closure_tolerance < absolute_tolerance:
        return (
            "canonical marker-target closure tolerance ordering is invalid: "
            f"closure={closure_tolerance}, absolute={absolute_tolerance}"
        )
    if scalars["density_kgm3"] <= 0.0:
        return (
            "canonical marker-target closure density is not positive: "
            f"{scalars['density_kgm3']}"
        )
    if scalars["final_max_adjustable_residual_mps"] > closure_tolerance:
        return (
            "canonical marker-target closure adjustable residual exceeds its "
            "tolerance: "
            f"residual={scalars['final_max_adjustable_residual_mps']}, "
            f"tolerance={closure_tolerance}"
        )
    if scalars["final_max_immutable_residual_mps"] > absolute_tolerance:
        return (
            "canonical marker-target closure immutable residual exceeds its "
            "tolerance: "
            f"residual={scalars['final_max_immutable_residual_mps']}, "
            f"tolerance={absolute_tolerance}"
        )
    if scalars["projection_only_max_residual_mps"] > absolute_tolerance:
        return (
            "canonical marker-target closure projection-only residual exceeds "
            "the absolute tolerance: "
            f"residual={scalars['projection_only_max_residual_mps']}, "
            f"tolerance={absolute_tolerance}"
        )
    return None


def _canonical_hibm_velocity_dirichlet_health_failure(
    report: Mapping[str, object],
) -> str | None:
    missing_runner_keys = tuple(
        key
        for key in CANONICAL_HIBM_VELOCITY_DIRICHLET_RUNNER_REPORT_KEYS
        if key not in report
    )
    if missing_runner_keys:
        return (
            "canonical velocity Dirichlet diagnostic is missing required key: "
            f"{missing_runner_keys[0]}"
        )
    if report.get("hibm_velocity_dirichlet_authority") != "canonical":
        return (
            "canonical velocity Dirichlet authority is invalid: "
            f"{report.get('hibm_velocity_dirichlet_authority')!r}"
        )
    generation_value = report["hibm_velocity_dirichlet_ledger_generation"]
    if isinstance(generation_value, (bool, np.bool_)) or not isinstance(
        generation_value,
        (int, np.integer),
    ):
        return (
            "canonical velocity Dirichlet generation must be an integral "
            f"non-bool value: {generation_value!r}"
        )
    generation = int(generation_value)
    if generation <= 0:
        return (
            "canonical velocity Dirichlet generation is not positive: "
            f"{generation}"
        )
    if report.get("hibm_velocity_dirichlet_authority_registered") is not True:
        return "canonical velocity Dirichlet consumers are not all registered"
    if report.get("hibm_velocity_dirichlet_authority_sealed") is not True:
        return "canonical velocity Dirichlet ledger is not sealed"

    device_report = report.get("canonical_velocity_dirichlet_report")
    if not isinstance(device_report, Mapping):
        return "canonical velocity Dirichlet device report is missing or invalid"
    schema_version = device_report.get("schema_version")
    if type(schema_version) is not int or schema_version != 5:
        return (
            "canonical velocity Dirichlet schema version is invalid: "
            f"{schema_version!r}"
        )
    identical_provenance_value = report[
        "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count"
    ]
    if isinstance(identical_provenance_value, (bool, np.bool_)) or not isinstance(
        identical_provenance_value,
        (int, np.integer),
    ):
        return (
            "canonical velocity Dirichlet identical segment-provenance merged "
            "count must be an integral non-bool value: "
            f"{identical_provenance_value!r}"
        )
    identical_provenance_count = int(identical_provenance_value)
    if identical_provenance_count < 0:
        return (
            "canonical velocity Dirichlet identical segment-provenance merged "
            f"count is negative: {identical_provenance_count}"
        )
    endpoint_clamped_value = report[
        "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count"
    ]
    if isinstance(endpoint_clamped_value, (bool, np.bool_)) or not isinstance(
        endpoint_clamped_value,
        (int, np.integer),
    ):
        return (
            "canonical velocity Dirichlet segment endpoint-clamped count must "
            "be an integral non-bool value: "
            f"{endpoint_clamped_value!r}"
        )
    endpoint_clamped_count = int(endpoint_clamped_value)
    if endpoint_clamped_count < 0:
        return (
            "canonical velocity Dirichlet segment endpoint-clamped count is "
            f"negative: {endpoint_clamped_count}"
        )
    clamp_ratio_value = report[
        "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio"
    ]
    if isinstance(clamp_ratio_value, (bool, np.bool_)) or not isinstance(
        clamp_ratio_value,
        (int, float, np.integer, np.floating),
    ):
        return (
            "canonical velocity Dirichlet segment endpoint-clamp ratio must be "
            f"numeric and non-bool: {clamp_ratio_value!r}"
        )
    max_endpoint_clamp_ratio = float(clamp_ratio_value)
    if (
        not math.isfinite(max_endpoint_clamp_ratio)
        or max_endpoint_clamp_ratio < 0.0
        or max_endpoint_clamp_ratio > 1.00002
    ):
        return (
            "canonical velocity Dirichlet segment endpoint-clamp ratio is "
            f"outside accepted MAC support: {max_endpoint_clamp_ratio}"
        )
    if (endpoint_clamped_count == 0) != (max_endpoint_clamp_ratio == 0.0):
        return (
            "canonical velocity Dirichlet segment endpoint-clamp count/ratio "
            "relation is inconsistent: "
            f"count={endpoint_clamped_count}, ratio={max_endpoint_clamp_ratio}"
        )

    expected_keys = set(CANONICAL_HIBM_VELOCITY_DIRICHLET_DEVICE_REPORT_KEYS)
    actual_keys = set(device_report)
    missing_device_keys = tuple(sorted(expected_keys - actual_keys))
    if missing_device_keys:
        return (
            "canonical velocity Dirichlet device report is missing required key: "
            f"{missing_device_keys[0]}"
        )
    unexpected_device_keys = tuple(sorted(actual_keys - expected_keys))
    if unexpected_device_keys:
        return (
            "canonical velocity Dirichlet device report has an unexpected key: "
            f"{unexpected_device_keys[0]}"
        )
    if device_report.get("authority") != "canonical_component_face":
        return (
            "canonical velocity Dirichlet device authority is invalid: "
            f"{device_report.get('authority')!r}"
        )
    closure_failure = _canonical_marker_target_closure_health_failure(
        device_report["marker_target_closure"]
    )
    if closure_failure is not None:
        return closure_failure

    extrema_keys = {
        "max_abs_claim_target_mps",
        "max_abs_committed_target_mps",
        "max_compatible_direct_target_spread_mps",
        "min_active_pressure_mobility",
        "max_active_pressure_mobility",
        "min_active_enforcement_weight",
        "max_active_enforcement_weight",
    }
    count_keys = tuple(
        key
        for key in CANONICAL_HIBM_VELOCITY_DIRICHLET_NUMERIC_DEVICE_REPORT_KEYS
        if key not in {"schema_version", "authority", *extrema_keys}
    )
    counts: dict[str, int] = {}
    for key in count_keys:
        value = device_report[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            return (
                "canonical velocity Dirichlet count must be an integral "
                f"non-bool value: {key}={value!r}"
            )
        counts[key] = int(value)
    negative_keys = tuple(key for key, value in counts.items() if value < 0)
    if negative_keys:
        return (
            "canonical velocity Dirichlet count is negative: "
            f"{negative_keys[0]}"
        )
    nonzero_invariants = {
        key: counts[key]
        for key in CANONICAL_HIBM_VELOCITY_DIRICHLET_ZERO_INVARIANT_KEYS
        if counts[key] != 0
    }
    if nonzero_invariants:
        return (
            "canonical velocity Dirichlet invariant counters are nonzero: "
            f"{nonzero_invariants}"
        )

    active = counts["final_active_component_count"]
    owned = counts["final_owned_component_count"]
    external = counts["final_external_exact_component_count"]
    hard = counts["final_hard_component_count"]
    soft = counts["final_soft_component_count"]
    active_rows = counts["final_active_storage_row_count"]
    if owned > active or external > active:
        return (
            "canonical velocity Dirichlet provenance subset is inconsistent: "
            f"owned={owned}, external={external}, active={active}"
        )
    if hard + soft != active:
        return (
            "canonical velocity Dirichlet hard/soft partition is inconsistent: "
            f"hard={hard}, soft={soft}, active={active}"
        )
    axis_total = sum(
        counts[key]
        for key in (
            "final_active_x_component_count",
            "final_active_y_component_count",
            "final_active_z_component_count",
        )
    )
    if axis_total != active:
        return (
            "canonical velocity Dirichlet axis partition is inconsistent: "
            f"axes={axis_total}, active={active}"
        )
    region_total = sum(
        counts[key]
        for key in (
            "primary_region_active_component_count",
            "secondary_region_active_component_count",
            "other_region_active_component_count",
            "unassigned_region_active_component_count",
        )
    )
    if region_total != active:
        return (
            "canonical velocity Dirichlet region partition is inconsistent: "
            f"regions={region_total}, active={active}"
        )
    if not (
        (active == 0 and active_rows == 0)
        or (active > 0 and active_rows <= active <= 3 * active_rows)
    ):
        return (
            "canonical velocity Dirichlet storage-row count is inconsistent: "
            f"rows={active_rows}, active={active}"
        )
    if counts["new_owned_claim_component_count"] > owned:
        return (
            "canonical velocity Dirichlet new-owned count exceeds final owned "
            f"components: new={counts['new_owned_claim_component_count']}, "
            f"owned={owned}"
        )
    reconstructed = counts["direct_geometry_reconstructed_component_count"]
    one_sided = counts["direct_geometry_one_sided_component_count"]
    duplicates = counts["duplicate_claim_component_count"]
    if identical_provenance_count + reconstructed > duplicates:
        return (
            "canonical velocity Dirichlet segment provenance/reconstruction "
            "counts exceed duplicate components: "
            f"identical={identical_provenance_count}, "
            f"reconstructed={reconstructed}, duplicates={duplicates}"
        )
    if endpoint_clamped_count + identical_provenance_count > duplicates:
        return (
            "canonical velocity Dirichlet segment endpoint-clamped and "
            "identical-provenance counts exceed duplicate components: "
            f"clamped={endpoint_clamped_count}, "
            f"identical={identical_provenance_count}, duplicates={duplicates}"
        )
    if one_sided > reconstructed or reconstructed > duplicates:
        return (
            "canonical velocity Dirichlet direct-geometry reconstruction counts "
            "are inconsistent: "
            f"one_sided={one_sided}, reconstructed={reconstructed}, "
            f"duplicates={duplicates}"
        )
    nominal_direct_claims = counts["nominal_direct_claim_count"]
    provenance_resolved_duplicates = identical_provenance_count + reconstructed
    if 2 * provenance_resolved_duplicates > nominal_direct_claims:
        return (
            "canonical velocity Dirichlet segment provenance/reconstruction "
            "lacks two nominal direct claims per resolved component: "
            f"identical={identical_provenance_count}, "
            f"reconstructed={reconstructed}, "
            f"nominal_direct_claims={nominal_direct_claims}"
        )

    extrema: dict[str, float] = {}
    for key in extrema_keys:
        value = device_report[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            return (
                "canonical velocity Dirichlet extrema must be numeric non-bool "
                f"values: {key}={value!r}"
            )
        extrema[key] = float(value)
    if any(not math.isfinite(value) for value in extrema.values()):
        return f"canonical velocity Dirichlet extrema are non-finite: {extrema}"
    if (
        extrema["max_abs_claim_target_mps"] < 0.0
        or extrema["max_abs_committed_target_mps"] < 0.0
        or extrema["max_compatible_direct_target_spread_mps"] < 0.0
    ):
        return f"canonical velocity Dirichlet target extrema are negative: {extrema}"
    compatible_spread = extrema["max_compatible_direct_target_spread_mps"]
    if provenance_resolved_duplicates == 0 and compatible_spread > 0.0:
        return (
            "canonical velocity Dirichlet segment provenance/reconstruction "
            "spread "
            "relation is inconsistent: "
            f"identical={identical_provenance_count}, "
            f"reconstructed={reconstructed}, spread={compatible_spread}"
        )
    for prefix in ("pressure_mobility", "enforcement_weight"):
        minimum = extrema[f"min_active_{prefix}"]
        maximum = extrema[f"max_active_{prefix}"]
        if not 0.0 <= minimum <= maximum <= 1.0:
            return (
                "canonical velocity Dirichlet active range is invalid: "
                f"{prefix}=({minimum}, {maximum})"
            )
    return None


def _hibm_velocity_dirichlet_health_failure(
    report: Mapping[str, object],
) -> str | None:
    if not bool(report.get("hibm_sharp_marker_boundary_enabled", True)):
        return None
    return _canonical_hibm_velocity_dirichlet_health_failure(report)


def _require_hibm_velocity_dirichlet_health(
    report: Mapping[str, object],
    *,
    context: str,
) -> None:
    failure = _hibm_velocity_dirichlet_health_failure(report)
    if failure is not None:
        raise RuntimeError(
            "HIBM velocity Dirichlet reconstruction health failure "
            f"({context}): {failure}"
        )


def _capture_velocity_dirichlet_row_ledger_reference(
    fluid: CartesianFluidSolver,
    *,
    context: str,
) -> int:
    capture = getattr(
        fluid,
        "capture_velocity_dirichlet_boundary_ledger_reference",
        None,
    )
    if not callable(capture):
        raise RuntimeError(
            "velocity Dirichlet row ledger capture is unavailable "
            f"({context})"
        )
    try:
        generation = int(capture())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "velocity Dirichlet row ledger snapshot generation is invalid "
            f"({context}): {exc}"
        ) from exc
    if generation <= 0:
        raise RuntimeError(
            "velocity Dirichlet row ledger snapshot generation must be positive "
            f"({context}): generation={generation}"
        )
    return generation


def _require_velocity_only_topology_reuse(
    report: Mapping[str, object],
    *,
    context: str,
) -> None:
    """Require an explicit, unchanged topology before reusing a soft map."""

    try:
        topology_reused = report["hibm_sharp_marker_boundary_topology_reused"]
        topology_mutated = report["hibm_preassembly_topology_mutated"]
    except KeyError as exc:
        raise RuntimeError(
            "velocity-only row topology diagnostic is incomplete "
            f"({context}): missing={exc.args[0]}"
        ) from exc
    if not isinstance(topology_reused, (bool, np.bool_)) or not isinstance(
        topology_mutated,
        (bool, np.bool_),
    ):
        raise RuntimeError(
            "velocity-only row topology diagnostic is not boolean "
            f"({context}): reused={topology_reused!r}, "
            f"mutated={topology_mutated!r}"
        )
    if not bool(topology_reused):
        raise RuntimeError(
            f"velocity-only row topology was not reused ({context})"
        )
    if bool(topology_mutated):
        raise RuntimeError(
            f"velocity-only row topology mutated ({context})"
        )


def _velocity_ledger_detail_integer(
    details: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    value = details[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{key} is not an exact integer: {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{key} is negative ({context}): {result}")
    return result


def _velocity_ledger_detail_boolean(
    details: Mapping[str, object],
    key: str,
) -> bool:
    value = details[key]
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{key} is not boolean: {value!r}")
    return bool(value)


def _validated_velocity_ledger_details(
    details: Mapping[str, object],
    *,
    reference_generation: int,
    context: str,
) -> dict[str, object]:
    schema_version = _velocity_ledger_detail_integer(
        details,
        "schema_version",
        context=context,
    )
    if schema_version != 1:
        raise ValueError(
            f"unsupported ledger comparison schema ({context}): {schema_version}"
        )
    normalized: dict[str, object] = {
        "schema_version": schema_version,
    }
    integer_keys = (
        "reference_generation",
        "device_content_mismatch_rows",
        "content_equivalence_mismatch_rows",
        "identity_mismatch_rows",
        "reference_component_generation",
        "current_component_generation",
        "reference_face_symmetric",
        "current_face_symmetric",
    )
    for key in integer_keys:
        normalized[key] = _velocity_ledger_detail_integer(
            details,
            key,
            context=context,
        )
    if int(normalized["reference_generation"]) != int(reference_generation):
        raise ValueError(
            "ledger comparison reference generation changed "
            f"({context}): expected={int(reference_generation)}, "
            f"actual={normalized['reference_generation']}"
        )

    boolean_keys = (
        "authority_changed",
        "component_generation_changed",
        "face_symmetric_changed",
    )
    for key in boolean_keys:
        normalized[key] = _velocity_ledger_detail_boolean(details, key)
    for key in ("reference_authority", "current_authority"):
        value = details[key]
        if not isinstance(value, str) or value not in {"legacy", "canonical"}:
            raise TypeError(f"{key} is not a valid authority: {value!r}")
        normalized[key] = value
    for key in (
        "first_identity_mismatch_field",
        "first_content_mismatch_field",
    ):
        value = details[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise TypeError(f"{key} is not a nonempty string or null: {value!r}")
        normalized[key] = value

    authority_changed = bool(normalized["authority_changed"])
    generation_changed = bool(normalized["component_generation_changed"])
    face_changed = bool(normalized["face_symmetric_changed"])
    device_rows = int(normalized["device_content_mismatch_rows"])
    content_rows = int(normalized["content_equivalence_mismatch_rows"])
    identity_rows = int(normalized["identity_mismatch_rows"])
    if authority_changed != (
        normalized["reference_authority"] != normalized["current_authority"]
    ):
        raise ValueError(f"authority change flag is inconsistent ({context})")
    if generation_changed != (
        normalized["reference_component_generation"]
        != normalized["current_component_generation"]
    ):
        raise ValueError(
            f"component generation change flag is inconsistent ({context})"
        )
    if face_changed != (
        normalized["reference_face_symmetric"]
        != normalized["current_face_symmetric"]
    ):
        raise ValueError(f"face-symmetric change flag is inconsistent ({context})")
    if content_rows != device_rows + int(authority_changed) + int(face_changed):
        raise ValueError(f"content mismatch count is inconsistent ({context})")
    if identity_rows != content_rows + int(generation_changed):
        raise ValueError(f"identity mismatch count is inconsistent ({context})")
    if (content_rows == 0) != (
        normalized["first_content_mismatch_field"] is None
    ):
        raise ValueError(f"first content mismatch field is inconsistent ({context})")
    if (identity_rows == 0) != (
        normalized["first_identity_mismatch_field"] is None
    ):
        raise ValueError(f"first identity mismatch field is inconsistent ({context})")
    return normalized


def _velocity_dirichlet_row_ledger_comparison(
    fluid: CartesianFluidSolver,
    *,
    reference_generation: int,
    comparison_mode: str = "strict_identity",
    context: str,
) -> dict[str, object]:
    if isinstance(reference_generation, (bool, np.bool_)) or not isinstance(
        reference_generation,
        (int, np.integer),
    ):
        raise TypeError(
            "velocity Dirichlet row ledger reference generation must be an "
            f"exact integer ({context}): {reference_generation!r}"
        )
    reference_generation_value = int(reference_generation)
    if reference_generation_value <= 0:
        raise ValueError(
            "velocity Dirichlet row ledger reference generation must be "
            f"positive ({context}): {reference_generation_value}"
        )
    if comparison_mode not in {"strict_identity", "content_equivalence"}:
        raise ValueError(
            "velocity Dirichlet row ledger comparison mode is invalid "
            f"({context}): {comparison_mode!r}"
        )
    detailed_compare = getattr(
        fluid,
        "velocity_dirichlet_boundary_ledger_comparison",
        None,
    )
    if callable(detailed_compare):
        try:
            details = detailed_compare(
                expected_generation=reference_generation_value
            )
            if not isinstance(details, Mapping):
                raise TypeError("comparison result is not a mapping")
            details = _validated_velocity_ledger_details(
                details,
                reference_generation=reference_generation_value,
                context=context,
            )
            identity_mismatch_rows = int(details["identity_mismatch_rows"])
            content_mismatch_rows = int(
                details["content_equivalence_mismatch_rows"]
            )
            device_content_mismatch_rows = int(
                details["device_content_mismatch_rows"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "velocity Dirichlet row ledger detailed comparison failed "
                f"({context}): {exc}"
            ) from exc
        mismatch_rows = (
            identity_mismatch_rows
            if comparison_mode == "strict_identity"
            else content_mismatch_rows
        )
        first_mismatch_field = details.get(
            (
                "first_identity_mismatch_field"
                if comparison_mode == "strict_identity"
                else "first_content_mismatch_field"
            )
        )
        if mismatch_rows > 0 and first_mismatch_field is None:
            first_mismatch_field = "unknown"
        return {
            "hibm_velocity_dirichlet_row_ledger_snapshot_generation": int(
                reference_generation_value
            ),
            "hibm_velocity_dirichlet_row_ledger_comparison_mode": (
                comparison_mode
            ),
            "hibm_velocity_dirichlet_row_ledger_matches_reference": (
                mismatch_rows == 0
            ),
            "hibm_velocity_dirichlet_row_ledger_mismatch_rows": mismatch_rows,
            "hibm_velocity_dirichlet_row_ledger_device_content_mismatch_rows": (
                device_content_mismatch_rows
            ),
            "hibm_velocity_dirichlet_row_ledger_content_equivalence_mismatch_rows": (
                content_mismatch_rows
            ),
            "hibm_velocity_dirichlet_row_ledger_identity_mismatch_rows": (
                identity_mismatch_rows
            ),
            "hibm_velocity_dirichlet_row_ledger_authority_changed": bool(
                details["authority_changed"]
            ),
            "hibm_velocity_dirichlet_row_ledger_component_generation_changed": bool(
                details["component_generation_changed"]
            ),
            "hibm_velocity_dirichlet_row_ledger_face_symmetric_changed": bool(
                details["face_symmetric_changed"]
            ),
            "hibm_velocity_dirichlet_row_ledger_reference_authority": str(
                details["reference_authority"]
            ),
            "hibm_velocity_dirichlet_row_ledger_current_authority": str(
                details["current_authority"]
            ),
            "hibm_velocity_dirichlet_row_ledger_reference_component_generation": int(
                details["reference_component_generation"]
            ),
            "hibm_velocity_dirichlet_row_ledger_current_component_generation": int(
                details["current_component_generation"]
            ),
            "hibm_velocity_dirichlet_row_ledger_reference_face_symmetric": int(
                details["reference_face_symmetric"]
            ),
            "hibm_velocity_dirichlet_row_ledger_current_face_symmetric": int(
                details["current_face_symmetric"]
            ),
            "hibm_velocity_dirichlet_row_ledger_first_mismatch_field": (
                first_mismatch_field
            ),
        }

    if comparison_mode == "content_equivalence":
        raise RuntimeError(
            "velocity Dirichlet row ledger content-equivalence comparison is "
            f"unavailable ({context})"
        )
    compare = getattr(
        fluid,
        "velocity_dirichlet_boundary_ledger_mismatch_rows",
        None,
    )
    if not callable(compare):
        raise RuntimeError(
            "velocity Dirichlet row ledger comparison is unavailable "
            f"({context})"
        )
    try:
        raw_mismatch_rows = compare(
            expected_generation=reference_generation_value
        )
        if isinstance(raw_mismatch_rows, (bool, np.bool_)) or not isinstance(
            raw_mismatch_rows,
            (int, np.integer),
        ):
            raise TypeError(
                "mismatch count is not an exact integer: "
                f"{raw_mismatch_rows!r}"
            )
        mismatch_rows = int(raw_mismatch_rows)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "velocity Dirichlet row ledger comparison failed "
            f"({context}): {exc}"
        ) from exc
    if mismatch_rows < 0:
        raise RuntimeError(
            "velocity Dirichlet row ledger mismatch count is invalid "
            f"({context}): mismatch_rows={mismatch_rows}"
        )
    return {
        "hibm_velocity_dirichlet_row_ledger_snapshot_generation": int(
            reference_generation_value
        ),
        "hibm_velocity_dirichlet_row_ledger_matches_reference": mismatch_rows == 0,
        "hibm_velocity_dirichlet_row_ledger_mismatch_rows": mismatch_rows,
    }


def _velocity_dirichlet_row_ledger_reference_diagnostics(
    *,
    reference_generation: int,
) -> dict[str, object]:
    if int(reference_generation) <= 0:
        raise RuntimeError(
            "velocity Dirichlet row ledger snapshot generation must be positive: "
            f"generation={int(reference_generation)}"
        )
    return {
        "hibm_velocity_dirichlet_row_ledger_snapshot_generation": int(
            reference_generation
        ),
        "hibm_velocity_dirichlet_row_ledger_matches_reference": True,
        "hibm_velocity_dirichlet_row_ledger_mismatch_rows": 0,
    }


def _require_velocity_only_consistency_row_reuse(
    reference_report: Mapping[str, object],
    consistency_report: Mapping[str, object],
    *,
    context: str,
) -> None:
    _require_velocity_only_topology_reuse(
        consistency_report,
        context=context,
    )

    try:
        reference_generation = int(
            reference_report[
                "hibm_velocity_dirichlet_row_ledger_snapshot_generation"
            ]
        )
        consistency_generation = int(
            consistency_report[
                "hibm_velocity_dirichlet_row_ledger_snapshot_generation"
            ]
        )
        reference_matches = bool(
            reference_report[
                "hibm_velocity_dirichlet_row_ledger_matches_reference"
            ]
        )
        consistency_matches = bool(
            consistency_report[
                "hibm_velocity_dirichlet_row_ledger_matches_reference"
            ]
        )
        reference_mismatch_rows = int(
            reference_report[
                "hibm_velocity_dirichlet_row_ledger_mismatch_rows"
            ]
        )
        consistency_mismatch_rows = int(
            consistency_report[
                "hibm_velocity_dirichlet_row_ledger_mismatch_rows"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "velocity-only consistency row ledger diagnostic is invalid "
            f"({context}): {exc}"
        ) from exc
    if (
        reference_generation <= 0
        or consistency_generation != reference_generation
    ):
        raise RuntimeError(
            "velocity-only consistency row ledger snapshot generation changed "
            f"({context}): reference={reference_generation}, "
            f"consistency={consistency_generation}"
        )
    if (
        not reference_matches
        or reference_mismatch_rows != 0
        or not consistency_matches
        or consistency_mismatch_rows != 0
    ):
        mismatch_field = consistency_report.get(
            "hibm_velocity_dirichlet_row_ledger_first_mismatch_field",
            "unknown",
        )
        raise RuntimeError(
            "velocity-only consistency row ledger changed "
            f"({context}): reference_matches={reference_matches}, "
            f"reference_mismatch_rows={reference_mismatch_rows}, "
            f"consistency_matches={consistency_matches}, "
            f"consistency_mismatch_rows={consistency_mismatch_rows}, "
            f"first_mismatch_field={mismatch_field}"
        )

    reference_is_canonical = bool(
        reference_report.get("hibm_velocity_dirichlet_authority") == "canonical"
        or "canonical_velocity_dirichlet_report" in reference_report
    )
    consistency_is_canonical = bool(
        consistency_report.get("hibm_velocity_dirichlet_authority") == "canonical"
        or "canonical_velocity_dirichlet_report" in consistency_report
    )
    if not (reference_is_canonical and consistency_is_canonical):
        raise RuntimeError(
            "canonical velocity-only consistency authority is required "
            f"({context})"
        )
    if reference_is_canonical or consistency_is_canonical:
        for label, report in (
            ("reference", reference_report),
            ("consistency", consistency_report),
        ):
            failure = _canonical_hibm_velocity_dirichlet_health_failure(report)
            if failure is not None:
                raise RuntimeError(
                    "canonical velocity-only consistency report is unhealthy "
                    f"({context}, {label}): {failure}"
                )
        reference_device_report = reference_report[
            "canonical_velocity_dirichlet_report"
        ]
        consistency_device_report = consistency_report[
            "canonical_velocity_dirichlet_report"
        ]
        reference_schema = int(reference_device_report["schema_version"])
        consistency_schema = int(consistency_device_report["schema_version"])
        if consistency_schema != reference_schema:
            raise RuntimeError(
                "canonical velocity-only consistency schema changed "
                f"({context}): reference={reference_schema}, "
                f"consistency={consistency_schema}"
            )
        comparison_keys = CANONICAL_HIBM_VELOCITY_DIRICHLET_DEVICE_REPORT_KEYS
        for key in (
            CANONICAL_HIBM_VELOCITY_DIRICHLET_SEGMENT_RUNNER_REPORT_KEYS
        ):
            reference_value = reference_report[key]
            consistency_value = consistency_report[key]
            if key.endswith("_ratio"):
                unchanged = bool(
                    math.isfinite(float(reference_value))
                    and math.isfinite(float(consistency_value))
                    and abs(float(consistency_value) - float(reference_value))
                    <= 2.0e-6
                )
            else:
                unchanged = consistency_value == reference_value
            if not unchanged:
                raise RuntimeError(
                    "canonical velocity-only consistency segment diagnostic "
                    f"changed ({context}): key={key}, "
                    f"reference={reference_value!r}, "
                    f"consistency={consistency_value!r}"
                )
        float_keys = {
            "max_abs_claim_target_mps",
            "max_abs_committed_target_mps",
            "min_active_pressure_mobility",
            "max_active_pressure_mobility",
            "min_active_enforcement_weight",
            "max_active_enforcement_weight",
        }
        for key in comparison_keys:
            reference_value = reference_device_report[key]
            consistency_value = consistency_device_report[key]
            if key == "marker_target_closure":
                unchanged = True
                for closure_key in (
                    CANONICAL_HIBM_VELOCITY_DIRICHLET_MARKER_TARGET_CLOSURE_REPORT_KEYS
                ):
                    reference_closure_value = reference_value[closure_key]
                    consistency_closure_value = consistency_value[closure_key]
                    if closure_key.endswith("_mps") or closure_key == "density_kgm3":
                        item_unchanged = bool(
                            math.isfinite(float(reference_closure_value))
                            and math.isfinite(float(consistency_closure_value))
                            and abs(
                                float(consistency_closure_value)
                                - float(reference_closure_value)
                            )
                            <= 2.0e-6
                        )
                    else:
                        item_unchanged = (
                            consistency_closure_value == reference_closure_value
                        )
                    unchanged = bool(unchanged and item_unchanged)
            elif key in float_keys:
                unchanged = bool(
                    math.isfinite(float(reference_value))
                    and math.isfinite(float(consistency_value))
                    and abs(float(consistency_value) - float(reference_value))
                    <= 2.0e-6
                )
            else:
                unchanged = consistency_value == reference_value
            if not unchanged:
                raise RuntimeError(
                    "canonical velocity-only consistency component diagnostic "
                    f"changed ({context}): key={key}, "
                    f"reference={reference_value!r}, "
                    f"consistency={consistency_value!r}"
                )
        return


def _hibm_marker_mac_constraint_iterations(config: Any) -> int:
    raw_value = getattr(config, "flow_hibm_marker_mac_constraint_iterations", 64)
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(
            "flow_hibm_marker_mac_constraint_iterations must be a positive integer"
        )
    try:
        value = int(raw_value)
        exact_value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "flow_hibm_marker_mac_constraint_iterations must be a positive integer"
        ) from exc
    if value <= 0 or not math.isfinite(exact_value) or exact_value != float(value):
        raise ValueError(
            "flow_hibm_marker_mac_constraint_iterations must be a positive integer"
        )
    return value


def _hibm_marker_mac_constraint_absolute_tolerance_mps(config: Any) -> float:
    raw_value = getattr(
        config,
        "flow_hibm_marker_mac_constraint_absolute_tolerance_mps",
        1.0e-4,
    )
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(
            "flow_hibm_marker_mac_constraint_absolute_tolerance_mps must be "
            "finite and positive"
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "flow_hibm_marker_mac_constraint_absolute_tolerance_mps must be "
            "finite and positive"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "flow_hibm_marker_mac_constraint_absolute_tolerance_mps must be "
            "finite and positive"
        )
    return value


def _hibm_marker_compatibility_closure_tolerance_mps(config: Any) -> float:
    raw_value = getattr(
        config,
        "flow_hibm_marker_compatibility_closure_tolerance_mps",
        1.0e-6,
    )
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(
            "flow_hibm_marker_compatibility_closure_tolerance_mps must be "
            "finite and positive"
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "flow_hibm_marker_compatibility_closure_tolerance_mps must be "
            "finite and positive"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "flow_hibm_marker_compatibility_closure_tolerance_mps must be "
            "finite and positive"
        )
    marker_mac_tolerance = _hibm_marker_mac_constraint_absolute_tolerance_mps(
        config
    )
    if value > marker_mac_tolerance:
        raise ValueError(
            "flow_hibm_marker_compatibility_closure_tolerance_mps must not "
            "exceed flow_hibm_marker_mac_constraint_absolute_tolerance_mps"
        )
    return value


class _HibmPreProjectionVelocityProjector:
    """Bind the generic fluid projection protocol to marker-space HIBM Q."""

    def __init__(
        self,
        *,
        markers: HibmMpmSurfaceMarkers,
        operator: HibmMpmMarkerMacConstraintOperator,
        max_iterations: int,
        absolute_tolerance_mps: float,
    ) -> None:
        self.markers_owner = markers
        self.operator = operator
        self.max_iterations = int(max_iterations)
        self.absolute_tolerance_mps = float(absolute_tolerance_mps)
        self._prepared_fluid: CartesianFluidSolver | None = None
        self._prepared_sampling_identity = None
        self._prepared_component_face_valid_mask = None
        self._prepared_sampling_obstacle = None
        self._prepared_topology_generation = -1
        self._prepared_component_face_valid_mask_generation = -1
        self._pressure_solve_context: dict[str, object] = {}
        self._pressure_nullspace_fluid: CartesianFluidSolver | None = None
        self._pressure_actuated_component_mobility = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuation_generation = -1
        self._pressure_nullspace_topology_generation = -1
        self._pressure_nullspace_component_face_valid_mask_generation = -1

    @property
    def last_prepared_sampling_identity(self):
        return self._prepared_sampling_identity

    @property
    def last_component_face_valid_mask(self):
        return self._prepared_component_face_valid_mask

    @property
    def last_sampling_obstacle(self):
        return self._prepared_sampling_obstacle

    def _current_generations(
        self,
        fluid: CartesianFluidSolver,
    ) -> tuple[int, int]:
        return (
            int(fluid.hibm_reachability_revision),
            int(fluid.velocity_dirichlet_component_ledger_generation),
        )

    def _require_prepared_fluid(self) -> CartesianFluidSolver:
        if self._prepared_fluid is None or self._prepared_sampling_identity is None:
            raise RuntimeError(
                "pre-projection velocity transaction has not been prepared"
            )
        return self._prepared_fluid

    def _clear_pressure_nullspace_transaction_state(self) -> None:
        self._pressure_nullspace_fluid = None
        self._pressure_actuated_component_mobility = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuation_generation = -1
        self._pressure_nullspace_topology_generation = -1
        self._pressure_nullspace_component_face_valid_mask_generation = -1

    def prepare_projection_transaction(
        self,
        *,
        fluid: CartesianFluidSolver,
        pressure_solve_context: Mapping[str, object],
    ) -> None:
        # A new affine Q transaction invalidates every pressure-nullspace
        # identity retained by the preceding Q/P cycle.  Clear the runner-side
        # references before any operation that can fail so stale A/mask fields
        # can never be reused after a partial prepare.
        self._clear_pressure_nullspace_transaction_state()
        if not isinstance(pressure_solve_context, Mapping):
            raise TypeError("pressure_solve_context must be a mapping")
        component_face_valid_mask = (
            fluid.prepare_hibm_no_slip_component_face_valid_mask()
        )
        sampling_obstacle = fluid.hibm_no_slip_sampling_obstacle
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        sampling_identity = self.markers_owner.prepare_no_slip_sampling_identity(
            obstacle_field=sampling_obstacle,
            component_face_valid_mask=component_face_valid_mask,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
        )
        self.operator.prepare(
            markers=self.markers_owner,
            fluid=fluid,
            component_face_valid_mask=component_face_valid_mask,
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
            prepared_sampling_identity=sampling_identity,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
        )
        self._prepared_fluid = fluid
        self._prepared_sampling_identity = sampling_identity
        self._prepared_component_face_valid_mask = component_face_valid_mask
        self._prepared_sampling_obstacle = sampling_obstacle
        self._prepared_topology_generation = topology_generation
        self._prepared_component_face_valid_mask_generation = valid_mask_generation
        self._pressure_solve_context = {
            str(key): value for key, value in pressure_solve_context.items()
        }

    def solve_projection_transaction(self) -> None:
        fluid = self._require_prepared_fluid()
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        self.operator.solve_device(
            max_iterations=self.max_iterations,
            absolute_tolerance_mps=self.absolute_tolerance_mps,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
            obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
        )

    def commit_projection_transaction(self) -> Mapping[str, object]:
        fluid = self._require_prepared_fluid()
        topology_generation, valid_mask_generation = self._current_generations(fluid)
        self.operator.commit_if_converged(
            fluid,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
            topology_generation=topology_generation,
            component_face_valid_mask_generation=valid_mask_generation,
            obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
        )
        report = asdict(self.operator.report())
        return {
            **report,
            "prepared": bool(report["prepared"]),
            "converged": bool(report["converged"]),
            "committed": bool(report["committed"]),
            "pressure_solve_context": dict(self._pressure_solve_context),
            "topology_generation": int(self._prepared_topology_generation),
            "component_face_valid_mask_generation": int(
                self._prepared_component_face_valid_mask_generation
            ),
        }

    def prepare_pressure_nullspace_transaction(
        self,
        *,
        fluid: CartesianFluidSolver,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> None:
        """Bind the pressure projector to the just-committed affine Q state."""

        self._clear_pressure_nullspace_transaction_state()
        prepared_fluid = self._require_prepared_fluid()
        if self.operator._phase != "committed":
            raise RuntimeError(
                "pressure nullspace prepare requires a committed marker Q transaction"
            )
        if fluid is not prepared_fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        if (
            component_face_valid_mask
            is not self._prepared_component_face_valid_mask
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if (
            pressure_actuated_component_mobility
            is not fluid.pressure_velocity_actuation_weight
        ):
            raise RuntimeError("pressure actuation weight owner changed")

        current_topology_generation, current_valid_mask_generation = (
            self._current_generations(fluid)
        )
        supplied_generations = (
            pressure_actuation_generation,
            topology_generation,
            component_face_valid_mask_generation,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or int(value) < 0
            for value in supplied_generations
        ):
            raise ValueError(
                "pressure nullspace generations must be non-negative integers"
            )
        if int(pressure_actuation_generation) != int(
            fluid.pressure_velocity_actuation_generation
        ):
            raise RuntimeError("pressure actuation generation changed")
        if int(topology_generation) != current_topology_generation or int(
            topology_generation
        ) != int(self._prepared_topology_generation):
            raise RuntimeError("pressure nullspace topology generation changed")
        if int(component_face_valid_mask_generation) != (
            current_valid_mask_generation
        ) or int(component_face_valid_mask_generation) != int(
            self._prepared_component_face_valid_mask_generation
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask generation changed"
            )

        self.operator.prepare_pressure_nullspace_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=int(pressure_actuation_generation),
            topology_generation=int(topology_generation),
            component_face_valid_mask_generation=int(
                component_face_valid_mask_generation
            ),
        )
        # Publish the identities only after the device-side factorization has
        # completed.  A failed prepare therefore leaves the wrapper unusable.
        self._pressure_nullspace_fluid = fluid
        self._pressure_actuated_component_mobility = (
            pressure_actuated_component_mobility
        )
        self._pressure_nullspace_component_face_valid_mask = (
            component_face_valid_mask
        )
        self._pressure_actuation_generation = int(pressure_actuation_generation)
        self._pressure_nullspace_topology_generation = int(topology_generation)
        self._pressure_nullspace_component_face_valid_mask_generation = int(
            component_face_valid_mask_generation
        )

    def _require_current_pressure_nullspace_transaction(
        self,
        *,
        component_face_valid_mask=None,
    ) -> tuple[CartesianFluidSolver, object, object]:
        """Validate immutable pressure/Q identities without device reads."""

        fluid = self._require_prepared_fluid()
        pressure_fluid = self._pressure_nullspace_fluid
        pressure_actuation_weight = self._pressure_actuated_component_mobility
        prepared_valid_mask = self._pressure_nullspace_component_face_valid_mask
        if (
            pressure_fluid is None
            or pressure_actuation_weight is None
            or prepared_valid_mask is None
        ):
            raise RuntimeError(
                "pressure constraint nullspace transaction is not prepared"
            )
        if self.operator._phase != "committed":
            raise RuntimeError(
                "pressure nullspace apply requires a committed marker Q transaction"
            )
        if fluid is not pressure_fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        if prepared_valid_mask is not self._prepared_component_face_valid_mask:
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if (
            component_face_valid_mask is not None
            and component_face_valid_mask is not prepared_valid_mask
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if pressure_actuation_weight is not fluid.pressure_velocity_actuation_weight:
            raise RuntimeError("pressure actuation weight owner changed")

        topology_generation, valid_mask_generation = self._current_generations(fluid)
        if int(fluid.pressure_velocity_actuation_generation) != int(
            self._pressure_actuation_generation
        ):
            raise RuntimeError("pressure actuation generation changed")
        if topology_generation != int(
            self._pressure_nullspace_topology_generation
        ):
            raise RuntimeError("pressure nullspace topology generation changed")
        if valid_mask_generation != int(
            self._pressure_nullspace_component_face_valid_mask_generation
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask generation changed"
            )
        return fluid, pressure_actuation_weight, prepared_valid_mask

    def project_pressure_actuated_grid_vector_to_marker_nullspace(
        self,
        *,
        input_velocity_mps,
        output_velocity_mps,
        max_iterations: int,
        absolute_tolerance_mps: float,
        component_face_valid_mask,
    ) -> None:
        """Apply one device-only matvec projection with immutable identities."""

        if isinstance(max_iterations, (bool, np.bool_)) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if int(max_iterations) != max_iterations:
            raise ValueError("max_iterations must be a positive integer")
        if isinstance(absolute_tolerance_mps, (bool, np.bool_)):
            raise ValueError(
                "absolute_tolerance_mps must be finite and positive"
            )
        tolerance = float(absolute_tolerance_mps)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "absolute_tolerance_mps must be finite and positive"
            )
        fluid, pressure_actuation_weight, prepared_valid_mask = (
            self._require_current_pressure_nullspace_transaction(
                component_face_valid_mask=component_face_valid_mask
            )
        )

        self.operator.apply_pressure_nullspace_transaction_device_only(
            input_face_correction=input_velocity_mps,
            output_face_correction=output_velocity_mps,
            fluid=fluid,
            pressure_actuated_component_mobility=pressure_actuation_weight,
            component_face_valid_mask=prepared_valid_mask,
            pressure_actuation_generation=int(self._pressure_actuation_generation),
            topology_generation=int(self._pressure_nullspace_topology_generation),
            component_face_valid_mask_generation=int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
        )

    def finalize_pressure_nullspace_transaction(self) -> Mapping[str, object]:
        """Read and validate accumulated device audits exactly once per solve."""

        fluid, pressure_actuation_weight, prepared_valid_mask = (
            self._require_current_pressure_nullspace_transaction()
        )
        report = self.operator.finalize_pressure_nullspace_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=pressure_actuation_weight,
            component_face_valid_mask=prepared_valid_mask,
            pressure_actuation_generation=int(self._pressure_actuation_generation),
            topology_generation=int(self._pressure_nullspace_topology_generation),
            component_face_valid_mask_generation=int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
            absolute_tolerance_mps=float(self.absolute_tolerance_mps),
        )
        scalar_report = asdict(report)
        if int(scalar_report["pressure_actuation_generation"]) != int(
            self._pressure_actuation_generation
        ):
            raise RuntimeError(
                "pressure nullspace report actuation generation changed"
            )
        return {
            **scalar_report,
            "topology_generation": int(
                self._pressure_nullspace_topology_generation
            ),
            "component_face_valid_mask_generation": int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
        }


def _empty_hibm_sharp_marker_boundary_report() -> dict[str, object]:
    return {
        "flow_solid_boundary_mode": FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS,
        "hibm_sharp_marker_boundary_enabled": False,
        "hibm_sharp_marker_boundary_search_radius_xyz_m": "",
        "hibm_dynamic_solid_volume_enabled": False,
        "hibm_sharp_marker_boundary_search_reused": False,
        "hibm_sharp_marker_boundary_topology_reused": False,
        "hibm_sharp_marker_boundary_topology_only": False,
        "hibm_sharp_marker_boundary_near_node_count": 0,
        "hibm_sharp_marker_boundary_external_node_count": 0,
        "hibm_sharp_marker_boundary_internal_node_count": 0,
        "hibm_sharp_marker_boundary_internal_obstacle_cell_count": 0,
        "hibm_sharp_marker_boundary_no_slip_rows": 0,
        "hibm_sharp_marker_boundary_pressure_neumann_rows": 0,
        "hibm_sharp_marker_boundary_pressure_gradient_updated": False,
        **_hibm_sharp_boundary_timing_report_fields(
            _empty_hibm_sharp_boundary_stage_wall_times()
        ),
        "hibm_preassembly_overflow_singleton_cleanup_cell_count": 0,
        "hibm_preassembly_overflow_singleton_cleanup_component_count": 0,
        "hibm_preassembly_tiny_unreached_cleanup_cell_count": 0,
        "hibm_preassembly_tiny_unreached_cleanup_component_count": 0,
        "hibm_preassembly_tiny_unreached_cleanup_pass_count": 0,
        "hibm_preassembly_remaining_unreached_cell_count": 0,
        "hibm_preassembly_cleanup_reused": False,
        "hibm_preassembly_topology_mutated": False,
        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": 0,
        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": 0,
        "hibm_pressure_neumann_skipped_obstacle_owner_count": 0,
        "hibm_pressure_neumann_relocated_obstacle_owner_count": 0,
        "hibm_pressure_neumann_duplicate_owner_count": 0,
        "hibm_pressure_neumann_invalid_reconstruction_count": 0,
        "hibm_pressure_neumann_invalid_unreconstructable_count": 0,
        "hibm_pressure_neumann_invalid_bad_marker_count": 0,
        "hibm_pressure_neumann_invalid_nonpositive_volume_count": 0,
    }


def _hibm_sharp_search_radius_m(config: Any) -> float:
    configured = getattr(config, "flow_hibm_sharp_search_radius_m", None)
    if configured is not None:
        return float(configured)
    return 2.5 * max(_grid_spacing_m(config))


def _hibm_sharp_search_radius_xyz_m(
    config: Any,
) -> tuple[float, float, float] | None:
    configured = getattr(config, "flow_hibm_sharp_search_radius_xyz_m", None)
    if configured is None:
        return None
    values = tuple(float(value) for value in configured)
    if len(values) != 3:
        raise ValueError(
            "flow_hibm_sharp_search_radius_xyz_m must contain exactly three values"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "flow_hibm_sharp_search_radius_xyz_m must contain finite positive values"
        )
    return values


def _hibm_sharp_interior_probe_distance_m(config: Any) -> float:
    configured = getattr(config, "flow_hibm_sharp_interior_probe_distance_m", None)
    if configured is not None:
        return float(configured)
    return 1.5 * max(_grid_spacing_m(config))


def _hibm_sharp_interior_probe_distance_xyz_m(
    config: Any,
) -> tuple[float, float, float] | None:
    configured = getattr(
        config,
        "flow_hibm_sharp_interior_probe_distance_xyz_m",
        None,
    )
    if configured is None:
        return None
    values = tuple(float(value) for value in configured)
    if len(values) != 3:
        raise ValueError(
            "flow_hibm_sharp_interior_probe_distance_xyz_m must contain "
            "exactly three values"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "flow_hibm_sharp_interior_probe_distance_xyz_m must contain "
            "finite positive values"
        )
    return values


def _apply_hibm_sharp_marker_boundary_to_fluid(
    markers: HibmMpmSurfaceMarkers | None,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    update_pressure_gradient: bool,
    boundary_cache: dict[str, object] | None = None,
    reuse_topology_from_previous_assembly: bool = False,
    topology_only: bool = False,
    measure_wall_times: bool = False,
    stage_observer: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if not _use_hibm_sharp_marker_boundary(config):
        return _empty_hibm_sharp_marker_boundary_report()
    if markers is None:
        raise ValueError("hibm_sharp_marker_rows requires surface markers")
    stage_wall_times = _empty_hibm_sharp_boundary_stage_wall_times()

    bounds_min, bounds_max = _domain_bounds(config)
    marker_capacity = max(
        int(getattr(markers, "marker_capacity", 0)),
        int(getattr(markers, "marker_count", 0)),
        1,
    )
    search_radius_m = _hibm_sharp_search_radius_m(config)
    search_radius_xyz_m = _hibm_sharp_search_radius_xyz_m(config)
    interior_probe_distance_m = _hibm_sharp_interior_probe_distance_m(config)
    interior_probe_distance_xyz_m = (
        _hibm_sharp_interior_probe_distance_xyz_m(config)
    )
    dynamic_solid_volume_enabled = bool(
        getattr(config, "flow_hibm_dynamic_solid_volume_enabled", False)
    )
    marker_mac_constraint_iterations = _hibm_marker_mac_constraint_iterations(
        config
    )
    marker_mac_constraint_absolute_tolerance_mps = (
        _hibm_marker_mac_constraint_absolute_tolerance_mps(config)
    )
    marker_compatibility_closure_tolerance_mps = (
        _hibm_marker_compatibility_closure_tolerance_mps(config)
    )
    # Resource identity is deliberately independent from the current marker
    # geometry and from search parameters.  Those values change frequently in
    # FSI but do not change the Taichi field shapes, so the allocated search,
    # boundary, and projection objects remain reusable.
    resource_cache_key = (
        tuple(config.grid_nodes),
        tuple(float(value) for value in bounds_min),
        tuple(float(value) for value in bounds_max),
        int(marker_capacity),
        int(marker_mac_constraint_iterations),
        float(marker_mac_constraint_absolute_tolerance_mps),
    )
    # Classified topology has a stricter identity.  In particular, marker
    # count alone cannot detect an in-place moving surface, so the marker-owned
    # geometry revision participates in every topology reuse decision.  The
    # fluid-owned external obstacle epoch closes the complementary case where
    # an obstacle changes without a marker write.  Cleanup-policy inputs are
    # included because the cached cleanup report is part of this identity.
    classified_topology_key = (
        int(getattr(markers, "marker_geometry_revision", 0)),
        int(getattr(fluid, "hibm_external_obstacle_topology_revision", 0)),
        str(fluid.velocity_dirichlet_boundary_authority),
        int(getattr(fluid, "velocity_dirichlet_face_symmetric", 0)),
        int(getattr(markers, "marker_count", 0)),
        int(getattr(markers, "projection_vertex_count", 0)),
        int(getattr(markers, "projection_triangle_count", 0)),
        int(getattr(markers, "projection_segment_count", 0)),
        float(search_radius_m),
        (
            None
            if search_radius_xyz_m is None
            else tuple(float(value) for value in search_radius_xyz_m)
        ),
        float(interior_probe_distance_m),
        False,  # classify_far_internal_nodes
        OUT_OF_PLANE_AXIS_INDEX,
        bool(dynamic_solid_volume_enabled),
        bool(
            getattr(
                config,
                "flow_hibm_sharp_interpolate_velocity_rows",
                True,
            )
        ),
        bool(getattr(config, "flow_pressure_outlet_enabled", True)),
        int(
            getattr(
                config,
                "flow_hibm_tiny_unreached_cleanup_component_cells",
                0,
            )
        ),
    )
    cache_entry = (
        boundary_cache.get("hibm_sharp_marker_boundary")
        if boundary_cache is not None
        else None
    )
    search_reused = bool(
        isinstance(cache_entry, dict)
        and cache_entry.get("cache_key") == resource_cache_key
        and cache_entry.get("markers_owner") is markers
        and isinstance(
            cache_entry.get("pre_projection_velocity_projector"),
            _HibmPreProjectionVelocityProjector,
        )
    )
    if search_reused:
        ib_search = cache_entry["ib_search"]
        ib_boundary = cache_entry["ib_boundary"]
        pre_projection_velocity_projector = cache_entry[
            "pre_projection_velocity_projector"
        ]
        if pre_projection_velocity_projector.markers_owner is not markers:
            raise RuntimeError(
                "cached HIBM pre-projection velocity projector marker owner changed"
            )
    else:
        if stage_observer is not None:
            stage_observer("hibm_resource_allocate_before")
        runtime = TaichiRuntimeConfig(arch="cuda")
        ib_search = HibmMpmIbNodeSearch(
            grid_nodes=tuple(config.grid_nodes),
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        ib_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=tuple(config.grid_nodes),
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        pre_projection_velocity_projector = _HibmPreProjectionVelocityProjector(
            markers=markers,
            operator=HibmMpmMarkerMacConstraintOperator(
                grid_nodes=tuple(config.grid_nodes),
                marker_capacity=marker_capacity,
            ),
            max_iterations=marker_mac_constraint_iterations,
            absolute_tolerance_mps=(
                marker_mac_constraint_absolute_tolerance_mps
            ),
        )
        cache_entry = {
            "cache_key": resource_cache_key,
            "markers_owner": markers,
            "ib_search": ib_search,
            "ib_boundary": ib_boundary,
            "pre_projection_velocity_projector": (
                pre_projection_velocity_projector
            ),
        }
        if boundary_cache is not None:
            boundary_cache["hibm_sharp_marker_boundary"] = cache_entry
        if stage_observer is not None:
            stage_observer("hibm_resource_allocate_after")
    topology_reused = bool(
        reuse_topology_from_previous_assembly
        and isinstance(cache_entry, dict)
        and cache_entry.get("cache_key") == resource_cache_key
        and cache_entry.get("markers_owner") is markers
        and cache_entry.get("classified_topology_key")
        == classified_topology_key
        and "search_report" in cache_entry
        and "internal_obstacle_cell_count" in cache_entry
    )
    if topology_reused:
        search_report = cache_entry["search_report"]
        internal_obstacle_cell_count = int(
            cache_entry["internal_obstacle_cell_count"]
        )
    else:
        if stage_observer is not None:
            stage_observer("hibm_search_classify_before")
        if isinstance(cache_entry, dict):
            # Invalidate host metadata before search overwrites its device
            # fields.  If search or obstacle publication raises, a later call
            # must reclassify instead of accepting the previous report for a
            # partially replaced topology.
            cache_entry.pop("classified_topology_key", None)
            cache_entry.pop("search_report", None)
            cache_entry.pop("internal_obstacle_cell_count", None)
            cache_entry.pop("cleanup_report", None)
        search_report = ib_search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            cell_width_x_m=fluid.cell_width_x_m,
            cell_width_y_m=fluid.cell_width_y_m,
            cell_width_z_m=fluid.cell_width_z_m,
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            classify_far_internal_nodes=False,
            search_radius_xyz_m=search_radius_xyz_m,
            search_inactive_axis=OUT_OF_PLANE_AXIS_INDEX,
        )
        if stage_observer is not None:
            stage_observer("hibm_search_classify_after")
            stage_observer("hibm_internal_obstacle_publish_before")
        internal_obstacle_cell_count = fluid.apply_hibm_internal_obstacles(
            ib_search.node_kind_code,
            internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
            external_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
            carve_external_nodes_from_dynamic_volume=dynamic_solid_volume_enabled,
            convert_internal_nodes=not dynamic_solid_volume_enabled,
        )
        if stage_observer is not None:
            stage_observer("hibm_internal_obstacle_publish_after")
        if isinstance(cache_entry, dict):
            cache_entry["classified_topology_key"] = classified_topology_key
            cache_entry["search_report"] = search_report
            cache_entry["internal_obstacle_cell_count"] = int(
                internal_obstacle_cell_count
            )
    if bool(topology_only):
        if isinstance(cache_entry, dict):
            # A topology-only observer does not execute reachability cleanup.
            # Never carry an earlier full-assembly cleanup claim through this
            # path, even when the classified search topology itself was reused.
            cache_entry.pop("cleanup_report", None)
        topology_report = _empty_hibm_sharp_marker_boundary_report()
        topology_report.update(
            {
                "flow_solid_boundary_mode": _flow_solid_boundary_mode(config),
                "hibm_sharp_marker_boundary_enabled": True,
                "hibm_dynamic_solid_volume_enabled": dynamic_solid_volume_enabled,
                "hibm_sharp_marker_boundary_search_reused": bool(search_reused),
                "hibm_sharp_marker_boundary_topology_reused": bool(
                    topology_reused
                ),
                "hibm_sharp_marker_boundary_near_node_count": (
                    search_report.near_boundary_node_count
                ),
                "hibm_sharp_marker_boundary_external_node_count": (
                    search_report.external_ib_node_count
                ),
                "hibm_sharp_marker_boundary_internal_node_count": (
                    search_report.internal_node_count
                ),
                "hibm_sharp_marker_boundary_internal_obstacle_cell_count": int(
                    internal_obstacle_cell_count
                ),
                "hibm_sharp_marker_boundary_topology_only": True,
            }
        )
        return topology_report
    if update_pressure_gradient:
        markers.update_pressure_neumann_gradient_from_fluid_predictor(
            ib_boundary.marker_pressure_neumann_gradient_field,
            velocity_field=fluid.velocity,
            obstacle_field=fluid.obstacle,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            density_kgm3=float(config.air_density_kgm3),
            dt_s=float(config.dt_s),
            probe_distance_m=interior_probe_distance_m,
            probe_distance_xyz_m=interior_probe_distance_xyz_m,
        )
    if stage_observer is not None:
        stage_observer("hibm_boundary_build_before")
    ib_boundary.build_from_search_device_fields(
        ib_search,
        markers,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            ib_boundary.marker_pressure_neumann_gradient_field
        ),
    )
    if update_pressure_gradient:
        ib_boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
            velocity_field=fluid.velocity,
            obstacle_field=fluid.obstacle,
            search=ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            density_kgm3=float(config.air_density_kgm3),
            dt_s=float(config.dt_s),
        )

    def assemble_velocity_rows() -> dict[str, object]:
        authority = str(fluid.velocity_dirichlet_boundary_authority)
        interpolate_interior_velocity = bool(
            getattr(config, "flow_hibm_sharp_interpolate_velocity_rows", True)
        )
        if authority != "canonical":
            raise RuntimeError(
                "unsupported velocity Dirichlet boundary authority for HIBM "
                f"assembly: {authority!r}"
            )

        canonical_ledger_observer_wall_time_s = 0.0
        canonical_ledger_stage_observer: Callable[[str], None] | None = None
        if stage_observer is not None:
            def emit_canonical_ledger_stage(stage: str) -> None:
                nonlocal canonical_ledger_observer_wall_time_s
                observer_started_s = time.perf_counter()
                try:
                    stage_observer(stage)
                finally:
                    canonical_ledger_observer_wall_time_s += max(
                        0.0, time.perf_counter() - observer_started_s
                    )

            canonical_ledger_stage_observer = emit_canonical_ledger_stage
        fluid._invalidate_velocity_dirichlet_component_ledger()
        builder_result = _measure_hibm_sharp_boundary_stage(
            stage_wall_times,
            "canonical_ledger_build",
            lambda: ib_boundary.assemble_velocity_dirichlet_component_face_ledger(
            velocity_dirichlet_active_component_mask=(
                fluid.velocity_dirichlet_boundary_active_component_mask
            ),
            velocity_dirichlet_value_mps=(
                fluid.velocity_dirichlet_boundary_value_mps
            ),
            velocity_dirichlet_pressure_mobility=(
                fluid.velocity_dirichlet_boundary_pressure_mobility
            ),
            velocity_dirichlet_component_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_component_enforcement_weight
            ),
            velocity_dirichlet_component_region_id=(
                fluid.velocity_dirichlet_boundary_component_region_id
            ),
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_component_mask=(
                fluid.velocity_dirichlet_boundary_owned_component_mask
            ),
            obstacle_field=fluid.obstacle,
            velocity_field=fluid.velocity,
            search=ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            marker_region_id=markers.region_id,
            surface_projection_inactive_axis=OUT_OF_PLANE_AXIS_INDEX,
            markers=markers,
            marker_compatibility_iterations_per_batch=64,
            marker_compatibility_absolute_tolerance_mps=(
                marker_mac_constraint_absolute_tolerance_mps
            ),
            marker_compatibility_closure_tolerance_mps=(
                marker_compatibility_closure_tolerance_mps
            ),
            marker_compatibility_density_kgm3=float(fluid.rho),
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
            interpolate_interior_velocity=interpolate_interior_velocity,
            stage_observer=canonical_ledger_stage_observer,
            ),
            enabled=measure_wall_times,
            excluded_wall_time=(
                (lambda: canonical_ledger_observer_wall_time_s)
                if canonical_ledger_stage_observer is not None
                else None
            ),
        )
        _measure_hibm_sharp_boundary_stage(
            stage_wall_times,
            "canonical_prepare_seal",
            lambda: _prepare_and_seal_canonical_velocity_dirichlet_component_ledger(
                fluid
            ),
            enabled=measure_wall_times,
        )
        return _canonical_hibm_velocity_dirichlet_report_fields(
            builder_result,
            fluid=fluid,
        )

    if stage_observer is not None:
        stage_observer("hibm_velocity_row_assembly_before")
    velocity_report = assemble_velocity_rows()
    if stage_observer is not None:
        stage_observer("hibm_velocity_row_assembly_after")
    fluid.clear_pressure_interface_matrix_terms()
    cleanup_report = {
        "hibm_preassembly_overflow_singleton_cleanup_cell_count": 0,
        "hibm_preassembly_overflow_singleton_cleanup_component_count": 0,
        "hibm_preassembly_tiny_unreached_cleanup_cell_count": 0,
        "hibm_preassembly_tiny_unreached_cleanup_component_count": 0,
        "hibm_preassembly_tiny_unreached_cleanup_pass_count": 0,
        "hibm_preassembly_remaining_unreached_cell_count": 0,
        "hibm_preassembly_cleanup_reused": False,
        "hibm_preassembly_topology_mutated": False,
    }

    def refresh_pressure_reachability() -> None:
        _measure_hibm_sharp_boundary_stage(
            stage_wall_times,
            "pressure_reachability_flood",
            lambda: fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            ),
            enabled=measure_wall_times,
        )

    def rebuild_velocity_rows_after_topology_mutation() -> None:
        nonlocal velocity_report
        if stage_observer is not None:
            stage_observer("hibm_velocity_row_assembly_before")
        velocity_report = assemble_velocity_rows()
        if stage_observer is not None:
            stage_observer("hibm_velocity_row_assembly_after")

    cleanup_reused = bool(
        topology_reused
        and isinstance(cache_entry, dict)
        and "cleanup_report" in cache_entry
    )
    if cleanup_reused:
        cleanup_report.update(dict(cache_entry.get("cleanup_report", {})))
        # The cached report records work performed by the assembly that
        # established this topology.  Reusing that saturated result performs
        # no obstacle conversion in the current assembly, so do not replay its
        # historical mutation bit as a new mutation.  The explicit reuse bit
        # keeps the cached provenance auditable while the current-call flag
        # remains safe for predictor/projection topology guards.
        cleanup_report["hibm_preassembly_cleanup_reused"] = True
        cleanup_report["hibm_preassembly_topology_mutated"] = False
        # Reusing the topology-cleanup result only proves that no obstacle or
        # row-cloud conversion has to be repeated.  Clearing the pressure
        # interface terms above deliberately invalidates the reachability
        # state, so refresh the flood before projection advertises it as
        # prepared.  This is independent of the particular solid geometry.
        if bool(getattr(config, "flow_pressure_outlet_enabled", True)):
            refresh_pressure_reachability()
            cleanup_report[
                "hibm_preassembly_remaining_unreached_cell_count"
            ] = int(fluid.last_hibm_pressure_unreached_cell_count)
    elif bool(getattr(config, "flow_pressure_outlet_enabled", True)):
        tiny_cleanup_threshold = int(
            getattr(
                config,
                "flow_hibm_tiny_unreached_cleanup_component_cells",
                0,
            )
        )
        topology_stable = False
        for _topology_cleanup_pass in range(8):
            refresh_pressure_reachability()
            converted_overflow_singletons = (
                fluid.convert_hibm_row_cloud_orphan_components(
                    max_component_cells=1,
                    overflow_singletons_only=True,
                    protect_velocity_dirichlet_radius_cells=2,
                )
            )
            overflow_component_count = 0
            if int(converted_overflow_singletons) > 0:
                overflow_component_count = int(
                    fluid.last_hibm_row_cloud_orphan_component_count
                )
                rebuild_velocity_rows_after_topology_mutation()
                refresh_pressure_reachability()
            tiny_cleanup_report = (
                fluid.cleanup_hibm_pressure_outlet_tiny_unreached_components(
                    max_component_cells=tiny_cleanup_threshold,
                    reachability_is_current=True,
                    after_topology_mutation=(
                        rebuild_velocity_rows_after_topology_mutation
                    ),
                )
            )
            cleanup_report[
                "hibm_preassembly_overflow_singleton_cleanup_cell_count"
            ] += int(converted_overflow_singletons)
            cleanup_report[
                "hibm_preassembly_overflow_singleton_cleanup_component_count"
            ] += int(overflow_component_count)
            for key in (
                "hibm_preassembly_tiny_unreached_cleanup_cell_count",
                "hibm_preassembly_tiny_unreached_cleanup_component_count",
                "hibm_preassembly_tiny_unreached_cleanup_pass_count",
            ):
                cleanup_report[key] += int(tiny_cleanup_report[key])
            topology_mutated_this_pass = bool(
                int(converted_overflow_singletons) > 0
                or int(
                    tiny_cleanup_report[
                        "hibm_preassembly_tiny_unreached_cleanup_cell_count"
                    ]
                )
                > 0
            )
            if not topology_mutated_this_pass:
                topology_stable = True
                break
            cleanup_report["hibm_preassembly_topology_mutated"] = True
        if not topology_stable:
            refresh_pressure_reachability()
            raise RuntimeError(
                "HIBM preassembly topology cleanup did not saturate after 8 passes"
            )
        cleanup_report[
            "hibm_preassembly_remaining_unreached_cell_count"
        ] = int(fluid.last_hibm_pressure_unreached_cell_count)
        if isinstance(cache_entry, dict):
            cache_entry["cleanup_report"] = dict(cleanup_report)
    pressure_report = _measure_hibm_sharp_boundary_stage(
        stage_wall_times,
        "pressure_neumann_assembly",
        lambda: ib_boundary.assemble_pressure_neumann_matrix_rows(
        fluid.pressure_interface_matrix_diagonal,
        fluid.pressure_interface_matrix_rhs,
        fluid.pressure_interface_coupling_active,
        fluid.pressure_interface_coupling_neighbor,
        fluid.pressure_interface_coupling_coefficient,
        fluid.obstacle,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        ib_search,
        markers,
        pressure_coupling_extra_neighbor=(
            fluid.pressure_interface_coupling_extra_neighbor
        ),
        pressure_coupling_extra_coefficient=(
            fluid.pressure_interface_coupling_extra_coefficient
        ),
        pressure_interface_row_count=fluid.pressure_interface_row_count,
        pressure_interface_row_owner=fluid.pressure_interface_row_owner,
        pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
        pressure_interface_row_transmissibility=(
            fluid.pressure_interface_row_transmissibility
        ),
        pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
        ),
        enabled=measure_wall_times,
    )
    if stage_observer is not None:
        stage_observer("hibm_boundary_build_after")
    return {
        "flow_solid_boundary_mode": _flow_solid_boundary_mode(config),
        "hibm_sharp_marker_boundary_enabled": True,
        "hibm_sharp_marker_boundary_search_radius_xyz_m": (
            list(search_radius_xyz_m) if search_radius_xyz_m is not None else ""
        ),
        "hibm_dynamic_solid_volume_enabled": dynamic_solid_volume_enabled,
        "hibm_sharp_marker_boundary_search_reused": bool(search_reused),
        "hibm_sharp_marker_boundary_topology_reused": bool(topology_reused),
        "hibm_sharp_marker_boundary_topology_only": False,
        "hibm_sharp_marker_boundary_near_node_count": (
            search_report.near_boundary_node_count
        ),
        "hibm_sharp_marker_boundary_external_node_count": (
            search_report.external_ib_node_count
        ),
        "hibm_sharp_marker_boundary_internal_node_count": (
            search_report.internal_node_count
        ),
        "hibm_sharp_marker_boundary_internal_obstacle_cell_count": (
            int(internal_obstacle_cell_count)
        ),
        "hibm_sharp_marker_boundary_no_slip_rows": int(
            velocity_report["canonical_velocity_dirichlet_report"][
                "final_active_storage_row_count"
            ]
        ),
        **_hibm_velocity_dirichlet_mapping_fields(velocity_report),
        "hibm_sharp_marker_boundary_pressure_neumann_rows": (
            pressure_report.active_pressure_neumann_rows
        ),
        "hibm_sharp_marker_boundary_pressure_gradient_updated": bool(
            update_pressure_gradient
        ),
        **_hibm_sharp_boundary_timing_report_fields(stage_wall_times),
        **cleanup_report,
        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
            pressure_report.skipped_velocity_dirichlet_row_count
        ),
        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
            pressure_report.skipped_pressure_boundary_adjacent_row_count
        ),
        "hibm_pressure_neumann_skipped_obstacle_owner_count": (
            pressure_report.skipped_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_relocated_obstacle_owner_count": (
            pressure_report.relocated_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_duplicate_owner_count": (
            pressure_report.duplicate_owner_row_count
        ),
        "hibm_pressure_neumann_invalid_reconstruction_count": (
            pressure_report.invalid_reconstruction_row_count
        ),
        "hibm_pressure_neumann_invalid_unreconstructable_count": (
            pressure_report.invalid_unreconstructable_row_count
        ),
        "hibm_pressure_neumann_invalid_bad_marker_count": (
            pressure_report.invalid_bad_marker_row_count
        ),
        "hibm_pressure_neumann_invalid_nonpositive_volume_count": (
            pressure_report.invalid_nonpositive_volume_row_count
        ),
    }


def _hibm_pre_projection_velocity_projector_from_cache(
    boundary_cache: dict[str, object] | None,
    *,
    markers: HibmMpmSurfaceMarkers,
) -> _HibmPreProjectionVelocityProjector:
    if boundary_cache is None:
        raise RuntimeError(
            "sharp HIBM pre-projection velocity projector cache is unavailable"
        )
    cache_entry = boundary_cache.get("hibm_sharp_marker_boundary")
    if not isinstance(cache_entry, dict):
        raise RuntimeError(
            "sharp HIBM pre-projection velocity projector cache entry is unavailable"
        )
    projector = cache_entry.get("pre_projection_velocity_projector")
    if not isinstance(projector, _HibmPreProjectionVelocityProjector):
        raise RuntimeError(
            "sharp HIBM pre-projection velocity projector is unavailable"
        )
    if projector.markers_owner is not markers:
        raise RuntimeError(
            "sharp HIBM pre-projection velocity projector marker owner changed"
        )
    return projector


def _flow_turbulence_model(config: Any) -> str:
    model = str(getattr(config, "flow_turbulence_model", "laminar")).lower()
    if model not in FLOW_TURBULENCE_MODELS:
        raise ValueError(f"unsupported flow_turbulence_model: {model!r}")
    return model


def _flow_predictor_kinematic_viscosity_m2_s(config: Any) -> float:
    molecular_nu = float(getattr(config, "air_viscosity_pa_s", 0.0)) / max(
        float(getattr(config, "air_density_kgm3", 1.0)),
        1.0e-30,
    )
    multiplier = float(
        getattr(config, "flow_predictor_kinematic_viscosity_multiplier", 1.0)
    )
    return molecular_nu * multiplier


def _flow_predictor_no_slip_domain_walls(
    config: Any,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    return _flow_domain_wall_flags(
        config,
        field_name="flow_predictor_no_slip_domain_walls",
    )


def _flow_symmetry_domain_walls(
    config: Any,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    return _flow_domain_wall_flags(
        config,
        field_name="flow_symmetry_domain_walls",
    )


def _flow_domain_wall_flags(
    config: Any,
    *,
    field_name: str,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    raw_walls = getattr(config, field_name, ())
    if raw_walls is None:
        names: tuple[str, ...] = ()
    elif isinstance(raw_walls, str):
        names = tuple(
            part.strip().lower()
            for part in raw_walls.split(",")
            if part.strip()
        )
    else:
        names = tuple(str(part).strip().lower() for part in raw_walls if str(part).strip())
    flags = [False, False, False, False, False, False]
    unsupported = sorted(
        {name for name in names if name not in FLOW_PREDICTOR_NO_SLIP_WALL_INDEX}
    )
    if unsupported:
        raise ValueError(
            f"unsupported {field_name} entries: {unsupported!r}"
        )
    for name in names:
        flags[FLOW_PREDICTOR_NO_SLIP_WALL_INDEX[name]] = True
    return tuple(flags)


def _apply_ymin_no_slip_rows(
    active: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    marker_regions: np.ndarray,
    hard_masks: np.ndarray,
    external_exact_masks: np.ndarray,
    owned_rows: np.ndarray,
    obstacle: np.ndarray,
    config: Any,
) -> None:
    rows = int(getattr(config, "flow_ymin_no_slip_rows", 0))
    if rows <= 0:
        return
    row_count = min(rows, active.shape[1])
    fluid_rows = obstacle[:, :row_count, :] == 0
    active_rows = active[:, :row_count, :]
    values_rows = values[:, :row_count, :, :]
    weights_rows = weights[:, :row_count, :]
    marker_region_rows = marker_regions[:, :row_count, :]
    hard_mask_rows = hard_masks[:, :row_count, :]
    external_exact_mask_rows = external_exact_masks[:, :row_count, :]
    owned_row_rows = owned_rows[:, :row_count, :]
    preserved_rows = active_rows != 0
    apply_rows = np.logical_and(fluid_rows, ~preserved_rows)
    clear_rows = np.logical_and(~fluid_rows, ~preserved_rows)

    active_rows[apply_rows] = 1
    values_rows[apply_rows, :] = 0.0
    weights_rows[apply_rows] = 1.0
    marker_region_rows[apply_rows] = -1
    hard_mask_rows[apply_rows] = 0b111
    external_exact_mask_rows[apply_rows] = 0
    ymin_external_mask = external_exact_mask_rows[:, 0, :]
    ymin_external_mask[apply_rows[:, 0, :]] = 0b010
    owned_row_rows[apply_rows] = 0

    active_rows[clear_rows] = 0
    values_rows[clear_rows, :] = 0.0
    weights_rows[clear_rows] = 0.0
    marker_region_rows[clear_rows] = -1
    hard_mask_rows[clear_rows] = 0
    external_exact_mask_rows[clear_rows] = 0
    owned_row_rows[clear_rows] = 0


def _apply_obstacle_no_slip_rows(
    active: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    obstacle: np.ndarray,
    config: Any,
) -> int:
    layers = int(getattr(config, "flow_obstacle_no_slip_layers", 0))
    wake_layers = int(getattr(config, "flow_obstacle_wake_no_slip_layers", 0))
    if layers <= 0 and wake_layers <= 0:
        return 0
    weight = float(getattr(config, "flow_obstacle_no_slip_weight", 1.0))
    cap_weight_config = getattr(config, "flow_obstacle_cap_no_slip_weight", None)
    cap_weight = weight if cap_weight_config is None else float(cap_weight_config)
    wake_weight = float(getattr(config, "flow_obstacle_wake_no_slip_weight", 0.5))
    fluid_mask = obstacle == 0
    solid_front = obstacle != 0
    selected = np.zeros_like(fluid_mask, dtype=bool)
    row_weights = np.zeros_like(weights, dtype=np.float32)
    for _layer in range(layers):
        x_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        x_adjacent[1:, :, :] |= solid_front[:-1, :, :]
        x_adjacent[:-1, :, :] |= solid_front[1:, :, :]
        y_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        y_adjacent[:, 1:, :] |= solid_front[:, :-1, :]
        y_adjacent[:, :-1, :] |= solid_front[:, 1:, :]
        z_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        z_adjacent[:, :, 1:] |= solid_front[:, :, :-1]
        z_adjacent[:, :, :-1] |= solid_front[:, :, 1:]
        adjacent_by_axis = (x_adjacent, y_adjacent, z_adjacent)
        adjacent = np.zeros_like(fluid_mask, dtype=bool)
        cap_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        for axis, axis_adjacent in enumerate(adjacent_by_axis):
            adjacent |= axis_adjacent
            if axis != STREAMWISE_AXIS_INDEX:
                cap_adjacent |= axis_adjacent
        layer_cells = adjacent & fluid_mask & ~selected
        selected |= layer_cells
        row_weights[layer_cells] = weight
        row_weights[layer_cells & cap_adjacent] = cap_weight
        solid_front = solid_front | layer_cells
    solid_mask = obstacle != 0
    for layer_index in range(1, wake_layers + 1):
        shifted = np.zeros_like(fluid_mask, dtype=bool)
        shifted[:, :, :-layer_index] |= solid_mask[:, :, layer_index:]
        layer_cells = shifted & fluid_mask & ~selected
        selected |= layer_cells
        row_weights[layer_cells] = wake_weight
    constrained = selected & (row_weights > 0.0)
    active[constrained] = 1
    values[constrained] = 0.0
    weights[constrained] = row_weights[constrained]
    return int(np.count_nonzero(constrained))


class _FlowPredictorSegmentConfig:
    """Read-only view of a flow config for one predictor/projection segment."""

    __slots__ = ("_base_config", "dt_s", "flow_predictor_substeps")

    def __init__(
        self,
        base_config: Any,
        *,
        dt_s: float,
    ) -> None:
        object.__setattr__(self, "_base_config", base_config)
        object.__setattr__(self, "dt_s", float(dt_s))
        object.__setattr__(self, "flow_predictor_substeps", 1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_config, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "flow predictor segment config is immutable; "
            f"cannot set {name!r}"
        )


def _combine_interleaved_flow_predictor_segment_reports(
    segment_reports: list[dict[str, object]],
    *,
    configured_substeps: int,
    segment_dt_s: float,
    reset_pressure: bool,
) -> dict[str, object]:
    """Combine full projection reports from interleaved physical segments."""

    if configured_substeps <= 0:
        raise ValueError("configured_substeps must be positive")
    if not math.isfinite(segment_dt_s) or segment_dt_s <= 0.0:
        raise ValueError("segment_dt_s must be finite and positive")
    if len(segment_reports) != configured_substeps:
        raise ValueError(
            "interleaved predictor requires exactly "
            f"{configured_substeps} segment reports, got {len(segment_reports)}"
        )
    for segment_report in segment_reports:
        _macro_time_accounting_report(
            requested_macro_dt_s=float(segment_dt_s),
            accepted_time_s=float(
                segment_report.get("requested_macro_dt_s", math.nan)
            ),
            accepted_substep_count=1,
            rejected_trial_count=0,
            component="fluid segment request",
        )
        _macro_time_accounting_report(
            requested_macro_dt_s=float(segment_dt_s),
            accepted_time_s=float(
                segment_report.get("fluid_accepted_time_s", math.nan)
            ),
            accepted_substep_count=segment_report[
                "flow_momentum_advection_substeps_total"
            ],
            rejected_trial_count=int(
                segment_report.get("fluid_rejected_trial_count", -1)
            ),
            component="fluid",
        )
    combined = dict(segment_reports[-1])
    combined.update(
        _macro_time_accounting_report(
            requested_macro_dt_s=math.fsum(
                float(segment_dt_s) for _ in range(configured_substeps)
            ),
            accepted_time_s=math.fsum(
                float(report["fluid_accepted_time_s"])
                for report in segment_reports
            ),
            accepted_substep_count=sum(
                int(report["flow_momentum_advection_substeps_total"])
                for report in segment_reports
            ),
            rejected_trial_count=sum(
                int(report["fluid_rejected_trial_count"])
                for report in segment_reports
            ),
            component="fluid",
        )
    )
    segment_projection_reports = [
        dict(report.get("projection_report", {})) for report in segment_reports
    ]
    combined_projection_report = _combine_flow_projection_reports(
        segment_projection_reports
    )
    for key in (
        "hibm_post_dirichlet_consistency_projection_count",
        "flow_sst_transport_substeps_total",
        "flow_sst_transport_rejected_trial_count_total",
        "flow_momentum_advection_substeps_total",
        "flow_momentum_advection_rejected_trial_count_total",
        "flow_sst_momentum_helmholtz_rejected_trial_count_total",
    ):
        if any(key in report for report in segment_reports):
            combined[key] = sum(int(report.get(key, 0)) for report in segment_reports)
    for key in (
        "flow_sst_transport_wall_time_s",
        "flow_momentum_predictor_wall_time_s",
        "hibm_pre_predictor_wall_time_s",
        "hibm_projection_cycle_wall_time_s",
    ):
        if any(key in report for report in segment_reports):
            combined[key] = math.fsum(
                float(report.get(key, 0.0)) for report in segment_reports
            )
    for key in (
        "flow_sst_transport_requested_time_s",
        "flow_sst_transport_accepted_time_s",
        "flow_sst_transport_remaining_unadvanced_time_s",
        "flow_momentum_advection_requested_time_s",
        "flow_momentum_advection_accepted_time_s",
        "flow_momentum_advection_remaining_unadvanced_time_s",
        "flow_sst_momentum_diffusion_requested_time_s",
        "flow_sst_momentum_diffusion_accepted_time_s",
        "flow_sst_momentum_diffusion_remaining_unadvanced_time_s",
        "flow_sst_requested_transport_time_s",
        "flow_sst_accepted_transport_time_s",
        "flow_sst_remaining_unadvanced_transport_time_s",
    ):
        if any(key in report for report in segment_reports):
            combined[key] = math.fsum(
                float(report.get(key, 0.0)) for report in segment_reports
            )
    for key in (
        "flow_sst_transport_diffusion_cfl_max",
        "flow_momentum_advection_cfl_max",
        "flow_momentum_advection_max_substep_cfl",
    ):
        if any(key in report for report in segment_reports):
            combined[key] = max(float(report.get(key, 0.0)) for report in segment_reports)
    consistency_count = int(
        combined.get("hibm_post_dirichlet_consistency_projection_count", 0)
    )
    combined_projection_report[
        "hibm_post_dirichlet_consistency_projection_count"
    ] = consistency_count
    combined_projection_report[
        "hibm_post_dirichlet_consistency_projection_applied"
    ] = bool(consistency_count)
    combined["projection_report"] = combined_projection_report
    combined["hibm_post_dirichlet_consistency_projection_applied"] = bool(
        consistency_count
    )
    combined["flow_pressure_reset_applied"] = bool(reset_pressure)
    # A topology mutation in any physical segment invalidates the whole outer
    # step for stationary/snapshot readiness.  Later cache-reuse segments must
    # not erase that evidence by contributing a terminal False value.
    combined["hibm_preassembly_topology_mutated"] = any(
        bool(report.get("hibm_preassembly_topology_mutated", False))
        for report in segment_reports
    )
    combined["flow_predictor_projection_segment_count"] = int(configured_substeps)
    combined["flow_predictor_projection_segment_dt_s"] = float(segment_dt_s)
    combined["flow_predictor_projection_segment_pre_projection_l2_max"] = max(
        float(
            segment_report.get(
                "flow_main_projection_pre_projection_l2",
                projection_report.get("pre_projection_l2", 0.0),
            )
        )
        for segment_report, projection_report in zip(
            segment_reports, segment_projection_reports, strict=True
        )
    )
    combined[
        "flow_predictor_projection_segment_pre_projection_max_abs_max"
    ] = max(
        float(
            segment_report.get(
                "flow_main_projection_pre_projection_max_abs",
                projection_report.get("pre_projection_max_abs", 0.0),
            )
        )
        for segment_report, projection_report in zip(
            segment_reports, segment_projection_reports, strict=True
        )
    )
    combined["flow_predictor_projection_segment_trace"] = [
        {
            "segment_index": int(segment_index),
            "pre_projection_l2": float(
                segment_report.get(
                    "flow_main_projection_pre_projection_l2",
                    report.get("pre_projection_l2", 0.0),
                )
            ),
            "pre_projection_max_abs": float(
                segment_report.get(
                    "flow_main_projection_pre_projection_max_abs",
                    report.get("pre_projection_max_abs", 0.0),
                )
            ),
            "projection_l2": float(
                segment_report.get(
                    "flow_main_projection_l2",
                    report.get("projection_l2", 0.0),
                )
            ),
            "projection_max_abs": float(
                segment_report.get(
                    "flow_main_projection_max_abs",
                    report.get("projection_max_abs", 0.0),
                )
            ),
            "cg_converged_all": bool(report.get("cg_converged_all", True)),
            "cg_iterations_total": int(report.get("cg_iterations_total", 0)),
        }
        for segment_index, (segment_report, report) in enumerate(
            zip(segment_reports, segment_projection_reports, strict=True),
            start=1,
        )
    ]
    combined["flow_predictor_note"] = (
        "core fluid predictor and full sharp-boundary pressure projection "
        f"interleaved across {configured_substeps} physical segments "
        f"(segment_dt_s={segment_dt_s:g}); each segment used: "
        f"{combined.get('flow_predictor_note', '')}"
    )
    return combined


def _invalidate_hibm_sharp_boundary_derived_cache(
    boundary_cache: dict[str, object] | None,
) -> None:
    if boundary_cache is None:
        return
    cache_entry = boundary_cache.get("hibm_sharp_marker_boundary")
    if not isinstance(cache_entry, dict):
        return
    for key in (
        "classified_topology_key",
        "search_report",
        "internal_obstacle_cell_count",
        "cleanup_report",
    ):
        cache_entry.pop(key, None)


def _flow_advance_current_step(
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    markers: HibmMpmSurfaceMarkers | None = None,
    sharp_boundary_cache: dict[str, object] | None = None,
    flow_phase: str,
    step_index_local: int,
    step_index_global: int,
    preflow_history: list[dict[str, object]],
    reset_pressure: bool,
    measure_wall_times: bool = False,
    preflow_stage_observer: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Advance one all-or-nothing fluid macro step."""

    fluid.save_state()
    try:
        mode = (
            _require_fsi_physical_flow_driver_mode(config)
            if str(flow_phase) == "fsi"
            else _effective_flow_driver_mode(config, flow_phase=flow_phase)
        )
        report = _flow_advance_current_step_trial(
            fluid,
            config,
            markers=markers,
            sharp_boundary_cache=sharp_boundary_cache,
            flow_phase=flow_phase,
            step_index_local=step_index_local,
            step_index_global=step_index_global,
            preflow_history=preflow_history,
            reset_pressure=reset_pressure,
            measure_wall_times=measure_wall_times,
            preflow_stage_observer=preflow_stage_observer,
        )
        if mode in FLOW_DRIVER_PHYSICAL_PREDICTOR_MODES:
            report.update(
                _macro_time_accounting_report(
                    requested_macro_dt_s=float(config.dt_s),
                    accepted_time_s=float(
                        report.get("fluid_accepted_time_s", math.nan)
                    ),
                    accepted_substep_count=report[
                        "flow_momentum_advection_substeps_total"
                    ],
                    rejected_trial_count=int(
                        report.get("fluid_rejected_trial_count", -1)
                    ),
                    component="fluid",
                )
            )
        return report
    except Exception as original_error:
        cleanup_errors: list[tuple[str, Exception]] = []
        for cleanup_name, cleanup in (
            ("restore_state", fluid.restore_state),
            (
                "invalidate_pressure_warmstart",
                fluid.invalidate_pressure_warmstart,
            ),
            (
                "invalidate_hibm_sharp_boundary_derived_cache",
                lambda: _invalidate_hibm_sharp_boundary_derived_cache(
                    sharp_boundary_cache
                ),
            ),
        ):
            try:
                cleanup()
            except Exception as cleanup_error:
                cleanup_errors.append((cleanup_name, cleanup_error))
        if cleanup_errors:
            cleanup_summary = "; ".join(
                f"{name}: {type(error).__name__}: {error}"
                for name, error in cleanup_errors
            )
            raise RuntimeError(
                "fluid macro rollback cleanup failed after "
                f"{type(original_error).__name__}: {original_error}; "
                f"cleanup errors: {cleanup_summary}"
            ) from original_error
        raise


def _flow_advance_current_step_trial(
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    markers: HibmMpmSurfaceMarkers | None = None,
    sharp_boundary_cache: dict[str, object] | None = None,
    flow_phase: str,
    step_index_local: int,
    step_index_global: int,
    preflow_history: list[dict[str, object]],
    reset_pressure: bool,
    measure_wall_times: bool = False,
    preflow_stage_observer: Callable[[str], None] | None = None,
) -> dict[str, object]:
    configured_predictor_substeps = int(
        getattr(config, "flow_predictor_substeps", 1)
    )
    mode = _effective_flow_driver_mode(config, flow_phase=flow_phase)
    if (
        configured_predictor_substeps > 1
        and mode in FLOW_DRIVER_PHYSICAL_PREDICTOR_MODES
    ):
        segment_dt_s = float(config.dt_s) / float(configured_predictor_substeps)
        segment_config = _FlowPredictorSegmentConfig(config, dt_s=segment_dt_s)
        segment_reports = [
            _flow_advance_current_step_trial(
                fluid,
                segment_config,
                markers=markers,
                sharp_boundary_cache=sharp_boundary_cache,
                flow_phase=flow_phase,
                step_index_local=step_index_local,
                step_index_global=step_index_global,
                preflow_history=preflow_history,
                reset_pressure=bool(reset_pressure and segment_index == 0),
                measure_wall_times=measure_wall_times,
                preflow_stage_observer=preflow_stage_observer,
            )
            for segment_index in range(configured_predictor_substeps)
        ]
        return _combine_interleaved_flow_predictor_segment_reports(
            segment_reports,
            configured_substeps=configured_predictor_substeps,
            segment_dt_s=segment_dt_s,
            reset_pressure=reset_pressure,
        )

    source_schedule_scope = _flow_source_schedule_scope(config)
    source_schedule_step_index = _flow_source_schedule_step_index(
        config,
        step_index_local=step_index_local,
        step_index_global=step_index_global,
    )
    mode = _effective_flow_driver_mode(config, flow_phase=flow_phase)
    fluid.clear_volume_source()
    driver_report = _flow_driver_report(
        mode=mode,
        full_field_reinitialized=_flow_driver_requires_full_field_reinitialize(mode),
        inlet_boundary_report={},
        volume_source_applied=False,
    )
    predictor_applied = False
    velocity_only_soft_rows = (
        _use_hibm_sharp_marker_boundary(config)
        and not bool(
            getattr(
                config,
                "flow_hibm_sharp_interpolate_velocity_rows",
                True,
            )
        )
    )

    # Geometry and the physical solid-volume mask can change after the prior
    # MPM step.  Refresh the carve and velocity rows before any predictor
    # substep consumes them.  The second assembly below runs after prediction
    # so the pressure-Neumann gradient is sampled from the current predictor.
    if preflow_stage_observer is not None:
        preflow_stage_observer("pre_predictor_hibm_before")
    pre_predictor_sharp_boundary_report = (
        _apply_hibm_sharp_marker_boundary_to_fluid(
            markers,
            fluid,
            config,
            update_pressure_gradient=False,
            boundary_cache=sharp_boundary_cache,
            reuse_topology_from_previous_assembly=True,
            measure_wall_times=measure_wall_times,
            stage_observer=preflow_stage_observer,
        )
    )
    if preflow_stage_observer is not None:
        preflow_stage_observer("pre_predictor_hibm_after")
    _require_hibm_velocity_dirichlet_health(
        pre_predictor_sharp_boundary_report,
        context=(
            f"{flow_phase} step {step_index_local} pre-predictor assembly"
        ),
    )
    pre_predictor_stage_wall_time_s = (
        _hibm_sharp_boundary_stage_wall_times_from_report(
            pre_predictor_sharp_boundary_report
        )
    )
    pre_predictor_wall_time_s = _hibm_stage_wall_time_sum(
        pre_predictor_stage_wall_time_s
    )
    pre_predictor_ledger_generation = None
    if velocity_only_soft_rows:
        pre_predictor_ledger_generation = (
            _capture_velocity_dirichlet_row_ledger_reference(
                fluid,
                context=(
                    f"{flow_phase} step {step_index_local} pre-predictor assembly"
                ),
            )
        )

    turbulence_model = _flow_turbulence_model(config)
    sst_transport_report: dict[str, object] = {}
    sst_transport_substeps_total = 0
    sst_transport_rejected_trial_count_total = 0
    sst_transport_diffusion_cfl_max = 0.0
    momentum_advection_scheme = "none"
    momentum_advection_substeps_total = 0
    momentum_advection_rejected_trial_count_total = 0
    momentum_advection_cfl_max = 0.0
    momentum_advection_max_substep_cfl = 0.0
    sst_transport_wall_time_s = 0.0
    momentum_predictor_wall_time_s = 0.0
    sst_transport_requested_time_parts_s: list[float] = []
    sst_transport_accepted_time_parts_s: list[float] = []
    momentum_advection_requested_time_parts_s: list[float] = []
    momentum_advection_accepted_time_parts_s: list[float] = []
    momentum_diffusion_requested_time_parts_s: list[float] = []
    momentum_diffusion_accepted_time_parts_s: list[float] = []
    momentum_diffusion_rejected_trial_count_total = 0
    if turbulence_model == "sst_2003" and markers is not None:
        if preflow_stage_observer is not None:
            preflow_stage_observer("sst_wall_distance_before")
        fluid.prepare_sst_wall_distance(
            no_slip_domain_walls=_flow_predictor_no_slip_domain_walls(config),
            marker_position_m=markers.x_gamma_m,
            marker_count=int(markers.marker_count),
            projection_segment_indices=markers.projection_triangle_indices,
            projection_segment_count=int(markers.projection_segment_count),
            inactive_axis=OUT_OF_PLANE_AXIS_INDEX,
        )
        if preflow_stage_observer is not None:
            preflow_stage_observer("sst_wall_distance_after")

    if mode == FLOW_DRIVER_PROJECTION_ONLY:
        pass
    elif mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC:
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=True,
            inlet_boundary_report=_zmax_inlet_boundary_report(fluid),
            volume_source_applied=False,
        )
    elif mode == FLOW_DRIVER_SUSTAINED_BOUNDARY:
        boundary_report = _refresh_zmax_inlet_boundary(fluid, config)
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=False,
            inlet_boundary_report=boundary_report,
            volume_source_applied=False,
        )
    elif mode in {
        FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
        FLOW_DRIVER_SUSTAINED_SOURCE,
        FLOW_DRIVER_SUSTAINED_PREDICTOR,
    }:
        boundary_report = _refresh_zmax_inlet_boundary(fluid, config)
        volume_source_applied = mode in {
            FLOW_DRIVER_SUSTAINED_SOURCE,
            FLOW_DRIVER_SUSTAINED_PREDICTOR,
        }
        if volume_source_applied:
            source_factor = _flow_inlet_source_factor(config, source_schedule_step_index)
            source_normal_velocity_mps = -float(config.inlet_velocity_mps) * source_factor
            fluid.add_zmax_velocity_inlet_volume_source(
                normal_velocity_mps=source_normal_velocity_mps,
            )
        else:
            source_factor = 0.0
            source_normal_velocity_mps = 0.0
        predictor_applied = mode in {
            FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
            FLOW_DRIVER_SUSTAINED_PREDICTOR,
        }
        predictor_note = ""
        predictor_kinematic_viscosity_m2_s = 0.0
        predictor_no_slip_domain_walls = _flow_predictor_no_slip_domain_walls(config)
        sst_stage_observer: Callable[[str], None] | None = None
        sst_stage_observer_wall_time_s = 0.0
        if preflow_stage_observer is not None:

            def emit_sst_stage(stage_name: str) -> None:
                nonlocal sst_stage_observer_wall_time_s
                if measure_wall_times:
                    _synchronize_hibm_sharp_boundary_stage_timing()
                observer_started_s = time.perf_counter()
                try:
                    preflow_stage_observer(f"sst_{stage_name}")
                finally:
                    sst_stage_observer_wall_time_s += max(
                        0.0, time.perf_counter() - observer_started_s
                    )

            sst_stage_observer = emit_sst_stage
        if predictor_applied:
            advection_scheme = str(
                getattr(config, "flow_advection_scheme", "euler")
            ).lower()
            predictor_substeps = int(getattr(config, "flow_predictor_substeps", 1))
            predictor_dt_s = float(config.dt_s) / float(predictor_substeps)
            predictor_kinematic_viscosity_m2_s = (
                _flow_predictor_kinematic_viscosity_m2_s(config)
            )
            for _predictor_substep in range(predictor_substeps):
                fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)
                if turbulence_model == "sst_2003":
                    if preflow_stage_observer is not None:
                        preflow_stage_observer("sst_transport_before")
                    sst_observer_wall_time_before_s = (
                        sst_stage_observer_wall_time_s
                    )
                    (
                        current_sst_report,
                        current_sst_wall_time_s,
                    ) = _measure_taichi_operation_wall_time(
                        lambda: fluid.advance_sst_transport(
                            dt_s=predictor_dt_s,
                            kinematic_viscosity_m2_s=(
                                predictor_kinematic_viscosity_m2_s
                            ),
                            no_slip_domain_walls=(
                                predictor_no_slip_domain_walls
                            ),
                            advection_scheme=advection_scheme,
                            stage_observer=sst_stage_observer,
                        ),
                        enabled=measure_wall_times,
                    )
                    current_sst_wall_time_s = max(
                        0.0,
                        current_sst_wall_time_s
                        - (
                            sst_stage_observer_wall_time_s
                            - sst_observer_wall_time_before_s
                        ),
                    )
                    if preflow_stage_observer is not None:
                        preflow_stage_observer("sst_transport_after")
                    sst_transport_wall_time_s += current_sst_wall_time_s
                    sst_transport_report = dict(current_sst_report)
                    current_sst_rejected_trial_count = int(
                        current_sst_report.get(
                            "rejected_transport_trial_count",
                            0,
                        )
                    )
                    current_sst_accepted_time_s = (
                        _validated_component_accepted_time(
                            requested_time_s=predictor_dt_s,
                            reported_requested_time_s=float(
                                current_sst_report.get(
                                    "requested_transport_time_s",
                                    math.nan,
                                )
                            ),
                            accepted_time_s=float(
                                current_sst_report.get(
                                    "accepted_transport_time_s",
                                    math.nan,
                                )
                            ),
                            accepted_substep_count=current_sst_report[
                                "diffusion_substeps"
                            ],
                            remaining_unadvanced_time_s=float(
                                current_sst_report.get(
                                    "remaining_unadvanced_transport_time_s",
                                    math.nan,
                                )
                            ),
                            rejected_trial_count=current_sst_rejected_trial_count,
                            component="fluid SST transport",
                        )
                    )
                    sst_transport_requested_time_parts_s.append(predictor_dt_s)
                    sst_transport_accepted_time_parts_s.append(
                        current_sst_accepted_time_s
                    )
                    sst_transport_substeps_total += int(
                        current_sst_report["diffusion_substeps"]
                    )
                    sst_transport_rejected_trial_count_total += (
                        current_sst_rejected_trial_count
                    )
                    sst_transport_diffusion_cfl_max = max(
                        sst_transport_diffusion_cfl_max,
                        float(
                            current_sst_report[
                                "diffusion_cfl_before_substeps"
                            ]
                        ),
                    )
                if preflow_stage_observer is not None:
                    preflow_stage_observer("momentum_predictor_before")
                _, current_predictor_wall_time_s = (
                    _measure_taichi_operation_wall_time(
                        lambda: fluid.predict(
                            dt_s=predictor_dt_s,
                            advection_scheme=advection_scheme,
                            kinematic_viscosity_m2_s=(
                                predictor_kinematic_viscosity_m2_s
                            ),
                            no_slip_domain_walls=(
                                predictor_no_slip_domain_walls
                            ),
                        ),
                        enabled=measure_wall_times,
                    )
                )
                if preflow_stage_observer is not None:
                    preflow_stage_observer("momentum_predictor_after")
                momentum_predictor_wall_time_s += current_predictor_wall_time_s
                current_momentum_rejected_trial_count = int(
                    getattr(
                        fluid,
                        "_last_momentum_advection_rejected_trial_count",
                        0,
                    )
                )
                current_momentum_accepted_time_s = (
                    _validated_component_accepted_time(
                        requested_time_s=predictor_dt_s,
                        reported_requested_time_s=float(
                            getattr(
                                fluid,
                                "_last_momentum_advection_requested_time_s",
                                math.nan,
                            )
                        ),
                        accepted_time_s=float(
                            getattr(
                                fluid,
                                "_last_momentum_advection_accepted_time_s",
                                math.nan,
                            )
                        ),
                        accepted_substep_count=(
                            fluid._last_momentum_advection_substeps
                        ),
                        remaining_unadvanced_time_s=float(
                            getattr(
                                fluid,
                                "_last_momentum_advection_remaining_unadvanced_time_s",
                                math.nan,
                            )
                        ),
                        rejected_trial_count=current_momentum_rejected_trial_count,
                        component="fluid momentum advection",
                    )
                )
                momentum_advection_requested_time_parts_s.append(predictor_dt_s)
                momentum_advection_accepted_time_parts_s.append(
                    current_momentum_accepted_time_s
                )
                if turbulence_model == "sst_2003":
                    current_diffusion_rejected_trial_count = int(
                        getattr(
                            fluid,
                            "_sst_last_momentum_helmholtz_rejected_trial_count",
                            0,
                        )
                    )
                    current_diffusion_accepted_time_s = (
                        _validated_component_accepted_time(
                            requested_time_s=predictor_dt_s,
                            reported_requested_time_s=float(
                                getattr(
                                    fluid,
                                    "_sst_last_momentum_diffusion_requested_time_s",
                                    math.nan,
                                )
                            ),
                            accepted_time_s=float(
                                getattr(
                                    fluid,
                                    "_sst_last_momentum_diffusion_accepted_time_s",
                                    math.nan,
                                )
                            ),
                            accepted_substep_count=(
                                fluid._sst_last_momentum_diffusion_substeps
                            ),
                            remaining_unadvanced_time_s=float(
                                getattr(
                                    fluid,
                                    "_sst_last_momentum_diffusion_remaining_unadvanced_time_s",
                                    math.nan,
                                )
                            ),
                            rejected_trial_count=(
                                current_diffusion_rejected_trial_count
                            ),
                            component="fluid momentum diffusion",
                        )
                    )
                    momentum_diffusion_requested_time_parts_s.append(
                        predictor_dt_s
                    )
                    momentum_diffusion_accepted_time_parts_s.append(
                        current_diffusion_accepted_time_s
                    )
                    momentum_diffusion_rejected_trial_count_total += (
                        current_diffusion_rejected_trial_count
                    )
                momentum_advection_scheme = str(
                    getattr(fluid, "_last_momentum_advection_scheme", advection_scheme)
                )
                momentum_advection_substeps_total += int(
                    fluid._last_momentum_advection_substeps
                )
                momentum_advection_rejected_trial_count_total += (
                    current_momentum_rejected_trial_count
                )
                momentum_advection_cfl_max = max(
                    momentum_advection_cfl_max,
                    float(getattr(fluid, "_last_momentum_advection_cfl", 0.0)),
                )
                momentum_advection_max_substep_cfl = max(
                    momentum_advection_max_substep_cfl,
                    float(
                        getattr(
                            fluid,
                            "_last_momentum_advection_max_substep_cfl",
                            0.0,
                        )
                    ),
                )
            fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)
            predictor_note = (
                "core fluid predictor applied before pressure projection "
                f"(advection_scheme={advection_scheme}, "
                f"outer_substeps={predictor_substeps}, "
                f"core_adaptive_substeps={momentum_advection_substeps_total}, "
                "core_rejected_trials="
                f"{momentum_advection_rejected_trial_count_total}, "
                f"core_initial_cfl_max={momentum_advection_cfl_max:g}, "
                f"nu={predictor_kinematic_viscosity_m2_s:g} m^2/s, "
                f"no_slip_domain_walls={predictor_no_slip_domain_walls})"
            )
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=False,
            inlet_boundary_report=boundary_report,
            volume_source_applied=volume_source_applied,
            source_factor=source_factor,
            source_normal_velocity_mps=source_normal_velocity_mps,
            predictor_applied=predictor_applied,
            predictor_note=predictor_note,
            predictor_kinematic_viscosity_m2_s=predictor_kinematic_viscosity_m2_s,
            predictor_no_slip_domain_walls=predictor_no_slip_domain_walls,
        )
    elif mode == FLOW_DRIVER_SHARP_REFERENCE:
        raise RuntimeError(
            "sharp_hibm_mpm_reference is reserved for a sharp-path runner"
        )
    else:  # pragma: no cover - protected by config validation.
        raise RuntimeError(f"unsupported flow_driver_mode: {mode!r}")

    sst_transport_requested_time_s = math.fsum(
        sst_transport_requested_time_parts_s
    )
    sst_transport_accepted_time_s = math.fsum(
        sst_transport_accepted_time_parts_s
    )
    momentum_advection_requested_time_s = math.fsum(
        momentum_advection_requested_time_parts_s
    )
    momentum_advection_accepted_time_s = math.fsum(
        momentum_advection_accepted_time_parts_s
    )
    momentum_diffusion_requested_time_s = math.fsum(
        momentum_diffusion_requested_time_parts_s
    )
    momentum_diffusion_accepted_time_s = math.fsum(
        momentum_diffusion_accepted_time_parts_s
    )
    fluid_rejected_trial_count = (
        sst_transport_rejected_trial_count_total
        + momentum_advection_rejected_trial_count_total
        + momentum_diffusion_rejected_trial_count_total
    )
    driver_report.update(
        {
            "flow_turbulence_model": turbulence_model,
            "flow_sst_near_wall_treatment": str(
                getattr(config, "flow_sst_near_wall_treatment", "resolved")
            ),
            "flow_sst_transport_applied": bool(sst_transport_report),
            "flow_sst_transport_substeps_total": int(
                sst_transport_substeps_total
            ),
            "flow_sst_transport_rejected_trial_count_total": int(
                sst_transport_rejected_trial_count_total
            ),
            "flow_sst_transport_requested_time_s": float(
                sst_transport_requested_time_s
            ),
            "flow_sst_transport_accepted_time_s": float(
                sst_transport_accepted_time_s
            ),
            "flow_sst_transport_remaining_unadvanced_time_s": float(
                sst_transport_requested_time_s - sst_transport_accepted_time_s
            ),
            "flow_sst_momentum_diffusion_requested_time_s": float(
                momentum_diffusion_requested_time_s
            ),
            "flow_sst_momentum_diffusion_accepted_time_s": float(
                momentum_diffusion_accepted_time_s
            ),
            "flow_sst_momentum_diffusion_remaining_unadvanced_time_s": float(
                momentum_diffusion_requested_time_s
                - momentum_diffusion_accepted_time_s
            ),
            "flow_sst_transport_diffusion_cfl_max": float(
                sst_transport_diffusion_cfl_max
            ),
            "flow_sst_momentum_diffusion_substeps_last": int(
                fluid._sst_last_momentum_diffusion_substeps
                if turbulence_model == "sst_2003"
                else 0
            ),
            "flow_sst_momentum_diffusion_integrator": str(
                getattr(fluid, "_sst_last_momentum_diffusion_integrator", "none")
            ),
            "flow_sst_momentum_diffusion_cfl_last": float(
                getattr(fluid, "_sst_last_momentum_diffusion_cfl", 0.0)
            ),
            "flow_sst_momentum_helmholtz_converged": bool(
                getattr(fluid, "_sst_last_momentum_helmholtz_converged", True)
            ),
            "flow_sst_momentum_helmholtz_iterations_last": int(
                getattr(fluid, "_sst_last_momentum_helmholtz_iterations", 0)
            ),
            "flow_sst_momentum_helmholtz_iterations_total_last": int(
                getattr(
                    fluid,
                    "_sst_last_momentum_helmholtz_iterations_total",
                    0,
                )
            ),
            "flow_sst_momentum_helmholtz_relative_residual_last": float(
                getattr(
                    fluid,
                    "_sst_last_momentum_helmholtz_relative_residual",
                    0.0,
                )
            ),
            "flow_sst_momentum_helmholtz_rejected_trial_count_last": int(
                getattr(
                    fluid,
                    "_sst_last_momentum_helmholtz_rejected_trial_count",
                    0,
                )
            ),
            "flow_sst_momentum_helmholtz_rejected_trial_count_total": int(
                momentum_diffusion_rejected_trial_count_total
            ),
            "flow_momentum_advection_scheme": momentum_advection_scheme,
            "flow_momentum_advection_substeps_total": int(
                momentum_advection_substeps_total
            ),
            "flow_momentum_advection_rejected_trial_count_total": int(
                momentum_advection_rejected_trial_count_total
            ),
            "flow_momentum_advection_requested_time_s": float(
                momentum_advection_requested_time_s
            ),
            "flow_momentum_advection_accepted_time_s": float(
                momentum_advection_accepted_time_s
            ),
            "flow_momentum_advection_remaining_unadvanced_time_s": float(
                momentum_advection_requested_time_s
                - momentum_advection_accepted_time_s
            ),
            "flow_momentum_advection_cfl_max": float(
                momentum_advection_cfl_max
            ),
            "flow_momentum_advection_max_substep_cfl": float(
                momentum_advection_max_substep_cfl
            ),
            "flow_sst_transport_wall_time_s": float(
                sst_transport_wall_time_s
            ),
            "flow_momentum_predictor_wall_time_s": float(
                momentum_predictor_wall_time_s
            ),
            **{
                f"flow_sst_{key}": value
                for key, value in sst_transport_report.items()
                if key != "turbulence_model"
            },
        }
    )

    if preflow_stage_observer is not None:
        preflow_stage_observer("projection_hibm_before")
    sharp_boundary_report = _apply_hibm_sharp_marker_boundary_to_fluid(
        markers,
        fluid,
        config,
        update_pressure_gradient=True,
        boundary_cache=sharp_boundary_cache,
        reuse_topology_from_previous_assembly=True,
        measure_wall_times=measure_wall_times,
        stage_observer=preflow_stage_observer,
    )
    if preflow_stage_observer is not None:
        preflow_stage_observer("projection_hibm_after")
    sharp_boundary_report = dict(sharp_boundary_report)
    _require_hibm_velocity_dirichlet_health(
        sharp_boundary_report,
        context=f"{flow_phase} step {step_index_local} projection assembly",
    )
    pre_projection_velocity_projector = None
    if _use_hibm_sharp_marker_boundary(config):
        if markers is None:
            raise RuntimeError(
                "sharp HIBM pre-projection velocity projector requires markers"
            )
        pre_projection_velocity_projector = (
            _hibm_pre_projection_velocity_projector_from_cache(
                sharp_boundary_cache,
                markers=markers,
            )
        )
    main_soft_rows_already_applied = False
    if velocity_only_soft_rows and predictor_applied:
        if pre_predictor_ledger_generation is None:
            raise RuntimeError(
                "velocity-only predictor row ledger reference is unavailable "
                f"({flow_phase} step {step_index_local})"
            )
        pre_predictor_comparison = _velocity_dirichlet_row_ledger_comparison(
            fluid,
            reference_generation=int(pre_predictor_ledger_generation),
            comparison_mode="content_equivalence",
            context=(
                f"{flow_phase} step {step_index_local} post-predictor assembly"
            ),
        )
        pre_predictor_mismatch_rows = int(
            pre_predictor_comparison[
                "hibm_velocity_dirichlet_row_ledger_mismatch_rows"
            ]
        )
        if pre_predictor_mismatch_rows == 0:
            _require_velocity_only_topology_reuse(
                sharp_boundary_report,
                context=(
                    f"{flow_phase} step {step_index_local} projection assembly"
                ),
            )
            main_soft_rows_already_applied = True
        sharp_boundary_report.update(
            {
                "hibm_velocity_dirichlet_pre_predictor_ledger_snapshot_generation": int(
                    pre_predictor_ledger_generation
                ),
                "hibm_velocity_dirichlet_pre_predictor_ledger_matches_projection_assembly": bool(
                    main_soft_rows_already_applied
                ),
                "hibm_velocity_dirichlet_pre_predictor_ledger_mismatch_rows": int(
                    pre_predictor_mismatch_rows
                ),
            }
        )

    consistency_ledger_generation = None
    if velocity_only_soft_rows:
        consistency_ledger_generation = (
            _capture_velocity_dirichlet_row_ledger_reference(
                fluid,
                context=f"{flow_phase} step {step_index_local} projection assembly",
            )
        )
        sharp_boundary_report.update(
            _velocity_dirichlet_row_ledger_reference_diagnostics(
                reference_generation=int(consistency_ledger_generation),
            )
        )
    terminal_sharp_boundary_report = sharp_boundary_report
    sharp_joint_qp_enabled = _use_hibm_sharp_marker_boundary(config)
    consistency_projection_count = (
        max(
            0,
            int(
                getattr(
                    config,
                    "flow_post_dirichlet_consistency_projection_iterations",
                    0,
                )
            ),
        )
        if sharp_joint_qp_enabled
        else 0
    )
    joint_qp_cycle_budget = 1 + consistency_projection_count
    reprojection_iterations = getattr(config, "flow_reprojection_iterations", None)
    reprojection_cg_tolerance = getattr(
        config,
        "flow_reprojection_cg_tolerance",
        None,
    )
    joint_qp_no_slip_tolerance = (
        _hibm_marker_mac_constraint_absolute_tolerance_mps(config)
        if sharp_joint_qp_enabled
        else 0.0
    )
    joint_qp_cycle_trace: list[dict[str, object]] = []
    terminal_no_slip_report: dict[str, object] | None = None

    if preflow_stage_observer is not None:
        preflow_stage_observer("main_pressure_projection_before")
    main_flow_report = _project_current_flow(
        fluid,
        config,
        reset_pressure=reset_pressure,
        pressure_solve_context={
            "phase": str(flow_phase),
            "step_index_local": int(step_index_local),
            "step_index_global": int(step_index_global),
            "hibm_sharp_marker_boundary_stage_wall_time_s": (
                _hibm_sharp_boundary_stage_wall_times_from_report(
                    sharp_boundary_report
                )
            ),
        },
        preserve_velocity_constraints=(
            False if _use_hibm_sharp_marker_boundary(config) else None
        ),
        velocity_dirichlet_soft_rows_already_applied=(
            main_soft_rows_already_applied
        ),
        pre_projection_velocity_projector=pre_projection_velocity_projector,
        pressure_velocity_nullspace_projector=(
            pre_projection_velocity_projector
        ),
    )
    if preflow_stage_observer is not None:
        preflow_stage_observer("main_pressure_projection_after")
    flow_report = main_flow_report
    main_projection_report = dict(main_flow_report["projection_report"])
    projection_reports = [main_projection_report]
    joint_qp_converged = False
    if sharp_joint_qp_enabled:
        terminal_no_slip_report = _sample_hibm_no_slip_report(
            markers,
            fluid,
            pre_projection_velocity_projector=pre_projection_velocity_projector,
        )
        main_cycle_diagnostics = _hibm_joint_qp_cycle_diagnostics(
            cycle_index=1,
            projection_stage="main",
            no_slip_report=terminal_no_slip_report,
            pressure_report=main_projection_report,
            no_slip_absolute_tolerance_mps=joint_qp_no_slip_tolerance,
            pressure_cg_tolerance=float(config.flow_cg_tolerance),
            sharp_boundary_report=sharp_boundary_report,
        )
        joint_qp_cycle_trace.append(main_cycle_diagnostics)
        joint_qp_converged = bool(main_cycle_diagnostics["converged"])

    for consistency_projection_index in range(consistency_projection_count):
        if joint_qp_converged:
            break
        if preflow_stage_observer is not None:
            preflow_stage_observer(
                f"consistency_hibm_before[{consistency_projection_index + 1}]"
            )
        consistency_boundary_report = _apply_hibm_sharp_marker_boundary_to_fluid(
            markers,
            fluid,
            config,
            update_pressure_gradient=False,
            boundary_cache=sharp_boundary_cache,
            reuse_topology_from_previous_assembly=True,
            measure_wall_times=measure_wall_times,
            stage_observer=preflow_stage_observer,
        )
        if preflow_stage_observer is not None:
            preflow_stage_observer(
                f"consistency_hibm_after[{consistency_projection_index + 1}]"
            )
        consistency_boundary_report = dict(consistency_boundary_report)
        _require_hibm_velocity_dirichlet_health(
            consistency_boundary_report,
            context=(
                f"{flow_phase} step {step_index_local} consistency projection "
                f"{consistency_projection_index + 1} assembly"
            ),
        )
        soft_rows_already_applied = not bool(
            getattr(
                config,
                "flow_hibm_sharp_interpolate_velocity_rows",
                True,
            )
        )
        if soft_rows_already_applied:
            if consistency_ledger_generation is None:
                raise RuntimeError(
                    "velocity-only consistency row ledger reference is unavailable "
                    f"({flow_phase} step {step_index_local} consistency projection "
                    f"{consistency_projection_index + 1})"
                )
            consistency_boundary_report.update(
                _velocity_dirichlet_row_ledger_comparison(
                    fluid,
                    reference_generation=int(consistency_ledger_generation),
                    comparison_mode="content_equivalence",
                    context=(
                        f"{flow_phase} step {step_index_local} consistency projection "
                        f"{consistency_projection_index + 1}"
                    ),
                )
            )
            _require_velocity_only_consistency_row_reuse(
                sharp_boundary_report,
                consistency_boundary_report,
                context=(
                    f"{flow_phase} step {step_index_local} consistency projection "
                    f"{consistency_projection_index + 1}"
                ),
            )
        terminal_sharp_boundary_report = consistency_boundary_report
        if preflow_stage_observer is not None:
            preflow_stage_observer(
                "consistency_pressure_projection_before["
                f"{consistency_projection_index + 1}]"
            )
        consistency_flow_report = _project_current_flow(
            fluid,
            config,
            reset_pressure=False,
            pressure_solve_context={
                "phase": str(flow_phase),
                "step_index_local": int(step_index_local),
                "step_index_global": int(step_index_global),
                "projection_stage": "post_dirichlet_consistency",
                "consistency_projection_index": int(consistency_projection_index + 1),
                "hibm_sharp_marker_boundary_stage_wall_time_s": (
                    _hibm_sharp_boundary_stage_wall_times_from_report(
                        consistency_boundary_report
                    )
                ),
            },
            projection_iterations=(
                int(reprojection_iterations)
                if reprojection_iterations is not None
                else int(config.flow_projection_iterations)
            ),
            cg_tolerance=(
                float(reprojection_cg_tolerance)
                if reprojection_cg_tolerance is not None
                else float(config.flow_cg_tolerance)
            ),
            accumulate_pressure_into_previous=True,
            homogenize_pressure_interface_rhs_for_increment=True,
            preserve_velocity_constraints=False,
            velocity_dirichlet_soft_rows_already_applied=soft_rows_already_applied,
            pre_projection_velocity_projector=pre_projection_velocity_projector,
            pressure_velocity_nullspace_projector=(
                pre_projection_velocity_projector
            ),
        )
        if preflow_stage_observer is not None:
            preflow_stage_observer(
                "consistency_pressure_projection_after["
                f"{consistency_projection_index + 1}]"
            )
        consistency_projection_report = dict(
            consistency_flow_report["projection_report"]
        )
        consistency_projection_report.update(
            {
                "hibm_projection_stage": (
                    "post_dirichlet_reconstruction_consistency"
                ),
                "hibm_post_dirichlet_consistency_projection_index": int(
                    consistency_projection_index + 1
                ),
                "hibm_post_dirichlet_consistency_projection_applied": True,
                "hibm_post_dirichlet_consistency_projection_count": 1,
            }
        )
        projection_reports.append(consistency_projection_report)
        flow_report = consistency_flow_report
        terminal_no_slip_report = _sample_hibm_no_slip_report(
            markers,
            fluid,
            pre_projection_velocity_projector=pre_projection_velocity_projector,
        )
        consistency_cycle_diagnostics = _hibm_joint_qp_cycle_diagnostics(
            cycle_index=len(projection_reports),
            projection_stage="post_dirichlet_consistency",
            no_slip_report=terminal_no_slip_report,
            pressure_report=consistency_projection_report,
            no_slip_absolute_tolerance_mps=joint_qp_no_slip_tolerance,
            pressure_cg_tolerance=(
                float(reprojection_cg_tolerance)
                if reprojection_cg_tolerance is not None
                else float(config.flow_cg_tolerance)
            ),
            sharp_boundary_report=consistency_boundary_report,
        )
        joint_qp_cycle_trace.append(consistency_cycle_diagnostics)
        joint_qp_converged = bool(consistency_cycle_diagnostics["converged"])

    combined_projection_report = _combine_flow_projection_reports(projection_reports)
    actual_consistency_projection_count = max(0, len(projection_reports) - 1)
    if sharp_joint_qp_enabled:
        joint_qp_diagnostics = _hibm_joint_qp_terminal_diagnostics(
            cycle_budget=joint_qp_cycle_budget,
            cycle_trace=joint_qp_cycle_trace,
        )
        combined_projection_report.update(joint_qp_diagnostics)
        _require_hibm_joint_qp_convergence(
            joint_qp_diagnostics,
            context=f"{flow_phase} step {step_index_local}",
        )
    flow_report["projection_report"] = combined_projection_report
    # Preserve the predictor -> main-projection transition explicitly.  The
    # combined report is intentionally terminal-state based and therefore its
    # pre_projection_* values may belong to a later consistency projection.
    for source_key, report_key in (
        ("pre_projection_l2", "flow_main_projection_pre_projection_l2"),
        (
            "pre_projection_max_abs",
            "flow_main_projection_pre_projection_max_abs",
        ),
        ("projection_l2", "flow_main_projection_l2"),
        ("projection_max_abs", "flow_main_projection_max_abs"),
    ):
        flow_report[report_key] = float(main_projection_report.get(source_key, 0.0))
    flow_report["hibm_post_dirichlet_consistency_projection_count"] = int(
        actual_consistency_projection_count
    )
    flow_report["hibm_post_dirichlet_consistency_projection_applied"] = bool(
        actual_consistency_projection_count
    )
    if sharp_joint_qp_enabled:
        if terminal_no_slip_report is None:
            raise RuntimeError("joint Q/P terminal no-slip report is unavailable")
        flow_report.update(terminal_no_slip_report)
    else:
        flow_report.update(
            {
                "hibm_no_slip_report": {},
                "hibm_no_slip_valid_marker_count": 0,
                "hibm_no_slip_invalid_marker_count": 0,
                "hibm_no_slip_max_residual_mps": 0.0,
                "hibm_no_slip_l2_residual_mps": 0.0,
                "hibm_no_slip_direct_sample_marker_count": 0,
                "hibm_no_slip_normal_walk_sample_marker_count": 0,
                "hibm_no_slip_nearest_fluid_sample_marker_count": 0,
                "hibm_no_slip_no_fluid_sample_marker_count": 0,
            }
        )
    flow_report.update(terminal_sharp_boundary_report)
    if sharp_joint_qp_enabled:
        flow_report.update(
            {
                "hibm_sharp_marker_boundary_terminal_stage_wall_time_s": (
                    joint_qp_diagnostics[
                        "hibm_sharp_marker_boundary_terminal_stage_wall_time_s"
                    ]
                ),
                "hibm_sharp_marker_boundary_total_stage_wall_time_s": (
                    joint_qp_diagnostics[
                        "hibm_sharp_marker_boundary_total_stage_wall_time_s"
                    ]
                ),
            }
        )
    flow_report["hibm_pre_predictor_stage_wall_time_s"] = dict(
        pre_predictor_stage_wall_time_s
    )
    flow_report["hibm_pre_predictor_wall_time_s"] = float(
        pre_predictor_wall_time_s
    )
    projection_cycle_stage_wall_time_s = flow_report.get(
        "hibm_sharp_marker_boundary_total_stage_wall_time_s",
        flow_report.get(
            "hibm_sharp_marker_boundary_stage_wall_time_s",
            {},
        ),
    )
    flow_report["hibm_projection_cycle_wall_time_s"] = (
        _hibm_stage_wall_time_sum(projection_cycle_stage_wall_time_s)
    )
    flow_report["hibm_sharp_marker_boundary_pre_predictor_refreshed"] = bool(
        pre_predictor_sharp_boundary_report.get(
            "hibm_sharp_marker_boundary_enabled",
            False,
        )
    )
    flow_report.update(driver_report)
    flow_report["flow_phase"] = str(flow_phase)
    flow_report["flow_step_index_local"] = int(step_index_local)
    flow_report["flow_step_index_global"] = int(step_index_global)
    flow_report["flow_pressure_reset_applied"] = bool(reset_pressure)
    flow_report["flow_source_schedule_step_index"] = int(source_schedule_step_index)
    flow_report["flow_source_schedule_scope"] = source_schedule_scope
    flow_report["flow_source_ramp_restarted_after_preflow"] = (
        _flow_source_ramp_restarted_after_preflow(
            config,
            flow_phase=flow_phase,
            step_index_local=step_index_local,
            step_index_global=step_index_global,
            source_schedule_step_index=source_schedule_step_index,
            preflow_history=preflow_history,
        )
    )
    flow_report["flow_obstacle_no_slip_layers"] = int(
        getattr(config, "flow_obstacle_no_slip_layers", 0)
    )
    flow_report["flow_obstacle_no_slip_weight"] = float(
        getattr(config, "flow_obstacle_no_slip_weight", 1.0)
    )
    cap_no_slip_weight = getattr(config, "flow_obstacle_cap_no_slip_weight", None)
    flow_report["flow_obstacle_cap_no_slip_weight"] = (
        None if cap_no_slip_weight is None else float(cap_no_slip_weight)
    )
    flow_report["flow_obstacle_wake_no_slip_layers"] = int(
        getattr(config, "flow_obstacle_wake_no_slip_layers", 0)
    )
    flow_report["flow_obstacle_wake_no_slip_weight"] = float(
        getattr(config, "flow_obstacle_wake_no_slip_weight", 0.5)
    )
    flow_report["flow_solid_boundary_mode"] = _flow_solid_boundary_mode(config)
    flow_report["flow_obstacle_normal_velocity_policy"] = (
        _flow_obstacle_normal_velocity_policy(config)
    )
    flow_report["flow_pressure_outlet_backflow_policy"] = (
        _flow_pressure_outlet_backflow_policy(config)
    )
    configured_velocity_inlet_zmax = getattr(
        config,
        "flow_projection_velocity_inlet_zmax",
        None,
    )
    flow_report["flow_projection_velocity_inlet_zmax"] = (
        None
        if configured_velocity_inlet_zmax is None
        else bool(configured_velocity_inlet_zmax)
    )
    if predictor_applied:
        flow_report.update(
            _macro_time_accounting_report(
                requested_macro_dt_s=float(config.dt_s),
                accepted_time_s=float(momentum_advection_accepted_time_s),
                accepted_substep_count=momentum_advection_substeps_total,
                rejected_trial_count=int(fluid_rejected_trial_count),
                component="fluid",
            )
        )
    else:
        requested_macro_dt_s = float(config.dt_s)
        flow_report.update(
            {
                "requested_macro_dt_s": requested_macro_dt_s,
                "fluid_accepted_time_s": 0.0,
                "fluid_rejected_trial_count": 0,
                "fluid_remaining_unadvanced_time_s": requested_macro_dt_s,
            }
        )
    return flow_report


def _effective_flow_driver_mode(config: Any, *, flow_phase: str = "fsi") -> str:
    if bool(getattr(config, "flow_reinitialize_inlet_each_step", False)):
        return FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
    if str(flow_phase) == "preflow":
        preflow_mode = getattr(config, "preflow_flow_driver_mode", None)
        if preflow_mode is not None and str(preflow_mode):
            return str(preflow_mode)
    return str(getattr(config, "flow_driver_mode", FLOW_DRIVER_PROJECTION_ONLY))


def _require_fsi_physical_flow_driver_mode(config: Any) -> str:
    """Reject FSI modes that do not advance the requested physical time."""

    mode = _effective_flow_driver_mode(config, flow_phase="fsi")
    if mode not in FLOW_DRIVER_PHYSICAL_PREDICTOR_MODES:
        _macro_time_accounting_report(
            requested_macro_dt_s=float(config.dt_s),
            accepted_time_s=0.0,
            accepted_substep_count=1,
            rejected_trial_count=0,
            component="fluid",
        )
    return mode


def _flow_driver_requires_full_field_reinitialize(mode: str) -> bool:
    return mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC


def _flow_inlet_source_factor(config: Any, step_index: int) -> float:
    strength = float(getattr(config, "flow_inlet_source_strength", 1.0))
    profile = str(getattr(config, "flow_inlet_source_profile", "constant"))
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if profile == "constant" or ramp_steps <= 0:
        return strength
    if profile == "linear_ramp":
        ramp_fraction = min(1.0, max(0.0, float(step_index + 1) / float(ramp_steps)))
        return strength * ramp_fraction
    raise ValueError(f"unsupported flow_inlet_source_profile: {profile!r}")


def _flow_source_schedule_scope(config: Any) -> str:
    return str(getattr(config, "flow_inlet_source_schedule_scope", "global"))


def _flow_source_schedule_step_index(
    config: Any,
    *,
    step_index_local: int,
    step_index_global: int,
) -> int:
    if _flow_source_schedule_scope(config) == "global":
        return int(step_index_global)
    return int(step_index_local)


def _flow_source_ramp_restarted_after_preflow(
    config: Any,
    *,
    flow_phase: str,
    step_index_local: int,
    step_index_global: int,
    source_schedule_step_index: int,
    preflow_history: list[dict[str, object]],
) -> bool:
    if str(flow_phase) != "fsi" or not preflow_history:
        return False
    if _flow_source_schedule_scope(config) != "phase_local":
        return False
    if str(getattr(config, "flow_inlet_source_profile", "constant")) != "linear_ramp":
        return False
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if ramp_steps <= 0:
        return False
    return (
        int(step_index_global) >= ramp_steps
        and int(source_schedule_step_index) < ramp_steps
        and int(step_index_local) == int(source_schedule_step_index)
    )


def _flow_driver_report(
    *,
    mode: str,
    full_field_reinitialized: bool,
    inlet_boundary_report: Mapping[str, object],
    volume_source_applied: bool,
    source_factor: float = 0.0,
    source_normal_velocity_mps: float = 0.0,
    predictor_applied: bool = False,
    predictor_note: str = "",
    predictor_kinematic_viscosity_m2_s: float = 0.0,
    predictor_no_slip_domain_walls: tuple[bool, bool, bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
        False,
        False,
    ),
) -> dict[str, object]:
    inlet_reapplied = bool(inlet_boundary_report)
    return {
        "flow_driver_mode": mode,
        "flow_driver_diagnostic_only": mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC,
        "flow_driver_uses_full_velocity_reset": bool(full_field_reinitialized),
        "flow_full_field_reinitialized": bool(full_field_reinitialized),
        "flow_inlet_boundary_reapplied": inlet_reapplied,
        "flow_volume_source_applied": bool(volume_source_applied),
        "flow_inlet_boundary_active_cell_count": int(
            inlet_boundary_report.get("flow_inlet_boundary_active_cell_count", 0)
        ),
        "flow_inlet_boundary_obstacle_cell_count": int(
            inlet_boundary_report.get("flow_inlet_boundary_obstacle_cell_count", 0)
        ),
        "flow_inlet_source_factor": float(source_factor),
        "flow_inlet_source_normal_velocity_mps": float(source_normal_velocity_mps),
        "flow_predictor_applied": bool(predictor_applied),
        "flow_predictor_note": str(predictor_note),
        "flow_predictor_kinematic_viscosity_m2_s": float(
            predictor_kinematic_viscosity_m2_s
        ),
        "flow_predictor_no_slip_domain_walls": [
            bool(flag) for flag in predictor_no_slip_domain_walls
        ],
    }


def _zmax_inlet_boundary_device_refresh_compatible(config: Any) -> bool:
    if int(getattr(config, "flow_ymin_no_slip_rows", 0)) > 0:
        return False
    if _use_hibm_sharp_marker_boundary(config):
        return True
    return (
        int(getattr(config, "flow_obstacle_no_slip_layers", 0)) <= 0
        and int(getattr(config, "flow_obstacle_wake_no_slip_layers", 0)) <= 0
    )


def _refresh_zmax_inlet_boundary(
    fluid: CartesianFluidSolver,
    config: Any,
) -> dict[str, object]:
    authority = str(fluid.velocity_dirichlet_boundary_authority)
    if authority == "canonical":
        report = dict(
            fluid.refresh_zmax_inlet_boundary_canonical(
                inlet_velocity_mps=float(config.inlet_velocity_mps),
                streamwise_axis_index=STREAMWISE_AXIS_INDEX,
            )
        )
        _prepare_and_seal_canonical_velocity_dirichlet_component_ledger(fluid)
        return report
    device_refresh = getattr(fluid, "refresh_zmax_inlet_boundary", None)
    if (
        device_refresh is not None
        and _zmax_inlet_boundary_device_refresh_compatible(config)
    ):
        return dict(
            device_refresh(
                inlet_velocity_mps=float(config.inlet_velocity_mps),
                streamwise_axis_index=STREAMWISE_AXIS_INDEX,
            )
        )

    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    values = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
    weights = fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
    enforcement_weights = (
        fluid.velocity_dirichlet_boundary_enforcement_weight.to_numpy()
    )
    marker_regions = fluid.velocity_dirichlet_boundary_marker_region_id.to_numpy()
    hard_masks = (
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
    )
    external_exact_masks = (
        fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
    )
    owned_rows = fluid.velocity_dirichlet_boundary_owned_row.to_numpy()
    obstacle = fluid.obstacle.to_numpy()
    k = int(config.grid_nodes[2]) - 1
    _apply_ymin_no_slip_rows(
        active,
        values,
        weights,
        marker_regions,
        hard_masks,
        external_exact_masks,
        owned_rows,
        obstacle,
        config,
    )
    if not _use_hibm_sharp_marker_boundary(config):
        _apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )
    direct_rows = owned_rows == 0
    enforcement_weights[direct_rows] = weights[direct_rows]
    enforcement_weights[active == 0] = 0.0
    fluid_mask = obstacle[:, :, k] == 0

    active[:, :, k] = fluid_mask.astype(np.int32)
    values[:, :, k, :] = 0.0
    values[:, :, k, STREAMWISE_AXIS_INDEX] = (
        -float(config.inlet_velocity_mps) * fluid_mask.astype(np.float32)
    )
    weights[:, :, k] = fluid_mask.astype(np.float32)
    enforcement_weights[:, :, k] = fluid_mask.astype(np.float32)
    marker_regions[:, :, k] = -1
    hard_masks[:, :, k] = np.where(fluid_mask, 0b111, 0).astype(np.int32)
    external_exact_masks[:, :, k] = np.where(
        fluid_mask,
        external_exact_masks[:, :, k] | np.int32(0b100),
        0,
    ).astype(np.int32)
    owned_rows[:, :, k] = 0

    fluid.velocity_dirichlet_boundary_active.from_numpy(active)
    fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
    fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
    fluid.velocity_dirichlet_boundary_enforcement_weight.from_numpy(
        enforcement_weights
    )
    fluid.velocity_dirichlet_boundary_marker_region_id.from_numpy(marker_regions)
    fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(hard_masks)
    fluid.velocity_dirichlet_boundary_external_exact_component_mask.from_numpy(
        external_exact_masks
    )
    fluid.velocity_dirichlet_boundary_owned_row.from_numpy(owned_rows)
    return _zmax_inlet_boundary_report(fluid)


def _zmax_inlet_boundary_report(
    fluid: CartesianFluidSolver,
) -> dict[str, object]:
    device_report = getattr(fluid, "zmax_inlet_boundary_report", None)
    if device_report is not None:
        return dict(device_report())
    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    obstacle = fluid.obstacle.to_numpy()
    k = active.shape[2] - 1
    active_slice = active[:, :, k] != 0
    obstacle_slice = obstacle[:, :, k] != 0
    return {
        "flow_inlet_boundary_active_cell_count": int(active_slice.sum()),
        "flow_inlet_boundary_obstacle_cell_count": int(
            np.logical_and(active_slice, obstacle_slice).sum()
        ),
    }


_PREFLOW_SNAPSHOT_FSI_ONLY_CONFIG_FIELDS = frozenset(
    {
        "step_count",
        "young_modulus_pa",
        "poisson_ratio",
        "solid_density_kgm3",
        "solid_constitutive_model",
        "solid_substeps",
        "solid_cfl_target",
        "solid_velocity_transfer_flip_blend",
        "kalman_writeback_mode",
        "kalman_interface_config",
        "kalman_fluid_config",
        "kalman_solid_config",
        "coupling_mode",
        "fsi_coupling_max_iterations",
        "fsi_coupling_absolute_tolerance_mps",
        "fsi_coupling_relative_tolerance",
        "iqn_history_limit",
        "iqn_initial_picard_relaxation",
        "iqn_svd_relative_cutoff",
        "iqn_reuse_previous_step_history",
        "initial_guess_mode",
        "initial_guess_kalman_config",
        "initial_guess_oracle_path",
        "iqn_kalman_oracle_interpolation_target_step",
        "iqn_kalman_oracle_interpolation_oracle_path",
        "iqn_kalman_oracle_interpolation_alphas",
        "detailed_preflow_stage_progress",
        "velocity_damping",
        "fixed_node_lock_policy",
        "displacement_tolerance",
        "velocity_peak_tolerance",
        "export_final_flow_snapshot",
        "preflow_snapshot_input_path",
        "preflow_snapshot_output_path",
    }
)


def _preflow_snapshot_config_payload(config: Any) -> dict[str, object]:
    if hasattr(config, "__dataclass_fields__"):
        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif hasattr(config, "__dict__"):
        payload = dict(vars(config))
    else:
        raise TypeError("preflow snapshot config must be a dataclass or mapping")
    return {
        str(key): value
        for key, value in payload.items()
        if str(key) not in _PREFLOW_SNAPSHOT_FSI_ONLY_CONFIG_FIELDS
    }


def _capture_preflow_snapshot_fields(fluid: Any) -> dict[str, np.ndarray]:
    fluid._require_velocity_dirichlet_component_ledger_sealed()
    return {
        name: np.asarray(getattr(fluid, name).to_numpy()).copy()
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }


_CANONICAL_SNAPSHOT_RESTORE_PREPARE_METHODS = (
    ("apply", "prepare_velocity_dirichlet_component_ledger_apply"),
    ("divergence", "prepare_velocity_dirichlet_component_ledger_divergence"),
    ("reachability", "prepare_velocity_dirichlet_component_ledger_reachability"),
    ("fv_operator", "prepare_velocity_dirichlet_component_ledger_fv_operator"),
    ("gradient", "prepare_velocity_dirichlet_component_ledger_gradient"),
    ("multigrid", "prepare_velocity_dirichlet_component_ledger_multigrid"),
    ("projection", "prepare_velocity_dirichlet_component_ledger_projection"),
    ("no_slip", "prepare_hibm_no_slip_component_face_valid_mask"),
    ("reference", "prepare_velocity_dirichlet_component_ledger_reference"),
    ("snapshot", "prepare_velocity_dirichlet_component_ledger_snapshot"),
)


def _canonical_snapshot_restore_prepare_plan(
    fluid: Any,
) -> tuple[tuple[str, str, Callable[[], object]], ...]:
    """Resolve every required canonical prepare path without mutating state.

    This preflight deliberately runs before the first ``from_numpy`` commit.
    During a phased migration, an incomplete physical-consumer set therefore
    rejects a canonical snapshot with zero runtime writes instead of entering
    a prepare/seal deadlock and relying on a second failing rebuild to unwind.
    """

    required_value = getattr(
        fluid,
        "_VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS",
        None,
    )
    if required_value is None:
        raise RuntimeError(
            "canonical snapshot restore is not ready before commit; missing "
            "the required physical-consumer registry"
        )
    try:
        required_consumers = frozenset(str(item) for item in required_value)
    except TypeError as exc:
        raise RuntimeError(
            "canonical snapshot restore is not ready before commit; invalid "
            "physical-consumer registry"
        ) from exc
    if not required_consumers:
        raise RuntimeError(
            "canonical snapshot restore is not ready before commit; missing "
            "required physical consumers"
        )

    prepare_plan: list[tuple[str, str, Callable[[], object]]] = []
    planned_consumers: set[str] = set()
    missing_prepare_consumers: list[str] = []
    for consumer, method_name in _CANONICAL_SNAPSHOT_RESTORE_PREPARE_METHODS:
        if consumer not in required_consumers:
            continue
        planned_consumers.add(consumer)
        prepare = getattr(fluid, method_name, None)
        if not callable(prepare):
            missing_prepare_consumers.append(consumer)
            continue
        prepare_plan.append((consumer, method_name, prepare))
    missing_prepare_consumers.extend(
        sorted(required_consumers - planned_consumers)
    )

    lifecycle_methods = (
        "_velocity_dirichlet_component_ledger_generation_errors",
        "seal_velocity_dirichlet_component_ledger",
        "_require_velocity_dirichlet_component_ledger_sealed",
    )
    missing_lifecycle_methods = [
        name for name in lifecycle_methods if not callable(getattr(fluid, name, None))
    ]
    if missing_prepare_consumers or missing_lifecycle_methods:
        details: list[str] = []
        if missing_prepare_consumers:
            details.append(
                "missing migrated consumer prepare APIs="
                f"{sorted(set(missing_prepare_consumers))}"
            )
        if missing_lifecycle_methods:
            details.append(
                "missing lifecycle APIs=" f"{sorted(missing_lifecycle_methods)}"
            )
        raise RuntimeError(
            "canonical snapshot restore is not ready before commit; "
            + ", ".join(details)
        )
    return tuple(prepare_plan)


def _prepare_canonical_preflow_snapshot_restore(
    fluid: Any,
    prepare_plan: tuple[tuple[str, str, Callable[[], object]], ...],
) -> None:
    """Prepare all migrated consumers, then seal, then permit sealed reads."""

    invalidate_reachability = getattr(
        fluid,
        "_invalidate_hibm_pressure_reachability",
        None,
    )
    if callable(invalidate_reachability):
        invalidate_reachability()

    no_slip_prepare_name = "prepare_hibm_no_slip_component_face_valid_mask"
    method_names = {method_name for _consumer, method_name, _prepare in prepare_plan}
    if no_slip_prepare_name not in method_names:
        raise RuntimeError(
            "canonical snapshot restore is missing the no-slip prepare API"
        )
    for _consumer, _method_name, prepare in prepare_plan:
        prepare()

    missing, unexpected, mismatched, invalid_capabilities = (
        fluid._velocity_dirichlet_component_ledger_generation_errors()
    )
    if missing or unexpected or mismatched or invalid_capabilities:
        raise RuntimeError(
            "canonical snapshot restore consumer preparation is incomplete "
            "before seal: "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}, "
            f"invalid_capabilities={invalid_capabilities}"
        )
    fluid.seal_velocity_dirichlet_component_ledger()
    # No sealed-only reader or diagnostic is permitted above this point.
    fluid._require_velocity_dirichlet_component_ledger_sealed()


def _prepare_and_seal_canonical_velocity_dirichlet_component_ledger(
    fluid: Any,
) -> None:
    """Prepare every physical consumer before any canonical ledger read."""

    authority = str(getattr(fluid, "velocity_dirichlet_boundary_authority", ""))
    if authority != "canonical":
        raise RuntimeError(
            "canonical velocity Dirichlet prepare/seal requires canonical authority"
        )
    prepare_plan = _canonical_snapshot_restore_prepare_plan(fluid)
    _prepare_canonical_preflow_snapshot_restore(fluid, prepare_plan)


def _rebuild_legacy_preflow_snapshot_derived_state(fluid: Any) -> None:
    """Preserve the established legacy restore ordering exactly."""

    invalidate_reachability = getattr(
        fluid,
        "_invalidate_hibm_pressure_reachability",
        None,
    )
    if callable(invalidate_reachability):
        invalidate_reachability()
    rebuild_no_slip_obstacle = getattr(
        fluid,
        "build_hibm_no_slip_sampling_obstacle",
        None,
    )
    if callable(rebuild_no_slip_obstacle):
        rebuild_no_slip_obstacle()
    rebuild_no_slip_component_mask = getattr(
        fluid,
        "build_hibm_no_slip_component_face_valid_mask",
        None,
    )
    if callable(rebuild_no_slip_component_mask):
        rebuild_no_slip_component_mask()


def _rollback_preflow_snapshot_restore(
    *,
    runtime_fields: Mapping[str, Any],
    runtime_backups: Mapping[str, np.ndarray],
    mirror_fields: Mapping[str, Any],
    mirror_backups: Mapping[str, np.ndarray],
    derived_fields: Mapping[str, Any],
    derived_backups: Mapping[str, np.ndarray],
    fluid: Any,
    metadata_backup: Mapping[str, object],
    missing_metadata: object,
) -> None:
    """Restore direct backups without re-entering prepare/seal failure paths."""

    rollback_errors: list[str] = []
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        try:
            runtime_fields[name].from_numpy(runtime_backups[name])
        except BaseException as exc:  # pragma: no cover - catastrophic backend loss.
            rollback_errors.append(f"field {name!r}: {exc!r}")
    for name, runtime_field in mirror_fields.items():
        try:
            runtime_field.from_numpy(mirror_backups[name])
        except BaseException as exc:  # pragma: no cover - catastrophic backend loss.
            rollback_errors.append(f"mirror {name!r}: {exc!r}")
    for name, runtime_field in derived_fields.items():
        try:
            runtime_field.from_numpy(derived_backups[name])
        except BaseException as exc:  # pragma: no cover - catastrophic backend loss.
            rollback_errors.append(f"derived {name!r}: {exc!r}")
    for name, previous in metadata_backup.items():
        try:
            if previous is missing_metadata:
                if hasattr(fluid, name):
                    delattr(fluid, name)
            else:
                setattr(fluid, name, previous)
        except BaseException as exc:  # pragma: no cover - exotic host proxy.
            rollback_errors.append(f"metadata {name!r}: {exc!r}")
    if rollback_errors:
        raise RuntimeError(
            "preflow snapshot restore rollback failed: " + "; ".join(rollback_errors)
        )


def _restore_preflow_snapshot_fields(
    fluid: Any,
    fields: Mapping[str, np.ndarray],
    *,
    velocity_dirichlet_boundary_authority: str | None = None,
    velocity_dirichlet_component_ledger_generation: int | None = None,
) -> None:
    if set(fields) != set(PREFLOW_SNAPSHOT_FIELD_NAMES):
        raise ValueError("preflow snapshot fields do not match the runtime contract")
    current_authority = str(fluid.velocity_dirichlet_boundary_authority)
    if velocity_dirichlet_boundary_authority is None:
        velocity_dirichlet_boundary_authority = current_authority
    if velocity_dirichlet_component_ledger_generation is None:
        velocity_dirichlet_component_ledger_generation = int(
            fluid.velocity_dirichlet_component_ledger_generation
        )
    if velocity_dirichlet_boundary_authority != current_authority:
        raise ValueError(
            "preflow snapshot velocity boundary authority changed after load: "
            f"snapshot={velocity_dirichlet_boundary_authority!r}, "
            f"solver={current_authority!r}"
        )
    if (
        not isinstance(velocity_dirichlet_component_ledger_generation, int)
        or isinstance(velocity_dirichlet_component_ledger_generation, bool)
        or velocity_dirichlet_component_ledger_generation < 0
    ):
        raise ValueError(
            "preflow snapshot velocity boundary generation must be a "
            "non-negative integer"
        )
    # Validate the complete host payload and every destination shape/dtype
    # before the first runtime mutation.  A direct helper caller therefore has
    # the same fail-closed schema contract as a disk-loaded PreflowSnapshot.
    validated_fields = validate_preflow_snapshot_fields(
        fields,
        velocity_dirichlet_boundary_authority=current_authority,
    )
    runtime_fields: dict[str, Any] = {}
    runtime_backups: dict[str, np.ndarray] = {}
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        runtime_field = getattr(fluid, name, None)
        if runtime_field is None or not callable(
            getattr(runtime_field, "to_numpy", None)
        ) or not callable(getattr(runtime_field, "from_numpy", None)):
            raise ValueError(
                f"preflow snapshot runtime field {name!r} is unavailable"
            )
        current = np.asarray(runtime_field.to_numpy()).copy()
        proposed = validated_fields[name]
        if current.shape != proposed.shape or current.dtype != proposed.dtype:
            raise ValueError(
                f"preflow snapshot runtime field {name!r} shape/dtype mismatch"
            )
        runtime_fields[name] = runtime_field
        runtime_backups[name] = current

    mirror_fields: dict[str, Any] = {}
    mirror_backups: dict[str, np.ndarray] = {}
    for name in ("pressure_accum", "pressure_tmp"):
        runtime_field = getattr(fluid, name, None)
        if runtime_field is None:
            continue
        current = np.asarray(runtime_field.to_numpy()).copy()
        proposed = validated_fields["pressure"]
        if current.shape != proposed.shape or current.dtype != proposed.dtype:
            raise ValueError(
                f"preflow snapshot runtime field {name!r} shape/dtype mismatch"
            )
        mirror_fields[name] = runtime_field
        mirror_backups[name] = current

    metadata_names = (
        "velocity_dirichlet_component_ledger_generation",
        "velocity_dirichlet_component_ledger_sealed",
        "_velocity_dirichlet_component_ledger_consumer_generations",
        "_velocity_dirichlet_component_ledger_consumer_capabilities",
        "_hibm_base_obstacle_initialized",
        "hibm_dynamic_solid_volume_enabled",
        "_sst_wall_distance_valid",
    )
    missing = object()
    metadata_backup = {
        name: getattr(fluid, name, missing) for name in metadata_names
    }

    derived_fields: dict[str, Any] = {}
    derived_backups: dict[str, np.ndarray] = {}
    for name in (
        "hibm_no_slip_sampling_obstacle",
        "hibm_no_slip_component_face_valid_mask",
    ):
        runtime_field = getattr(fluid, name, None)
        if runtime_field is None:
            continue
        to_numpy = getattr(runtime_field, "to_numpy", None)
        from_numpy = getattr(runtime_field, "from_numpy", None)
        if not callable(to_numpy) or not callable(from_numpy):
            raise ValueError(
                f"preflow snapshot derived runtime field {name!r} is incomplete"
            )
        derived_fields[name] = runtime_field
        derived_backups[name] = np.asarray(to_numpy()).copy()

    canonical_prepare_plan = (
        _canonical_snapshot_restore_prepare_plan(fluid)
        if current_authority == "canonical"
        else ()
    )

    try:
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
            runtime_fields[name].from_numpy(validated_fields[name])
        for runtime_field in mirror_fields.values():
            runtime_field.from_numpy(validated_fields["pressure"])
        if (
            getattr(fluid, "turbulence_model", "laminar") == "sst_2003"
            and hasattr(fluid, "_sst_wall_distance_valid")
        ):
            # Schema v8 restores the physical wall-distance field itself, so
            # do not discard it and rebuild a different field before FSI.
            fluid._sst_wall_distance_valid = True
        fluid.velocity_dirichlet_component_ledger_generation = int(
            velocity_dirichlet_component_ledger_generation
        )
        fluid.velocity_dirichlet_component_ledger_sealed = False
        fluid._velocity_dirichlet_component_ledger_consumer_generations = {}
        fluid._velocity_dirichlet_component_ledger_consumer_capabilities = {}
        if hasattr(fluid, "_hibm_base_obstacle_initialized"):
            fluid._hibm_base_obstacle_initialized = True
        if hasattr(fluid, "hibm_dynamic_solid_volume_enabled"):
            fluid.hibm_dynamic_solid_volume_enabled = bool(
                np.any(
                    validated_fields[
                        "hibm_dynamic_solid_volume_obstacle"
                    ]
                )
            )
        if current_authority == "canonical":
            _prepare_canonical_preflow_snapshot_restore(
                fluid,
                canonical_prepare_plan,
            )
        else:
            _rebuild_legacy_preflow_snapshot_derived_state(fluid)
            fluid._require_velocity_dirichlet_component_ledger_sealed()
    except BaseException as restore_error:
        # Rollback restores direct backups only.  Re-entering prepare/seal here
        # can reproduce the same failure and obscure the original exception.
        try:
            _rollback_preflow_snapshot_restore(
                runtime_fields=runtime_fields,
                runtime_backups=runtime_backups,
                mirror_fields=mirror_fields,
                mirror_backups=mirror_backups,
                derived_fields=derived_fields,
                derived_backups=derived_backups,
                fluid=fluid,
                metadata_backup=metadata_backup,
                missing_metadata=missing,
            )
        except BaseException as rollback_error:
            raise RuntimeError(
                "preflow snapshot restore failed and rollback was incomplete: "
                f"{rollback_error}"
            ) from restore_error
        raise


def _preflow_snapshot_source_payload() -> dict[str, bytes]:
    repo_root = Path(__file__).resolve().parents[2]
    # Snapshot compatibility follows the executable solver dependency surface,
    # not every case and benchmark that happens to share the repository.  The
    # canonical config and geometry hashes already identify the active case.
    # Include this runner explicitly plus all reusable solver modules; future
    # runtime callbacks must opt in through an explicit dependency manifest.
    runner_path = Path(__file__).resolve()
    paths: set[Path] = {runner_path} if runner_path.is_file() else set()
    solver_root = repo_root / "simulation_core"
    if solver_root.is_dir():
        paths.update(
            path for path in solver_root.rglob("*.py") if path.is_file()
        )
    return {
        path.relative_to(repo_root).as_posix(): path.read_bytes()
        for path in sorted(paths)
    }


def _preflow_snapshot_geometry_payload(
    *,
    fluid: Any,
    markers: Any,
    solid: Any,
) -> dict[str, np.ndarray]:
    marker_count = int(markers.marker_count)
    projection_vertex_count = int(markers.projection_vertex_count)
    projection_triangle_count = int(markers.projection_triangle_count)
    projection_segment_count = int(markers.projection_segment_count)
    projection_indices = markers.projection_triangle_indices.to_numpy()
    solid_count = int(solid.particle_count)
    geometry = {
        "cell_face_x_m": fluid.cell_face_x_m.to_numpy(),
        "cell_face_y_m": fluid.cell_face_y_m.to_numpy(),
        "cell_face_z_m": fluid.cell_face_z_m.to_numpy(),
        "cell_center_x_m": fluid.cell_center_x_m.to_numpy(),
        "cell_center_y_m": fluid.cell_center_y_m.to_numpy(),
        "cell_center_z_m": fluid.cell_center_z_m.to_numpy(),
        "cell_width_x_m": fluid.cell_width_x_m.to_numpy(),
        "cell_width_y_m": fluid.cell_width_y_m.to_numpy(),
        "cell_width_z_m": fluid.cell_width_z_m.to_numpy(),
        "initial_obstacle": fluid.obstacle.to_numpy(),
        "initial_hibm_base_obstacle": fluid.hibm_base_obstacle.to_numpy(),
        "initial_hibm_dynamic_solid_volume_obstacle": (
            fluid.hibm_dynamic_solid_volume_obstacle.to_numpy()
        ),
        "marker_population_count": np.asarray(
            (marker_count, projection_vertex_count),
            dtype=np.int32,
        ),
        "external_velocity_boundary_x_face_active_component_mask": (
            fluid.external_velocity_boundary_x_face_active_component_mask.to_numpy()
        ),
        "external_velocity_boundary_x_face_value_mps": (
            fluid.external_velocity_boundary_x_face_value_mps.to_numpy()
        ),
        "external_velocity_boundary_y_face_active_component_mask": (
            fluid.external_velocity_boundary_y_face_active_component_mask.to_numpy()
        ),
        "external_velocity_boundary_y_face_value_mps": (
            fluid.external_velocity_boundary_y_face_value_mps.to_numpy()
        ),
        "external_velocity_boundary_z_face_active_component_mask": (
            fluid.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        ),
        "external_velocity_boundary_z_face_value_mps": (
            fluid.external_velocity_boundary_z_face_value_mps.to_numpy()
        ),
        "marker_position_m": markers.x_gamma_m.to_numpy()[:marker_count],
        "marker_normal": markers.n_gamma.to_numpy()[:marker_count],
        "marker_area_m2": markers.A_gamma_m2.to_numpy()[:marker_count],
        "marker_region_id": markers.region_id.to_numpy()[:marker_count],
        "marker_projection_position_m": markers.x_gamma_m.to_numpy()[
            :projection_vertex_count
        ],
        "marker_projection_velocity_mps": markers.v_gamma_mps.to_numpy()[
            :projection_vertex_count
        ],
        "marker_projection_normal": markers.n_gamma.to_numpy()[
            :projection_vertex_count
        ],
        "marker_projection_area_m2": markers.A_gamma_m2.to_numpy()[
            :projection_vertex_count
        ],
        "marker_projection_region_id": markers.region_id.to_numpy()[
            :projection_vertex_count
        ],
        "marker_projection_pressure_owner_index": (
            markers.projection_vertex_pressure_owner_index.to_numpy()[
                :projection_vertex_count
            ]
        ),
        "marker_projection_topology_count": np.asarray(
            (projection_triangle_count, projection_segment_count),
            dtype=np.int32,
        ),
        "marker_projection_triangle_indices": projection_indices[
            :projection_triangle_count
        ],
        "marker_projection_segment_indices": projection_indices[
            :projection_segment_count, :2
        ],
        "marker_pressure_probe_origin_m": (
            markers.pressure_probe_origin_m.to_numpy()[:marker_count]
        ),
        "marker_pressure_probe_origin_explicit": (
            markers.pressure_probe_origin_explicit.to_numpy()[:marker_count]
        ),
        "solid_rest_position_m": solid.rest_x.to_numpy()[:solid_count],
        "solid_fixed_particle": solid.fixed_particle.to_numpy()[:solid_count],
    }
    return {name: np.asarray(value) for name, value in geometry.items()}


def _preflow_snapshot_identity(
    *,
    markers: Any,
    fluid: Any,
    solid: Any,
    config: Any,
) -> PreflowSnapshotIdentity:
    return PreflowSnapshotIdentity.from_inputs(
        config=_preflow_snapshot_config_payload(config),
        sources=_preflow_snapshot_source_payload(),
        geometry=_preflow_snapshot_geometry_payload(
            fluid=fluid,
            markers=markers,
            solid=solid,
        ),
    )


_PREFLOW_OPTIONAL_NAN_PROJECTION_DIAGNOSTIC_FIELDS = frozenset(
    {
        "zmin_unreached_source_centroid_x_m",
        "zmin_unreached_source_centroid_y_m",
        "zmin_unreached_source_centroid_z_m",
        "zmin_unreached_source_min_x_m",
        "zmin_unreached_source_min_y_m",
        "zmin_unreached_source_min_z_m",
        "zmin_unreached_source_max_x_m",
        "zmin_unreached_source_max_y_m",
        "zmin_unreached_source_max_z_m",
    }
)


def _mutable_preflow_report_value(
    value: object,
    *,
    path: tuple[object, ...] = (),
) -> object:
    """Copy a report into mutable, strict-JSON-safe snapshot metadata.

    Projection diagnostics use NaN as an explicit "unavailable"
    sentinel when, for example, no unreached-source centroid exists.  Those
    optional diagnostics are metadata rather than solver state, so persist the
    sentinel as JSON ``null`` while leaving the live report untouched.  The
    generic :class:`PreflowSnapshot` validator remains strict and continues to
    reject non-finite history supplied directly by other callers.  Infinite
    diagnostics are not valid sentinels and remain fail-closed.
    """

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        if (
            len(path) >= 2
            and path[-2] == "flow_projection_report"
            and path[-1] in _PREFLOW_OPTIONAL_NAN_PROJECTION_DIAGNOSTIC_FIELDS
        ):
            return None
        return value
    if isinstance(value, Mapping):
        return {
            key: _mutable_preflow_report_value(item, path=(*path, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _mutable_preflow_report_value(item, path=(*path, index))
            for index, item in enumerate(value)
        ]
    return value


def _preflow_expected_no_slip_marker_count(config: Any) -> int:
    if not _use_hibm_sharp_marker_boundary(config):
        return 0
    expected_marker_count = int(getattr(config, "marker_count", 0))
    if _traction_marker_layout(config) == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
        expected_marker_count *= 2
    return expected_marker_count


def _preflow_expected_tip_cap_traction_marker_count(config: Any) -> int:
    if not _use_hibm_sharp_marker_boundary(config):
        return 0
    if _traction_marker_layout(config) != TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
        return 0
    return 2 if _traction_tip_cap_pressure_enabled(config) else 0


def _preflow_traction_readiness_mode(config: Any) -> str:
    mode = str(
        getattr(
            config,
            "preflow_traction_readiness_mode",
            PREFLOW_TRACTION_READINESS_FLOW_ONLY,
        )
    )
    if mode not in PREFLOW_TRACTION_READINESS_MODES:
        raise ValueError(f"unsupported preflow_traction_readiness_mode: {mode!r}")
    return mode


def _preflow_traction_readiness(
    rows: list[Mapping[str, object]],
    config: Any,
) -> str:
    if not rows:
        return PREFLOW_TRACTION_NOT_EVALUATED

    expected_side_count = _preflow_expected_no_slip_marker_count(config)
    expected_tip_cap_count = _preflow_expected_tip_cap_traction_marker_count(config)

    def nonnegative_count(row: Mapping[str, object], key: str, default: int) -> int | None:
        value = row.get(key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            return None
        count = int(value)
        return count if count >= 0 else None

    readiness_states: list[str] = []
    for row in rows:
        valid_count = nonnegative_count(row, "stress_valid_marker_count", -1)
        invalid_count = nonnegative_count(row, "stress_invalid_marker_count", -1)
        tip_cap_count = nonnegative_count(row, "tip_cap_marker_count", 0)
        tip_cap_valid = nonnegative_count(row, "tip_cap_valid_marker_count", 0)
        tip_cap_invalid = nonnegative_count(row, "tip_cap_invalid_marker_count", 0)
        counts = (
            valid_count,
            invalid_count,
            tip_cap_count,
            tip_cap_valid,
            tip_cap_invalid,
        )
        if None in counts:
            return PREFLOW_TRACTION_INVALID
        parsed_counts = tuple(int(count) for count in counts)
        (
            valid_count,
            invalid_count,
            tip_cap_count,
            tip_cap_valid,
            tip_cap_invalid,
        ) = parsed_counts
        if (
            tip_cap_count != expected_tip_cap_count
            or tip_cap_valid != expected_tip_cap_count
            or tip_cap_invalid != 0
            or tip_cap_valid + tip_cap_invalid != tip_cap_count
            or tip_cap_valid > valid_count
            or tip_cap_invalid > invalid_count
        ):
            return PREFLOW_TRACTION_INVALID

        side_valid = valid_count - tip_cap_valid
        side_invalid = invalid_count - tip_cap_invalid
        if expected_side_count == 0:
            if side_valid != 0 or side_invalid != 0:
                return PREFLOW_TRACTION_INVALID
            readiness_states.append(
                PREFLOW_TRACTION_EVALUATED
                if expected_tip_cap_count > 0
                else PREFLOW_TRACTION_NOT_EVALUATED
            )
        elif side_valid == expected_side_count and side_invalid == 0:
            readiness_states.append(PREFLOW_TRACTION_EVALUATED)
        elif side_valid == 0 and side_invalid == expected_side_count:
            readiness_states.append(PREFLOW_TRACTION_NOT_EVALUATED)
        else:
            return PREFLOW_TRACTION_INVALID
    first_state = readiness_states[0]
    if any(state != first_state for state in readiness_states[1:]):
        return PREFLOW_TRACTION_INVALID
    return first_state


def _preflow_no_slip_limit_mps(config: Any) -> float:
    return float(
        abs(float(getattr(config, "inlet_velocity_mps", 0.0)))
        * float(
            getattr(
                config,
                "preflow_stationary_no_slip_tolerance_fraction",
                0.05,
            )
        )
    )


def _preflow_snapshot_rejection_error(
    *,
    payload: Mapping[str, object],
    gate: str,
    message: str,
) -> PreflowSnapshotValidationError:
    history_value = _mutable_preflow_report_value(payload.get("preflow_history", []))
    history = history_value if isinstance(history_value, list) else []
    terminal_value = history[-1] if history else {}
    terminal = terminal_value if isinstance(terminal_value, dict) else {}
    return PreflowSnapshotValidationError(
        message,
        diagnostics={
            "preflow_snapshot_rejection": {
                "status": "rejected",
                "gate": str(gate),
                "preflow_steps_completed": int(
                    payload.get("preflow_steps_completed", len(history))
                ),
                "preflow_history": history,
                "terminal_preflow_diagnostics": dict(terminal),
            }
        },
    )


def _strict_pressure_health_report_int(
    report: Mapping[str, object],
    key: str,
) -> int | None:
    value = report.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        return None
    return int(value)


def _preflow_pressure_operator_health_failure(
    row: Mapping[str, object],
) -> str | None:
    """Validate final pressure topology using the operator that was solved.

    The preassembly Cartesian flood deliberately runs before HIBM pressure
    interface rows exist.  Its remaining-cell count is therefore diagnostic,
    not the final nullspace authority.  A nonzero count is acceptable only
    when the later exact operator graph was actually prepared from a valid
    interface matrix.  All paths still fail closed on missing or inconsistent
    exact-graph diagnostics.
    """

    projection_value = row.get("flow_projection_report")
    if not isinstance(projection_value, Mapping):
        return "final pressure projection report is missing"
    projection_report = projection_value

    if projection_report.get("pressure_nullspace_compatibility_measured") is not True:
        return "final pressure nullspace compatibility was not measured"
    if projection_report.get("pressure_nullspace_component_labels_converged") is not True:
        return "final pressure operator component labels did not converge"
    if projection_report.get("pressure_nullspace_component_overflow") is not False:
        return "final pressure operator component graph overflowed"

    component_count = _strict_pressure_health_report_int(
        projection_report,
        "pressure_nullspace_component_count",
    )
    incompatible_count = _strict_pressure_health_report_int(
        projection_report,
        "pressure_nullspace_incompatible_component_count",
    )
    if component_count is None or component_count < 0:
        return "final pressure operator component count is unavailable"
    if incompatible_count is None or incompatible_count != 0:
        return "final pressure operator has incompatible nullspace components"
    if component_count > 0 and not (
        projection_report.get(
            "pressure_nullspace_componentwise_projection_applied"
        )
        is True
        or projection_report.get("pressure_nullspace_zero_mean_projection_applied")
        is True
    ):
        return "final pressure nullspace components were not projected"

    remaining_unreached_cells = _strict_pressure_health_report_int(
        row,
        "hibm_preassembly_remaining_unreached_cell_count",
    )
    if remaining_unreached_cells is None or remaining_unreached_cells < 0:
        return "preassembly pressure reachability count is unavailable"
    if remaining_unreached_cells == 0:
        return None

    if projection_report.get("pressure_outlet_operator_graph_prepared") is not True:
        return "nonzero Cartesian pockets lack a prepared exact operator graph"
    if projection_report.get("pressure_interface_matrix_active") is not True:
        return "nonzero Cartesian pockets lack an active pressure interface matrix"
    invalid_interface_row_count = _strict_pressure_health_report_int(
        projection_report,
        "pressure_interface_matrix_row_invalid_count",
    )
    if invalid_interface_row_count is None or invalid_interface_row_count != 0:
        return "pressure interface matrix contains invalid rows"
    overflowed_interface_row_count = _strict_pressure_health_report_int(
        projection_report,
        "pressure_interface_matrix_row_overflow_count",
    )
    if (
        overflowed_interface_row_count is None
        or overflowed_interface_row_count != 0
    ):
        return "pressure interface matrix row storage overflowed"

    interface_diagonal_cell_count = _strict_pressure_health_report_int(
        projection_report,
        "unreached_cells_with_interface_diagonal",
    )
    interface_coupling_cell_count = _strict_pressure_health_report_int(
        projection_report,
        "unreached_cells_with_interface_coupling",
    )
    raw_component_count = _strict_pressure_health_report_int(
        projection_report,
        "cg_unreached_component_raw_count",
    )
    compact_component_count = _strict_pressure_health_report_int(
        projection_report,
        "cg_unreached_component_count",
    )
    interface_hit_component_count = _strict_pressure_health_report_int(
        projection_report,
        "unreached_components_with_interface_hits",
    )
    if (
        interface_diagonal_cell_count is None
        or interface_diagonal_cell_count < 0
        or interface_diagonal_cell_count != remaining_unreached_cells
    ):
        return "Cartesian pressure pockets lack complete interface diagonal coverage"
    if (
        interface_coupling_cell_count is None
        or interface_coupling_cell_count < 0
        or interface_coupling_cell_count != remaining_unreached_cells
    ):
        return "Cartesian pressure pockets lack complete interface coupling coverage"
    if (
        raw_component_count is None
        or raw_component_count <= 0
        or raw_component_count > remaining_unreached_cells
    ):
        return "Cartesian pressure pocket component count is inconsistent"
    if (
        compact_component_count is None
        or compact_component_count <= 0
        or compact_component_count > raw_component_count
    ):
        return "Cartesian pressure pocket compact component count is inconsistent"
    if (
        interface_hit_component_count is None
        or interface_hit_component_count < 0
        or interface_hit_component_count != compact_component_count
    ):
        return "Cartesian pressure pocket components lack complete interface coverage"
    if component_count > 0 and (
        projection_report.get(
            "pressure_nullspace_componentwise_projection_applied"
        )
        is not True
        or projection_report.get("pressure_nullspace_policy")
        != "outlet_disconnected_fv_cg_operator_componentwise_zero_mean"
    ):
        return "outlet pressure nullspace components lack exact componentwise projection"
    return None


def _preflow_report_snapshot_payload(
    report: Mapping[str, object],
    config: Any,
) -> dict[str, object]:
    payload = _mutable_preflow_report_value(
        {
            key: value
            for key, value in report.items()
            if key != "final_flow_field_snapshot"
        }
    )
    if not isinstance(payload, dict):  # pragma: no cover - mapping above is a dict.
        raise TypeError("preflow snapshot report root must be a mapping")
    payload["final_flow_field_snapshot"] = {}
    history = payload.get("preflow_history")
    completed = int(payload.get("preflow_steps_completed", -1))
    if not isinstance(history, list) or not history or completed != len(history):
        raise ValueError("preflow snapshot requires a complete non-empty history")
    if not all(isinstance(row, Mapping) for row in history):
        raise ValueError("preflow snapshot history rows must be mappings")
    final_row = history[-1]
    if not isinstance(final_row, Mapping):
        raise ValueError("preflow snapshot final history row must be a mapping")
    traction_mode = _preflow_traction_readiness_mode(config)
    traction_readiness = _preflow_traction_readiness([final_row], config)
    traction_ready = traction_readiness == PREFLOW_TRACTION_EVALUATED
    flow_only_not_evaluated = (
        traction_mode == PREFLOW_TRACTION_READINESS_FLOW_ONLY
        and traction_readiness == PREFLOW_TRACTION_NOT_EVALUATED
    )
    if not (traction_ready or flow_only_not_evaluated):
        raise _preflow_snapshot_rejection_error(
            payload=payload,
            gate="final_traction_readiness",
            message=(
                "preflow snapshot final traction readiness failed: "
                f"mode={traction_mode!r}, readiness={traction_readiness!r}"
            ),
        )
    payload["preflow_traction_readiness_mode"] = traction_mode
    payload["preflow_traction_readiness"] = traction_readiness
    convergence_mode = str(
        getattr(config, "preflow_convergence_mode", "single_step_legacy")
    )
    if convergence_mode == "windowed_stationary":
        if payload.get("preflow_convergence_mode") != "windowed_stationary":
            raise _preflow_snapshot_rejection_error(
                payload=payload,
                gate="windowed_convergence_mode",
                message=(
                    "windowed preflow snapshot report must record "
                    "preflow_convergence_mode='windowed_stationary'"
                ),
            )
        if payload.get("preflow_converged") is not True:
            raise _preflow_snapshot_rejection_error(
                payload=payload,
                gate="windowed_converged",
                message="windowed preflow snapshot requires preflow_converged=True",
            )
        if payload.get("preflow_stop_reason") != "windowed_stationary":
            raise _preflow_snapshot_rejection_error(
                payload=payload,
                gate="windowed_stop_reason",
                message=(
                    "windowed preflow snapshot requires "
                    "preflow_stop_reason='windowed_stationary'"
                ),
            )
        try:
            stationary_report = _preflow_windowed_stationary_report(
                history,
                config,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise _preflow_snapshot_rejection_error(
                payload=payload,
                gate="windowed_stationary_certificate_unverifiable",
                message=(
                    "windowed preflow snapshot terminal stationary window "
                    "certificate could not be verified"
                ),
            ) from exc
        if not bool(stationary_report.get("stationary", False)):
            raise _preflow_snapshot_rejection_error(
                payload=payload,
                gate="windowed_stationary_certificate",
                message=(
                    "windowed preflow snapshot terminal stationary window "
                    "certificate failed: "
                    f"{stationary_report.get('reason', 'unknown')}"
                ),
            )
        payload["preflow_stationary_certificate"] = (
            _mutable_preflow_report_value(stationary_report)
        )
    expected_valid_markers = _preflow_expected_no_slip_marker_count(config)
    valid_markers = int(final_row.get("hibm_no_slip_valid_marker_count", -1))
    if valid_markers != expected_valid_markers:
        raise _preflow_snapshot_rejection_error(
            payload=payload,
            gate="final_valid_marker_count",
            message=(
                "preflow snapshot final valid marker count does not match the "
                f"configured interface: {valid_markers} != {expected_valid_markers}"
            ),
        )
    no_slip_residual_mps = float(
        final_row.get("hibm_no_slip_max_residual_mps", math.inf)
    )
    no_slip_limit_mps = _preflow_no_slip_limit_mps(config)
    velocity_dirichlet_health_failure = (
        _hibm_velocity_dirichlet_health_failure(final_row)
    )
    pressure_operator_health_failure = (
        _preflow_pressure_operator_health_failure(final_row)
    )
    q_health_ok = not _use_hibm_sharp_marker_boundary(config) or (
        bool(
            final_row.get(
                "flow_projection_pre_projection_velocity_projector_prepared_all",
                False,
            )
        )
        and bool(
            final_row.get(
                "flow_projection_pre_projection_velocity_projector_converged_all",
                False,
            )
        )
        and bool(
            final_row.get(
                "flow_projection_pre_projection_velocity_projector_committed_all",
                False,
            )
        )
    )
    if (
        not math.isfinite(no_slip_residual_mps)
        or no_slip_residual_mps > no_slip_limit_mps
    ):
        raise _preflow_snapshot_rejection_error(
            payload=payload,
            gate="final_no_slip_residual",
            message=(
                "preflow snapshot final no-slip residual exceeds the configured "
                f"limit: {no_slip_residual_mps} > {no_slip_limit_mps} m/s"
            ),
        )
    if (
        not q_health_ok
        or not bool(final_row.get("flow_projection_cg_converged_all", False))
        or int(final_row.get("flow_projection_cg_breakdown_count", -1)) != 0
        or bool(final_row.get("flow_projection_pressure_solve_failed", True))
        or bool(
            final_row.get(
                "flow_projection_pressure_projection_physical_failure",
                True,
            )
        )
        or int(final_row.get("hibm_no_slip_invalid_marker_count", -1)) != 0
        or bool(final_row.get("hibm_preassembly_topology_mutated", True))
        or pressure_operator_health_failure is not None
        or velocity_dirichlet_health_failure is not None
    ):
        health_failures = tuple(
            failure
            for failure in (
                pressure_operator_health_failure,
                velocity_dirichlet_health_failure,
            )
            if failure is not None
        )
        detail = f": {'; '.join(health_failures)}" if health_failures else ""
        raise _preflow_snapshot_rejection_error(
            payload=payload,
            gate="final_numerical_health",
            message=(
                "preflow snapshot final state failed a numerical health gate" + detail
            ),
        )
    return payload


def _write_fixed_solid_preflow_snapshot(
    *,
    path: str | Path,
    report: Mapping[str, object],
    markers: Any,
    fluid: Any,
    solid: Any,
    config: Any,
    identity: PreflowSnapshotIdentity | None = None,
) -> None:
    if identity is None:
        identity = _preflow_snapshot_identity(
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
        )
    snapshot_fields = _capture_preflow_snapshot_fields(fluid)
    snapshot = PreflowSnapshot(
        fields=snapshot_fields,
        identity=identity,
        history=_preflow_report_snapshot_payload(report, config),
        velocity_dirichlet_boundary_authority=str(
            fluid.velocity_dirichlet_boundary_authority
        ),
        velocity_dirichlet_component_ledger_generation=int(
            fluid.velocity_dirichlet_component_ledger_generation
        ),
    )
    files = save_preflow_snapshot(path, snapshot)
    if isinstance(report, dict):
        report["preflow_snapshot_loaded"] = False
        report["preflow_snapshot_npz_path"] = str(files.npz_path)
        report["preflow_snapshot_metadata_path"] = str(files.metadata_path)
        report["preflow_snapshot_identity"] = {
            "config_sha256": identity.config_sha256,
            "source_sha256": identity.source_sha256,
            "geometry_sha256": identity.geometry_sha256,
        }


def _restore_fixed_solid_preflow_snapshot(
    *,
    path: str | Path,
    markers: Any,
    fluid: Any,
    solid: Any,
    config: Any,
    expected_identity: PreflowSnapshotIdentity | None = None,
) -> dict[str, object]:
    if expected_identity is None:
        expected_identity = _preflow_snapshot_identity(
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
        )
    snapshot = load_preflow_snapshot(
        path,
        expected_identity=expected_identity,
        expected_velocity_dirichlet_boundary_authority=str(
            fluid.velocity_dirichlet_boundary_authority
        ),
    )
    if not isinstance(snapshot.history, Mapping):
        raise ValueError("preflow snapshot history root must be a mapping")
    report = _preflow_report_snapshot_payload(snapshot.history, config)
    _restore_preflow_snapshot_fields(
        fluid,
        snapshot.fields,
        velocity_dirichlet_boundary_authority=(
            snapshot.velocity_dirichlet_boundary_authority
        ),
        velocity_dirichlet_component_ledger_generation=(
            snapshot.velocity_dirichlet_component_ledger_generation
        ),
    )
    report["preflow_status"] = "snapshot_loaded"
    report["preflow_stop_reason"] = "snapshot_loaded"
    report["preflow_snapshot_loaded"] = True
    report["preflow_snapshot_input_path"] = str(path)
    report["preflow_snapshot_identity"] = {
        "config_sha256": expected_identity.config_sha256,
        "source_sha256": expected_identity.source_sha256,
        "geometry_sha256": expected_identity.geometry_sha256,
    }
    return report


def _run_or_restore_fixed_solid_preflow(
    *,
    markers: Any,
    fluid: Any,
    solid: Any,
    config: Any,
    progress_observer: Callable[[dict[str, object]], None] | None = None,
    run_started_s: float | None = None,
    profile_wall_time: bool = False,
    particle_position_generation: int | None = None,
) -> dict[str, object]:
    input_path, output_path = _preflow_snapshot_paths(config)
    identity = (
        _preflow_snapshot_identity(
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
        )
        if input_path or output_path
        else None
    )
    if input_path is not None:
        return _restore_fixed_solid_preflow_snapshot(
            path=input_path,
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
            expected_identity=identity,
        )
    report = _run_fixed_solid_preflow(
        markers,
        fluid,
        solid,
        config,
        progress_observer=progress_observer,
        run_started_s=run_started_s,
        profile_wall_time=profile_wall_time,
        particle_position_generation=particle_position_generation,
    )
    report["preflow_snapshot_loaded"] = False
    if output_path is not None:
        _write_fixed_solid_preflow_snapshot(
            path=output_path,
            report=report,
            markers=markers,
            fluid=fluid,
            solid=solid,
            config=config,
            identity=identity,
        )
    return report


def _run_fixed_solid_preflow(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    config: Any,
    *,
    progress_observer: Callable[[dict[str, object]], None] | None = None,
    run_started_s: float | None = None,
    profile_wall_time: bool = False,
    particle_position_generation: int | None = None,
) -> dict[str, object]:
    requested_steps = int(getattr(config, "preflow_steps", 0))
    tolerance = float(getattr(config, "preflow_convergence_tolerance", 0.0))
    convergence_mode = str(
        getattr(config, "preflow_convergence_mode", "single_step_legacy")
    )
    if convergence_mode not in {"single_step_legacy", "windowed_stationary"}:
        raise ValueError(f"unsupported preflow_convergence_mode: {convergence_mode!r}")
    history: list[dict[str, object]] = []
    previous_row: dict[str, object] | None = None
    converged = requested_steps == 0
    stop_reason = "not_requested" if requested_steps == 0 else "max_steps"
    sharp_boundary_cache: dict[str, object] = {}
    effective_run_started_s = (
        time.perf_counter()
        if run_started_s is None
        else float(run_started_s)
    )

    def emit_completed_step(row: Mapping[str, object]) -> None:
        _emit_run_progress(
            progress_observer,
            run_started_s=effective_run_started_s,
            phase="preflow_step",
            preflow_step=int(row["preflow_step"]),
            preflow_steps_requested=requested_steps,
            preflow_steps_completed=len(history),
            preflow_step_wall_time_s=float(
                row["preflow_step_wall_time_s"]
            ),
            preflow_flow_advance_wall_time_s=float(
                row["preflow_flow_advance_wall_time_s"]
            ),
            flow_momentum_advection_substeps_total=int(
                row.get("flow_momentum_advection_substeps_total", 0)
            ),
            flow_sst_transport_substeps_total=int(
                row.get("flow_sst_transport_substeps_total", 0)
            ),
        )

    for preflow_index in range(requested_steps):
        _emit_run_progress(
            progress_observer,
            run_started_s=effective_run_started_s,
            phase="preflow_step",
            preflow_step=preflow_index + 1,
            preflow_steps_requested=requested_steps,
            preflow_steps_completed=len(history),
        )
        observer_wall_time_s = 0.0
        preflow_stage_observer: Callable[[str], None] | None = None
        if progress_observer is not None and bool(
            getattr(config, "detailed_preflow_stage_progress", False)
        ):
            def emit_preflow_stage(preflow_stage: str) -> None:
                nonlocal observer_wall_time_s
                observer_started_s = time.perf_counter()
                try:
                    _emit_run_progress(
                        progress_observer,
                        run_started_s=effective_run_started_s,
                        phase="preflow_stage",
                        preflow_step=preflow_index + 1,
                        preflow_steps_requested=requested_steps,
                        preflow_steps_completed=len(history),
                        preflow_stage=preflow_stage,
                    )
                except Exception as exc:
                    raise PreflowStageObserverError(
                        f"preflow stage observer failed at {preflow_stage!r}"
                    ) from exc
                finally:
                    observer_wall_time_s += max(
                        0.0, time.perf_counter() - observer_started_s
                    )
            preflow_stage_observer = emit_preflow_stage
        preflow_step_started_s = time.perf_counter()
        if _flow_driver_requires_full_field_reinitialize(
            _effective_flow_driver_mode(config, flow_phase="preflow")
        ):
            _initialize_computed_flow(fluid, config)
        feedback_constraint_report = _apply_marker_feedback_to_fluid(
            markers,
            fluid,
            config,
            feedback_available=bool(getattr(config, "apply_marker_feedback_to_fluid", True)),
        )
        if profile_wall_time:
            _synchronize_hibm_sharp_boundary_stage_timing()
        flow_advance_started_s = time.perf_counter()
        try:
            flow_report = _flow_advance_current_step(
                fluid,
                config,
                markers=markers,
                sharp_boundary_cache=sharp_boundary_cache,
                flow_phase="preflow",
                step_index_local=preflow_index,
                step_index_global=preflow_index,
                preflow_history=history,
                reset_pressure=(
                    bool(getattr(config, "flow_reset_pressure_each_step", False))
                    or preflow_index == 0
                ),
                measure_wall_times=profile_wall_time,
                preflow_stage_observer=preflow_stage_observer,
            )
        except (FloatingPointError, ValueError) as exc:
            previous = history[-1] if history else {}
            previous_summary = {
                key: previous.get(key, "unavailable")
                for key in (
                    "preflow_step",
                    "local_velocity_peak_mps",
                    "computed_pressure_min_pa",
                    "computed_pressure_max_pa",
                    "flow_sst_turbulent_kinetic_energy_max_m2_s2",
                    "flow_sst_specific_dissipation_rate_max_s",
                    "flow_sst_eddy_viscosity_max_pa_s",
                )
            }
            raise type(exc)(
                "fixed-solid preflow advance failed at "
                f"step={preflow_index + 1}/{requested_steps}, "
                f"completed_steps={len(history)}, "
                f"previous_step={previous_summary}: {exc}"
            ) from exc
        if profile_wall_time:
            _synchronize_hibm_sharp_boundary_stage_timing()
        preflow_flow_advance_wall_time_s = (
            max(0.0, time.perf_counter() - flow_advance_started_s - observer_wall_time_s)
        )
        feedback_constraint_report[
            "no_slip_projected_residual_after_projection_mps"
        ] = (
            float(flow_report["hibm_no_slip_max_residual_mps"])
            if _use_hibm_sharp_marker_boundary(config)
            else _measure_projected_no_slip_residual(
                markers,
                fluid,
                config,
                feedback_consumed=bool(
                    feedback_constraint_report[
                        "fluid_marker_velocity_constraints_enabled"
                    ]
                ),
            )
        )
        stress_report = _sample_stress_to_marker_forces(markers, fluid, config)
        force_report = markers.aggregate_region_forces(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
        )
        markers.clear_mpm_external_forces(
            solid.external_force_n,
            particle_count=solid.particle_count,
        )
        scatter_report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=config.mpm_support_radius_m,
            particle_position_generation=particle_position_generation,
        )
        if profile_wall_time:
            _synchronize_hibm_sharp_boundary_stage_timing()
        preflow_step_wall_time_s = (
            max(0.0, time.perf_counter() - preflow_step_started_s - observer_wall_time_s)
        )
        row = {
            "preflow_step": preflow_index + 1,
            "preflow_step_wall_time_s": float(preflow_step_wall_time_s),
            "preflow_flow_advance_wall_time_s": float(
                preflow_flow_advance_wall_time_s
            ),
            "flow_sst_transport_wall_time_s": float(
                flow_report.get("flow_sst_transport_wall_time_s", 0.0)
            ),
            "flow_momentum_predictor_wall_time_s": float(
                flow_report.get("flow_momentum_predictor_wall_time_s", 0.0)
            ),
            "fluid_recomputed": True,
            "flow_driver_mode": flow_report["flow_driver_mode"],
            "flow_driver_diagnostic_only": flow_report["flow_driver_diagnostic_only"],
            "flow_driver_uses_full_velocity_reset": flow_report[
                "flow_driver_uses_full_velocity_reset"
            ],
            "flow_full_field_reinitialized": flow_report[
                "flow_full_field_reinitialized"
            ],
            "flow_inlet_boundary_reapplied": flow_report[
                "flow_inlet_boundary_reapplied"
            ],
            "flow_volume_source_applied": flow_report["flow_volume_source_applied"],
            "flow_inlet_source_strength": float(
                getattr(config, "flow_inlet_source_strength", 1.0)
            ),
            "flow_inlet_source_profile": str(
                getattr(config, "flow_inlet_source_profile", "constant")
            ),
            "flow_inlet_source_ramp_steps": int(
                getattr(config, "flow_inlet_source_ramp_steps", 0)
            ),
            "flow_inlet_source_schedule_scope": str(
                getattr(config, "flow_inlet_source_schedule_scope", "global")
            ),
            "flow_inlet_source_factor": flow_report["flow_inlet_source_factor"],
            "flow_inlet_source_normal_velocity_mps": flow_report[
                "flow_inlet_source_normal_velocity_mps"
            ],
            "flow_pressure_outlet_enabled": bool(
                getattr(config, "flow_pressure_outlet_enabled", True)
            ),
            "flow_outlet_balance_policy": str(
                getattr(config, "flow_outlet_balance_policy", "report_only")
            ),
            "flow_predictor_applied": flow_report["flow_predictor_applied"],
            "flow_predictor_note": flow_report["flow_predictor_note"],
            "flow_predictor_projection_segment_count": int(
                flow_report.get("flow_predictor_projection_segment_count", 1)
            ),
            "flow_predictor_projection_segment_dt_s": float(
                flow_report.get(
                    "flow_predictor_projection_segment_dt_s",
                    config.dt_s,
                )
            ),
            "flow_predictor_projection_segment_pre_projection_l2_max": float(
                flow_report.get(
                    "flow_predictor_projection_segment_pre_projection_l2_max",
                    flow_report.get("flow_main_projection_pre_projection_l2", 0.0),
                )
            ),
            "flow_predictor_projection_segment_pre_projection_max_abs_max": float(
                flow_report.get(
                    "flow_predictor_projection_segment_pre_projection_max_abs_max",
                    flow_report.get(
                        "flow_main_projection_pre_projection_max_abs",
                        0.0,
                    ),
                )
            ),
            "flow_predictor_projection_segment_trace": list(
                flow_report.get("flow_predictor_projection_segment_trace", [])
            ),
            "flow_predictor_kinematic_viscosity_m2_s": flow_report[
                "flow_predictor_kinematic_viscosity_m2_s"
            ],
            "flow_predictor_no_slip_domain_walls": flow_report[
                "flow_predictor_no_slip_domain_walls"
            ],
            "flow_obstacle_no_slip_layers": flow_report["flow_obstacle_no_slip_layers"],
            "flow_obstacle_no_slip_weight": flow_report["flow_obstacle_no_slip_weight"],
            "flow_solid_boundary_mode": flow_report["flow_solid_boundary_mode"],
            "flow_obstacle_normal_velocity_policy": flow_report[
                "flow_obstacle_normal_velocity_policy"
            ],
            "flow_pressure_outlet_backflow_policy": flow_report[
                "flow_pressure_outlet_backflow_policy"
            ],
            "hibm_sharp_marker_boundary_enabled": flow_report[
                "hibm_sharp_marker_boundary_enabled"
            ],
            "hibm_sharp_marker_boundary_search_reused": flow_report[
                "hibm_sharp_marker_boundary_search_reused"
            ],
            "hibm_sharp_marker_boundary_topology_reused": flow_report[
                "hibm_sharp_marker_boundary_topology_reused"
            ],
            "hibm_preassembly_overflow_singleton_cleanup_cell_count": flow_report[
                "hibm_preassembly_overflow_singleton_cleanup_cell_count"
            ],
            "hibm_preassembly_overflow_singleton_cleanup_component_count": (
                flow_report[
                    "hibm_preassembly_overflow_singleton_cleanup_component_count"
                ]
            ),
            "hibm_preassembly_tiny_unreached_cleanup_cell_count": flow_report[
                "hibm_preassembly_tiny_unreached_cleanup_cell_count"
            ],
            "hibm_preassembly_tiny_unreached_cleanup_component_count": flow_report[
                "hibm_preassembly_tiny_unreached_cleanup_component_count"
            ],
            "hibm_preassembly_tiny_unreached_cleanup_pass_count": flow_report[
                "hibm_preassembly_tiny_unreached_cleanup_pass_count"
            ],
            "hibm_preassembly_remaining_unreached_cell_count": flow_report[
                "hibm_preassembly_remaining_unreached_cell_count"
            ],
            "hibm_preassembly_cleanup_reused": flow_report[
                "hibm_preassembly_cleanup_reused"
            ],
            "hibm_preassembly_topology_mutated": flow_report[
                "hibm_preassembly_topology_mutated"
            ],
            "hibm_sharp_marker_boundary_near_node_count": flow_report[
                "hibm_sharp_marker_boundary_near_node_count"
            ],
            "hibm_sharp_marker_boundary_external_node_count": flow_report[
                "hibm_sharp_marker_boundary_external_node_count"
            ],
            "hibm_sharp_marker_boundary_internal_node_count": flow_report[
                "hibm_sharp_marker_boundary_internal_node_count"
            ],
            "hibm_sharp_marker_boundary_internal_obstacle_cell_count": flow_report[
                "hibm_sharp_marker_boundary_internal_obstacle_cell_count"
            ],
            "hibm_sharp_marker_boundary_no_slip_rows": flow_report[
                "hibm_sharp_marker_boundary_no_slip_rows"
            ],
            **_hibm_velocity_dirichlet_mapping_fields(flow_report),
            "hibm_sharp_marker_boundary_pressure_neumann_rows": flow_report[
                "hibm_sharp_marker_boundary_pressure_neumann_rows"
            ],
            "hibm_sharp_marker_boundary_pressure_gradient_updated": flow_report[
                "hibm_sharp_marker_boundary_pressure_gradient_updated"
            ],
            "hibm_pressure_neumann_skipped_velocity_dirichlet_count": flow_report[
                "hibm_pressure_neumann_skipped_velocity_dirichlet_count"
            ],
            "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
                flow_report[
                    "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count"
                ]
            ),
            "hibm_pressure_neumann_skipped_obstacle_owner_count": flow_report[
                "hibm_pressure_neumann_skipped_obstacle_owner_count"
            ],
            "hibm_pressure_neumann_relocated_obstacle_owner_count": flow_report[
                "hibm_pressure_neumann_relocated_obstacle_owner_count"
            ],
            "hibm_pressure_neumann_duplicate_owner_count": flow_report[
                "hibm_pressure_neumann_duplicate_owner_count"
            ],
            "hibm_pressure_neumann_invalid_reconstruction_count": flow_report[
                "hibm_pressure_neumann_invalid_reconstruction_count"
            ],
            "hibm_pressure_neumann_invalid_unreconstructable_count": flow_report[
                "hibm_pressure_neumann_invalid_unreconstructable_count"
            ],
            "hibm_pressure_neumann_invalid_bad_marker_count": flow_report[
                "hibm_pressure_neumann_invalid_bad_marker_count"
            ],
            "hibm_pressure_neumann_invalid_nonpositive_volume_count": flow_report[
                "hibm_pressure_neumann_invalid_nonpositive_volume_count"
            ],
            "flow_inlet_boundary_active_cell_count": flow_report[
                "flow_inlet_boundary_active_cell_count"
            ],
            "flow_inlet_boundary_obstacle_cell_count": flow_report[
                "flow_inlet_boundary_obstacle_cell_count"
            ],
            "apply_marker_feedback_to_fluid": bool(
                getattr(config, "apply_marker_feedback_to_fluid", True)
            ),
            "fluid_marker_velocity_constraints_enabled": (
                feedback_constraint_report["fluid_marker_velocity_constraints_enabled"]
            ),
            "fluid_marker_velocity_constraint_active_cell_count": (
                feedback_constraint_report[
                    "fluid_marker_velocity_constraint_active_cell_count"
                ]
            ),
            "fluid_marker_feedback_enforcement_mode": feedback_constraint_report[
                "fluid_marker_feedback_enforcement_mode"
            ],
            "legacy_constraint_active_cell_count": feedback_constraint_report[
                "legacy_constraint_active_cell_count"
            ],
            "fluid_feedback_constraint_marker_count": (
                feedback_constraint_report["fluid_feedback_constraint_marker_count"]
            ),
            "fluid_feedback_constraint_active_cell_count": (
                feedback_constraint_report["fluid_feedback_constraint_active_cell_count"]
            ),
            "fluid_feedback_constraint_cleared_cell_count": (
                feedback_constraint_report["fluid_feedback_constraint_cleared_cell_count"]
            ),
            "fluid_feedback_constraint_obstacle_cell_count": (
                feedback_constraint_report["fluid_feedback_constraint_obstacle_cell_count"]
            ),
            "fluid_feedback_constraint_non_obstacle_cell_count": (
                feedback_constraint_report[
                    "fluid_feedback_constraint_non_obstacle_cell_count"
                ]
            ),
            "fluid_feedback_constraint_projection_participating_cell_count": (
                feedback_constraint_report[
                    "fluid_feedback_constraint_projection_participating_cell_count"
                ]
            ),
            "no_slip_residual_before_mps": (
                feedback_constraint_report["no_slip_residual_before_mps"]
            ),
            "no_slip_residual_after_mps": (
                feedback_constraint_report["no_slip_residual_after_mps"]
            ),
            "no_slip_target_residual_after_assembly_mps": (
                feedback_constraint_report["no_slip_target_residual_after_assembly_mps"]
            ),
            "no_slip_projected_residual_after_projection_mps": (
                feedback_constraint_report[
                    "no_slip_projected_residual_after_projection_mps"
                ]
            ),
            "hibm_no_slip_valid_marker_count": flow_report[
                "hibm_no_slip_valid_marker_count"
            ],
            "hibm_no_slip_invalid_marker_count": flow_report[
                "hibm_no_slip_invalid_marker_count"
            ],
            "hibm_no_slip_max_residual_mps": flow_report[
                "hibm_no_slip_max_residual_mps"
            ],
            "hibm_no_slip_l2_residual_mps": flow_report[
                "hibm_no_slip_l2_residual_mps"
            ],
            "hibm_post_dirichlet_consistency_projection_count": flow_report[
                "hibm_post_dirichlet_consistency_projection_count"
            ],
            "hibm_post_dirichlet_consistency_projection_applied": flow_report[
                "hibm_post_dirichlet_consistency_projection_applied"
            ],
            "flow_phase": flow_report["flow_phase"],
            "flow_step_index_local": flow_report["flow_step_index_local"],
            "flow_step_index_global": flow_report["flow_step_index_global"],
            "flow_source_schedule_step_index": flow_report[
                "flow_source_schedule_step_index"
            ],
            "flow_source_schedule_scope": flow_report["flow_source_schedule_scope"],
            "flow_source_ramp_restarted_after_preflow": flow_report[
                "flow_source_ramp_restarted_after_preflow"
            ],
            "flow_pressure_reset_applied": flow_report["flow_pressure_reset_applied"],
            "solid_fixed": True,
            "solid_advanced": False,
            "local_velocity_peak_mps": flow_report["local_velocity_peak_mps"],
            "fluid_speed_p99_mps": flow_report["fluid_speed_p99_mps"],
            "fluid_speed_p999_mps": flow_report["fluid_speed_p999_mps"],
            "pressure_min_pa": flow_report["pressure_min_pa"],
            "pressure_max_pa": flow_report["pressure_max_pa"],
            "flow_projection_report": flow_report["projection_report"],
            **_flow_projection_report_fields(flow_report),
            **_flow_source_report_fields(flow_report),
            **_flow_transport_report_fields(flow_report),
            "stress_valid_marker_count": stress_report.valid_marker_count,
            "stress_invalid_marker_count": stress_report.invalid_marker_count,
            **_marker_projection_boundary_report_fields(
                markers,
                traction_tip_cap_pressure_enabled=(
                    _traction_tip_cap_pressure_enabled(config)
                ),
                canonical_velocity_dirichlet_report=flow_report.get(
                    "canonical_velocity_dirichlet_report"
                ),
            ),
            "two_sided_pressure_marker_count": (
                stress_report.two_sided_pressure_marker_count
            ),
            "marker_total_area_m2": _marker_total_area_m2(markers),
            "total_marker_force_n": force_report.total_marker_force_n,
            "mpm_external_force_n": scatter_report.total_mpm_external_force_n,
            "scatter_invalid_marker_count": scatter_report.invalid_marker_count,
            "scatter_active_marker_count": scatter_report.active_marker_count,
            "scatter_active_particle_count": scatter_report.active_pair_count,
            **_marker_force_report_fields(force_report),
            **_stress_sampling_report_fields(stress_report),
            **_marker_traction_report_fields(
                markers, include_face_diagnostics=False
            ),
            **_scatter_report_fields(scatter_report),
        }
        row["preflow_traction_readiness"] = _preflow_traction_readiness(
            [row],
            config,
        )
        if previous_row is not None:
            row["velocity_peak_relative_delta"] = _relative_delta(
                row["local_velocity_peak_mps"],
                previous_row["local_velocity_peak_mps"],
            )
            row["pressure_range_relative_delta"] = _relative_delta(
                _pressure_range(row),
                _pressure_range(previous_row),
            )
            if convergence_mode == "single_step_legacy" and tolerance > 0.0 and (
                float(row["velocity_peak_relative_delta"]) <= tolerance
                and float(row["pressure_range_relative_delta"]) <= tolerance
            ):
                converged = True
                stop_reason = "converged"
                history.append(row)
                emit_completed_step(row)
                break
        else:
            row["velocity_peak_relative_delta"] = ""
            row["pressure_range_relative_delta"] = ""
        history.append(row)
        emit_completed_step(row)
        if convergence_mode == "windowed_stationary":
            stationary_report = _preflow_windowed_stationary_report(history, config)
            row["preflow_stationary_gate_passed"] = bool(
                stationary_report["stationary"]
            )
            row["preflow_stationary_gate_reason"] = str(
                stationary_report["reason"]
            )
            row["preflow_stationary_consecutive_windows_passed"] = int(
                stationary_report["consecutive_windows_passed"]
            )
            row["preflow_stationary_window_metrics"] = dict(
                stationary_report["window_metrics"]
            )
            if bool(stationary_report["stationary"]):
                converged = True
                stop_reason = "windowed_stationary"
                break
        previous_row = row

    return {
        "preflow_steps_requested": requested_steps,
        "preflow_steps_completed": len(history),
        "preflow_convergence_tolerance": tolerance,
        "preflow_convergence_mode": convergence_mode,
        "preflow_converged": converged,
        "preflow_status": stop_reason,
        "preflow_stop_reason": stop_reason,
        "preflow_traction_readiness_mode": _preflow_traction_readiness_mode(
            config
        ),
        "preflow_traction_readiness": (
            _preflow_traction_readiness([history[-1]], config)
            if history
            else PREFLOW_TRACTION_NOT_EVALUATED
        ),
        "preflow_history": history,
        "final_stress_marker_diagnostics": (
            markers.stress_marker_diagnostics() if history else []
        ),
        "final_stress_face_diagnostics": (
            markers.stress_face_diagnostics(
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_REGION_ID,
                include_face_diagnostics=True,
            )
            if history
            else {}
        ),
        "final_flow_field_snapshot": (
            _synchronized_flow_boundary_snapshot(
                _flow_field_snapshot(fluid),
                stage="fixed_solid_preflow_terminal_projection",
            )
            if history and bool(getattr(config, "export_final_flow_snapshot", False))
            else {}
        ),
    }


def _pressure_range(row: Mapping[str, object]) -> float:
    return float(row["pressure_max_pa"]) - float(row["pressure_min_pa"])


def _preflow_windowed_stationary_report(
    history: list[Mapping[str, object]],
    config: Any,
) -> dict[str, object]:
    """Evaluate a conservative post-burn-in stationary-flow window gate."""

    min_steps = int(getattr(config, "preflow_stationary_min_steps", 20))
    window_steps = int(getattr(config, "preflow_stationary_window_steps", 10))
    consecutive_windows = int(
        getattr(config, "preflow_stationary_consecutive_windows", 3)
    )
    tolerance = float(getattr(config, "preflow_stationary_tolerance", 0.05))
    divergence_tolerance = float(
        getattr(config, "preflow_stationary_divergence_tolerance", 0.05)
    )
    no_slip_fraction = float(
        getattr(config, "preflow_stationary_no_slip_tolerance_fraction", 0.05)
    )
    if min_steps < 0 or window_steps <= 0 or consecutive_windows <= 0:
        raise ValueError("preflow stationary window sizes must be positive")
    if not (0.0 < tolerance < 1.0):
        raise ValueError("preflow_stationary_tolerance must be in (0, 1)")
    if not (0.0 < divergence_tolerance < 1.0):
        raise ValueError(
            "preflow_stationary_divergence_tolerance must be in (0, 1)"
        )
    if not (0.0 < no_slip_fraction < 1.0):
        raise ValueError(
            "preflow_stationary_no_slip_tolerance_fraction must be in (0, 1)"
        )

    required_steps = min_steps + window_steps + consecutive_windows - 1
    base_report: dict[str, object] = {
        "stationary": False,
        "reason": "insufficient_post_burn_in_windows",
        "required_steps": required_steps,
        "completed_steps": len(history),
        "first_evaluated_window_start_step": min_steps + 1,
        "consecutive_windows_passed": 0,
        "window_metrics": {},
        "consecutive_window_union_metrics": {},
        "traction_readiness": PREFLOW_TRACTION_NOT_EVALUATED,
        "marker_force_metric_evaluated": False,
        "marker_force_reference_area_m2": None,
        "sst_transport_stationarity_evaluated": False,
        "excluded_window_metrics": ["marker_force_relative_span"],
    }
    if len(history) < required_steps:
        return base_report

    inlet_speed = abs(float(getattr(config, "inlet_velocity_mps", 0.0)))
    if not math.isfinite(inlet_speed) or inlet_speed <= 0.0:
        raise ValueError("windowed preflow convergence requires positive inlet speed")
    density = float(getattr(config, "air_density_kgm3", 1.0))
    dynamic_pressure = 0.5 * density * inlet_speed * inlet_speed
    expected_marker_count = _preflow_expected_no_slip_marker_count(config)
    min_grid_spacing = min(_grid_spacing_m(config))
    convective_rate_s_inv = inlet_speed / min_grid_spacing

    def marker_force_reference_area_m2(
        rows: list[Mapping[str, object]],
    ) -> float:
        areas: list[float] = []
        for row in rows:
            try:
                area = float(row["marker_total_area_m2"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "preflow history marker_total_area_m2 must be present, "
                    "finite, and positive when marker force is evaluated"
                ) from exc
            if not math.isfinite(area) or area <= 0.0:
                raise ValueError(
                    "preflow history marker_total_area_m2 must be finite and "
                    "positive when marker force is evaluated"
                )
            areas.append(area)
        if not areas:
            raise ValueError(
                "preflow history marker_total_area_m2 requires at least one row"
            )
        reference_area = areas[0]
        if any(
            not math.isclose(
                area,
                reference_area,
                rel_tol=1.0e-12,
                abs_tol=1.0e-18,
            )
            for area in areas[1:]
        ):
            raise ValueError(
                "preflow history marker_total_area_m2 must remain consistent "
                "throughout each evaluated stationary window"
            )
        return reference_area

    def projection_l2(row: Mapping[str, object]) -> float:
        if "flow_projection_l2" in row:
            return float(row["flow_projection_l2"])
        projection_report = row.get("flow_projection_report", {})
        if isinstance(projection_report, Mapping):
            return float(projection_report.get("projection_l2", math.nan))
        return math.nan

    def relative_span(values: list[float], physical_floor: float = 0.0) -> float:
        if not values or not all(math.isfinite(value) for value in values):
            return math.inf
        scale = max(max(abs(value) for value in values), physical_floor, 1.0e-30)
        return (max(values) - min(values)) / scale

    sst_stationary_metric_specs = (
        (
            "flow_sst_turbulent_kinetic_energy_max_m2_s2",
            "flow_sst_turbulent_kinetic_energy_max_relative_span",
            False,
        ),
        (
            "flow_sst_turbulent_kinetic_energy_volume_mean_m2_s2",
            "flow_sst_turbulent_kinetic_energy_volume_mean_relative_span",
            False,
        ),
        (
            "flow_sst_turbulent_kinetic_energy_volume_rms_m2_s2",
            "flow_sst_turbulent_kinetic_energy_volume_rms_relative_span",
            False,
        ),
        (
            "flow_sst_specific_dissipation_rate_max_s",
            "flow_sst_specific_dissipation_rate_max_relative_span",
            True,
        ),
        (
            "flow_sst_specific_dissipation_rate_volume_mean_s",
            "flow_sst_specific_dissipation_rate_volume_mean_relative_span",
            True,
        ),
        (
            "flow_sst_specific_dissipation_rate_volume_rms_s",
            "flow_sst_specific_dissipation_rate_volume_rms_relative_span",
            True,
        ),
        (
            "flow_sst_eddy_viscosity_max_pa_s",
            "flow_sst_eddy_viscosity_max_relative_span",
            False,
        ),
        (
            "flow_sst_eddy_viscosity_volume_mean_pa_s",
            "flow_sst_eddy_viscosity_volume_mean_relative_span",
            False,
        ),
        (
            "flow_sst_eddy_viscosity_volume_rms_pa_s",
            "flow_sst_eddy_viscosity_volume_rms_relative_span",
            False,
        ),
    )

    def physical_sst_value(
        row: Mapping[str, object],
        field: str,
        *,
        strictly_positive: bool,
    ) -> float:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"preflow SST history field {field} must be present and finite"
            ) from exc
        lower_bound_ok = value > 0.0 if strictly_positive else value >= 0.0
        if not math.isfinite(value) or not lower_bound_ok:
            constraint = "positive" if strictly_positive else "nonnegative"
            raise ValueError(
                f"preflow SST history field {field} must be finite and {constraint}"
            )
        return value

    def vector_relative_span(
        values: list[np.ndarray],
        physical_floor: float = 0.0,
    ) -> float:
        if not values:
            return math.inf
        vectors = np.asarray(values, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            return math.inf
        if not bool(np.all(np.isfinite(vectors))):
            return math.inf
        magnitudes = np.linalg.norm(vectors, axis=1)
        scale = max(float(np.max(magnitudes)), physical_floor, 1.0e-30)
        component_span = np.max(vectors, axis=0) - np.min(vectors, axis=0)
        return float(np.linalg.norm(component_span)) / scale

    def window_metrics(
        rows: list[Mapping[str, object]],
        *,
        include_marker_force: bool,
        include_sst_transport: bool,
    ) -> dict[str, float]:
        velocity = [float(row["local_velocity_peak_mps"]) for row in rows]
        pressure = [_pressure_range(row) for row in rows]
        divergence = [projection_l2(row) for row in rows]
        metrics = {
            "velocity_peak_relative_span": relative_span(velocity, inlet_speed),
            "pressure_range_relative_span": relative_span(
                pressure, dynamic_pressure
            ),
            # projection_l2 has units of 1/s.  Near zero, scale its window
            # variation by the physical convective rate U/h instead of by
            # numerical residual jitter itself.  The absolute |L2|h/U guard
            # below remains the authority for excessive divergence.
            "projection_l2_relative_span": relative_span(
                divergence,
                convective_rate_s_inv,
            ),
        }
        if include_marker_force:
            force = [
                np.asarray(row["total_marker_force_n"], dtype=float)
                for row in rows
            ]
            metrics["marker_force_relative_span"] = vector_relative_span(
                force,
                force_floor,
            )
        if include_sst_transport:
            for field, metric_name, strictly_positive in sst_stationary_metric_specs:
                values = [
                    physical_sst_value(
                        row,
                        field,
                        strictly_positive=strictly_positive,
                    )
                    for row in rows
                ]
                metrics[metric_name] = relative_span(values)
        return metrics

    union_start = len(history) - (window_steps + consecutive_windows - 1)
    guarded_rows = history[union_start:]
    sst_transport_flags = [
        bool(row.get("flow_sst_transport_applied", False)) for row in guarded_rows
    ]
    sst_transport_stationarity_evaluated = any(sst_transport_flags)
    if sst_transport_stationarity_evaluated and not all(sst_transport_flags):
        raise ValueError(
            "flow_sst_transport_applied must remain true throughout each "
            "evaluated stationary window"
        )
    if sst_transport_stationarity_evaluated:
        for row in guarded_rows:
            for field, _, strictly_positive in sst_stationary_metric_specs:
                physical_sst_value(
                    row,
                    field,
                    strictly_positive=strictly_positive,
                )
    traction_mode = _preflow_traction_readiness_mode(config)
    traction_readiness = _preflow_traction_readiness(guarded_rows, config)
    marker_force_metric_evaluated = (
        traction_readiness == PREFLOW_TRACTION_EVALUATED
    )
    marker_force_reference_area = None
    force_floor = 0.0
    if marker_force_metric_evaluated:
        marker_force_reference_area = marker_force_reference_area_m2(guarded_rows)
        force_floor = dynamic_pressure * marker_force_reference_area
    traction_guard_ok = marker_force_metric_evaluated or (
        traction_mode == PREFLOW_TRACTION_READINESS_FLOW_ONLY
        and traction_readiness == PREFLOW_TRACTION_NOT_EVALUATED
    )
    base_report.update(
        {
            "traction_readiness": traction_readiness,
            "marker_force_metric_evaluated": marker_force_metric_evaluated,
            "marker_force_reference_area_m2": marker_force_reference_area,
            "sst_transport_stationarity_evaluated": (
                sst_transport_stationarity_evaluated
            ),
            "excluded_window_metrics": (
                []
                if marker_force_metric_evaluated
                else ["marker_force_relative_span"]
            ),
        }
    )
    no_slip_limit = _preflow_no_slip_limit_mps(config)
    for row in guarded_rows:
        row_projection_l2 = projection_l2(row)
        nondimensional_divergence = (
            abs(row_projection_l2) * min_grid_spacing / inlet_speed
        )
        volume_source_applied = bool(row.get("flow_volume_source_applied", False))
        inlet_ready = (
            math.isclose(
                float(row.get("flow_inlet_source_factor", math.nan)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            if volume_source_applied
            else bool(row.get("flow_inlet_boundary_reapplied", False))
        )
        valid_markers = int(row.get("hibm_no_slip_valid_marker_count", -1))
        invalid_markers = int(row.get("hibm_no_slip_invalid_marker_count", -1))
        velocity_dirichlet_health_failure = (
            _hibm_velocity_dirichlet_health_failure(row)
        )
        pressure_operator_health_failure = (
            _preflow_pressure_operator_health_failure(row)
        )
        q_health_ok = not _use_hibm_sharp_marker_boundary(config) or (
            bool(
                row.get(
                    "flow_projection_pre_projection_velocity_projector_prepared_all",
                    False,
                )
            )
            and bool(
                row.get(
                    "flow_projection_pre_projection_velocity_projector_converged_all",
                    False,
                )
            )
            and bool(
                row.get(
                    "flow_projection_pre_projection_velocity_projector_committed_all",
                    False,
                )
            )
        )
        guards_ok = (
            math.isfinite(row_projection_l2)
            and nondimensional_divergence <= divergence_tolerance
            and float(row.get("hibm_no_slip_max_residual_mps", math.inf))
            <= no_slip_limit
            and valid_markers == expected_marker_count
            and invalid_markers == 0
            and bool(row.get("flow_projection_cg_converged_all", False))
            and int(row.get("flow_projection_cg_breakdown_count", -1)) == 0
            and not bool(row.get("flow_projection_pressure_solve_failed", True))
            and not bool(
                row.get(
                    "flow_projection_pressure_projection_physical_failure",
                    True,
                )
            )
            and not bool(row.get("hibm_preassembly_topology_mutated", True))
            and pressure_operator_health_failure is None
            and velocity_dirichlet_health_failure is None
            and q_health_ok
            and inlet_ready
            and traction_guard_ok
        )
        if not guards_ok:
            return {
                **base_report,
                "reason": "physical_guard_failed",
                "guard_failure_step": int(row.get("preflow_step", -1)),
                "velocity_dirichlet_health_failure": (
                    velocity_dirichlet_health_failure or ""
                ),
                "pressure_operator_health_failure": (
                    pressure_operator_health_failure or ""
                ),
            }

    reports: list[dict[str, float]] = []
    for window_offset in range(consecutive_windows):
        end = len(history) - consecutive_windows + window_offset + 1
        start = end - window_steps
        reports.append(
            window_metrics(
                history[start:end],
                include_marker_force=marker_force_metric_evaluated,
                include_sst_transport=sst_transport_stationarity_evaluated,
            )
        )
    latest_metrics = reports[-1]
    union_metrics = window_metrics(
        guarded_rows,
        include_marker_force=marker_force_metric_evaluated,
        include_sst_transport=sst_transport_stationarity_evaluated,
    )
    passes = [
        all(metric <= tolerance for metric in report.values()) for report in reports
    ]
    consecutive_passed = 0
    for passed in reversed(passes):
        if not passed:
            break
        consecutive_passed += 1
    union_passed = all(metric <= tolerance for metric in union_metrics.values())
    stationary = (
        consecutive_passed == consecutive_windows and union_passed
    )
    if stationary:
        reason = "stationary"
    elif consecutive_passed == consecutive_windows and not union_passed:
        reason = "consecutive_window_union_span_exceeded"
    else:
        reason = "window_span_exceeded"
    return {
        **base_report,
        "stationary": stationary,
        "reason": reason,
        "consecutive_windows_passed": consecutive_passed,
        "window_metrics": latest_metrics,
        "consecutive_window_union_metrics": union_metrics,
        "window_reports": reports,
    }


def _relative_delta(current: object, previous: object) -> float:
    current_value = float(current)
    previous_value = float(previous)
    scale = max(abs(current_value), abs(previous_value), 1.0e-30)
    return abs(current_value - previous_value) / scale


_IMMUTABLE_FLOW_GEOMETRY_FIELD_NAMES = (
    "cell_face_x_m",
    "cell_face_y_m",
    "cell_face_z_m",
    "cell_center_x_m",
    "cell_center_y_m",
    "cell_center_z_m",
    "cell_width_x_m",
    "cell_width_y_m",
    "cell_width_z_m",
)
_PARITY_FLOW_GEOMETRY_FIELD_NAMES = (
    "cell_center_y_m",
    "cell_center_z_m",
)


def _flow_geometry_snapshot_cache_required(
    *,
    step_count: int,
    has_step_observer: bool,
    export_final_flow_snapshot: bool,
) -> bool:
    return step_count > 0 and (
        has_step_observer or export_final_flow_snapshot
    )


def _immutable_flow_geometry_snapshot(
    fluid: CartesianFluidSolver,
    *,
    include_full_geometry: bool,
) -> dict[str, np.ndarray]:
    """Download immutable grid geometry once for host-side snapshot export."""

    field_names = (
        _IMMUTABLE_FLOW_GEOMETRY_FIELD_NAMES
        if include_full_geometry
        else _PARITY_FLOW_GEOMETRY_FIELD_NAMES
    )
    snapshot: dict[str, np.ndarray] = {}
    for field_name in field_names:
        value = np.asarray(getattr(fluid, field_name).to_numpy()).copy()
        value.setflags(write=False)
        snapshot[field_name] = value
    return snapshot


def _flow_field_snapshot(
    fluid: CartesianFluidSolver,
    *,
    immutable_geometry: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    snapshot = {
        "pressure": _fluid_feedback_pressure_numpy(fluid),
        "velocity": fluid.velocity.to_numpy(),
        "obstacle": fluid.obstacle.to_numpy(),
        "hibm_base_obstacle": fluid.hibm_base_obstacle.to_numpy(),
        "hibm_dynamic_solid_volume_obstacle": (
            fluid.hibm_dynamic_solid_volume_obstacle.to_numpy()
        ),
        "hibm_dynamic_solid_volume_external_carve": (
            fluid.hibm_dynamic_solid_volume_external_carve.to_numpy()
        ),
        "velocity_dirichlet_boundary_active": (
            fluid.velocity_dirichlet_boundary_active.to_numpy()
        ),
        "velocity_dirichlet_boundary_projection_weight": (
            fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        ),
        "velocity_dirichlet_boundary_enforcement_weight": (
            fluid.velocity_dirichlet_boundary_enforcement_weight.to_numpy()
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        ),
        "velocity_dirichlet_boundary_external_exact_component_mask": (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        ),
        "velocity_dirichlet_boundary_owned_row": (
            fluid.velocity_dirichlet_boundary_owned_row.to_numpy()
        ),
        "velocity_dirichlet_boundary_marker_region_id": (
            fluid.velocity_dirichlet_boundary_marker_region_id.to_numpy()
        ),
    }
    snapshot.update(
        {
            field_name: (
                immutable_geometry[field_name]
                if immutable_geometry is not None
                else getattr(fluid, field_name).to_numpy()
            )
            for field_name in _IMMUTABLE_FLOW_GEOMETRY_FIELD_NAMES
        }
    )
    sampling_obstacle = getattr(fluid, "sampling_obstacle", None)
    if sampling_obstacle is not None:
        snapshot["sampling_obstacle"] = sampling_obstacle.to_numpy()
    return snapshot


def _flow_parity_snapshot(
    fluid: CartesianFluidSolver,
    *,
    immutable_geometry: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return only the arrays needed for per-step Fluent-parity frames."""

    return {
        "pressure": _fluid_feedback_pressure_numpy(fluid),
        "velocity": fluid.velocity.to_numpy(),
        "obstacle": fluid.obstacle.to_numpy(),
        "velocity_dirichlet_boundary_active": (
            fluid.velocity_dirichlet_boundary_active.to_numpy()
        ),
        "velocity_dirichlet_boundary_projection_weight": (
            fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        ),
        "velocity_dirichlet_boundary_enforcement_weight": (
            fluid.velocity_dirichlet_boundary_enforcement_weight.to_numpy()
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        ),
        "velocity_dirichlet_boundary_owned_row": (
            fluid.velocity_dirichlet_boundary_owned_row.to_numpy()
        ),
        "velocity_dirichlet_boundary_marker_region_id": (
            fluid.velocity_dirichlet_boundary_marker_region_id.to_numpy()
        ),
        "cell_center_y_m": (
            immutable_geometry["cell_center_y_m"]
            if immutable_geometry is not None
            else fluid.cell_center_y_m.to_numpy()
        ),
        "cell_center_z_m": (
            immutable_geometry["cell_center_z_m"]
            if immutable_geometry is not None
            else fluid.cell_center_z_m.to_numpy()
        ),
    }


def _synchronized_flow_boundary_snapshot(
    snapshot: Mapping[str, np.ndarray],
    *,
    stage: str,
) -> dict[str, np.ndarray]:
    """Tag a flow field and its boundary ledger as one synchronized stage."""

    normalized_stage = str(stage).strip()
    if not normalized_stage:
        raise ValueError("synchronized flow snapshot stage must be non-empty")
    return {
        **dict(snapshot),
        "flow_solution_stage": np.asarray(normalized_stage),
        "boundary_topology_stage": np.asarray(normalized_stage),
        "flow_boundary_state_synchronized": np.asarray(True),
    }


def _direct_step_observer_snapshot(
    flow_snapshot: Mapping[str, np.ndarray],
    solid: NeoHookeanMpmState,
    markers: HibmMpmSurfaceMarkers,
    *,
    solid_positions_m: np.ndarray,
    solid_rest_positions_m: np.ndarray,
    fixed_mask: np.ndarray,
    tip_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Combine the projected direct-flow stage with post-solid geometry."""

    return _stage_aware_step_observer_snapshot(
        flow_snapshot,
        solid,
        markers,
        solid_positions_m=solid_positions_m,
        solid_rest_positions_m=solid_rest_positions_m,
        fixed_mask=fixed_mask,
        tip_mask=tip_mask,
        expected_flow_stage="pre_solid_projection",
        structure_geometry_stage="post_solid_observer",
        error_message=(
            "direct step observer requires a pre-solid synchronized flow snapshot"
        ),
    )


def _stage_aware_step_observer_snapshot(
    flow_snapshot: Mapping[str, np.ndarray],
    solid: NeoHookeanMpmState,
    markers: HibmMpmSurfaceMarkers,
    *,
    solid_positions_m: np.ndarray,
    solid_rest_positions_m: np.ndarray,
    fixed_mask: np.ndarray,
    tip_mask: np.ndarray,
    expected_flow_stage: str,
    structure_geometry_stage: str,
    error_message: str,
) -> dict[str, np.ndarray]:
    """Validate one synchronized flow stage and attach structure geometry."""

    stages = tuple(
        str(np.asarray(flow_snapshot.get(key, "")).item())
        for key in ("flow_solution_stage", "boundary_topology_stage")
    )
    synchronized = np.asarray(
        flow_snapshot.get("flow_boundary_state_synchronized", False)
    )
    if (
        stages != (expected_flow_stage, expected_flow_stage)
        or synchronized.shape != ()
        or not isinstance(synchronized.item(), (bool, np.bool_))
        or not bool(synchronized.item())
    ):
        raise RuntimeError(error_message)
    snapshot = dict(flow_snapshot)
    solid_count = int(solid.particle_count)
    marker_count = int(markers.marker_count)
    snapshot.update(
        {
            "structure_geometry_stage": np.asarray(
                structure_geometry_stage
            ),
            "solid_position_m": np.asarray(solid_positions_m)[:solid_count].copy(),
            "solid_velocity_mps": solid.v.to_numpy()[:solid_count],
            "solid_rest_position_m": np.asarray(solid_rest_positions_m)[
                :solid_count
            ].copy(),
            "solid_fixed_mask": np.asarray(fixed_mask, dtype=bool)[
                :solid_count
            ].copy(),
            "solid_tip_mask": np.asarray(tip_mask, dtype=bool)[:solid_count].copy(),
            "marker_position_m": markers.x_gamma_m.to_numpy()[:marker_count],
            "marker_velocity_mps": markers.v_gamma_mps.to_numpy()[:marker_count],
            "marker_normal": markers.n_gamma.to_numpy()[:marker_count],
            "marker_area_m2": markers.A_gamma_m2.to_numpy()[:marker_count],
            "marker_region_id": markers.region_id.to_numpy()[:marker_count],
        }
    )
    return snapshot


def _fluid_feedback_pressure_field(fluid: CartesianFluidSolver):
    return getattr(fluid, "fsi_pressure", fluid.pressure)


def _fluid_feedback_pressure_numpy(fluid: CartesianFluidSolver) -> np.ndarray:
    return _fluid_feedback_pressure_field(fluid).to_numpy()


_ACTIVE_KALMAN_CONFIG_FIELDS = {
    INTERFACE_MARKER_VELOCITY_OWNER: "kalman_interface_config",
    FLUID_FSI_PRESSURE_FEEDBACK_OWNER: "kalman_fluid_config",
    SOLID_PARTICLE_VELOCITY_OWNER: "kalman_solid_config",
}


def _modified_physics_kalman_mode(config: Any) -> str:
    mode = str(getattr(config, "kalman_writeback_mode", "off"))
    if mode not in ACTIVE_KALMAN_MODE_OWNERS:
        raise ValueError(f"unsupported kalman_writeback_mode: {mode!r}")
    return mode


def _modified_physics_kalman_configs(config: Any) -> dict[str, object]:
    mode = _modified_physics_kalman_mode(config)
    enabled = ACTIVE_KALMAN_MODE_OWNERS[mode]
    all_configs = {
        owner: getattr(config, field_name, None)
        for owner, field_name in _ACTIVE_KALMAN_CONFIG_FIELDS.items()
    }
    unexpected = [
        owner
        for owner, owner_config in all_configs.items()
        if owner not in enabled and owner_config is not None
    ]
    if unexpected:
        raise ValueError(
            f"Kalman mode {mode!r} has unused configs for {unexpected!r}"
        )
    configs = {owner: all_configs[owner] for owner in enabled}
    missing = [owner for owner in enabled if configs.get(owner) is None]
    if missing:
        raise ValueError(
            f"Kalman mode {mode!r} is missing configs for {missing!r}"
        )
    return configs


def _kalman_fluid_observation(fluid: CartesianFluidSolver) -> np.ndarray:
    if not hasattr(fluid, "fsi_pressure"):
        raise RuntimeError(
            "modified-physics fluid Kalman requires fluid.fsi_pressure"
        )
    values = np.asarray(fluid.fsi_pressure.to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("raw fluid.fsi_pressure observation must be finite")
    return values


def _kalman_solid_observation(solid: NeoHookeanMpmState) -> np.ndarray:
    particle_count = int(solid.particle_count)
    values = np.asarray(solid.v.to_numpy()[:particle_count], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("raw solid velocity observation must be finite")
    return values


def _kalman_interface_observation(
    markers: HibmMpmSurfaceMarkers,
) -> np.ndarray:
    marker_count = int(markers.marker_count)
    values = np.asarray(
        markers.v_gamma_mps.to_numpy()[:marker_count],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("raw marker velocity observation must be finite")
    return values


def _finite_kalman_values(
    values: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
    dtype: object = np.float32,
) -> np.ndarray:
    candidate = np.asarray(values, dtype=np.float64)
    if tuple(candidate.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{name} shape changed: expected={expected_shape!r}, "
            f"actual={tuple(candidate.shape)!r}"
        )
    if not np.all(np.isfinite(candidate)):
        raise RuntimeError(f"{name} must be finite before f32 writeback")
    converted = np.ascontiguousarray(candidate, dtype=dtype)
    if not np.all(np.isfinite(converted)):
        raise RuntimeError(f"{name} overflowed during writeback conversion")
    return converted


def _apply_kalman_fluid_writeback(
    fluid: CartesianFluidSolver,
    filtered_pressure_pa: object,
) -> np.ndarray:
    if not hasattr(fluid, "fsi_pressure"):
        raise RuntimeError(
            "modified-physics fluid Kalman requires fluid.fsi_pressure"
    )
    field = fluid.fsi_pressure
    field_shape = getattr(field, "shape", None)
    expected_shape = (
        tuple(int(value) for value in field_shape)
        if field_shape is not None
        else tuple(np.asarray(filtered_pressure_pa).shape)
    )
    field_dtype_name = str(getattr(field, "dtype", "")).lower()
    target_dtype = np.float64 if "f64" in field_dtype_name else np.float32
    converted = _finite_kalman_values(
        filtered_pressure_pa,
        expected_shape=expected_shape,
        name="filtered fluid.fsi_pressure",
        dtype=target_dtype,
    )
    field.from_numpy(converted)
    return converted.astype(np.float64, copy=True)


def _apply_kalman_solid_writeback(
    solid: NeoHookeanMpmState,
    filtered_velocity_mps: object,
    *,
    fixed_mask: np.ndarray,
    enforce_plane_strain_x: bool,
) -> np.ndarray:
    particle_count = int(solid.particle_count)
    full_velocity = np.asarray(solid.v.to_numpy()).copy()
    converted = _finite_kalman_values(
        filtered_velocity_mps,
        expected_shape=(particle_count, 3),
        name="filtered solid velocity",
    )
    fixed = np.asarray(fixed_mask, dtype=bool)[:particle_count]
    if tuple(fixed.shape) != (particle_count,):
        raise RuntimeError("solid fixed mask shape changed during Kalman writeback")
    converted[fixed] = 0.0
    if bool(enforce_plane_strain_x):
        converted[:, 0] = 0.0
    full_velocity[:particle_count] = converted
    solid.v.from_numpy(np.ascontiguousarray(full_velocity, dtype=np.float32))
    return converted.astype(np.float64, copy=True)


def _refresh_kalman_interface_derived_vertices(
    markers: HibmMpmSurfaceMarkers,
) -> None:
    if getattr(markers, "_open_ribbon_tip_cap_binding", None) is not None:
        markers.refresh_open_ribbon_tip_cap_projection_vertices()


def _apply_kalman_interface_writeback(
    markers: HibmMpmSurfaceMarkers,
    filtered_velocity_mps: object,
) -> np.ndarray:
    marker_count = int(markers.marker_count)
    full_velocity = np.asarray(markers.v_gamma_mps.to_numpy()).copy()
    converted = _finite_kalman_values(
        filtered_velocity_mps,
        expected_shape=(marker_count, 3),
        name="filtered marker velocity",
    )
    full_velocity[:marker_count] = converted
    markers.v_gamma_mps.from_numpy(
        np.ascontiguousarray(full_velocity, dtype=np.float32)
    )
    _refresh_kalman_interface_derived_vertices(markers)
    return converted.astype(np.float64, copy=True)


def _initialize_modified_physics_kalman_controller(
    config: Any,
    *,
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    markers: HibmMpmSurfaceMarkers,
) -> ActiveKalmanWritebackController | None:
    mode = _modified_physics_kalman_mode(config)
    if mode == "off":
        return None
    configs = _modified_physics_kalman_configs(config)
    observations: dict[str, np.ndarray] = {}
    for owner in ACTIVE_KALMAN_MODE_OWNERS[mode]:
        if owner == FLUID_FSI_PRESSURE_FEEDBACK_OWNER:
            observations[owner] = _kalman_fluid_observation(fluid)
        elif owner == SOLID_PARTICLE_VELOCITY_OWNER:
            observations[owner] = _kalman_solid_observation(solid)
        elif owner == INTERFACE_MARKER_VELOCITY_OWNER:
            observations[owner] = _kalman_interface_observation(markers)
        else:  # pragma: no cover - static owner table is exhaustively tested
            raise RuntimeError(f"unsupported active Kalman owner: {owner!r}")
    return ActiveKalmanWritebackController(
        mode=mode,
        configs=configs,
        initial_observations=observations,
    )


def _empty_modified_physics_kalman_step_report(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "modified_physics": False,
        "owners": {},
        "filter_wall_time_s": 0.0,
        "state_transfer_wall_time_s": 0.0,
        "total_overhead_s": 0.0,
    }


def _kalman_controller_filter_wall_time_s(
    controller: ActiveKalmanWritebackController | None,
) -> float:
    if controller is None:
        return 0.0
    summary = controller.summary()
    return float(
        math.fsum(
            float(owner_report.get("filter_wall_time_s", 0.0))
            for owner_report in summary.get("owners", {}).values()
        )
    )


def _restore_modified_physics_kalman_targets(
    raw_targets: Mapping[str, np.ndarray],
    *,
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    markers: HibmMpmSurfaceMarkers,
) -> None:
    if FLUID_FSI_PRESSURE_FEEDBACK_OWNER in raw_targets:
        fluid.fsi_pressure.from_numpy(raw_targets[FLUID_FSI_PRESSURE_FEEDBACK_OWNER])
    if SOLID_PARTICLE_VELOCITY_OWNER in raw_targets:
        solid.v.from_numpy(raw_targets[SOLID_PARTICLE_VELOCITY_OWNER])
    if INTERFACE_MARKER_VELOCITY_OWNER in raw_targets:
        markers.v_gamma_mps.from_numpy(
            raw_targets[INTERFACE_MARKER_VELOCITY_OWNER]
        )
        _refresh_kalman_interface_derived_vertices(markers)


def _discard_modified_physics_kalman_step(
    controller: ActiveKalmanWritebackController | None,
    raw_targets: Mapping[str, np.ndarray],
    *,
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    markers: HibmMpmSurfaceMarkers,
) -> None:
    if controller is None:
        return
    try:
        _restore_modified_physics_kalman_targets(
            raw_targets,
            fluid=fluid,
            solid=solid,
            markers=markers,
        )
    finally:
        if controller.has_active_step:
            controller.discard_step()


def _apply_marker_feedback_to_fluid(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    feedback_available: bool,
) -> dict[str, object]:
    sharp_reconstructed_rows = _use_hibm_sharp_marker_boundary(config)
    authority = str(fluid.velocity_dirichlet_boundary_authority)
    if sharp_reconstructed_rows:
        if authority != "canonical":
            raise RuntimeError(
                "component-face HIBM marker feedback requires canonical "
                f"velocity rows, got {authority!r}"
            )
        report = _empty_feedback_constraint_report()
        report.update(
            {
                "fluid_projection_consumed_feedback": bool(
                    feedback_available and int(markers.marker_count) > 0
                ),
                "fluid_feedback_constraint_marker_count": (
                    int(markers.marker_count) if feedback_available else 0
                ),
                "fluid_marker_feedback_collocated_writer_used": False,
                "fluid_marker_feedback_enforcement_mode": (
                    "hibm_sharp_reconstructed_rows"
                ),
                "legacy_constraint_active_cell_count": 0,
            }
        )
        return report
    if authority != "legacy":
        raise RuntimeError(
            "collocated marker feedback requires legacy velocity rows, "
            f"got {authority!r}"
        )
    report = fluid.apply_marker_feedback_constraints(
        markers.x_gamma_m,
        markers.v_gamma_mps,
        markers.region_id,
        int(markers.marker_count),
        feedback_available=bool(feedback_available),
        preserve_velocity_constraints=bool(
            getattr(config, "preserve_marker_velocity_constraints", True)
        ),
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
    )
    report["fluid_marker_feedback_collocated_writer_used"] = True
    report["fluid_marker_feedback_enforcement_mode"] = (
        "legacy_collocated_marker_cells"
    )
    report["legacy_constraint_active_cell_count"] = int(
        report.get("fluid_feedback_constraint_active_cell_count", 0)
    )
    return report


def _empty_feedback_constraint_report(
    cleared_cell_count: int = 0,
) -> dict[str, object]:
    return {
        "fluid_projection_consumed_feedback": False,
        "fluid_feedback_constraint_marker_count": 0,
        "fluid_feedback_constraint_active_cell_count": 0,
        "fluid_feedback_constraint_cleared_cell_count": cleared_cell_count,
        "fluid_feedback_constraint_obstacle_cell_count": 0,
        "fluid_feedback_constraint_non_obstacle_cell_count": 0,
        "fluid_feedback_constraint_projection_participating_cell_count": 0,
        "fluid_marker_velocity_constraints_enabled": False,
        "fluid_marker_velocity_constraint_active_cell_count": 0,
        "no_slip_residual_before_mps": "",
        "no_slip_residual_after_mps": "",
        "no_slip_target_residual_after_assembly_mps": "",
        "no_slip_projected_residual_after_projection_mps": 0.0,
    }


def _measure_projected_no_slip_residual(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    feedback_consumed: bool,
) -> float:
    if not feedback_consumed:
        return 0.0

    marker_count = int(markers.marker_count)
    if marker_count <= 0:
        return 0.0

    measure_device = getattr(fluid, "marker_feedback_projected_residual_mps", None)
    if measure_device is not None:
        return float(
            measure_device(
                markers.x_gamma_m,
                markers.v_gamma_mps,
                marker_count,
            )
        )

    marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
    marker_velocities = markers.v_gamma_mps.to_numpy()[:marker_count]
    marker_cells = _marker_grid_cells(marker_positions, config)
    velocity = fluid.velocity.to_numpy()

    residuals = []
    for cell, marker_velocity in zip(marker_cells, marker_velocities):
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        residuals.append(float(np.linalg.norm(velocity[i, j, k] - marker_velocity)))
    return max(residuals, default=0.0)


def _marker_grid_cells(
    marker_positions: np.ndarray,
    config: Any,
) -> np.ndarray:
    bounds_min, bounds_max = _domain_bounds(config)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    grid_nodes = np.asarray(config.grid_nodes, dtype=np.int32)
    cell_width = (upper - lower) / grid_nodes.astype(np.float64)
    marker_cells = np.floor((marker_positions - lower) / cell_width).astype(np.int32)
    return np.clip(marker_cells, 0, grid_nodes - 1)


def _flow_state_report(
    fluid: CartesianFluidSolver,
    projection_report: Any,
    *,
    include_percentiles: bool = False,
) -> dict[str, object]:
    device_report = getattr(fluid, "flow_state_report", None)
    if device_report is not None:
        state = dict(
            device_report(
                pressure_field=_fluid_feedback_pressure_field(fluid),
                include_percentiles=bool(include_percentiles),
            )
        )
        return {
            "mode": FLOW_SOLUTION_MODE,
            "projection_report": projection_report,
            "obstacle_cell_count": int(state["obstacle_cell_count"]),
            "fluid_cell_count": int(state["fluid_cell_count"]),
            "local_velocity_peak_mps": float(state["local_velocity_peak_mps"]),
            "fluid_speed_p99_mps": state["fluid_speed_p99_mps"],
            "fluid_speed_p999_mps": state["fluid_speed_p999_mps"],
            "pressure_min_pa": float(state["pressure_min_pa"]),
            "pressure_max_pa": float(state["pressure_max_pa"]),
            "pressure_sign_convention": "fluid.fsi_pressure feedback field is sampled for reports and traction",
            **_flow_source_report_fields(projection_report),
        }

    obstacle = fluid.obstacle.to_numpy()
    velocity = fluid.velocity.to_numpy()
    pressure = _fluid_feedback_pressure_numpy(fluid)
    non_obstacle = obstacle == 0
    speed = np.linalg.norm(velocity, axis=3)
    active_speed = speed[non_obstacle]
    if bool(include_percentiles) and active_speed.size:
        speed_p99 = float(np.percentile(active_speed, 99.0))
        speed_p999 = float(np.percentile(active_speed, 99.9))
    else:
        speed_p99 = "" if active_speed.size else 0.0
        speed_p999 = "" if active_speed.size else 0.0
    active_pressure = pressure[non_obstacle]
    if active_pressure.size:
        pressure_min_pa = float(active_pressure.min())
        pressure_max_pa = float(active_pressure.max())
    else:
        pressure_min_pa = 0.0
        pressure_max_pa = 0.0
    return {
        "mode": FLOW_SOLUTION_MODE,
        "projection_report": projection_report,
        "obstacle_cell_count": int(obstacle.sum()),
        "fluid_cell_count": int(non_obstacle.sum()),
        "local_velocity_peak_mps": float(active_speed.max(initial=0.0)),
        "fluid_speed_p99_mps": speed_p99,
        "fluid_speed_p999_mps": speed_p999,
        "pressure_min_pa": pressure_min_pa,
        "pressure_max_pa": pressure_max_pa,
        "pressure_sign_convention": "fluid.fsi_pressure feedback field is sampled for reports and traction",
        **_flow_source_report_fields(projection_report),
    }


def _flow_source_report_fields(report: Any) -> dict[str, object]:
    if not isinstance(report, Mapping):
        return {key: "" for key in FLOW_SOURCE_REPORT_KEYS}
    fields = {key: report.get(key, "") for key in FLOW_SOURCE_REPORT_KEYS}
    if fields["pressure_outlet_flux_ratio"] == "":
        fields["pressure_outlet_flux_ratio"] = report.get(
            "zmin_pressure_outlet_to_abs_source_ratio",
            report.get("zmin_pressure_outlet_to_positive_source_ratio", ""),
        )
    if fields["velocity_outlet_flux_ratio"] == "":
        fields["velocity_outlet_flux_ratio"] = report.get(
            "zmin_velocity_outlet_to_abs_source_ratio",
            report.get("zmin_velocity_outlet_to_positive_source_ratio", ""),
        )
    return fields


def _flow_transport_report_fields(report: Any) -> dict[str, object]:
    """Persist core transport identity and CFL evidence in every artifact row."""

    if not isinstance(report, Mapping):
        return {}
    return {
        key: value
        for key, value in report.items()
        if key == "flow_turbulence_model"
        or key
        in {
            "requested_macro_dt_s",
            "fluid_accepted_time_s",
            "fluid_rejected_trial_count",
            "fluid_remaining_unadvanced_time_s",
        }
        or key.startswith("flow_sst_")
        or key.startswith("flow_momentum_advection_")
    }


def _flow_projection_report_fields(report: Any) -> dict[str, object]:
    if not isinstance(report, Mapping):
        return {f"flow_projection_{key}": "" for key in FLOW_PROJECTION_REPORT_KEYS}
    projection_report = report.get("projection_report", report)
    if not isinstance(projection_report, Mapping):
        projection_report = {}
    fields = {
        f"flow_projection_{key}": projection_report.get(key, "")
        for key in FLOW_PROJECTION_REPORT_KEYS
    }
    if fields["flow_projection_fsi_pressure_snapshot_updated"] == "":
        fields["flow_projection_fsi_pressure_snapshot_updated"] = report.get(
            "fsi_pressure_snapshot_updated",
            "",
        )
    return fields


def _marker_force_report_fields(report: Any) -> dict[str, object]:
    primary_force = tuple(report.primary_marker_force_n)
    secondary_force = tuple(report.secondary_marker_force_n)
    tip_cap_force = tuple(getattr(report, "tip_cap_marker_force_n", (0.0, 0.0, 0.0)))
    total_force = tuple(report.total_marker_force_n)
    fluid_reaction = tuple(report.fluid_reaction_force_n)
    primary_plus_secondary_z = (
        float(primary_force[2]) + float(secondary_force[2]) + float(tip_cap_force[2])
    )
    total_z = float(total_force[2])
    return {
        "primary_face_force_n": primary_force,
        "secondary_face_force_n": secondary_force,
        "primary_face_force_z_N": float(primary_force[2]),
        "secondary_face_force_z_N": float(secondary_force[2]),
        "tip_cap_force_n": tip_cap_force,
        "tip_cap_force_z_N": float(tip_cap_force[2]),
        "primary_plus_secondary_force_z_N": primary_plus_secondary_z,
        "force_decomposition_residual_N": abs(primary_plus_secondary_z - total_z),
        "marker_force_z_N": float(total_force[2]),
        "fluid_reaction_force_n": fluid_reaction,
        "fluid_reaction_force_z_N": float(fluid_reaction[2]),
        "marker_action_reaction_residual_n": float(
            report.action_reaction_residual_n
        ),
        "marker_action_reaction_residual_N": float(
            report.action_reaction_residual_n
        ),
        "primary_face_marker_count": int(report.primary_marker_count),
        "secondary_face_marker_count": int(report.secondary_marker_count),
        "total_marker_count": int(report.total_marker_count),
        "tip_cap_marker_count": int(getattr(report, "tip_cap_marker_count", 0)),
        "primary_face_valid_marker_count": int(
            report.primary_stress_valid_marker_count
        ),
        "secondary_face_valid_marker_count": int(
            report.secondary_stress_valid_marker_count
        ),
        "primary_face_invalid_marker_count": int(
            report.primary_stress_invalid_marker_count
        ),
        "secondary_face_invalid_marker_count": int(
            report.secondary_stress_invalid_marker_count
        ),
        "tip_cap_valid_marker_count": int(
            getattr(report, "tip_cap_stress_valid_marker_count", 0)
        ),
        "tip_cap_invalid_marker_count": int(
            getattr(report, "tip_cap_stress_invalid_marker_count", 0)
        ),
        "primary_face_force_norm_sum_N": float(
            report.primary_marker_force_norm_sum_n
        ),
        "secondary_face_force_norm_sum_N": float(
            report.secondary_marker_force_norm_sum_n
        ),
        "total_marker_force_norm_sum_N": float(
            report.total_marker_force_norm_sum_n
        ),
        "primary_face_force_norm_max_N": float(
            report.primary_marker_force_norm_max_n
        ),
        "secondary_face_force_norm_max_N": float(
            report.secondary_marker_force_norm_max_n
        ),
        "total_marker_force_norm_max_N": float(
            report.total_marker_force_norm_max_n
        ),
    }


def _stress_sampling_report_fields(report: Any) -> dict[str, object]:
    return {
        "max_abs_traction_pa": float(report.max_abs_traction_pa),
        "two_sided_pressure_marker_count": int(
            report.two_sided_pressure_marker_count
        ),
        "one_sided_pressure_marker_count": int(
            report.one_sided_pressure_marker_count
        ),
        "tip_cap_marker_count": int(getattr(report, "tip_cap_marker_count", 0)),
        "tip_cap_valid_marker_count": int(
            getattr(report, "tip_cap_valid_marker_count", 0)
        ),
        "tip_cap_invalid_marker_count": int(
            getattr(report, "tip_cap_invalid_marker_count", 0)
        ),
        "two_sided_extended_marker_count": int(
            getattr(report, "two_sided_extended_marker_count", 0)
        ),
        "one_sided_extended_marker_count": int(
            getattr(report, "one_sided_extended_marker_count", 0)
        ),
    }


def _marker_traction_report_fields(
    markers: HibmMpmSurfaceMarkers,
    *,
    include_face_diagnostics: bool = True,
) -> dict[str, object]:
    return markers.stress_face_diagnostics(
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
        streamwise_axis_index=STREAMWISE_AXIS_INDEX,
        include_face_diagnostics=include_face_diagnostics,
    )


def _marker_projection_boundary_report_fields(
    markers: HibmMpmSurfaceMarkers,
    *,
    traction_tip_cap_pressure_enabled: bool,
    canonical_velocity_dirichlet_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    physical_count = int(markers.marker_count)
    projection_count = int(markers.projection_vertex_count)
    boundary_only_count = projection_count - physical_count
    tip_cap_enabled = boundary_only_count == 4
    closure_report = (
        canonical_velocity_dirichlet_report.get("marker_target_closure")
        if isinstance(canonical_velocity_dirichlet_report, Mapping)
        else None
    )
    closure_healthy = (
        _canonical_marker_target_closure_health_failure(closure_report) is None
    )
    closure_projection_count = (
        int(closure_report["projection_only_marker_count"])
        if closure_healthy and isinstance(closure_report, Mapping)
        else 0
    )
    closure_evaluated_axis_count = (
        int(closure_report["projection_only_evaluated_axis_count"])
        if closure_healthy and isinstance(closure_report, Mapping)
        else 0
    )
    closure_invalid_axis_count = (
        int(closure_report["projection_only_invalid_axis_count"])
        if closure_healthy and isinstance(closure_report, Mapping)
        else 0
    )
    closure_constraint_count = (
        int(closure_report["projection_only_constraint_count"])
        if closure_healthy and isinstance(closure_report, Mapping)
        else 0
    )
    closure_max_residual_mps = (
        float(closure_report["projection_only_max_residual_mps"])
        if closure_healthy and isinstance(closure_report, Mapping)
        else None
    )
    tip_cap_closure_included = bool(
        tip_cap_enabled
        and closure_healthy
        and closure_projection_count == boundary_only_count
        and closure_evaluated_axis_count == 3 * boundary_only_count
        and closure_invalid_axis_count == 0
    )
    tip_cap_force_included = bool(
        tip_cap_enabled and traction_tip_cap_pressure_enabled
    )
    return {
        "marker_physical_traction_count": physical_count,
        "marker_projection_vertex_count": projection_count,
        "marker_boundary_only_vertex_count": boundary_only_count,
        "tip_cap_boundary_enabled": tip_cap_enabled,
        "tip_cap_boundary_region_id": (
            TIP_CAP_BOUNDARY_REGION_ID if tip_cap_enabled else None
        ),
        "tip_cap_force_included": tip_cap_force_included,
        "tip_cap_traction_policy": (
            "one_sided_gauge_pressure_outward_normal"
            if tip_cap_force_included
            else "projection_only_no_traction"
            if tip_cap_enabled
            else "not_applicable"
        ),
        "tip_cap_no_slip_closure_included": tip_cap_closure_included,
        "tip_cap_no_slip_health_policy": (
            "canonical_marker_target_closure_kernel_evidence"
            if tip_cap_closure_included
            else "missing_kernel_evidence"
            if tip_cap_enabled
            else "not_applicable"
        ),
        "tip_cap_marker_target_closure_projection_vertex_count": (
            closure_projection_count
        ),
        "tip_cap_marker_target_closure_evaluated_axis_count": (
            closure_evaluated_axis_count
        ),
        "tip_cap_marker_target_closure_invalid_axis_count": (
            closure_invalid_axis_count
        ),
        "tip_cap_marker_target_closure_constraint_count": (
            closure_constraint_count
        ),
        "tip_cap_marker_target_closure_max_residual_mps": (
            closure_max_residual_mps
        ),
    }


def _scatter_report_fields(report: Any) -> dict[str, object]:
    return {
        "scatter_action_reaction_residual_n": float(
            report.action_reaction_residual_n
        ),
        "scatter_action_reaction_residual_N": float(
            report.action_reaction_residual_n
        ),
    }


def _build_markers(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> HibmMpmSurfaceMarkers:
    markers_per_face = int(config.marker_count)
    marker_layout = _traction_marker_layout(config)
    physical_marker_capacity = (
        markers_per_face
        if marker_layout == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE
        else 2 * markers_per_face
    )
    open_ribbon_tip_cap_enabled = (
        marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
    )
    if open_ribbon_tip_cap_enabled and markers_per_face < 2:
        raise ValueError(
            "dual physical-face tip-cap projection requires at least two "
            "markers per face"
        )
    marker_capacity = physical_marker_capacity + (
        4 if open_ribbon_tip_cap_enabled else 0
    )
    markers = HibmMpmSurfaceMarkers(
        marker_capacity=marker_capacity,
        runtime=runtime,
    )
    solid_min, solid_max = _solid_box(config)
    x_center = 0.5 * (solid_min[0] + solid_max[0])
    segment = config.flap_height_m / markers_per_face
    area = config.flap_height_m * (solid_max[0] - solid_min[0]) / markers_per_face
    dz = _grid_spacing_m(config)[2]
    offset = _traction_marker_face_offset_cells(config) * dz
    probe_origin_mode = _traction_pressure_probe_origin_mode(config)
    probe_origin_offset_cells = _traction_pressure_probe_origin_offset_cells(config)
    probe_origin_offset = (
        0.0 if probe_origin_offset_cells is None else probe_origin_offset_cells * dz
    )
    if marker_layout == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE:
        face_specs = (
            (
                0.5 * (solid_min[2] + solid_max[2]),
                0.5 * (solid_min[2] + solid_max[2]),
                (0.0, 0.0, 1.0),
                PRIMARY_REGION_ID,
            ),
        )
    else:
        face_specs = (
            (
                solid_max[2] + offset,
                solid_max[2],
                (0.0, 0.0, 1.0),
                PRIMARY_REGION_ID,
            ),
            (
                solid_min[2] - offset,
                solid_min[2],
                (0.0, 0.0, -1.0),
                SECONDARY_REGION_ID,
            ),
        )
    positions = []
    probe_origins = []
    velocities = []
    normals = []
    areas = []
    regions = []
    for z, physical_face_z, normal, region_id in face_specs:
        for marker in range(markers_per_face):
            y = solid_min[1] + (float(marker) + 0.5) * segment
            positions.append((x_center, y, z))
            if (
                probe_origin_mode
                == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
            ):
                probe_origin_z = physical_face_z + probe_origin_offset * normal[2]
                probe_origins.append((x_center, y, probe_origin_z))
            velocities.append((0.0, 0.0, 0.0))
            normals.append(normal)
            areas.append(area)
            regions.append(region_id)
    markers.load_markers(
        positions_m=positions,
        velocities_mps=velocities,
        normals=normals,
        areas_m2=areas,
        region_ids=regions,
        pressure_probe_origins_m=(
            probe_origins
            if probe_origin_mode == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
            else None
        ),
    )
    projection_segments = tuple(
        (
            face_index * markers_per_face + marker_index,
            face_index * markers_per_face + marker_index + 1,
        )
        for face_index in range(len(face_specs))
        for marker_index in range(markers_per_face - 1)
    )
    if open_ribbon_tip_cap_enabled:
        markers.configure_open_ribbon_tip_cap(
            primary_previous_marker_index=markers_per_face - 2,
            primary_tip_marker_index=markers_per_face - 1,
            secondary_previous_marker_index=2 * markers_per_face - 2,
            secondary_tip_marker_index=2 * markers_per_face - 1,
            cap_region_id=TIP_CAP_BOUNDARY_REGION_ID,
            cap_area_m2=(solid_max[0] - solid_min[0])
            * (solid_max[2] - solid_min[2]),
            inactive_axis=OUT_OF_PLANE_AXIS_INDEX,
        )
        projection_segments += (
            (markers_per_face - 1, physical_marker_capacity),
            (2 * markers_per_face - 1, physical_marker_capacity + 1),
            (physical_marker_capacity + 2, physical_marker_capacity + 3),
        )
    markers.set_projection_segments(projection_segments)
    return markers


def _install_selected_pressure_pair_anchor_markers(
    markers: HibmMpmSurfaceMarkers,
    config: Any,
) -> dict[str, object]:
    anchor_markers_json = _traction_pressure_pair_anchor_markers_json(config)
    if anchor_markers_json is None:
        if (
            _is_selected_traction_formulation_coupled_smoke(config)
            and _traction_pressure_pair_runtime_provider_mode(config)
            == TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR
        ):
            runtime_pair_map = _runtime_pressure_pair_anchor_map(markers, config)
            runtime_pair_map.require_current_marker_geometry(markers)
            markers.set_pressure_pair_anchor_cells(
                inside_cells=runtime_pair_map.inside_cells,
                outside_cells=runtime_pair_map.outside_cells,
                source_marker_geometry_revision=(
                    runtime_pair_map.marker_geometry_revision
                ),
                source_marker_geometry_sha256=(
                    runtime_pair_map.marker_geometry_sha256
                ),
            )
            return _pressure_pair_anchor_install_report(
                status="installed",
                source="runtime_generated",
                marker_count=int(markers.marker_count),
                active_marker_count=runtime_pair_map.selected_count,
                anchor_map_sha256=runtime_pair_map.pair_map_sha256,
                source_marker_geometry_sha256=(
                    runtime_pair_map.marker_geometry_sha256
                ),
                source_marker_geometry_revision=(
                    runtime_pair_map.marker_geometry_revision
                ),
                current_marker_geometry_sha256=(
                    runtime_pair_map.marker_geometry_sha256
                ),
                current_marker_geometry_revision=(
                    runtime_pair_map.marker_geometry_revision
                ),
                pair_map_diagnostics=runtime_pair_map.as_diagnostics(),
                fixed_solid_snapshot_policy="runtime_marker_geometry",
            )
        if _is_selected_traction_formulation_coupled_smoke(config):
            raise ValueError(
                "selected coupled smoke requires "
                "traction_pressure_pair_anchor_markers_json"
            )
        return _pressure_pair_anchor_install_report(
            status="not_requested",
            source="unset",
            marker_count=int(markers.marker_count),
        )
    if not _is_selected_traction_formulation_coupled_smoke(config):
        raise ValueError(
            "traction_pressure_pair_anchor_markers_json is selected coupled smoke only"
        )

    (
        marker_payload,
        resolved_markers_json,
        wrapper_payloads,
        wrapper_paths,
    ) = _load_pressure_pair_anchor_marker_payload(Path(anchor_markers_json))
    _assert_pressure_pair_anchor_marker_geometry_matches(markers, marker_payload)
    inside_cells, outside_cells = _pressure_pair_anchor_cells_from_marker_payload(
        marker_payload,
    )
    markers.set_pressure_pair_anchor_cells(
        inside_cells=inside_cells,
        outside_cells=outside_cells,
    )

    metadata_sources = list(wrapper_payloads) + [marker_payload]
    return _pressure_pair_anchor_install_report(
        status="installed",
        source="marker_diagnostics_json",
        marker_count=int(markers.marker_count),
        active_marker_count=len(inside_cells),
        source_json=anchor_markers_json,
        resolved_json=resolved_markers_json.as_posix(),
        wrapper_jsons=[path.as_posix() for path in wrapper_paths],
        wrapper_depth=len(wrapper_paths),
        anchor_map_sha256=_first_metadata_value(
            metadata_sources,
            "anchor_map_sha256",
        ),
        source_flow_snapshot_sha256=_first_metadata_value(
            metadata_sources,
            "anchor_source_flow_snapshot_sha256",
            "flow_snapshot_sha256",
            "new_or_confirmed_flow_snapshot_sha256",
        ),
        source_marker_geometry_sha256=_first_metadata_value(
            metadata_sources,
            "anchor_source_marker_geometry_sha256",
            "marker_geometry_sha256",
        ),
        fixed_solid_snapshot_policy=_first_metadata_value(
            metadata_sources,
            "fixed_solid_snapshot_policy",
        ),
    )


def _runtime_pressure_pair_anchor_map(
    markers: HibmMpmSurfaceMarkers,
    config: Any,
    fluid_state: Any = None,
) -> PressureSamplePairMap:
    solid_min, solid_max = _solid_box(config)
    inside_axis_position_m = 0.5 * (
        float(solid_min[STREAMWISE_AXIS_INDEX])
        + float(solid_max[STREAMWISE_AXIS_INDEX])
    )
    provider = RuntimeAnchoredCellPairProvider(
        domain_bounds_m=_domain_bounds(config),
        grid_nodes=tuple(int(value) for value in config.grid_nodes),
        anchor_axis=STREAMWISE_AXIS_INDEX,
        inside_axis_position_m=inside_axis_position_m,
        outside_axis_offset_cells=1,
        normal_aware_rays=True,
    )
    return provider.compute_pairs(markers, fluid_state=fluid_state)


def _refresh_runtime_pressure_pair_anchor_markers(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    refresh_count: int,
) -> tuple[dict[str, object], PressureSamplePairMap] | None:
    if not (
        _is_selected_traction_formulation_coupled_smoke(config)
        and _traction_pressure_pair_runtime_provider_mode(config)
        == TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR
    ):
        return None

    # Retire the previous device map before reading or validating the new
    # geometry.  Any failure below therefore leaves anchors inactive instead
    # of silently sampling cells owned by an older interface revision.
    markers.reset_pressure_pair_anchor_cells()
    runtime_pair_map = _runtime_pressure_pair_anchor_map(
        markers,
        config,
        fluid_state=fluid,
    )
    runtime_pair_map.require_current_marker_geometry(markers)
    markers.set_pressure_pair_anchor_cells(
        inside_cells=runtime_pair_map.inside_cells,
        outside_cells=runtime_pair_map.outside_cells,
        source_marker_geometry_revision=runtime_pair_map.marker_geometry_revision,
        source_marker_geometry_sha256=runtime_pair_map.marker_geometry_sha256,
    )
    report = _pressure_pair_anchor_install_report(
        status="installed",
        source="runtime_generated",
        marker_count=int(markers.marker_count),
        active_marker_count=runtime_pair_map.selected_count,
        anchor_map_sha256=runtime_pair_map.pair_map_sha256,
        source_marker_geometry_sha256=runtime_pair_map.marker_geometry_sha256,
        source_marker_geometry_revision=runtime_pair_map.marker_geometry_revision,
        current_marker_geometry_sha256=runtime_pair_map.marker_geometry_sha256,
        current_marker_geometry_revision=runtime_pair_map.marker_geometry_revision,
        pair_map_diagnostics=runtime_pair_map.as_diagnostics(),
        fixed_solid_snapshot_policy="runtime_current_marker_geometry",
        runtime_refresh_count=int(refresh_count),
    )
    return report, runtime_pair_map


def _load_pressure_pair_anchor_marker_payload(
    path: Path,
) -> tuple[dict[str, Any], Path, tuple[dict[str, Any], ...], tuple[Path, ...]]:
    current = path
    wrappers: list[dict[str, Any]] = []
    wrapper_paths: list[Path] = []
    seen: set[str] = set()
    for _depth in range(8):
        current_key = current.resolve().as_posix()
        if current_key in seen:
            raise ValueError("pressure pair anchor marker diagnostics source cycle")
        seen.add(current_key)
        payload = json.loads(current.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("pressure pair anchor marker diagnostics must be an object")
        if isinstance(payload.get("markers"), list):
            return payload, current, tuple(wrappers), tuple(wrapper_paths)
        source = payload.get("source_marker_diagnostics_json")
        if not source:
            raise ValueError(
                "pressure pair anchor marker diagnostics must contain markers "
                "or source_marker_diagnostics_json"
            )
        wrappers.append(payload)
        wrapper_paths.append(current)
        current = Path(str(source))
    raise ValueError("pressure pair anchor marker diagnostics source chain too deep")


def _pressure_pair_anchor_cells_from_marker_payload(
    payload: Mapping[str, Any],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    marker_payloads = _pressure_pair_anchor_marker_entries(payload)
    inside_cells: list[tuple[int, int, int]] = []
    outside_cells: list[tuple[int, int, int]] = []
    for index, marker in enumerate(marker_payloads):
        if not bool(marker.get("pressure_pair_anchor_active", False)):
            raise ValueError(
                "pressure pair anchor marker payload contains inactive marker "
                f"{index}"
            )
        inside_cells.append(
            _pressure_pair_anchor_cell(
                marker.get("pressure_pair_anchor_inside_cell"),
                marker_index=index,
                field_name="pressure_pair_anchor_inside_cell",
            )
        )
        outside_cells.append(
            _pressure_pair_anchor_cell(
                marker.get("pressure_pair_anchor_outside_cell"),
                marker_index=index,
                field_name="pressure_pair_anchor_outside_cell",
            )
        )
    return inside_cells, outside_cells


def _pressure_pair_anchor_marker_entries(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    markers_payload = payload.get("markers")
    if not isinstance(markers_payload, list) or not markers_payload:
        raise ValueError("pressure pair anchor marker payload must contain markers")
    entries: list[Mapping[str, Any]] = []
    for index, marker in enumerate(markers_payload):
        if not isinstance(marker, Mapping):
            raise ValueError(f"pressure pair anchor marker {index} must be an object")
        entries.append(marker)
    declared_count = payload.get("marker_count")
    if declared_count is not None and int(declared_count) != len(entries):
        raise ValueError("pressure pair anchor marker_count does not match markers")
    return entries


def _pressure_pair_anchor_cell(
    value: object,
    *,
    marker_index: int,
    field_name: str,
) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} for marker {marker_index} must have 3 cells")
    cell = tuple(int(component) for component in value)
    if any(component < 0 for component in cell):
        raise ValueError(f"{field_name} for marker {marker_index} must be in-bounds")
    return cell


def _assert_pressure_pair_anchor_marker_geometry_matches(
    markers: HibmMpmSurfaceMarkers,
    payload: Mapping[str, Any],
) -> None:
    marker_payloads = _pressure_pair_anchor_marker_entries(payload)
    marker_count = int(markers.marker_count)
    if len(marker_payloads) != marker_count:
        raise ValueError("pressure pair anchor marker count must match live markers")
    positions = markers.x_gamma_m.to_numpy()[:marker_count]
    normals = markers.n_gamma.to_numpy()[:marker_count]
    regions = markers.region_id.to_numpy()[:marker_count]
    for index, marker in enumerate(marker_payloads):
        marker_index = int(marker.get("marker_index", index))
        if marker_index != index:
            raise ValueError("pressure pair anchor marker indices must be ordered")
        if int(marker.get("region_id", -1)) != int(regions[index]):
            raise ValueError(
                "pressure pair anchor marker region mismatch at marker "
                f"{index}"
            )
        expected_position = _pressure_pair_anchor_vector3(
            marker.get("position_m"),
            marker_index=index,
            field_name="position_m",
        )
        expected_normal = _pressure_pair_anchor_vector3(
            marker.get("normal"),
            marker_index=index,
            field_name="normal",
        )
        if not np.allclose(
            positions[index],
            np.asarray(expected_position, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(
                "pressure pair anchor marker position mismatch at marker "
                f"{index}"
            )
        if not np.allclose(
            normals[index],
            np.asarray(expected_normal, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(
                "pressure pair anchor marker normal mismatch at marker "
                f"{index}"
            )


def _pressure_pair_anchor_vector3(
    value: object,
    *,
    marker_index: int,
    field_name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} for marker {marker_index} must be length 3")
    return tuple(float(component) for component in value)


def _first_metadata_value(
    payloads: list[Mapping[str, Any]],
    *keys: str,
) -> str:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value is not None and str(value) != "":
                return str(value)
    return ""


def _pressure_pair_anchor_install_report(
    *,
    status: str,
    source: str,
    marker_count: int,
    active_marker_count: int = 0,
    source_json: str = "",
    resolved_json: str = "",
    wrapper_jsons: list[str] | None = None,
    wrapper_depth: int = 0,
    anchor_map_sha256: str = "",
    source_flow_snapshot_sha256: str = "",
    source_marker_geometry_sha256: str = "",
    source_marker_geometry_revision: int | None = None,
    current_marker_geometry_sha256: str = "",
    current_marker_geometry_revision: int | None = None,
    pair_map_diagnostics: Mapping[str, Any] | None = None,
    fixed_solid_snapshot_policy: str = "",
    runtime_refresh_count: int = 0,
) -> dict[str, object]:
    full_map_installed = active_marker_count == marker_count
    current_sha256 = str(current_marker_geometry_sha256)
    if not current_sha256 and full_map_installed:
        current_sha256 = str(source_marker_geometry_sha256)
    current_revision = current_marker_geometry_revision
    if current_revision is None and full_map_installed:
        current_revision = source_marker_geometry_revision
    return {
        "pressure_pair_anchor_install_status": status,
        "pressure_pair_anchor_source": source,
        "pressure_pair_anchor_markers_json": source_json,
        "pressure_pair_anchor_resolved_markers_json": resolved_json,
        "pressure_pair_anchor_wrapper_jsons": list(wrapper_jsons or []),
        "pressure_pair_anchor_wrapper_depth": int(wrapper_depth),
        "pressure_pair_anchor_active_marker_count": int(active_marker_count),
        "pressure_pair_anchor_expected_marker_count": int(marker_count),
        "pressure_pair_anchor_map_sha256": anchor_map_sha256,
        "pressure_pair_anchor_source_flow_snapshot_sha256": (
            source_flow_snapshot_sha256
        ),
        "pressure_pair_anchor_source_marker_geometry_sha256": (
            source_marker_geometry_sha256
        ),
        "pressure_pair_anchor_source_marker_geometry_revision": (
            source_marker_geometry_revision
        ),
        "pressure_pair_anchor_current_marker_geometry_sha256": current_sha256,
        "pressure_pair_anchor_current_marker_geometry_revision": current_revision,
        "pressure_pair_anchor_runtime_refresh_count": int(runtime_refresh_count),
        "pressure_pair_anchor_pair_map": dict(pair_map_diagnostics or {}),
        "pressure_pair_anchor_fixed_solid_snapshot_policy": (
            fixed_solid_snapshot_policy
        ),
    }


def _build_solid(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> NeoHookeanMpmState:
    bounds_min, bounds_max = _solid_mpm_bounds(config)
    capacity = math.prod(config.solid_particle_counts)
    solid = NeoHookeanMpmState(
        particle_capacity=capacity,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        grid_nodes=config.grid_nodes,
        runtime=runtime,
    )
    solid_min, solid_max = _solid_box(config)
    solid.initialize_box(
        particle_counts=config.solid_particle_counts,
        box_min_m=solid_min,
        box_max_m=solid_max,
        density_kgm3=config.solid_density_kgm3,
    )
    _configure_solid_fields(solid, config)
    return solid


def _configure_solid_fields(
    solid: NeoHookeanMpmState,
    config: Any,
) -> None:
    particle_count = int(solid.particle_count)
    positions = solid.x.to_numpy()
    normals = np.zeros((solid.particle_capacity, 3), dtype=np.float32)
    areas = np.zeros((solid.particle_capacity,), dtype=np.float32)
    region_ids = np.zeros((solid.particle_capacity,), dtype=np.int32)
    fixed = np.zeros((solid.particle_capacity,), dtype=np.int32)

    solid_min, solid_max = _solid_box(config)
    root_row_height = config.flap_height_m / float(config.solid_particle_counts[1])
    root_limit = solid_min[1] + 1.01 * root_row_height
    mid_z = 0.5 * (solid_min[2] + solid_max[2])
    particle_area = config.flap_height_m * (solid_max[0] - solid_min[0]) / max(
        float(particle_count),
        1.0,
    )
    for particle in range(particle_count):
        region_ids[particle] = PRIMARY_REGION_ID
        normals[particle] = (
            0.0,
            0.0,
            -1.0 if positions[particle, 2] < mid_z else 1.0,
        )
        areas[particle] = particle_area
        if positions[particle, 1] <= root_limit:
            fixed[particle] = 1

    solid.region_id.from_numpy(region_ids)
    solid.fixed_particle.from_numpy(fixed)
    solid.surface_normal.from_numpy(normals)
    solid.rest_surface_normal.from_numpy(normals)
    solid.area_weight_m2.from_numpy(areas)
    solid.rest_area_weight_m2.from_numpy(areas)


def _sample_stress_to_marker_forces(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any | None = None,
) -> Any:
    pressure_sampling_mode = (
        TRACTION_PRESSURE_TWO_SIDED
        if config is None
        else _traction_pressure_sampling_mode(config)
    )
    one_sided_policy = (
        TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED
        if config is None
        else _traction_one_sided_pressure_policy(config)
    )
    one_sided_region_id = (
        PRIMARY_REGION_ID
        if pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
        and one_sided_policy == TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED
        else -1
    )
    per_face_one_sided = (
        pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
        and one_sided_policy == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED
    )
    primary_side_sign = (
        0.0
        if config is None
        else _traction_one_sided_primary_fluid_side_normal_sign(config) or 0.0
    )
    secondary_side_sign = (
        0.0
        if config is None
        else _traction_one_sided_secondary_fluid_side_normal_sign(config) or 0.0
    )
    report = markers.sample_fluid_stress_to_marker_tractions(
        fluid.velocity,
        _fluid_feedback_pressure_field(fluid),
        fluid.obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        fluid.grid.grid_nodes,
        viscosity_pa_s=0.0 if config is None else _traction_viscosity_pa_s(config),
        two_sided_pressure=True,
        one_sided_pressure_region_id=one_sided_region_id,
        one_sided_reference_pressure_pa=0.0,
        one_sided_pressure_primary_region_id=(
            PRIMARY_REGION_ID if per_face_one_sided else -1
        ),
        one_sided_pressure_secondary_region_id=(
            SECONDARY_REGION_ID if per_face_one_sided else -1
        ),
        one_sided_primary_reference_pressure_pa=(
            0.0
            if config is None
            else _traction_one_sided_primary_reference_pressure_pa(config)
        ),
        one_sided_secondary_reference_pressure_pa=(
            0.0
            if config is None
            else _traction_one_sided_secondary_reference_pressure_pa(config)
        ),
        one_sided_primary_fluid_side_normal_sign=primary_side_sign,
        one_sided_secondary_fluid_side_normal_sign=secondary_side_sign,
        tip_cap_pressure_enabled=(
            False if config is None else _traction_tip_cap_pressure_enabled(config)
        ),
        tip_cap_region_id=TIP_CAP_BOUNDARY_REGION_ID,
        tip_cap_reference_pressure_pa=0.0,
        pressure_probe_ladder_start_offset_cells=(
            None
            if config is None
            else _traction_pressure_probe_start_offset_cells(config)
        ),
        pressure_probe_ladder_spacing_cells=(
            0.5
            if config is None
            else _traction_pressure_probe_ladder_spacing_cells(config)
        ),
        pressure_probe_ladder_rung_count=(
            5 if config is None else _traction_pressure_probe_ladder_rung_count(config)
        ),
        pressure_probe_ladder_mode=(
            TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL
            if config is None
            else _traction_pressure_probe_ladder_mode(config)
        ),
        pressure_pair_policy=(
            TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER
            if config is None
            else _traction_pressure_pair_policy(config)
        ),
        pressure_pair_max_cell_delta=(
            1 if config is None else _traction_pressure_pair_max_cell_delta(config)
        ),
        pressure_pair_require_opposite_sides=(
            True
            if config is None
            else _traction_pressure_pair_require_opposite_sides(config)
        ),
    )
    markers.compute_marker_forces()
    return report


def _solid_displacement_report(
    solid: NeoHookeanMpmState,
    fixed_mask: np.ndarray,
    tip_mask: np.ndarray,
    rest: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> dict[str, object]:
    if positions is None:
        positions = solid.x.to_numpy()[: solid.particle_count]
    if rest is None:
        # rest positions are constant; per-step callers pass a cached copy so
        # the whole rest array is not re-fetched from the device every step
        rest = solid.rest_x.to_numpy()[: solid.particle_count]
    displacement = positions - rest
    norms = np.linalg.norm(displacement, axis=1)
    tip_displacement = displacement[tip_mask]
    if tip_displacement.size == 0:
        raise RuntimeError("tip particle mask is empty")
    root_norms = norms[fixed_mask]
    return {
        "max_displacement_m": float(norms.max(initial=0.0)),
        "tip_mean_displacement_m": tuple(float(v) for v in tip_displacement.mean(axis=0)),
        "tip_displacement_norm_m": float(np.linalg.norm(tip_displacement.mean(axis=0))),
        "root_max_displacement_m": float(root_norms.max(initial=0.0)),
    }


def _solid_masks(
    solid: NeoHookeanMpmState,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    rest = solid.rest_x.to_numpy()[: solid.particle_count]
    fixed = solid.fixed_particle.to_numpy()[: solid.particle_count] != 0
    _, solid_max = _solid_box(config)
    tip_row_height = config.flap_height_m / float(config.solid_particle_counts[1])
    tip_mask = rest[:, 1] >= solid_max[1] - 1.01 * tip_row_height
    return fixed, tip_mask
