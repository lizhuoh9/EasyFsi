from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import taichi as ti

from simulation_core import InterfaceReactionRelaxationState

from .history import _final_row_int, _final_row_number, _row_bool, read_csv_rows
from .schedules import pressure_schedule_dict

RUN_CHECKPOINT_VERSION = 3

RUN_CHECKPOINT_FILENAME = "run_checkpoint.npz"

CHECKPOINT_MARKER_STATE_FIELD_NAMES = (
    "x_gamma_m",
    "v_gamma_mps",
    "n_gamma",
    "A_gamma_m2",
)

CHECKPOINT_ARG_FINGERPRINT_FIELDS = (
    "source_config",
    "steps_explicit",
    "projection_iterations",
    "fluid_advection_scheme",
    "pressure_solver",
    "pressure_solve_failure_policy",
    "cg_preconditioner",
    "cg_tolerance",
    "multigrid_cycles",
    "divergence_cleanup_iterations",
    "divergence_cleanup_relaxation",
    "projection_divergence_tolerance",
    "grid_scale",
    "use_graded_grid",
    "graded_grid_target_spacing_m",
    "graded_grid_farfield_spacing_m",
    "graded_grid_growth_ratio",
    "graded_grid_max_cells",
    "use_tail_refinement",
    "tail_refinement_target_spacing_m",
    "tail_refinement_padding_m",
    "time_step_scale",
    "solid_model",
    "solid_mpm_layers",
    "solid_mpm_substeps",
    "membrane_thickness_scale",
    "solid_density_scale",
    "solid_mpm_cfl",
    "solid_mpm_velocity_damping",
    "solid_mpm_flip_blend",
    "mooney_membrane_force_scale",
    "poissons_ratio",
    "constraint_force_scale",
    "fsi_constraint_force_solid_mobility_ratio",
    "fsi_solid_response_mobility_coupling",
    "fsi_velocity_target_solid_mobility_ratio",
    "fsi_solid_response_velocity_mobility_coupling",
    "fsi_velocity_constraint_blend",
    "fsi_velocity_constraint_solid_mobility_ratio",
    "interface_reaction_relaxation",
    "interface_reaction_aitken",
    "interface_reaction_aitken_lower_bound",
    "interface_reaction_aitken_upper_bound",
    "interface_reaction_passivity_limit",
    "interface_reaction_robin_impedance_ns_m",
    "interface_reaction_robin_matrix_impedance_ns_m",
    "interface_reaction_robin_target_mode",
    "fsi_coupling_mode",
    "fsi_stabilization_preset",
    "fsi_coupling_solver",
    "fsi_coupling_target_map_relaxation",
    "fsi_coupling_rejected_trial_backtrack",
    "fsi_coupling_residual_growth_rejection_factor",
    "fsi_coupling_max_accepted_residual_n",
    "fsi_coupling_trust_region_force_increment_n",
    "fsi_coupling_trust_region_adaptive",
    "fsi_coupling_trust_region_shrink_factor",
    "fsi_coupling_trust_region_growth_factor",
    "fsi_coupling_trust_region_rebound_factor",
    "fsi_coupling_trust_region_rebound_backtrack",
    "fsi_coupling_trust_region_rebound_stop_factor",
    "fsi_coupling_trust_region_rebound_stop_max_residual_n",
    "reuse_accepted_fsi_trial_state",
    "min_outlet_to_main_volume_flux_ratio",
    "pressure_outlet_source_ratio_tolerance",
    "fluid_substeps",
    "adaptive_fluid_substeps",
    "adaptive_fluid_substeps_target_cfl",
    "adaptive_fluid_substeps_max",
    "adaptive_fluid_substeps_safety",
    "ibm_correction_iterations",
    "fsi_coupling_iterations",
    "fsi_coupling_adaptive_iterations_max",
    "fsi_coupling_adaptive_iterations_residual_threshold_n",
    "fsi_coupling_adaptive_iterations_cfl_threshold",
    "fsi_coupling_same_step_rerun_iterations_max",
    "fsi_coupling_same_step_rerun_residual_threshold_n",
    "fsi_coupling_same_step_rerun_fluid_substep_factor",
    "fsi_coupling_residual_continuation_iterations_max",
    "fsi_coupling_residual_continuation_threshold_n",
    "fsi_coupling_residual_continuation_rebound_secant_from_best",
    "fsi_coupling_residual_continuation_rebound_secant_factor",
    "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max",
    "fsi_coupling_trial_interior_divergence_tolerance",
    "fsi_coupling_tolerance_n",
    "fsi_marker_coupling_tolerance_mps",
    "disable_pressure_outlet_zmin",
    "disable_reduced_obstacles",
    "source_config_intersect_reduced_water_domain",
    "source_config_connect_surface_seeds_to_zmin",
    "source_config_surface_seed_zmin_connection_max_carve_cells",
    "use_region14_aperture_carve",
    "disable_region14_aperture_carve",
    "open_downstream_farfield",
    "use_nozzle_taper",
    "nozzle_taper_length_m",
    "nozzle_taper_inlet_radius_m",
    "pressure_t0_s",
    "pressure_t1_s",
    "pressure_t2_s",
    "pressure_p0_pa",
    "pressure_p1_pa",
    "pressure_p2_pa",
    "diagnostic_disable_pressure_neumann_matrix_rows",
    "arch",
)

def resume_history_rows_for_checkpoint(
    rows: list[dict[str, object]],
    *,
    completed_step: int,
) -> list[dict[str, object]]:
    checkpoint_step = int(completed_step)
    if checkpoint_step < 0:
        raise ValueError("completed_step must be non-negative")
    if len(rows) < checkpoint_step:
        raise ValueError(
            "resume requires history.csv to contain at least the checkpointed "
            f"steps: len(history)={len(rows)} checkpoint={checkpoint_step}"
        )
    return list(rows[:checkpoint_step])

def validate_resume_history_checkpoint_alignment(
    rows: list[dict[str, object]],
    *,
    completed_step: int,
    checkpoint_time_s: float,
    dt_s: float,
) -> None:
    checkpoint_step = int(completed_step)
    if checkpoint_step == 0:
        if rows:
            raise ValueError("resume history must be empty for a zero-step checkpoint")
        return
    if len(rows) != checkpoint_step:
        raise ValueError(
            "resume history row count must equal the checkpointed step count after truncation: "
            f"len(history)={len(rows)} checkpoint={checkpoint_step}"
        )
    try:
        history_step = int(rows[-1]["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume history final row must contain an integer step") from exc
    if history_step != checkpoint_step:
        raise ValueError(
            "resume history final row step does not match checkpoint: "
            f"history={history_step} checkpoint={checkpoint_step}"
        )
    try:
        history_time_s = float(rows[-1]["time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume history final row must contain finite time_s") from exc
    if not math.isfinite(history_time_s):
        raise ValueError("resume history final row must contain finite time_s")
    tolerance_s = max(abs(float(dt_s)) * 1.0e-4, 1.0e-7)
    if abs(history_time_s - float(checkpoint_time_s)) > tolerance_s:
        raise ValueError(
            "resume history final row time_s does not match checkpoint: "
            f"history={history_time_s:.9g} checkpoint={float(checkpoint_time_s):.9g}"
        )

def checkpoint_path_for_args(args: argparse.Namespace, output_dir: Path) -> Path:
    raw_path = getattr(args, "checkpoint_path", None)
    if raw_path:
        return Path(raw_path).resolve()
    return output_dir / RUN_CHECKPOINT_FILENAME

def _checkpoint_normalized_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_checkpoint_normalized_value(item) for item in value]
    if isinstance(value, list):
        return [_checkpoint_normalized_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _checkpoint_normalized_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    return value

def checkpoint_run_fingerprint(
    *,
    args: argparse.Namespace,
    spec: SquidReducedSpec,
    step_count: int,
    full_pressure_waveform_steps: int,
) -> dict[str, object]:
    spec_payload = asdict(spec)
    spec_payload["source_config_path"] = str(Path(spec.source_config_path).resolve())
    arg_payload = {
        name: getattr(args, name, None)
        for name in CHECKPOINT_ARG_FINGERPRINT_FIELDS
    }
    if arg_payload.get("source_config") is not None:
        arg_payload["source_config"] = str(Path(str(arg_payload["source_config"])).resolve())
    payload = {
        "requested_steps": int(step_count),
        "full_pressure_waveform_steps": int(full_pressure_waveform_steps),
        "spec": spec_payload,
        "args": arg_payload,
    }
    return _checkpoint_normalized_value(payload)  # type: ignore[return-value]

def _checkpoint_resume_physical_fingerprint(fingerprint: object) -> object:
    if not isinstance(fingerprint, dict):
        return fingerprint
    comparable = dict(fingerprint)
    comparable.pop("requested_steps", None)
    return _checkpoint_normalized_value(comparable)

def validate_checkpoint_run_fingerprint(
    metadata: dict[str, object],
    *,
    args: argparse.Namespace,
    spec: SquidReducedSpec,
    step_count: int,
    full_pressure_waveform_steps: int,
) -> None:
    actual = metadata.get("run_fingerprint")
    expected = checkpoint_run_fingerprint(
        args=args,
        spec=spec,
        step_count=step_count,
        full_pressure_waveform_steps=full_pressure_waveform_steps,
    )
    if _checkpoint_resume_physical_fingerprint(
        actual
    ) != _checkpoint_resume_physical_fingerprint(expected):
        raise ValueError(
            "checkpoint run fingerprint does not match current configuration; "
            "restart with the same source config, pressure schedule, grid, solver, "
            "solid, and FSI options"
        )

def _array_to_payload(payload: dict[str, np.ndarray], name: str, value: object) -> None:
    payload[name] = np.asarray(value).copy()

def _read_scalar_field(field: ti.template()) -> float:
    return float(field[None])

def _write_scalar_field(field: ti.template(), value: object) -> None:
    field[None] = float(np.asarray(value))

def _read_vector_field(field: ti.template()) -> np.ndarray:
    value = field[None]
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)

def _write_vector_field(field: ti.template(), value: object) -> None:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    field[None] = ti.Vector([float(array[0]), float(array[1]), float(array[2])])

def _checkpoint_interface_state_dict(
    state: InterfaceReactionRelaxationState,
) -> dict[str, object]:
    return {
        "relaxation": float(state.relaxation),
        "previous_residual_n": (
            None
            if state.previous_residual_n is None
            else [float(value) for value in state.previous_residual_n]
        ),
        "previous_velocity_mps": (
            None
            if state.previous_velocity_mps is None
            else [float(value) for value in state.previous_velocity_mps]
        ),
    }

def _checkpoint_interface_vector(
    data: object,
    *,
    name: str,
) -> tuple[float, ...] | None:
    if data is None:
        return None
    try:
        vector = tuple(float(value) for value in data)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(f"checkpoint {name} must be a vector or null") from exc
    except ValueError as exc:
        raise ValueError(f"checkpoint {name} must contain numeric values") from exc
    if not vector:
        raise ValueError(f"checkpoint {name} must not be empty")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"checkpoint {name} must contain only finite values")
    return vector

def _interface_state_from_checkpoint(data: object) -> InterfaceReactionRelaxationState:
    if not isinstance(data, dict):
        raise ValueError("checkpoint interface_reaction_state must be an object")
    residual = _checkpoint_interface_vector(
        data.get("previous_residual_n"),
        name="previous_residual_n",
    )
    velocity = _checkpoint_interface_vector(
        data.get("previous_velocity_mps"),
        name="previous_velocity_mps",
    )
    if residual is not None and velocity is not None and len(residual) != len(velocity):
        raise ValueError(
            "checkpoint previous_residual_n and previous_velocity_mps must have the same length"
        )
    try:
        relaxation = float(data.get("relaxation", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint relaxation must be finite") from exc
    if not math.isfinite(relaxation):
        raise ValueError("checkpoint relaxation must be finite")
    return InterfaceReactionRelaxationState(
        previous_residual_n=residual,
        previous_velocity_mps=velocity,
        relaxation=relaxation,
    )

def sharp_marker_state_arrays(markers) -> dict[str, np.ndarray]:
    """Export the dynamic HIBM sharp marker state for checkpointing.

    Markers advance by dt*v surface-state updates, so their state cannot be rebuilt
    from rest solid particles on resume. Checkpoint read/write is case-level
    host I/O, matching the existing fluid/solid checkpoint transfers.
    """
    count = int(markers.marker_count)
    state: dict[str, np.ndarray] = {}
    for name in CHECKPOINT_MARKER_STATE_FIELD_NAMES:
        state[name] = np.asarray(getattr(markers, name).to_numpy())[:count].copy()
    return state

def sharp_pressure_neumann_gradient_state_array(sharp_coupling_state) -> np.ndarray:
    """Export the active marker pressure-Neumann gradients for trial restore."""
    count = int(sharp_coupling_state.markers.marker_count)
    field = sharp_coupling_state.marker_pressure_neumann_gradient_pa_per_m
    return np.asarray(field.to_numpy())[:count].copy()

def restore_sharp_pressure_neumann_gradient_state_array(
    sharp_coupling_state,
    state: object,
) -> None:
    """Restore active marker pressure-Neumann gradients exported above."""
    count = int(sharp_coupling_state.markers.marker_count)
    field = sharp_coupling_state.marker_pressure_neumann_gradient_pa_per_m
    full = field.to_numpy()
    array = np.asarray(state, dtype=full.dtype)
    expected_shape = tuple(full[:count].shape)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            "sharp pressure-Neumann gradient state shape mismatch: "
            f"{tuple(array.shape)} != {expected_shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("sharp pressure-Neumann gradient state must be finite")
    full[:count] = array
    field.from_numpy(full)

def relaxed_sharp_pressure_neumann_gradient_state_array(
    guess: object,
    candidate: object,
    *,
    relaxation: float,
) -> np.ndarray:
    omega = float(relaxation)
    if not math.isfinite(omega) or not 0.0 <= omega <= 1.5:
        raise ValueError("relaxation must be finite and in [0, 1.5]")
    guess_array = np.asarray(guess)
    candidate_array = np.asarray(candidate)
    if tuple(candidate_array.shape) != tuple(guess_array.shape):
        raise ValueError(
            "sharp pressure-Neumann gradient state shape mismatch: "
            f"{tuple(candidate_array.shape)} != {tuple(guess_array.shape)}"
        )
    if not bool(np.all(np.isfinite(guess_array))) or not bool(
        np.all(np.isfinite(candidate_array))
    ):
        raise ValueError("sharp pressure-Neumann gradient state must be finite")
    relaxed = guess_array + omega * (candidate_array - guess_array)
    return relaxed.astype(guess_array.dtype, copy=False)

def _sharp_marker_state_array(
    state: Mapping[str, object],
    name: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if name not in state:
        raise ValueError(f"sharp marker state is missing {name!r}")
    array = np.asarray(state[name], dtype=np.float64)
    if expected_shape is not None and tuple(array.shape) != expected_shape:
        raise ValueError(
            f"sharp marker state {name!r} shape mismatch: "
            f"{tuple(array.shape)} != {expected_shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"sharp marker state {name!r} must be finite")
    return array

def _sharp_marker_fixed_point_residual_vector_mps(
    guess: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    dt_s: float,
) -> np.ndarray:
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    guess_x = _sharp_marker_state_array(guess, "x_gamma_m")
    candidate_x = _sharp_marker_state_array(
        candidate,
        "x_gamma_m",
        expected_shape=tuple(guess_x.shape),
    )
    guess_v = _sharp_marker_state_array(guess, "v_gamma_mps")
    candidate_v = _sharp_marker_state_array(
        candidate,
        "v_gamma_mps",
        expected_shape=tuple(guess_v.shape),
    )
    if guess_x.ndim != 2 or guess_x.shape[1] != 3:
        raise ValueError("x_gamma_m must have shape (marker_count, 3)")
    if guess_v.ndim != 2 or guess_v.shape[1] != 3:
        raise ValueError("v_gamma_mps must have shape (marker_count, 3)")
    if guess_x.shape[0] != guess_v.shape[0]:
        raise ValueError("x_gamma_m and v_gamma_mps marker counts must match")
    position_residual_mps = (candidate_x - guess_x) / dt
    velocity_residual_mps = candidate_v - guess_v
    return np.concatenate(
        [position_residual_mps, velocity_residual_mps],
        axis=1,
    )

def sharp_marker_fixed_point_residual_mps(
    guess: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    dt_s: float,
) -> dict[str, float | int]:
    """Measure marker fixed-point mismatch in velocity units.

    The residual combines position mismatch divided by dt and velocity mismatch,
    so it directly measures whether the marker boundary state used by the fluid
    agrees with the MPM surface state returned by the solid response.
    """
    residual_vector = _sharp_marker_fixed_point_residual_vector_mps(
        guess,
        candidate,
        dt_s=dt_s,
    )
    if residual_vector.shape[0] <= 0:
        return {
            "l2_mps": 0.0,
            "max_mps": 0.0,
            "sample_count": 0,
        }
    marker_norms = np.linalg.norm(residual_vector, axis=1)
    return {
        "l2_mps": float(np.sqrt(np.mean(marker_norms * marker_norms))),
        "max_mps": float(np.max(marker_norms)),
        "sample_count": int(marker_norms.shape[0]),
    }

def _marker_group_l2_mps(
    marker_norms_mps: np.ndarray,
    mask: np.ndarray,
) -> float:
    if marker_norms_mps.shape[0] <= 0 or not bool(np.any(mask)):
        return 0.0
    values = marker_norms_mps[mask]
    return float(np.sqrt(np.mean(values * values)))

def sharp_marker_fixed_point_residual_diagnostics_mps(
    guess: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    dt_s: float,
    marker_region_ids: object,
    primary_region_id: int,
    secondary_region_id: int,
) -> dict[str, float | int]:
    residual_vector = _sharp_marker_fixed_point_residual_vector_mps(
        guess,
        candidate,
        dt_s=dt_s,
    )
    marker_count = int(residual_vector.shape[0])
    if marker_count <= 0:
        return {
            "position_l2_mps": 0.0,
            "position_max_mps": 0.0,
            "velocity_l2_mps": 0.0,
            "velocity_max_mps": 0.0,
            "combined_l2_mps": 0.0,
            "combined_max_mps": 0.0,
            "primary_region_l2_mps": 0.0,
            "secondary_region_l2_mps": 0.0,
            "other_region_l2_mps": 0.0,
            "max_marker_index": -1,
            "max_marker_region_id": -1,
            "max_marker_position_mps": 0.0,
            "max_marker_velocity_mps": 0.0,
            "max_marker_combined_mps": 0.0,
        }
    regions = np.asarray(marker_region_ids, dtype=np.int64)
    if regions.shape[0] < marker_count:
        raise ValueError("marker_region_ids must contain at least marker_count values")
    regions = regions[:marker_count]
    position_norms = np.linalg.norm(residual_vector[:, :3], axis=1)
    velocity_norms = np.linalg.norm(residual_vector[:, 3:], axis=1)
    marker_norms = np.linalg.norm(residual_vector, axis=1)
    primary_mask = regions == int(primary_region_id)
    secondary_mask = regions == int(secondary_region_id)
    other_mask = ~(primary_mask | secondary_mask)
    max_index = int(np.argmax(marker_norms))
    return {
        "position_l2_mps": float(np.sqrt(np.mean(position_norms * position_norms))),
        "position_max_mps": float(np.max(position_norms)),
        "velocity_l2_mps": float(np.sqrt(np.mean(velocity_norms * velocity_norms))),
        "velocity_max_mps": float(np.max(velocity_norms)),
        "combined_l2_mps": float(np.sqrt(np.mean(marker_norms * marker_norms))),
        "combined_max_mps": float(np.max(marker_norms)),
        "primary_region_l2_mps": _marker_group_l2_mps(marker_norms, primary_mask),
        "secondary_region_l2_mps": _marker_group_l2_mps(marker_norms, secondary_mask),
        "other_region_l2_mps": _marker_group_l2_mps(marker_norms, other_mask),
        "max_marker_index": max_index,
        "max_marker_region_id": int(regions[max_index]),
        "max_marker_position_mps": float(position_norms[max_index]),
        "max_marker_velocity_mps": float(velocity_norms[max_index]),
        "max_marker_combined_mps": float(marker_norms[max_index]),
    }

def relaxed_sharp_marker_state_arrays(
    guess: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    relaxation: float,
) -> dict[str, np.ndarray]:
    """Return a relaxed marker state without mutating either input mapping."""
    omega = float(relaxation)
    if not math.isfinite(omega) or not 0.0 <= omega <= 1.5:
        raise ValueError("relaxation must be finite and in [0, 1.5]")
    relaxed: dict[str, np.ndarray] = {}
    for name in CHECKPOINT_MARKER_STATE_FIELD_NAMES:
        guess_array = _sharp_marker_state_array(guess, name)
        candidate_array = _sharp_marker_state_array(
            candidate,
            name,
            expected_shape=tuple(guess_array.shape),
        )
        if name == "A_gamma_m2":
            next_array = guess_array + omega * (candidate_array - guess_array)
            relaxed[name] = np.maximum(next_array, 0.0).astype(
                np.asarray(guess[name]).dtype,
                copy=False,
            )
            continue
        next_array = guess_array + omega * (candidate_array - guess_array)
        if name == "n_gamma":
            norms = np.linalg.norm(next_array, axis=1)
            invalid = norms <= 1.0e-12
            safe_norms = np.where(invalid, 1.0, norms)
            next_array = next_array / safe_norms[:, None]
            if np.any(invalid):
                next_array[invalid] = guess_array[invalid]
        relaxed[name] = next_array.astype(np.asarray(guess[name]).dtype, copy=False)
    return relaxed

def _sharp_marker_aitken_relaxation(
    *,
    previous_relaxation: float,
    previous_residual_mps: np.ndarray,
    current_residual_mps: np.ndarray,
    lower: float = 0.01,
    upper: float = 1.0,
) -> float:
    previous = np.asarray(previous_residual_mps, dtype=np.float64).reshape(-1)
    current = np.asarray(current_residual_mps, dtype=np.float64).reshape(-1)
    if previous.shape != current.shape:
        raise ValueError("Aitken residual vectors must have the same shape")
    delta = current - previous
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-30:
        return float(previous_relaxation)
    raw = -float(previous_relaxation) * float(np.dot(previous, delta)) / denominator
    if not math.isfinite(raw):
        return float(previous_relaxation)
    return max(float(lower), min(float(upper), raw))

def restore_sharp_marker_state_arrays(
    markers,
    state: Mapping[str, object],
) -> None:
    """Restore dynamic HIBM sharp marker state exported by sharp_marker_state_arrays."""
    count = int(markers.marker_count)
    for name in CHECKPOINT_MARKER_STATE_FIELD_NAMES:
        if name not in state:
            raise ValueError(f"checkpoint sharp marker state is missing {name!r}")
        field = getattr(markers, name)
        full = field.to_numpy()
        array = np.asarray(state[name], dtype=full.dtype)
        expected_shape = tuple(full[:count].shape)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                "checkpoint sharp marker state shape does not match the current "
                f"marker layout for {name!r}: {tuple(array.shape)} != {expected_shape}"
            )
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"checkpoint sharp marker state {name!r} must be finite")
        full[:count] = array
        field.from_numpy(full)

def write_run_checkpoint(
    path: Path,
    *,
    completed_step: int,
    step_count: int,
    full_pressure_waveform_steps: int,
    args: argparse.Namespace,
    simulator: ReducedSquidFSI,
    solid_mpm: object,
    interface_reaction_state: InterfaceReactionRelaxationState,
    sharp_coupling_state=None,
) -> None:
    payload: dict[str, np.ndarray] = {}
    metadata = {
        "version": RUN_CHECKPOINT_VERSION,
        "completed_step": int(completed_step),
        "requested_steps": int(step_count),
        "full_pressure_waveform_steps": int(full_pressure_waveform_steps),
        "solid_model": str(args.solid_model),
        "grid_nodes": [int(value) for value in simulator.spec.grid_nodes],
        "particle_count": int(getattr(solid_mpm, "particle_count", 0)),
        "run_fingerprint": checkpoint_run_fingerprint(
            args=args,
            spec=simulator.spec,
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
        ),
        "interface_reaction_state": _checkpoint_interface_state_dict(
            interface_reaction_state
        ),
    }
    _array_to_payload(payload, "__metadata__", np.asarray(json.dumps(metadata)))

    for name in (
        "time_s",
        "pressure_load_pa",
        "hydraulic_pressure_pa",
        "main_w_m",
        "main_v_mps",
        "tail_w_m",
        "tail_v_mps",
        "volume_flux_m3s",
        "nozzle_velocity_z_mps",
        "max_speed_mps",
        "lip_flow_z_m3s",
        "outlet_flow_z_m3s",
        "downstream_flow_z_m3s",
    ):
        _array_to_payload(payload, f"sim_{name}", _read_scalar_field(getattr(simulator, name)))
    for name in (
        "lip_sample_count",
        "outlet_sample_count",
        "downstream_sample_count",
    ):
        _array_to_payload(payload, f"sim_{name}", int(getattr(simulator, name)[None]))
    for name in (
        "primary_interface_reaction_force_n",
        "secondary_interface_reaction_force_n",
    ):
        _array_to_payload(payload, f"sim_{name}", _read_vector_field(getattr(simulator, name)))

    fluid = simulator.fluid
    for name in ("velocity", "velocity_prev", "pressure"):
        _array_to_payload(payload, f"fluid_{name}", getattr(fluid, name).to_numpy())

    if args.solid_model == "tri_mooney_shell_mpm":
        for name in ("x", "u", "v"):
            _array_to_payload(payload, f"solid_{name}", getattr(solid_mpm, name).to_numpy())
    elif args.solid_model == "neo_hookean_mpm":
        for name in ("x", "v", "C", "F"):
            _array_to_payload(payload, f"solid_{name}", getattr(solid_mpm, name).to_numpy())
    else:
        raise ValueError(f"unsupported solid model for checkpoint: {args.solid_model!r}")

    if sharp_coupling_state is not None:
        marker_state = sharp_marker_state_arrays(sharp_coupling_state.markers)
        for name in CHECKPOINT_MARKER_STATE_FIELD_NAMES:
            _array_to_payload(payload, f"marker_{name}", marker_state[name])
    _array_to_payload(
        payload,
        "has_marker_state",
        np.asarray(sharp_coupling_state is not None),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(path)

def load_run_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    simulator: ReducedSquidFSI,
    solid_mpm: object,
    step_count: int | None = None,
    full_pressure_waveform_steps: int | None = None,
    sharp_coupling_state=None,
) -> tuple[int, InterfaceReactionRelaxationState]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["__metadata__"]))
        if int(metadata.get("version", -1)) != RUN_CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {metadata.get('version')!r}"
            )
        if str(metadata.get("solid_model")) != str(args.solid_model):
            raise ValueError(
                "checkpoint solid model does not match --solid-model: "
                f"{metadata.get('solid_model')!r} != {args.solid_model!r}"
            )
        if tuple(int(value) for value in metadata.get("grid_nodes", ())) != tuple(
            int(value) for value in simulator.spec.grid_nodes
        ):
            raise ValueError("checkpoint grid shape does not match current configuration")
        if int(metadata.get("particle_count", -1)) != int(getattr(solid_mpm, "particle_count", 0)):
            raise ValueError("checkpoint solid particle count does not match current configuration")
        validate_checkpoint_run_fingerprint(
            metadata,
            args=args,
            spec=simulator.spec,
            step_count=(
                int(metadata["requested_steps"])
                if step_count is None
                else int(step_count)
            ),
            full_pressure_waveform_steps=(
                int(metadata["full_pressure_waveform_steps"])
                if full_pressure_waveform_steps is None
                else int(full_pressure_waveform_steps)
            ),
        )

        for name in (
            "time_s",
            "pressure_load_pa",
            "hydraulic_pressure_pa",
            "main_w_m",
            "main_v_mps",
            "tail_w_m",
            "tail_v_mps",
            "volume_flux_m3s",
            "nozzle_velocity_z_mps",
            "max_speed_mps",
            "lip_flow_z_m3s",
            "outlet_flow_z_m3s",
            "downstream_flow_z_m3s",
        ):
            _write_scalar_field(getattr(simulator, name), checkpoint[f"sim_{name}"])
        for name in (
            "lip_sample_count",
            "outlet_sample_count",
            "downstream_sample_count",
        ):
            getattr(simulator, name)[None] = int(np.asarray(checkpoint[f"sim_{name}"]))
        for name in (
            "primary_interface_reaction_force_n",
            "secondary_interface_reaction_force_n",
        ):
            _write_vector_field(getattr(simulator, name), checkpoint[f"sim_{name}"])

        fluid = simulator.fluid
        for name in ("velocity", "velocity_prev", "pressure"):
            getattr(fluid, name).from_numpy(checkpoint[f"fluid_{name}"])
        fluid.pressure_tmp.from_numpy(checkpoint["fluid_pressure"])
        fluid.pressure_accum.from_numpy(checkpoint["fluid_pressure"])

        if args.solid_model == "tri_mooney_shell_mpm":
            for name in ("x", "u", "v"):
                getattr(solid_mpm, name).from_numpy(checkpoint[f"solid_{name}"])
        elif args.solid_model == "neo_hookean_mpm":
            for name in ("x", "v", "C", "F"):
                getattr(solid_mpm, name).from_numpy(checkpoint[f"solid_{name}"])
        else:
            raise ValueError(f"unsupported solid model for checkpoint: {args.solid_model!r}")

        if sharp_coupling_state is not None:
            missing_marker_keys = [
                f"marker_{name}"
                for name in CHECKPOINT_MARKER_STATE_FIELD_NAMES
                if f"marker_{name}" not in checkpoint
            ]
            if missing_marker_keys:
                raise ValueError(
                    "checkpoint does not contain HIBM sharp marker state "
                    f"(missing {', '.join(missing_marker_keys)}); resuming a "
                    "sharp-coupling run from it would rebuild the immersed "
                    "boundary from rest geometry against a deformed fluid state"
                )
            restore_sharp_marker_state_arrays(
                sharp_coupling_state.markers,
                {
                    name: checkpoint[f"marker_{name}"]
                    for name in CHECKPOINT_MARKER_STATE_FIELD_NAMES
                },
            )

        return (
            int(metadata["completed_step"]),
            _interface_state_from_checkpoint(metadata.get("interface_reaction_state")),
        )
