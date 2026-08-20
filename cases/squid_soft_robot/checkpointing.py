from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from simulation_core.coupling.hibm_mpm.interface_state import (
    MARKER_INTERFACE_STATE_FIELDS,
    capture_marker_interface_state,
    restore_marker_interface_state,
)

from .source_config import (
    _selection_ids_as_int_tuple,
    source_config_requests_fluid_active_mask,
    source_config_volume_particle_cache_path,
)

if TYPE_CHECKING:
    import argparse

    from .runtime_state import ReducedSquidFSI
    from .spec import SquidReducedSpec


RUN_CHECKPOINT_VERSION = 7

RUN_CHECKPOINT_FILENAME = "run_checkpoint.npz"

CHECKPOINT_SIM_SCALAR_FIELD_NAMES = (
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
)
CHECKPOINT_SIM_COUNT_FIELD_NAMES = (
    "lip_sample_count",
    "outlet_sample_count",
    "downstream_sample_count",
)
CHECKPOINT_FLUID_FIELD_NAMES = ("velocity", "velocity_prev", "pressure")
CHECKPOINT_SOLID_FIELD_NAMES = {
    "tri_mooney_shell_mpm": ("x", "u", "v"),
    "neo_hookean_mpm": ("x", "v", "C", "F", "position_increment_residual_m"),
}

CHECKPOINT_ARG_FINGERPRINT_FIELDS = (
    "source_config",
    "steps_explicit",
    "projection_iterations",
    "hibm_post_dirichlet_consistency_projections",
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
    "fixed_rim_region_id",
    "neo_fixed_node_lock_policy",
    "solid_mpm_layers",
    "solid_mpm_substeps",
    "membrane_thickness_scale",
    "solid_density_scale",
    "solid_mpm_cfl",
    "solid_mpm_velocity_damping",
    "solid_mpm_flip_blend",
    "mooney_membrane_force_scale",
    "poissons_ratio",
    "interface_reaction_relaxation",
    "interface_reaction_aitken",
    "min_outlet_to_main_volume_flux_ratio",
    "pressure_outlet_source_ratio_tolerance",
    "fluid_substeps",
    "adaptive_fluid_substeps",
    "adaptive_fluid_substeps_target_cfl",
    "adaptive_fluid_substeps_max",
    "adaptive_fluid_substeps_safety",
    "fsi_coupling_iterations",
    "fsi_marker_coupling_tolerance_mps",
    "disable_pressure_outlet_zmin",
    "far_pressure_air_backed",
    "far_pressure_inside_probe_max_multiplier",
    "two_sided_probe_max_multiplier",
    "one_sided_probe_max_multiplier",
    "far_pressure_air_backed_probe_normal_sign",
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
    if checkpoint_step < 0:
        raise ValueError("completed_step must be non-negative")
    resolved_dt_s = float(dt_s)
    checkpoint_time = float(checkpoint_time_s)
    if not math.isfinite(resolved_dt_s) or resolved_dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(checkpoint_time):
        raise ValueError("checkpoint time_s must be finite")
    tolerance_s = max(abs(resolved_dt_s) * 1.0e-4, 1.0e-7)
    expected_checkpoint_time_s = float(checkpoint_step) * resolved_dt_s
    if abs(checkpoint_time - expected_checkpoint_time_s) > tolerance_s:
        raise ValueError(
            "checkpoint time_s does not match completed_step * dt_s: "
            f"checkpoint={checkpoint_time:.9g} "
            f"expected={expected_checkpoint_time_s:.9g}"
        )
    if checkpoint_step == 0:
        if rows:
            raise ValueError("resume history must be empty for a zero-step checkpoint")
        return
    if len(rows) != checkpoint_step:
        raise ValueError(
            "resume history row count must equal the checkpointed step count after truncation: "
            f"len(history)={len(rows)} checkpoint={checkpoint_step}"
        )
    for expected_step, row in enumerate(rows, start=1):
        try:
            raw_step = row["step"]
            history_step = (
                int(raw_step)
                if isinstance(raw_step, str)
                else _checkpoint_integral_number(
                    raw_step,
                    field=f"resume history row {expected_step} step",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"resume history row {expected_step} step must be an integer"
            ) from exc
        if history_step != expected_step:
            raise ValueError(
                f"resume history row {expected_step} step is out of sequence: "
                f"got {history_step}"
            )
    try:
        history_time_s = float(rows[-1]["time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume history final row must contain finite time_s") from exc
    if not math.isfinite(history_time_s):
        raise ValueError("resume history final row must contain finite time_s")
    if abs(history_time_s - checkpoint_time) > tolerance_s:
        raise ValueError(
            "resume history final row time_s does not match checkpoint: "
            f"history={history_time_s:.9g} checkpoint={checkpoint_time:.9g}"
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

def _checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _checkpoint_source_inputs(
    source_config_path: str | Path,
    *,
    reduced_obstacles_enabled: bool,
) -> dict[str, object]:
    config_path = Path(source_config_path).resolve()
    result: dict[str, object] = {
        "source_config_path": str(config_path),
        "source_config_sha256": None,
        "geometry_files": {},
    }
    if not config_path.is_file():
        return result

    raw_config = config_path.read_bytes()
    result["source_config_sha256"] = hashlib.sha256(raw_config).hexdigest()
    try:
        config = json.loads(raw_config.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"source config is not valid UTF-8 JSON: {config_path}") from exc
    if not isinstance(config, Mapping):
        raise ValueError(f"source config must contain a JSON object: {config_path}")

    geometry_files: dict[str, dict[str, str]] = {}
    for field_name in ("mesh_path", "surface_mesh_cache_path"):
        raw_path = config.get(field_name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint fingerprint input {field_name} not found: {path}"
            )
        geometry_files[field_name] = {
            "path": str(path),
            "sha256": _checkpoint_file_sha256(path),
        }
    analysis = config.get("analysis_settings", {})
    surface_only_region_ids = (
        _selection_ids_as_int_tuple(
            analysis.get("solid_obstacle_surface_only_region_ids", ())
        )
        if isinstance(analysis, Mapping)
        else ()
    )
    if (
        reduced_obstacles_enabled
        and source_config_requests_fluid_active_mask(config)
        and not surface_only_region_ids
    ):
        volume_cache = source_config_volume_particle_cache_path(config_path)
        geometry_files["volume_particle_cache_path"] = {
            "path": str(volume_cache.resolve()),
            "sha256": _checkpoint_file_sha256(volume_cache),
        }
    result["geometry_files"] = geometry_files
    return result

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
    if (
        arg_payload.get("solid_model") == "neo_hookean_mpm"
        and arg_payload.get("neo_fixed_node_lock_policy") is None
    ):
        # Fingerprint the effective runtime policy, so the case default and an
        # explicit equivalent value remain resume-compatible.
        arg_payload["neo_fixed_node_lock_policy"] = "pure_fixed_mass"
    if arg_payload.get("source_config") is not None:
        arg_payload["source_config"] = str(Path(str(arg_payload["source_config"])).resolve())
    payload = {
        "requested_steps": int(step_count),
        "full_pressure_waveform_steps": int(full_pressure_waveform_steps),
        "spec": spec_payload,
        "args": arg_payload,
        "source_inputs": _checkpoint_source_inputs(
            spec.source_config_path,
            reduced_obstacles_enabled=not bool(
                arg_payload.get("disable_reduced_obstacles")
            ),
        ),
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
    frozen_run_fingerprint: Mapping[str, object] | None = None,
) -> None:
    actual = metadata.get("run_fingerprint")
    expected = (
        frozen_run_fingerprint
        if frozen_run_fingerprint is not None
        else checkpoint_run_fingerprint(
            args=args,
            spec=spec,
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
        )
    )
    if _checkpoint_resume_physical_fingerprint(
        actual
    ) != _checkpoint_resume_physical_fingerprint(expected):
        raise ValueError(
            "checkpoint run fingerprint does not match current configuration; "
            "restart with the same source config, pressure schedule, grid, solver, "
            "solid, and FSI options"
        )

def validate_frozen_checkpoint_run_fingerprint(
    frozen_run_fingerprint: Mapping[str, object],
    *,
    args: argparse.Namespace,
    spec: SquidReducedSpec,
    step_count: int,
    full_pressure_waveform_steps: int,
) -> None:
    current = checkpoint_run_fingerprint(
        args=args,
        spec=spec,
        step_count=step_count,
        full_pressure_waveform_steps=full_pressure_waveform_steps,
    )
    if _checkpoint_normalized_value(
        frozen_run_fingerprint
    ) != _checkpoint_normalized_value(current):
        raise RuntimeError(
            "checkpoint source inputs changed during initialization; "
            "restart setup with stable source files"
        )

def _array_to_payload(payload: dict[str, np.ndarray], name: str, value: object) -> None:
    array = np.asarray(value).copy()
    if array.dtype.kind in "biufc" and not bool(np.all(np.isfinite(array))):
        raise ValueError(f"checkpoint {name!r} must be finite")
    payload[name] = array


def _write_checkpoint_payload_atomically(
    path: Path,
    payload: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as temporary_file:
        temp_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temp_path, **payload)
        with np.load(temp_path, allow_pickle=False) as stored:
            if set(stored.files) != set(payload):
                raise ValueError("checkpoint temporary archive has an invalid field set")
            for name, expected in payload.items():
                actual = np.asarray(stored[name])
                if (
                    actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                    or not np.array_equal(actual, expected)
                ):
                    raise ValueError(
                        f"checkpoint temporary archive failed validation for {name!r}"
                    )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)

def _read_scalar_field(field: ti.template()) -> float:
    return float(field[None])

def _write_scalar_field(field: ti.template(), value: object) -> None:
    field[None] = float(np.asarray(value))


def sharp_marker_state_arrays(markers) -> dict[str, np.ndarray]:
    """Capture the dynamic sharp marker state used by a fixed-point trial."""
    return capture_marker_interface_state(markers)


def restore_sharp_marker_state_arrays(
    markers,
    state: Mapping[str, object],
) -> None:
    """Restore a marker state captured by :func:`sharp_marker_state_arrays`."""
    restore_marker_interface_state(markers, state)


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
    """Measure marker fixed-point mismatch in velocity units."""
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
        "secondary_region_l2_mps": _marker_group_l2_mps(
            marker_norms,
            secondary_mask,
        ),
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
    for name in MARKER_INTERFACE_STATE_FIELDS:
        guess_array = _sharp_marker_state_array(guess, name)
        candidate_array = _sharp_marker_state_array(
            candidate,
            name,
            expected_shape=tuple(guess_array.shape),
        )
        next_array = guess_array + omega * (candidate_array - guess_array)
        if name == "A_gamma_m2":
            next_array = np.maximum(next_array, 0.0)
        elif name == "n_gamma":
            norms = np.linalg.norm(next_array, axis=1)
            invalid = norms <= 1.0e-12
            safe_norms = np.where(invalid, 1.0, norms)
            next_array = next_array / safe_norms[:, None]
            if np.any(invalid):
                next_array[invalid] = guess_array[invalid]
        relaxed[name] = next_array.astype(
            np.asarray(guess[name]).dtype,
            copy=False,
        )
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

def _checkpoint_value(checkpoint, key: str) -> np.ndarray:
    if key not in checkpoint:
        raise ValueError(f"checkpoint is missing {key!r}")
    return np.asarray(checkpoint[key])

def _checkpoint_scalar(checkpoint, key: str) -> float:
    value = _checkpoint_value(checkpoint, key)
    if value.size != 1:
        raise ValueError(f"checkpoint {key!r} must be a scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"checkpoint {key!r} must be numeric") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"checkpoint {key!r} must be finite")
    return scalar

def _checkpoint_count(checkpoint, key: str) -> int:
    value = _checkpoint_scalar(checkpoint, key)
    if not value.is_integer() or value < 0.0:
        raise ValueError(f"checkpoint {key!r} must be a non-negative integer")
    return int(value)

def _checkpoint_integral_number(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"checkpoint {field} must be an integral non-boolean number")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"checkpoint {field} must be an integral non-boolean number")

def _checkpoint_integral_sequence(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"checkpoint {field} must be a sequence of integers")
    return tuple(
        _checkpoint_integral_number(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )

def _checkpoint_field_array(
    checkpoint,
    key: str,
    field,
    *,
    active_count: int | None = None,
) -> np.ndarray:
    expected = np.asarray(field.to_numpy())
    if active_count is not None:
        expected = expected[:active_count]
    value = _checkpoint_value(checkpoint, key)
    if value.shape != expected.shape:
        raise ValueError(
            f"checkpoint {key!r} shape mismatch: {value.shape} != {expected.shape}"
        )
    try:
        decoded = value.astype(expected.dtype, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"checkpoint {key!r} has an incompatible dtype") from exc
    if not bool(np.all(np.isfinite(decoded))):
        raise ValueError(f"checkpoint {key!r} must be finite")
    return decoded

def write_run_checkpoint(
    path: Path,
    *,
    completed_step: int,
    step_count: int,
    full_pressure_waveform_steps: int,
    args: argparse.Namespace,
    simulator: ReducedSquidFSI,
    solid_mpm: object,
    sharp_coupling_state=None,
    frozen_run_fingerprint: Mapping[str, object] | None = None,
) -> None:
    completed = _checkpoint_integral_number(completed_step, field="completed_step")
    requested_steps = _checkpoint_integral_number(step_count, field="requested_steps")
    waveform_steps = _checkpoint_integral_number(
        full_pressure_waveform_steps,
        field="full_pressure_waveform_steps",
    )
    if requested_steps <= 0 or waveform_steps <= 0:
        raise ValueError("checkpoint step counts must be positive")
    if completed < 0 or completed > requested_steps:
        raise ValueError(
            "checkpoint completed_step must be between zero and requested_steps"
        )
    grid_nodes = _checkpoint_integral_sequence(
        simulator.spec.grid_nodes,
        field="grid_nodes",
    )
    particle_count = _checkpoint_integral_number(
        getattr(solid_mpm, "particle_count", 0),
        field="particle_count",
    )
    payload: dict[str, np.ndarray] = {}
    metadata = {
        "version": RUN_CHECKPOINT_VERSION,
        "completed_step": completed,
        "requested_steps": requested_steps,
        "full_pressure_waveform_steps": waveform_steps,
        "solid_model": str(args.solid_model),
        "grid_nodes": list(grid_nodes),
        "particle_count": particle_count,
        "run_fingerprint": _checkpoint_normalized_value(
            frozen_run_fingerprint
            if frozen_run_fingerprint is not None
            else checkpoint_run_fingerprint(
                args=args,
                spec=simulator.spec,
                step_count=step_count,
                full_pressure_waveform_steps=full_pressure_waveform_steps,
            )
        ),
    }
    try:
        encoded_metadata = json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be finite and JSON-serializable") from exc
    _array_to_payload(payload, "__metadata__", np.asarray(encoded_metadata))

    for name in CHECKPOINT_SIM_SCALAR_FIELD_NAMES:
        _array_to_payload(payload, f"sim_{name}", _read_scalar_field(getattr(simulator, name)))
    for name in CHECKPOINT_SIM_COUNT_FIELD_NAMES:
        _array_to_payload(payload, f"sim_{name}", int(getattr(simulator, name)[None]))
    fluid = simulator.fluid
    for name in CHECKPOINT_FLUID_FIELD_NAMES:
        _array_to_payload(payload, f"fluid_{name}", getattr(fluid, name).to_numpy())

    solid_field_names = CHECKPOINT_SOLID_FIELD_NAMES.get(str(args.solid_model))
    if solid_field_names is None:
        raise ValueError(f"unsupported solid model for checkpoint: {args.solid_model!r}")
    for name in solid_field_names:
        _array_to_payload(payload, f"solid_{name}", getattr(solid_mpm, name).to_numpy())

    if sharp_coupling_state is not None:
        marker_state = capture_marker_interface_state(sharp_coupling_state.markers)
        for name in MARKER_INTERFACE_STATE_FIELDS:
            _array_to_payload(payload, f"marker_{name}", marker_state[name])
    _array_to_payload(
        payload,
        "has_marker_state",
        np.asarray(sharp_coupling_state is not None),
    )

    _write_checkpoint_payload_atomically(path, payload)

def load_run_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    simulator: ReducedSquidFSI,
    solid_mpm: object,
    step_count: int | None = None,
    full_pressure_waveform_steps: int | None = None,
    sharp_coupling_state=None,
    frozen_run_fingerprint: Mapping[str, object] | None = None,
) -> int:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as checkpoint:
        try:
            metadata = json.loads(str(checkpoint["__metadata__"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint metadata is missing or invalid") from exc
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        version = _checkpoint_integral_number(
            metadata.get("version"),
            field="version",
        )
        if version != RUN_CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {metadata.get('version')!r}"
            )
        if str(metadata.get("solid_model")) != str(args.solid_model):
            raise ValueError(
                "checkpoint solid model does not match --solid-model: "
                f"{metadata.get('solid_model')!r} != {args.solid_model!r}"
            )
        grid_nodes = _checkpoint_integral_sequence(
            metadata.get("grid_nodes"),
            field="grid_nodes",
        )
        if grid_nodes != tuple(int(value) for value in simulator.spec.grid_nodes):
            raise ValueError("checkpoint grid shape does not match current configuration")
        particle_count = _checkpoint_integral_number(
            metadata.get("particle_count"),
            field="particle_count",
        )
        if particle_count != int(getattr(solid_mpm, "particle_count", 0)):
            raise ValueError("checkpoint solid particle count does not match current configuration")
        stored_requested_steps = _checkpoint_integral_number(
            metadata.get("requested_steps"),
            field="requested_steps",
        )
        if stored_requested_steps <= 0:
            raise ValueError("checkpoint requested_steps must be positive")
        effective_requested_steps = _checkpoint_integral_number(
            stored_requested_steps if step_count is None else step_count,
            field="requested_steps",
        )
        if effective_requested_steps <= 0:
            raise ValueError("checkpoint requested_steps must be positive")
        stored_waveform_steps = _checkpoint_integral_number(
            metadata.get("full_pressure_waveform_steps"),
            field="full_pressure_waveform_steps",
        )
        effective_waveform_steps = _checkpoint_integral_number(
            (
                stored_waveform_steps
                if full_pressure_waveform_steps is None
                else full_pressure_waveform_steps
            ),
            field="full_pressure_waveform_steps",
        )
        if stored_waveform_steps <= 0 or effective_waveform_steps <= 0:
            raise ValueError(
                "checkpoint full_pressure_waveform_steps must be positive"
            )
        validate_checkpoint_run_fingerprint(
            metadata,
            args=args,
            spec=simulator.spec,
            step_count=effective_requested_steps,
            full_pressure_waveform_steps=effective_waveform_steps,
            frozen_run_fingerprint=frozen_run_fingerprint,
        )
        completed_step = _checkpoint_integral_number(
            metadata.get("completed_step"),
            field="completed_step",
        )
        if completed_step < 0:
            raise ValueError("checkpoint completed_step must be non-negative")
        if completed_step >= effective_requested_steps:
            raise ValueError(
                "checkpoint completed_step must be less than requested_steps: "
                f"{completed_step} >= {effective_requested_steps}"
            )
        solid_field_names = CHECKPOINT_SOLID_FIELD_NAMES.get(str(args.solid_model))
        if solid_field_names is None:
            raise ValueError(
                f"unsupported solid model for checkpoint: {args.solid_model!r}"
            )

        scalar_values = {
            name: _checkpoint_scalar(checkpoint, f"sim_{name}")
            for name in CHECKPOINT_SIM_SCALAR_FIELD_NAMES
        }
        count_values = {
            name: _checkpoint_count(checkpoint, f"sim_{name}")
            for name in CHECKPOINT_SIM_COUNT_FIELD_NAMES
        }
        fluid = simulator.fluid
        fluid_values = {
            name: _checkpoint_field_array(
                checkpoint,
                f"fluid_{name}",
                getattr(fluid, name),
            )
            for name in CHECKPOINT_FLUID_FIELD_NAMES
        }
        solid_values = {
            name: _checkpoint_field_array(
                checkpoint,
                f"solid_{name}",
                getattr(solid_mpm, name),
            )
            for name in solid_field_names
        }

        marker_values: dict[str, object] = {}
        if sharp_coupling_state is not None:
            missing_marker_keys = [
                f"marker_{name}"
                for name in MARKER_INTERFACE_STATE_FIELDS
                if f"marker_{name}" not in checkpoint
            ]
            if missing_marker_keys:
                raise ValueError(
                    "checkpoint does not contain HIBM sharp marker state "
                    f"(missing {', '.join(missing_marker_keys)}); resuming a "
                    "sharp-coupling run from it would rebuild the immersed "
                    "boundary from rest geometry against a deformed fluid state"
                )
            marker_count = int(sharp_coupling_state.markers.marker_count)
            marker_values = capture_marker_interface_state(
                sharp_coupling_state.markers
            )
            marker_values.update(
                {
                    name: _checkpoint_field_array(
                        checkpoint,
                        f"marker_{name}",
                        getattr(sharp_coupling_state.markers, name),
                        active_count=marker_count,
                    )
                    for name in MARKER_INTERFACE_STATE_FIELDS
                }
            )

    # Commit only after every metadata field and array has been decoded and
    # validated. Malformed checkpoints therefore leave the live solver intact.
    for name, value in scalar_values.items():
        _write_scalar_field(getattr(simulator, name), value)
    for name, value in count_values.items():
        getattr(simulator, name)[None] = value
    for name, value in fluid_values.items():
        getattr(fluid, name).from_numpy(value)
    fluid.pressure_tmp.from_numpy(fluid_values["pressure"])
    fluid.pressure_accum.from_numpy(fluid_values["pressure"])
    for name, value in solid_values.items():
        getattr(solid_mpm, name).from_numpy(value)
    if sharp_coupling_state is not None:
        restore_marker_interface_state(
            sharp_coupling_state.markers,
            marker_values,
        )
    return completed_step
