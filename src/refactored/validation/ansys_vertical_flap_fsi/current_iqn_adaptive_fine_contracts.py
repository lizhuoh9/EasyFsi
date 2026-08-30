"""Exact method identity and raw-trial evidence for current IQN/adaptive fine50.

Shared native comparator gates still own pressure, projection, Fluent inputs,
completeness and the existing five-percent diagnostic. This module alone never
certifies numerical success or Fluent parity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .native_fine_final_contracts import (
    FINAL_FINE_CONFIG_IDENTITY,
    FINAL_FINE_EXPORT_IDENTITY,
    _identity_values_equal,
)

PROFILE_ID = "current_iqn_adaptive_fine50_v1"
EXPECTED_STEPS = 50
EXPECTED_DT_S = 5.0e-4
CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY = {
    **FINAL_FINE_CONFIG_IDENTITY,
    "dt_s": EXPECTED_DT_S,
    "solid_substeps": None,
    "coupling_mode": "iqn_ils",
    "initial_guess_mode": "carry_forward",
    "fsi_coupling_max_iterations": 16,
    "fsi_coupling_absolute_tolerance_mps": 0.0,
    "fsi_coupling_relative_tolerance": 1.0e-3,
    "iqn_history_limit": 8,
    "iqn_initial_picard_relaxation": 0.5,
    "iqn_svd_relative_cutoff": 1.0e-10,
    "iqn_reuse_previous_step_history": False,
    "kalman_writeback_mode": "off",
    "traction_marker_layout": "dual_physical_faces",
    "flow_hibm_marker_compatibility_closure_tolerance_mps": 1.1e-6,
}
IQN_TRIAL_VECTOR_FRAME_KEYS = (
    "iqn_trial_guess_mps", "iqn_trial_candidate_mps", "iqn_trial_residual_mps",
    "iqn_trial_index", "iqn_trial_layout_sha256", "iqn_trial_step",
    "iqn_trial_time_s", "iqn_trial_dt_s",
)
PHYSICAL_MARKER_FRAME_KEYS = (
    "marker_position_m", "marker_velocity_mps", "marker_normal",
    "marker_area_m2", "marker_region_id",
)
PROFILE_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    {"profile": PROFILE_ID, "config": CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY,
     "export": FINAL_FINE_EXPORT_IDENTITY},
    sort_keys=True, separators=(",", ":"),
).encode("utf-8")).hexdigest()


class CurrentIqnAdaptiveFineContractError(RuntimeError):
    """Artifacts do not establish the declared current-method contract."""


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise CurrentIqnAdaptiveFineContractError(label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool)
             and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _time(value: Any, expected: float, label: str) -> None:
    # Accumulation-roundoff accounting only, never a residual tolerance.
    _require(math.isclose(_number(value, label), expected, rel_tol=0.0, abs_tol=1e-12),
             f"{label} does not consume the complete physical time")


def _array(value: Any, label: str, *, dtype=None, shape=None) -> np.ndarray:
    _require(isinstance(value, np.ndarray), f"{label} must be an ndarray")
    if dtype is not None:
        _require(value.dtype == np.dtype(dtype), f"{label} must have dtype {dtype}")
    if shape is not None:
        _require(value.shape == shape, f"{label} has an invalid shape")
    return value


def _physical_markers(frame: Mapping[str, Any], count: int, step: int) -> np.ndarray:
    _require(set(PHYSICAL_MARKER_FRAME_KEYS) <= set(frame),
             f"IQN frame {step} lacks independent physical marker arrays")
    for key in PHYSICAL_MARKER_FRAME_KEYS[:-1]:
        shape = (count,) if key == "marker_area_m2" else (count, 3)
        values = _array(frame[key], key, shape=shape)
        _require(values.dtype.kind == "f" and np.all(np.isfinite(values)),
                 f"IQN frame {step} has invalid physical {key}")
    regions = _array(frame["marker_region_id"], "marker_region_id", shape=(count,))
    _require(regions.dtype.kind in "iu", "physical marker regions must be integral")
    _require(np.all(frame["marker_area_m2"] > 0.0), "physical marker areas must be positive")
    _require(np.all(np.linalg.norm(frame["marker_normal"], axis=1) > 0.0),
             "physical marker normals must be nonzero")
    return regions


def validate_iqn_trial_vector_frame(
    frame: Mapping[str, Any], *, step: int, marker_count: int,
    layout_sha256: str | None,
) -> dict[str, Any]:
    _require(set(IQN_TRIAL_VECTOR_FRAME_KEYS) <= set(frame),
             f"IQN frame {step} lacks required raw trial arrays")
    regions = _physical_markers(frame, marker_count, step)
    guess, candidate, residual = (
        _array(frame[key], key, dtype=np.float64)
        for key in IQN_TRIAL_VECTOR_FRAME_KEYS[:3]
    )
    _require(guess.ndim == 3 and 1 <= guess.shape[0] <= 16
             and guess.shape[1:] == (marker_count, 3)
             and candidate.shape == guess.shape and residual.shape == guess.shape,
             f"IQN frame {step} has invalid (T,M,3) shape")
    _require(all(np.all(np.isfinite(x)) for x in (guess, candidate, residual))
             and np.array_equal(residual, candidate - guess),
             f"IQN frame {step} residual must equal candidate minus guess")
    indices = _array(frame["iqn_trial_index"], "iqn_trial_index", dtype=np.int64,
                     shape=(len(guess),))
    _require(np.array_equal(indices, np.arange(len(guess), dtype=np.int64)),
             f"IQN frame {step} indices must be contiguous")
    raw_step = _array(frame["iqn_trial_step"], "iqn_trial_step", dtype=np.int64, shape=())
    _require(raw_step.item() == step, f"IQN frame {step} step identity changed")
    for key, expected in (("iqn_trial_time_s", step * EXPECTED_DT_S),
                          ("iqn_trial_dt_s", EXPECTED_DT_S)):
        _time(_array(frame[key], key, dtype=np.float64, shape=()).item(), expected, key)
    raw_layout = _array(frame["iqn_trial_layout_sha256"], "IQN layout", shape=())
    _require(raw_layout.dtype.kind == "U", "IQN layout must be a Unicode scalar")
    layout = raw_layout.item()
    _require(re.fullmatch(r"[0-9a-f]{64}", layout) is not None
             and (layout_sha256 is None or layout == layout_sha256),
             f"IQN frame {step} layout identity changed")
    _require(np.array_equal(candidate[-1], frame["marker_velocity_mps"]),
             f"IQN frame {step} accepted candidate differs from physical marker velocity")
    with np.errstate(over="ignore", invalid="ignore"):
        residual_rms = np.sqrt(np.mean(np.sum(residual**2, axis=2), axis=1))
        candidate_rms = np.sqrt(np.mean(np.sum(candidate**2, axis=2), axis=1))
    _require(np.all(np.isfinite(residual_rms)) and np.all(np.isfinite(candidate_rms)),
             f"IQN frame {step} RMS overflow")
    effective = 1e-3 * np.maximum(candidate_rms, 1e-30)
    return {"T": len(guess), "M": marker_count, "layout": layout,
            "regions": regions.tolist(), "residual": residual_rms.tolist(),
            "candidate": candidate_rms.tolist(), "effective": effective.tolist()}


def _history_vectors(history: Mapping[str, Any], trace: Mapping[str, Any], step: int) -> None:
    count = trace["T"]
    used = _integer(history.get("hibm_fsi_coupling_iterations_used"), "iterations", minimum=1)
    _require(used == count, f"history {step} iterations disagree with raw trial count")
    for key, trace_key in (
        ("hibm_fsi_coupling_residual_history_mps", "residual"),
        ("hibm_fsi_coupling_candidate_velocity_rms_history_mps", "candidate"),
        ("hibm_fsi_coupling_effective_tolerance_history_mps", "effective"),
    ):
        values = history.get(key)
        _require(isinstance(values, list) and len(values) == count, f"history {step} invalid {key}")
        numeric = [_number(value, key) for value in values]
        _require(np.allclose(numeric, trace[trace_key], rtol=1e-12, atol=0.0),
                 f"history {step} {key} disagrees with independently recomputed raw trials")
    _require(trace["residual"][-1] <= trace["effective"][-1],
             f"history {step} final raw residual is above the unchanged tolerance")
    _require(all(r > t for r, t in zip(trace["residual"][:-1], trace["effective"][:-1])),
             f"history {step} continued after an already-converged trial")
    modes = history.get("hibm_fsi_coupling_update_mode_history")
    ranks = history.get("hibm_fsi_coupling_iqn_rank_history")
    _require(isinstance(modes, list) and len(modes) == count - 1
             and all(mode in {"picard", "iqn_ils"} for mode in modes),
             f"history {step} has invalid non-reuse update modes")
    _require(isinstance(ranks, list) and len(ranks) == count - 1,
             f"history {step} has invalid rank history")
    for rank, mode in zip(ranks, modes):
        _require(_integer(rank, "IQN rank") <= 8 and (mode != "iqn_ils" or rank > 0),
                 f"history {step} has impossible IQN rank")
    fallback = _integer(history.get("hibm_fsi_coupling_iqn_fallback_count"), "IQN fallback count")
    _require(fallback <= modes.count("picard"), f"history {step} invalid IQN fallback count")


def _history(history: Mapping[str, Any], trace: Mapping[str, Any], step: int) -> None:
    _require(_integer(history.get("step"), "history step", minimum=1) == step,
             f"history {step} step identity changed")
    for key, expected in (
        ("time_s", step * EXPECTED_DT_S), ("requested_macro_dt_s", EXPECTED_DT_S),
        ("fluid_accepted_time_s", EXPECTED_DT_S), ("solid_accepted_time_s", EXPECTED_DT_S),
    ):
        _time(history.get(key), expected, f"history {step} {key}")
    for key in ("fluid_remaining_unadvanced_time_s", "solid_remaining_unadvanced_time_s"):
        _require(_number(history.get(key), key) == 0.0, f"history {step} has unadvanced physical time")
    accepted = _integer(history.get("solid_accepted_substep_count"), "accepted substeps", minimum=1)
    selected = _integer(history.get("solid_substeps_selected"), "selected substeps", minimum=1)
    _require(accepted == selected, f"history {step} solid substep count mismatch")
    _time(_number(history.get("solid_substep_dt_s"), "solid substep dt") * accepted,
          EXPECTED_DT_S, f"history {step} solid time sum")
    _require(history.get("hibm_coupling_scheme") == "iterative_marker_velocity_iqn_ils"
             and history.get("hibm_fsi_coupling_explicit_single_pass") is False
             and history.get("hibm_fsi_coupling_converged") is True,
             f"history {step} is not a converged IQN route")
    reuse = _mapping(history.get("hibm_iqn_reuse"), "IQN reuse report")
    _require(reuse.get("enabled") is False and reuse.get("used") is False,
             f"history {step} reused previous-step IQN history")
    _history_vectors(history, trace, step)


def validate_current_iqn_adaptive_fine50(
    manifest: Mapping[str, Any], summary: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    trial_frames: Sequence[Mapping[str, Any]] | Callable[[int], Mapping[str, Any]],
    *, pressure_semantics_mode: str,
) -> dict[str, Any]:
    _require(pressure_semantics_mode == "strict", "current IQN profile requires strict pressure")
    config = _mapping(manifest.get("config"), "manifest config")
    for key, expected in CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY.items():
        _require(key in config and _identity_values_equal(config[key], expected),
                 f"current IQN fine identity mismatch for config {key}")
    for key, expected in (
        ("status", "completed"), ("step_count_requested", 50), ("step_count_completed", 50),
        ("kalman_writeback_mode", "off"), ("kalman_modified_physics", False),
    ):
        _require(_identity_values_equal(summary.get(key), expected), f"summary {key} mismatch")
    _time(summary.get("dt_s"), EXPECTED_DT_S, "summary dt")
    _time(summary.get("final_time_s"), EXPECTED_STEPS * EXPECTED_DT_S, "summary final time")
    export = _mapping(summary.get("solver_npz_summary"), "solver export")
    for key, expected in FINAL_FINE_EXPORT_IDENTITY.items():
        _require(_identity_values_equal(export.get(key), expected), f"solver export {key} mismatch")
    _require(len(histories) == EXPECTED_STEPS, "current IQN profile requires 50 histories")
    if not callable(trial_frames):
        _require(len(trial_frames) == EXPECTED_STEPS, "current IQN profile requires 50 trial frames")
    marker_count = 2 * config["marker_count"]
    layout = None
    regions = None
    traces = []
    for step, history in enumerate(histories, 1):
        frame = trial_frames(step) if callable(trial_frames) else trial_frames[step - 1]
        trace = validate_iqn_trial_vector_frame(
            _mapping(frame, f"frame {step}"), step=step,
            marker_count=marker_count, layout_sha256=layout,
        )
        _require(regions is None or regions == trace["regions"], "physical marker region order changed")
        layout, regions = trace["layout"], trace["regions"]
        _history(_mapping(history, f"history {step}"), trace, step)
        traces.append({key: value for key, value in trace.items() if key != "regions"})
    return {
        "schema": "current_iqn_adaptive_fine50_identity_v3", "status": "passed",
        "comparison_profile": PROFILE_ID, "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "legacy_final_identity_satisfied": False, "legacy_final_acceptance_claimed": False,
        "requires_iqn_trial_vectors": True, "physical_marker_count": marker_count,
        "physical_marker_count_cross_check": "config_and_exported_arrays",
        "marker_layout_sha256": layout, "trial_trace_reports": traces,
    }
