"""Fail-closed evidence contracts for the R24B oracle-headroom campaign.

The module is deliberately independent of Taichi.  It audits already-produced
strict-CUDA artifacts and prepares non-deployable oracle blend trajectories;
it never changes the production predictor, solver, or accepted state.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .kalman_oracle_headroom_integrity import (
    production_preflow_source_sha256,
    validate_current_source_files,
    validate_frozen_step_array_shapes,
)


EXPECTED_STEPS = 8
TRIAL_REDUCTION_MIN = 0.10
CG_REDUCTION_MIN = 0.10
WARM_WALL_REDUCTION_MIN = 0.05
MARKER_STATE_NRMSE_MAX = 5.0e-3
FIELD_STATE_NRMSE_MAX = 1.0e-2
NO_SLIP_MAX_MPS = 1.0e-4
CLOSURE_TOLERANCE_MPS = 1.1e-6

_ALLOWED_CONFIG_DIFFERENCES = frozenset(
    {
        "initial_guess_mode",
        "initial_guess_oracle_path",
    }
)
_FROZEN_CONFIG_EXACT = {
    "coupling_mode": "iqn_ils",
    "flow_cg_preconditioner": "fv_multigrid",
    "flow_pressure_solver": "fv_jacobi",
    "fsi_coupling_absolute_tolerance_mps": 0.0,
    "fsi_coupling_max_iterations": 16,
    "grid_nodes": [4, 256, 320],
    "initial_guess_kalman_config": None,
    "iqn_kalman_oracle_interpolation_oracle_path": None,
    "iqn_kalman_oracle_interpolation_target_step": None,
    "iqn_reuse_previous_step_history": False,
    "kalman_writeback_mode": "off",
    "marker_count": 64,
    "solid_particle_counts": [1, 256, 20],
    "solid_substeps": None,
    "step_count": EXPECTED_STEPS,
}
_FROZEN_CONFIG_FLOATS = {
    "dt_s": 5.0e-4,
    "flow_cg_tolerance": 1.0e-6,
    "flow_hibm_marker_compatibility_closure_tolerance_mps": CLOSURE_TOLERANCE_MPS,
    "fsi_coupling_relative_tolerance": 1.0e-3,
    "solid_cfl_target": 0.14,
}
_FIELD_KEYS = ("marker_velocity_mps", "marker_position_m", "solid_position_m")
_FLOW_FIELD_KEYS = ("u", "v", "p", "speed")
_WORK_KEYS = (
    "trial_count",
    "fluid_solve_count",
    "solid_macro_solve_count",
    "cg_iterations_total",
    "flow_momentum_advection_substeps_total",
    "flow_sst_transport_substeps_total",
    "solid_substeps_executed_total",
    "flow_wall_time_s_total",
    "hibm_wall_time_s_total",
    "solid_wall_time_s_total",
)
_REQUIRED_ARTIFACTS = (
    "oracle_source_manifest.json",
    "oracle_step_metrics.csv",
    "oracle_headroom_summary.json",
    "oracle_blend_response.json",
)


class OracleHeadroomContractError(RuntimeError):
    """Raised when campaign evidence cannot prove a required invariant."""


@dataclass(frozen=True)
class _StepEvidence:
    step: int
    frame_path: Path
    history_path: Path
    arrays: Mapping[str, np.ndarray]
    history: Mapping[str, Any]
    layout_sha256: str


@dataclass(frozen=True)
class _RunEvidence:
    root: Path
    repo_root: Path
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    source_sha256: Mapping[str, str]
    steps: tuple[_StepEvidence, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleHeadroomContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleHeadroomContractError(f"JSON object required: {path}")
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OracleHeadroomContractError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def _as_finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleHeadroomContractError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise OracleHeadroomContractError(f"{label} must be finite")
    return number


def _as_int(value: Any, *, label: str) -> int:
    number = _as_finite_float(value, label=label)
    integer = int(number)
    if number != integer:
        raise OracleHeadroomContractError(f"{label} must be an integer")
    return integer


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleHeadroomContractError(message)


def _validate_frozen_config(config: Mapping[str, Any], *, label: str) -> None:
    for key, expected in _FROZEN_CONFIG_EXACT.items():
        actual = config.get(key)
        _require(
            type(actual) is type(expected) and actual == expected,
            f"{label} frozen config {key} must be {expected!r}",
        )
    for key, expected in _FROZEN_CONFIG_FLOATS.items():
        actual = _as_finite_float(config.get(key), label=f"{label} config {key}")
        _require(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15),
            f"{label} frozen config {key} must be {expected!r}",
        )


def _validate_requested_runtime(payload: Mapping[str, Any], *, label: str) -> None:
    runtime = payload.get("taichi_runtime")
    _require(isinstance(runtime, dict), f"{label} runtime identity missing")
    expected = {
        "default_fp": "f32",
        "random_seed": 0,
        "requested_arch": "cuda",
        "strict_arch": True,
    }
    for key, value in expected.items():
        actual = runtime.get(key)
        _require(
            type(actual) is type(value) and actual == value,
            f"{label} runtime identity {key} must be {value!r}",
        )


def _validate_summary_identity(summary: Mapping[str, Any], *, label: str) -> None:
    runtime = summary.get("taichi_runtime_identity")
    _require(isinstance(runtime, dict), f"{label} runtime identity missing")
    expected_runtime = {
        "actual_arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "requested_arch": "cuda",
        "strict_arch_verified": True,
    }
    for key, value in expected_runtime.items():
        actual = runtime.get(key)
        _require(
            type(actual) is type(value) and actual == value,
            f"{label} runtime identity {key} must be {value!r}",
        )
    compiler = runtime.get("compiler_configuration")
    _require(
        isinstance(compiler, dict)
        and compiler.get("taichi_version") == "1.7.4",
        f"{label} runtime identity requires Taichi 1.7.4",
    )

    grid = summary.get("grid")
    _require(
        isinstance(grid, dict) and grid.get("grid_nodes") == [4, 256, 320],
        f"{label} summary identity grid_nodes mismatch",
    )
    expected_summary = {
        "hibm_coupling_scheme": "iterative_marker_velocity_iqn_ils",
        "kalman_modified_physics": False,
        "kalman_writeback_mode": "off",
        "marker_count": 64,
        "solid_particle_counts": [1, 256, 20],
        "solid_substeps": None,
        "solid_substeps_mode": "adaptive",
    }
    for key, value in expected_summary.items():
        actual = summary.get(key)
        _require(
            type(actual) is type(value) and actual == value,
            f"{label} summary identity {key} must be {value!r}",
        )


def _resolve_run_path(run: _RunEvidence, value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        repo_text = run.manifest.get("repo_root")
        _require(
            isinstance(repo_text, str) and repo_text,
            f"{label} repo root missing",
        )
        repo_root = Path(repo_text).expanduser()
        _require(repo_root.is_absolute(), f"{label} repo root must be absolute")
        path = repo_root / path
    return path.resolve()


def _exact_numbered_files(root: Path, directory: str, suffix: str) -> tuple[Path, ...]:
    folder = root / directory
    _require(folder.is_dir(), f"missing {directory} directory: {root}")
    actual = tuple(sorted(folder.glob(f"step_*{suffix}")))
    expected = tuple(
        folder / f"step_{step:04d}{suffix}" for step in range(1, EXPECTED_STEPS + 1)
    )
    _require(actual == expected, f"{directory} must contain exact8 numbered artifacts")
    return actual


def _normalised_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    _require(reference.shape == candidate.shape, "accepted-state shape mismatch")
    _require(np.all(np.isfinite(reference)), "non-finite reference accepted state")
    _require(np.all(np.isfinite(candidate)), "non-finite candidate accepted state")
    rmse = float(np.sqrt(np.mean(np.square(candidate - reference))))
    scale = max(float(np.sqrt(np.mean(np.square(reference)))), 1.0e-12)
    return rmse / scale


@lru_cache(maxsize=8)
def _preflow_snapshot_identity(prefix_text: str) -> dict[str, Any]:
    prefix = Path(prefix_text).expanduser().resolve()
    manifest_path = Path(f"{prefix}.json")
    manifest = _read_json(manifest_path)
    identity = manifest.get("identity")
    _require(
        isinstance(identity, dict),
        f"preflow snapshot identity missing: {manifest_path}",
    )
    for key in ("config_sha256", "geometry_sha256", "source_sha256"):
        digest = identity.get(key)
        _require(
            isinstance(digest, str) and len(digest) == 64,
            f"preflow snapshot {key} invalid: {manifest_path}",
        )
    npz_name = manifest.get("npz_file")
    _require(
        isinstance(npz_name, str) and npz_name,
        f"preflow snapshot payload missing: {manifest_path}",
    )
    npz_path = (manifest_path.parent / npz_name).resolve()
    _require(npz_path.is_file(), f"preflow snapshot NPZ missing: {npz_path}")
    npz_sha256 = _sha256_file(npz_path)
    declared_npz_sha256 = manifest.get("npz_sha256")
    if declared_npz_sha256 is not None:
        _require(
            declared_npz_sha256 == npz_sha256,
            f"preflow snapshot NPZ SHA mismatch: {npz_path}",
        )
    return {
        "prefix": str(prefix),
        "manifest_sha256": _sha256_file(manifest_path),
        "npz_file": npz_path.name,
        "npz_sha256": npz_sha256,
        "identity": {str(key): str(value) for key, value in sorted(identity.items())},
    }


def _scalar_text(value: np.ndarray, *, label: str) -> str:
    _require(value.size == 1, f"{label} must be scalar")
    item = value.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("ascii")
    _require(isinstance(item, str) and item, f"{label} must be text")
    return item


def _load_step(frame_path: Path, history_path: Path, step: int) -> _StepEvidence:
    required = {
        *_FIELD_KEYS,
        *_FLOW_FIELD_KEYS,
        "iqn_trial_guess_mps",
        "iqn_trial_candidate_mps",
        "iqn_trial_residual_mps",
        "iqn_trial_index",
        "iqn_trial_layout_sha256",
        "iqn_trial_step",
        "iqn_trial_time_s",
        "iqn_trial_dt_s",
    }
    try:
        with np.load(frame_path, allow_pickle=False) as frame:
            missing = required.difference(frame.files)
            _require(not missing, f"step {step} missing arrays: {sorted(missing)}")
            arrays = {name: np.array(frame[name], copy=True) for name in required}
    except (OSError, ValueError) as exc:
        raise OracleHeadroomContractError(f"invalid step frame: {frame_path}") from exc

    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number):
            _require(np.all(np.isfinite(array)), f"step {step} non-finite {name}")
    try:
        validate_frozen_step_array_shapes(arrays)
    except ValueError as exc:
        raise OracleHeadroomContractError(f"step {step} {exc}") from exc
    marker_velocity = arrays["marker_velocity_mps"]
    guesses = arrays["iqn_trial_guess_mps"]
    candidates = arrays["iqn_trial_candidate_mps"]
    residuals = arrays["iqn_trial_residual_mps"]
    _require(
        marker_velocity.ndim == 2 and marker_velocity.shape[1] == 3,
        f"step {step} marker velocity shape invalid",
    )
    _require(
        guesses.ndim == 3 and guesses.shape[1:] == marker_velocity.shape,
        f"step {step} trial guess shape invalid",
    )
    _require(
        candidates.shape == guesses.shape and residuals.shape == guesses.shape,
        f"step {step} trial vector shapes disagree",
    )
    trial_count = guesses.shape[0]
    _require(trial_count > 0, f"step {step} has no IQN trial vectors")
    _require(
        arrays["iqn_trial_index"].shape == (trial_count,),
        f"step {step} trial index shape invalid",
    )
    _require(
        np.array_equal(arrays["iqn_trial_index"], np.arange(trial_count)),
        f"step {step} trial index is not contiguous",
    )
    _require(
        _as_int(arrays["iqn_trial_step"].reshape(()), label="trial step") == step,
        f"step {step} trial metadata disagrees",
    )
    dt_s = _as_finite_float(arrays["iqn_trial_dt_s"].reshape(()), label="trial dt")
    time_s = _as_finite_float(
        arrays["iqn_trial_time_s"].reshape(()), label="trial time"
    )
    _require(dt_s > 0.0, f"step {step} trial dt must be positive")
    _require(
        math.isclose(time_s, step * dt_s, rel_tol=0.0, abs_tol=1.0e-14),
        f"step {step} trial time is not source-matched",
    )

    wrapper = _read_json(history_path)
    _require(
        _as_int(wrapper.get("step_index"), label="history step") == step,
        f"step {step} history index disagrees",
    )
    history = wrapper.get("history")
    _require(isinstance(history, dict), f"step {step} history object missing")
    iterations = _as_int(
        history.get("hibm_fsi_coupling_iterations_used"),
        label=f"step {step} coupling iterations",
    )
    _require(iterations == trial_count, f"step {step} trial vectors/history disagree")
    layout = _scalar_text(
        arrays["iqn_trial_layout_sha256"],
        label=f"step {step} layout SHA",
    )
    _require(len(layout) == 64, f"step {step} layout SHA length invalid")
    return _StepEvidence(
        step=step,
        frame_path=frame_path,
        history_path=history_path,
        arrays=arrays,
        history=history,
        layout_sha256=layout,
    )


def _validate_step_runtime(step: _StepEvidence, *, expected_mode: str) -> None:
    history = step.history
    _require(
        history.get("initial_guess_mode_requested") == expected_mode
        and history.get("initial_guess_mode_used") == expected_mode,
        f"step {step.step} runtime initial guess mode mismatch",
    )
    reuse = history.get("hibm_iqn_reuse")
    _require(isinstance(reuse, dict), f"step {step.step} runtime IQN reuse missing")
    expected_reuse = {
        "enabled": False,
        "imported_pair_count": 0,
        "retained_pair_count": 0,
        "source_step": None,
        "used": False,
    }
    _require(
        all(
            type(reuse.get(key)) is type(value) and reuse.get(key) == value
            for key, value in expected_reuse.items()
        ),
        f"step {step.step} runtime IQN reuse must be inactive",
    )


def _load_run(root: Path | str, *, expected_mode: str) -> _RunEvidence:
    resolved = Path(root).expanduser().resolve()
    _require(resolved.is_dir(), f"run root does not exist: {resolved}")
    manifest = _read_json(resolved / "run_manifest.json")
    progress = _read_json(resolved / "progress.json")
    summary = _read_json(resolved / "our_solver_summary.json")
    config = manifest.get("config")
    _require(isinstance(config, dict), f"run config missing: {resolved}")
    _validate_frozen_config(config, label=str(resolved))
    _validate_requested_runtime(manifest, label=f"{resolved} manifest")
    _validate_requested_runtime(progress, label=f"{resolved} progress")
    _validate_summary_identity(summary, label=str(resolved))

    _require(progress.get("status") == "completed", f"run not completed: {resolved}")
    _require(summary.get("status") == "completed", f"summary not completed: {resolved}")
    for label, value in (
        ("config step_count", config.get("step_count")),
        ("progress completed", progress.get("step_completed")),
        ("summary requested", summary.get("step_count_requested")),
        ("summary completed", summary.get("step_count_completed")),
    ):
        _require(
            _as_int(value, label=label) == EXPECTED_STEPS,
            f"{resolved} is not exact8",
        )
    _require(
        config.get("initial_guess_mode") == expected_mode,
        f"{resolved} initial guess mode must be {expected_mode}",
    )
    if expected_mode == "carry_forward":
        _require(
            config.get("initial_guess_oracle_path") is None,
            f"{resolved} Q0 oracle path must be null",
        )
    _require(
        manifest.get("save_step_fields") is True
        and manifest.get("save_iqn_trial_vectors") is True,
        f"{resolved} must save step fields and IQN trial vectors",
    )
    _require(
        summary.get("profile_wall_time_enabled") is True,
        f"{resolved} must enable wall-time profiling",
    )
    _require(
        manifest.get("profile_wall_time") is True,
        f"{resolved} manifest must enable wall-time profiling",
    )
    _require(
        summary.get("initial_guess_mode") == expected_mode,
        f"{resolved} summary initial guess mode disagrees",
    )

    source_sha256 = manifest.get("source_sha256")
    _require(
        isinstance(source_sha256, dict) and source_sha256,
        f"{resolved} source SHA map missing",
    )
    for name, digest in source_sha256.items():
        _require(isinstance(name, str) and name, f"{resolved} source path invalid")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            f"{resolved} source SHA invalid for {name}",
        )
    normalized_sources = {
        str(key): str(value) for key, value in sorted(source_sha256.items())
    }
    try:
        repo_root = validate_current_source_files(
            manifest.get("repo_root"),
            normalized_sources,
        )
    except (OSError, ValueError) as exc:
        raise OracleHeadroomContractError(f"{resolved} {exc}") from exc

    frames = _exact_numbered_files(resolved, "step_fields", ".npz")
    histories = _exact_numbered_files(resolved, "step_history", ".json")
    steps = tuple(
        _load_step(frame, history, index)
        for index, (frame, history) in enumerate(zip(frames, histories), start=1)
    )
    for step in steps:
        _validate_step_runtime(step, expected_mode=expected_mode)
    layouts = {item.layout_sha256 for item in steps}
    _require(len(layouts) == 1, f"{resolved} layout SHA changes within exact8")
    return _RunEvidence(
        root=resolved,
        repo_root=repo_root,
        manifest=manifest,
        config=config,
        summary=summary,
        source_sha256=normalized_sources,
        steps=steps,
    )


def _config_without_control_surface(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in config.items()
        if key not in _ALLOWED_CONFIG_DIFFERENCES
    }


def _validate_pair(q0: _RunEvidence, q3: _RunEvidence) -> None:
    _require(
        q0.source_sha256 == q3.source_sha256,
        "Q0/Q3 source SHA maps disagree",
    )
    _require(
        _config_without_control_surface(q0.config)
        == _config_without_control_surface(q3.config),
        "Q0/Q3 config differs outside the initial-guess control surface",
    )
    q0_preflow_path = _resolve_run_path(
        q0,
        q0.config.get("preflow_snapshot_input_path"),
        label="Q0 preflow",
    )
    q0_preflow = _preflow_snapshot_identity(str(q0_preflow_path))
    q3_preflow_path = q3.config.get("preflow_snapshot_input_path")
    q3_preflow = _preflow_snapshot_identity(
        str(_resolve_run_path(q3, q3_preflow_path, label="Q3 preflow"))
    )
    _require(
        q0_preflow == q3_preflow,
        "Q0/Q3 preflow snapshot identities disagree",
    )
    for label, run in (("Q0", q0), ("Q3", q3)):
        try:
            expected_source = production_preflow_source_sha256(
                run.repo_root,
                run.source_sha256,
            )
        except (OSError, ValueError) as exc:
            raise OracleHeadroomContractError(
                f"{label} preflow source identity cannot be recomputed: {exc}"
            ) from exc
        _require(
            q0_preflow["identity"]["source_sha256"] == expected_source,
            f"{label} preflow source identity mismatches executable surface",
        )
    oracle_path = q3.config.get("initial_guess_oracle_path")
    _require(
        _resolve_run_path(q3, oracle_path, label="Q3 oracle") == q0.root,
        "Q3 oracle path must resolve to the paired Q0 root",
    )
    for q0_step, q3_step in zip(q0.steps, q3.steps):
        _require(
            q0_step.layout_sha256 == q3_step.layout_sha256,
            f"step {q0_step.step} Q0/Q3 layout SHA differs",
        )
        oracle_guess = q3_step.arrays["iqn_trial_guess_mps"][0]
        accepted = q0_step.arrays["marker_velocity_mps"]
        _require(
            np.array_equal(oracle_guess, accepted),
            f"step {q0_step.step} oracle guess is not Q0 same-step accepted velocity",
        )


def _work_metrics(step: _StepEvidence) -> dict[str, float | int]:
    history = step.history
    work = history.get("hibm_fsi_trial_work_report")
    _require(isinstance(work, dict), f"step {step.step} work report missing")
    metrics: dict[str, float | int] = {}
    for key in _WORK_KEYS:
        value = _as_finite_float(work.get(key), label=f"step {step.step} {key}")
        _require(value >= 0.0, f"step {step.step} {key} must be non-negative")
        if key.endswith("_count") or key.endswith("_total") and "wall_time" not in key:
            metrics[key] = _as_int(value, label=f"step {step.step} {key}")
        else:
            metrics[key] = value
    iterations = _as_int(
        history.get("hibm_fsi_coupling_iterations_used"),
        label=f"step {step.step} coupling iterations",
    )
    _require(
        metrics["trial_count"] == iterations,
        f"step {step.step} work trial count disagrees",
    )
    metrics["coupling_iterations"] = iterations
    metrics["rejected_trials"] = _as_int(
        history.get("hibm_fsi_coupling_rejected_trial_count"),
        label=f"step {step.step} rejected trials",
    )
    metrics["first_absolute_residual_mps"] = _as_finite_float(
        history.get("hibm_fsi_coupling_first_absolute_residual_mps"),
        label=f"step {step.step} first absolute residual",
    )
    metrics["first_relative_residual"] = _as_finite_float(
        history.get("hibm_fsi_coupling_first_relative_residual"),
        label=f"step {step.step} first relative residual",
    )
    metrics["component_wall_s"] = sum(
        float(metrics[key])
        for key in (
            "flow_wall_time_s_total",
            "hibm_wall_time_s_total",
            "solid_wall_time_s_total",
        )
    )
    return metrics


def _physics_health(step: _StepEvidence, *, dt_s: float) -> dict[str, Any]:
    history = step.history
    requested = _as_finite_float(
        history.get("requested_macro_dt_s"),
        label=f"step {step.step} requested macro dt",
    )
    fluid = _as_finite_float(
        history.get("fluid_accepted_time_s"),
        label=f"step {step.step} fluid accepted time",
    )
    solid = _as_finite_float(
        history.get("solid_accepted_time_s"),
        label=f"step {step.step} solid accepted time",
    )
    fluid_remaining = _as_finite_float(
        history.get("fluid_remaining_unadvanced_time_s"),
        label=f"step {step.step} fluid remaining time",
    )
    solid_remaining = _as_finite_float(
        history.get("solid_remaining_unadvanced_time_s"),
        label=f"step {step.step} solid remaining time",
    )
    time_ok = all(
        math.isclose(value, dt_s, rel_tol=1.0e-12, abs_tol=1.0e-14)
        for value in (requested, fluid, solid)
    ) and all(abs(value) <= 1.0e-14 for value in (fluid_remaining, solid_remaining))

    closure_report = history.get("canonical_velocity_dirichlet_report")
    _require(
        isinstance(closure_report, dict),
        f"step {step.step} canonical closure report missing",
    )
    closure = closure_report.get("marker_target_closure")
    _require(
        isinstance(closure, dict),
        f"step {step.step} marker target closure missing",
    )
    closure_residual = _as_finite_float(
        closure.get("final_max_residual_mps"),
        label=f"step {step.step} closure residual",
    )
    closure_tolerance = _as_finite_float(
        closure.get("closure_tolerance_mps"),
        label=f"step {step.step} closure tolerance",
    )
    _require(
        math.isclose(
            closure_tolerance,
            CLOSURE_TOLERANCE_MPS,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        f"step {step.step} closure tolerance must remain frozen",
    )
    history_tolerance = history.get(
        "flow_hibm_marker_compatibility_closure_tolerance_mps"
    )
    if history_tolerance is not None:
        _require(
            math.isclose(
                _as_finite_float(
                    history_tolerance,
                    label=f"step {step.step} history closure tolerance",
                ),
                CLOSURE_TOLERANCE_MPS,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            f"step {step.step} history closure tolerance must remain frozen",
        )
    invalid_closure = _as_int(
        closure.get("projection_only_invalid_axis_count"),
        label=f"step {step.step} invalid closure axes",
    )
    no_slip = _as_finite_float(
        history.get("hibm_no_slip_max_residual_mps"),
        label=f"step {step.step} no-slip residual",
    )
    health = {
        "coupling_converged": history.get("hibm_fsi_coupling_converged") is True,
        "pressure_converged": (
            history.get("flow_projection_cg_converged_all") is True
            and _as_int(
                history.get("flow_projection_cg_breakdown_count"),
                label=f"step {step.step} CG breakdown count",
            )
            == 0
            and history.get("flow_projection_pressure_solve_failed") is False
        ),
        "physical_time_complete": time_ok,
        "out_of_bounds_zero": _as_int(
            history.get("mpm_grid_out_of_bounds_particle_count"),
            label=f"step {step.step} out-of-bounds count",
        )
        == 0,
        "deformation_clamp_zero": _as_int(
            history.get("mpm_deformation_clamp_count"),
            label=f"step {step.step} deformation clamp count",
        )
        == 0,
        "solid_retry_zero": _as_int(
            history.get("solid_retry_count"),
            label=f"step {step.step} solid retry count",
        )
        == 0,
        "no_slip_valid": (
            _as_int(
                history.get("hibm_no_slip_invalid_marker_count"),
                label=f"step {step.step} invalid no-slip markers",
            )
            == 0
            and no_slip <= NO_SLIP_MAX_MPS
        ),
        "closure_valid": (
            invalid_closure == 0
            and closure_residual <= CLOSURE_TOLERANCE_MPS
        ),
    }
    health["all"] = all(bool(value) for value in health.values())
    health["no_slip_max_residual_mps"] = no_slip
    health["closure_max_residual_mps"] = closure_residual
    health["closure_tolerance_mps"] = CLOSURE_TOLERANCE_MPS
    return health


def _reduction(reference: float, candidate: float) -> float:
    _require(reference > 0.0, "Q0 reference metric must be positive")
    return (reference - candidate) / reference
