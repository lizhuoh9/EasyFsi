from __future__ import annotations

from collections.abc import Mapping
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import re
import stat
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").exists():
            return parent
    raise RuntimeError("could not locate repo root from validation script path")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TAICHI_OFFLINE_CACHE_DIR = (
    REPO_ROOT
    / "validation_runs"
    / ".taichi_cache"
    / "ansys_vertical_flap_cuda_f32"
)
TAICHI_OFFLINE_CACHE_MAX_SIZE_BYTES = 512 * 1024 * 1024
TAICHI_OFFLINE_CACHE_CLEANING_POLICY = "lru"
TAICHI_OFFLINE_CACHE_CLEANING_FACTOR = 0.25

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    InterfaceKalmanConfig,
    VerticalFlapFsiConfig,
    run_ansys_vertical_flap_benchmark,
    selected_formulation_solver_config,
    with_local_surface_force_support,
)
from benchmarks.official.solid_mpm_fsi_runner import (  # noqa: E402
    _fsi_checkpoint_config_payload,
    _fsi_checkpoint_paths,
    _preflow_snapshot_source_payload,
)
from simulation_core.drivers import FsiCouplingConvergenceError  # noqa: E402
from simulation_core.coupling.accepted_fsi_checkpoint import load_accepted_fsi_checkpoint  # noqa: E402
from simulation_core.diagnostics.atomic_file import (  # noqa: E402
    publish_file_create_only,
    replace_file_atomically,
)
from simulation_core.diagnostics.checkpoint_store import (  # noqa: E402
    CheckpointHead,
    read_checkpoint_head,
)
from simulation_core.fluids.preflow_snapshot import canonical_source_sha256  # noqa: E402
from simulation_core.diagnostics.run_attempt import (  # noqa: E402
    prepare_resume_attempt,
    require_completed_output,
)
from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (  # noqa: E402
    save_solver_npz_from_flow_snapshot,
)
from src.refactored.validation.ansys_vertical_flap_fsi.kalman_oracle_headroom_contracts import (  # noqa: E402
    _load_run as _load_oracle_headroom_run,
)
from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_lineage import (  # noqa: E402
    q0_probe_identity as _q0_probe_identity,
)


PHYSICAL_SOLID_BOUNDS = {
    "streamwise_min_m": 0.050,
    "streamwise_max_m": 0.053,
    "y_min_m": 0.0,
    "y_max_m": 0.010,
}

STEP_FRAME_STRUCTURE_KEYS = (
    "solid_x_m",
    "solid_y_m",
    "solid_rest_x_m",
    "solid_rest_y_m",
    "solid_vx_mps",
    "solid_vy_mps",
    "solid_position_m",
    "solid_velocity_mps",
    "solid_rest_position_m",
    "solid_fixed_mask",
    "solid_tip_mask",
    "marker_x_m",
    "marker_y_m",
    "marker_position_m",
    "marker_velocity_mps",
    "marker_normal",
    "marker_area_m2",
    "marker_region_id",
)

STEP_FRAME_DIAGNOSTIC_KEYS = (
    "velocity_dirichlet_boundary_active",
    "velocity_dirichlet_boundary_projection_weight",
    "velocity_dirichlet_boundary_enforcement_weight",
    "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_owned_row",
    "velocity_dirichlet_boundary_marker_region_id",
    "flow_solution_stage",
    "boundary_topology_stage",
    "flow_boundary_state_synchronized",
    "structure_geometry_stage",
)

IQN_TRIAL_VECTOR_STEP_KEYS = (
    "iqn_trial_guess_mps",
    "iqn_trial_candidate_mps",
    "iqn_trial_residual_mps",
    "iqn_trial_index",
    "iqn_trial_layout_sha256",
    "iqn_trial_step",
    "iqn_trial_time_s",
    "iqn_trial_dt_s",
    "iqn_trial_marker_reference_positions_m",
)

_JSON_CANONICAL_KEY_ALIASES = {
    "marker_action_reaction_residual_n": "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_n": "scatter_action_reaction_residual_N",
}


def _json_values_equal(left: Any, right: Any) -> bool:
    return json.dumps(
        left,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        right,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        key_by_casefold: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            source_key = str(raw_key)
            key = _JSON_CANONICAL_KEY_ALIASES.get(source_key, source_key)
            safe_value = _json_safe(raw_value)
            folded = key.casefold()
            existing_key = key_by_casefold.get(folded)
            if existing_key is not None:
                if existing_key != key or not _json_values_equal(
                    safe[existing_key], safe_value
                ):
                    raise ValueError(
                        "case-colliding JSON keys disagree: "
                        f"{existing_key!r} and {source_key!r}"
                    )
                continue
            key_by_casefold[folded] = key
            safe[key] = safe_value
        return safe
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    diagnostics = getattr(exc, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return {}
    return dict(_json_safe(diagnostics))


def _fsi_coupling_diagnostics(exc: BaseException) -> dict[str, Any] | None:
    """Return the complete report only for the concrete FSI convergence error."""

    if type(exc) is not FsiCouplingConvergenceError:
        return None
    return dict(
        _json_safe(
            {
                "context": asdict(exc.context),
                "report": asdict(exc.report),
            }
        )
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    owned_descriptor: int | None = descriptor
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        owned_descriptor = None
        with handle:
            handle.write(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        replace_file_atomically(temporary, path)
    except BaseException:
        if owned_descriptor is not None:
            try:
                os.close(owned_descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise


def _record_failure_artifacts(
    *,
    output_dir: Path,
    exc: BaseException,
    elapsed_s: float,
    config_payload: dict[str, Any],
    config: VerticalFlapFsiConfig | None,
) -> None:
    interrupted = isinstance(exc, KeyboardInterrupt)
    exception_status = "interrupted" if interrupted else "failed"
    exception_artifact_name = (
        "interruption.json" if interrupted else "failure.json"
    )
    reporting_errors: list[str] = []
    try:
        exception_diagnostics = _exception_diagnostics(exc)
    except BaseException as diagnostics_exc:
        exception_diagnostics = {}
        reporting_errors.append(
            "exception diagnostics failed: "
            f"{type(diagnostics_exc).__name__}: {diagnostics_exc}"
        )
    try:
        fsi_coupling_diagnostics = _fsi_coupling_diagnostics(exc)
    except BaseException as diagnostics_exc:
        fsi_coupling_diagnostics = None
        reporting_errors.append(
            "FSI coupling diagnostics failed: "
            f"{type(diagnostics_exc).__name__}: {diagnostics_exc}"
        )
    failure_payload = {
        "status": exception_status,
        "elapsed_s": float(elapsed_s),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "pressure_solve_diagnostics": exception_diagnostics,
        "traceback": traceback.format_exc(),
        "config": config_payload,
        "grid": _grid_summary(config) if config is not None else None,
    }
    if fsi_coupling_diagnostics is not None:
        failure_payload["fsi_coupling_diagnostics"] = fsi_coupling_diagnostics
    try:
        _write_json_atomic(
            output_dir / exception_artifact_name,
            failure_payload,
        )
    except BaseException as artifact_exc:
        reporting_errors.append(
            f"{exception_artifact_name} write failed: "
            f"{type(artifact_exc).__name__}: {artifact_exc}"
        )

    progress_path = output_dir / "progress.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        progress = {"step_completed": 0, "time_s": 0.0}
    try:
        progress_payload = {
            **{
                key: value
                for key, value in progress.items()
                if key != "fsi_coupling_diagnostics"
            },
            "status": exception_status,
            "phase": exception_status,
            "elapsed_s": float(elapsed_s),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pressure_solve_diagnostics": exception_diagnostics,
            "reporting_errors": reporting_errors,
        }
        if fsi_coupling_diagnostics is not None:
            progress_payload["fsi_coupling_diagnostics"] = fsi_coupling_diagnostics
        _write_json_atomic(
            progress_path,
            progress_payload,
        )
    except BaseException as progress_exc:
        reporting_errors.append(
            "progress write failed: "
            f"{type(progress_exc).__name__}: {progress_exc}"
        )
    if reporting_errors:
        print(
            json.dumps(
                {
                    "status": "artifact_reporting_degraded",
                    "primary_error_type": type(exc).__name__,
                    "primary_error": str(exc),
                    "reporting_errors": reporting_errors,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    roots = (
        REPO_ROOT / "cases",
        REPO_ROOT / "benchmarks" / "official",
        REPO_ROOT / "simulation_core",
        REPO_ROOT / "src" / "refactored" / "validation" / "ansys_vertical_flap_fsi",
    )
    paths = {
        Path(__file__).resolve(),
        REPO_ROOT / "tools" / "audit_ansys_vertical_flap_oracle_threshold.py",
        REPO_ROOT / "tools" / "validation" / "compare_solid_substep_ab.py",
        (
            REPO_ROOT
            / "tools"
            / "validation"
            / "gru_kalman_live"
            / "__init__.py"
        ),
        (
            REPO_ROOT
            / "tools"
            / "validation"
            / "gru_kalman_live"
            / "candidate_bundle.py"
        ),
    }
    for root in roots:
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    return {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }


def _prepare_output_dir(
    output_dir: Path,
    *,
    resume: bool,
    canonical_output_dir: Path | None = None,
) -> None:
    resolved_output_dir = output_dir.resolve(strict=False)
    if resume:
        if canonical_output_dir is None:
            raise ValueError("checkpoint resume requires --resume-run-dir")
        canonical_resolved = canonical_output_dir.resolve(strict=False)
        if resolved_output_dir == canonical_resolved:
            raise ValueError("resume attempt output directory must be distinct from canonical root")
        if resolved_output_dir.is_relative_to(canonical_resolved):
            raise ValueError("resume attempt output directory must be outside canonical root")
    output_entry = _lstat_or_none(output_dir)
    if output_entry is not None:
        if (
            _is_link_or_reparse_point(output_entry)
            or not stat.S_ISDIR(output_entry.st_mode)
        ):
            raise RuntimeError(f"output path is not a real directory: {output_dir}")
        if resume:
            if any(output_dir.iterdir()):
                raise RuntimeError(
                    "checkpoint resume attempt output directory must be fresh and empty"
                )
            return
        if any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to reuse non-empty output directory: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=False)


def _checkpoint_observer_identity(
    *,
    output_dir: Path,
    span_reduction: str,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float,
    save_iqn_trial_vectors: bool,
) -> str:
    payload = {
        "format": "ansys-vertical-flap-step-observer-v1",
        "output_dir": str(output_dir.resolve()),
        "span_reduction": str(span_reduction),
        "streamwise_velocity_sign": float(streamwise_velocity_sign),
        "reverse_streamwise_axis": bool(reverse_streamwise_axis),
        "streamwise_length_m": float(streamwise_length_m),
        "save_iqn_trial_vectors": bool(save_iqn_trial_vectors),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "ansys-vertical-flap-step-observer-v1:" + hashlib.sha256(encoded).hexdigest()


def _saved_checkpoint_config_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read original run config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("original run config must be a JSON object")
    operational = {
        "step_count",
        "fsi_checkpoint_input_path",
        "fsi_checkpoint_output_path",
        "fsi_checkpoint_expected_generation",
        "preflow_snapshot_input_path",
        "preflow_snapshot_output_path",
    }
    return {key: value for key, value in payload.items() if key not in operational}


def _resume_artifact_steps(directory: Path, suffix: str) -> set[int]:
    root_resolved = _require_real_directory_path(
        directory.parent,
        role="checkpoint canonical root",
    )
    directory_entry = _lstat_or_none(directory)
    if directory_entry is None:
        return set()
    if (
        _is_link_or_reparse_point(directory_entry)
        or not stat.S_ISDIR(directory_entry.st_mode)
    ):
        raise ValueError(
            f"checkpoint artifact path is not a real directory: {directory}"
        )
    directory_resolved = directory.resolve(strict=True)
    if not directory_resolved.is_relative_to(root_resolved):
        raise ValueError(f"checkpoint artifact path escapes canonical root: {directory}")
    pattern = re.compile(rf"step_(\d+){re.escape(suffix)}")
    steps: set[int] = set()
    for artifact in directory.iterdir():
        if not artifact.name.startswith("step_") or not artifact.name.endswith(suffix):
            continue
        match = pattern.fullmatch(artifact.name)
        if match is None:
            raise ValueError(f"invalid checkpoint step artifact: {artifact}")
        step = int(match.group(1))
        entry = _lstat_or_none(artifact)
        if (
            artifact.name != f"step_{step:04d}{suffix}"
            or step in steps
            or entry is None
            or _is_link_or_reparse_point(entry)
            or not stat.S_ISREG(entry.st_mode)
            or not artifact.resolve(strict=True).is_relative_to(root_resolved)
        ):
            raise ValueError(f"invalid checkpoint step artifact: {artifact}")
        steps.add(step)
    return steps


def _validate_resume_artifact_prefix(
    *,
    output_dir: Path,
    accepted_step: int,
    require_iqn_trial_vectors: bool,
    artifact_validator=None,
) -> None:
    if artifact_validator is None:
        artifact_validator = _validate_step_artifacts
    expected = set(range(1, accepted_step))
    allowed = expected | {accepted_step}
    fields = _resume_artifact_steps(output_dir / "step_fields", ".npz")
    histories = _resume_artifact_steps(output_dir / "step_history", ".json")
    if fields - allowed or histories - allowed:
        raise ValueError("checkpoint output contains a forward step artifact")
    if fields & expected != expected or histories & expected != expected:
        raise ValueError("checkpoint output is missing an earlier accepted step artifact")
    if accepted_step > 0:
        try:
            artifact_validator(
                output_dir,
                expected_steps=accepted_step - 1,
                require_iqn_trial_vectors=require_iqn_trial_vectors,
                allow_partial_terminal_step=accepted_step,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError("checkpoint output has unreadable accepted artifacts") from exc


def _preflight_checkpoint_resume(
    *,
    output_dir: Path,
    config: VerticalFlapFsiConfig,
    checkpoint_input_path: Path,
    step_observer: object | None,
    checkpoint_head_reader=read_checkpoint_head,
    checkpoint_loader=load_accepted_fsi_checkpoint,
    artifact_validator=None,
) -> CheckpointHead:
    output_dir = _require_real_directory_path(
        output_dir, role="checkpoint canonical root"
    )
    saved_config = _saved_checkpoint_config_payload(output_dir / "our_solver_config.json")
    current_config = _fsi_checkpoint_config_payload(config)
    if not _json_values_equal(saved_config, current_config):
        raise ValueError("checkpoint resume config changes physics or algorithm settings")
    head = checkpoint_head_reader(checkpoint_input_path)
    if head is None or not isinstance(head.metadata, Mapping):
        raise ValueError("complete accepted FSI checkpoint manifest is required")
    identity = head.metadata.get("identity")
    current_source_sha256 = canonical_source_sha256(
        _preflow_snapshot_source_payload()
    )
    if (
        not isinstance(identity, Mapping)
        or identity.get("source_sha256") != current_source_sha256
    ):
        raise ValueError("checkpoint source identity does not match current source")
    loaded = checkpoint_loader(
        checkpoint_input_path,
        expected_identity=identity,
        target_step_count=int(config.step_count),
        expected_generation=head.generation,
    )
    accepted_step = int(loaded.state.macro_state.accepted_step_index)
    if head.accepted_step != accepted_step:
        raise ValueError("checkpoint head accepted step differs from loaded state")
    if loaded.generation != head.generation:
        raise ValueError("checkpoint head generation differs from loaded state")
    saved_observer_identity = loaded.state.runner_state.get("observer_identity")
    observer_identity = (
        None if step_observer is None else getattr(step_observer, "checkpoint_identity", None)
    )
    if saved_observer_identity != observer_identity:
        raise ValueError("checkpoint observer destination differs from the original output directory")
    if step_observer is not None:
        _validate_resume_artifact_prefix(
            output_dir=output_dir,
            accepted_step=accepted_step,
            require_iqn_trial_vectors=bool(
                getattr(step_observer, "record_iqn_trial_vectors", False)
            ),
            artifact_validator=artifact_validator,
        )
    return head


def _configure_taichi_offline_cache(
    *,
    enabled: bool,
    cache_dir: Path,
) -> dict[str, object]:
    resolved_cache_dir = cache_dir.resolve()
    os.environ["TI_OFFLINE_CACHE"] = "1" if enabled else "0"
    os.environ["SIMULATION_TAICHI_OFFLINE_CACHE"] = (
        "1" if enabled else "0"
    )
    os.environ["TI_OFFLINE_CACHE_MAX_SIZE_OF_FILES"] = str(
        TAICHI_OFFLINE_CACHE_MAX_SIZE_BYTES
    )
    os.environ["TI_OFFLINE_CACHE_CLEANING_POLICY"] = (
        TAICHI_OFFLINE_CACHE_CLEANING_POLICY
    )
    os.environ["TI_OFFLINE_CACHE_CLEANING_FACTOR"] = str(
        TAICHI_OFFLINE_CACHE_CLEANING_FACTOR
    )
    if enabled:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ[
            "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH"
        ] = str(resolved_cache_dir)
    else:
        os.environ.pop(
            "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
            None,
        )
    return {
        "configuration_state": "requested_before_taichi_init",
        "offline_cache_enabled": bool(enabled),
        "offline_cache_file_path": (
            str(resolved_cache_dir) if enabled else ""
        ),
        "offline_cache_max_size_bytes": (
            TAICHI_OFFLINE_CACHE_MAX_SIZE_BYTES
        ),
        "offline_cache_cleaning_policy": (
            TAICHI_OFFLINE_CACHE_CLEANING_POLICY
        ),
        "offline_cache_cleaning_factor": TAICHI_OFFLINE_CACHE_CLEANING_FACTOR,
    }


_PROGRESS_FAILURE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "error",
        "error_type",
        "fsi_coupling_diagnostics",
        "pressure_solve_diagnostics",
        "reporting_errors",
        "traceback",
    }
)


def _merge_progress_event(
    existing: dict[str, object], event: dict[str, object]
) -> dict[str, object]:
    base = (
        {
            key: value
            for key, value in existing.items()
            if key not in _PROGRESS_FAILURE_DIAGNOSTIC_FIELDS
        }
        if event.get("status") == "running"
        else dict(existing)
    )
    return {**base, **event}


def _make_run_progress_observer(
    *,
    output_dir: Path,
):
    def observe(event: dict[str, object]) -> None:
        progress_path = output_dir / "progress.json"
        try:
            existing = json.loads(progress_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            existing = {}
        _write_json_atomic(
            progress_path,
            _merge_progress_event(
                {
                    "status": "running",
                    "phase": "initializing",
                    "step_completed": 0,
                    "time_s": 0.0,
                    **existing,
                },
                event,
            ),
        )

    return observe


def _require_vector_rows(snapshot: dict[str, Any], key: str) -> np.ndarray:
    array = np.asarray(snapshot[key])
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{key} must have shape (count, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key} contains non-finite values")
    return array


def _step_structure_arrays(
    snapshot: dict[str, Any],
    *,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float,
) -> dict[str, np.ndarray]:
    solid_position = _require_vector_rows(snapshot, "solid_position_m")
    solid_velocity = _require_vector_rows(snapshot, "solid_velocity_mps")
    solid_rest = _require_vector_rows(snapshot, "solid_rest_position_m")
    marker_position = _require_vector_rows(snapshot, "marker_position_m")
    marker_velocity = _require_vector_rows(snapshot, "marker_velocity_mps")
    marker_normal = _require_vector_rows(snapshot, "marker_normal")
    solid_count = solid_position.shape[0]
    marker_count = marker_position.shape[0]
    if solid_velocity.shape[0] != solid_count or solid_rest.shape[0] != solid_count:
        raise ValueError("solid position/velocity/rest counts must match")
    if marker_velocity.shape[0] != marker_count or marker_normal.shape[0] != marker_count:
        raise ValueError("marker position/velocity/normal counts must match")

    fixed_mask = np.asarray(snapshot["solid_fixed_mask"], dtype=bool)
    tip_mask = np.asarray(snapshot["solid_tip_mask"], dtype=bool)
    marker_area = np.asarray(snapshot["marker_area_m2"])
    marker_region = np.asarray(snapshot["marker_region_id"])
    if fixed_mask.shape != (solid_count,) or tip_mask.shape != (solid_count,):
        raise ValueError("solid fixed/tip masks must match the solid particle count")
    if marker_area.shape != (marker_count,) or marker_region.shape != (marker_count,):
        raise ValueError("marker area/region arrays must match the marker count")
    if not np.all(np.isfinite(marker_area)):
        raise ValueError("marker_area_m2 contains non-finite values")

    def streamwise(position_z: np.ndarray) -> np.ndarray:
        if reverse_streamwise_axis:
            return float(streamwise_length_m) - position_z
        return position_z.copy()

    return {
        "solid_x_m": streamwise(solid_position[:, 2]),
        "solid_y_m": solid_position[:, 1].copy(),
        "solid_rest_x_m": streamwise(solid_rest[:, 2]),
        "solid_rest_y_m": solid_rest[:, 1].copy(),
        "solid_vx_mps": float(streamwise_velocity_sign) * solid_velocity[:, 2],
        "solid_vy_mps": solid_velocity[:, 1].copy(),
        "solid_position_m": solid_position.copy(),
        "solid_velocity_mps": solid_velocity.copy(),
        "solid_rest_position_m": solid_rest.copy(),
        "solid_fixed_mask": fixed_mask.copy(),
        "solid_tip_mask": tip_mask.copy(),
        "marker_x_m": streamwise(marker_position[:, 2]),
        "marker_y_m": marker_position[:, 1].copy(),
        "marker_position_m": marker_position.copy(),
        "marker_velocity_mps": marker_velocity.copy(),
        "marker_normal": marker_normal.copy(),
        "marker_area_m2": marker_area.copy(),
        "marker_region_id": marker_region.copy(),
    }


def _append_npz_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(array), allow_pickle=False)
            archive.writestr(f"{key}.npy", buffer.getvalue())


def _is_link_or_reparse_point(entry: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return stat.S_ISLNK(entry.st_mode) or bool(
        getattr(entry, "st_file_attributes", 0) & reparse_point
    )


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _require_real_directory_path(path: Path, *, role: str) -> Path:
    entry = _lstat_or_none(path)
    if (
        entry is None
        or _is_link_or_reparse_point(entry)
        or not stat.S_ISDIR(entry.st_mode)
    ):
        raise ValueError(f"{role} must be an existing real directory: {path}")
    return path.resolve(strict=True)


def _real_artifact_directory(root: Path, name: str) -> Path:
    root_entry = _lstat_or_none(root)
    if root_entry is None:
        root.mkdir(parents=True, exist_ok=False)
        root_entry = root.lstat()
    if (
        _is_link_or_reparse_point(root_entry)
        or not stat.S_ISDIR(root_entry.st_mode)
    ):
        raise ValueError(f"accepted artifact root must be a real directory: {root}")
    root_resolved = root.resolve(strict=True)
    directory = root / name
    directory_entry = _lstat_or_none(directory)
    if directory_entry is None:
        directory.mkdir()
        directory_entry = directory.lstat()
    if (
        _is_link_or_reparse_point(directory_entry)
        or not stat.S_ISDIR(directory_entry.st_mode)
    ):
        raise ValueError(f"accepted artifact directory must be a real directory: {directory}")
    resolved = directory.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"accepted artifact directory escapes canonical root: {directory}")
    return resolved


def _require_real_artifact_file(path: Path, root: Path) -> None:
    entry = _lstat_or_none(path)
    if entry is None:
        return
    if _is_link_or_reparse_point(entry) or not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"accepted artifact must be a real file: {path}")
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"accepted artifact escapes canonical root: {path}")


def _arrays_semantically_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _npz_semantically_equal(existing: Path, candidate: Path) -> bool:
    try:
        with np.load(existing, allow_pickle=False) as current, np.load(
            candidate, allow_pickle=False
        ) as proposed:
            if (
                len(current.files) != len(set(current.files))
                or len(proposed.files) != len(set(proposed.files))
            ):
                return False
            if set(current.files) != set(proposed.files):
                return False
            return all(
                _arrays_semantically_equal(
                    np.asarray(current[key]), np.asarray(proposed[key])
                )
                for key in current.files
            )
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _json_semantically_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_semantically_equal(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _load_unique_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key!r}")
            payload[key] = value
        return payload

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("step history must be a JSON object")
    return payload


def _require_existing_history_semantic_match(
    path: Path,
    payload: dict[str, Any],
) -> None:
    if not path.exists():
        return
    try:
        matches = _json_semantically_equal(
            _load_unique_json_object(path), _json_safe(payload)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing accepted step history replay mismatch: {path}") from exc
    if not matches:
        raise ValueError(f"existing accepted step history replay mismatch: {path}")


def _publish_create_or_semantic_same(
    *,
    temporary: Path,
    destination: Path,
    matches_existing,
    artifact_name: str,
) -> None:
    try:
        publish_file_create_only(temporary, destination)
    except FileExistsError:
        if not matches_existing(destination, temporary):
            raise ValueError(
                f"existing accepted {artifact_name} replay mismatch: {destination}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _write_step_history_create_or_semantic_same(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        expected = _json_safe(payload)
        _publish_create_or_semantic_same(
            temporary=temporary,
            destination=path,
            matches_existing=lambda existing, _candidate: _json_semantically_equal(
                _load_unique_json_object(existing), expected
            ),
            artifact_name="step history",
        )
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _save_step_frame_atomic(
    path: Path,
    snapshot: dict[str, Any],
    *,
    span_reduction: str,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float,
    save_iqn_trial_vectors: bool = False,
) -> dict[str, Any]:
    if save_iqn_trial_vectors:
        missing_trial_vectors = sorted(
            set(IQN_TRIAL_VECTOR_STEP_KEYS) - set(snapshot)
        )
        if missing_trial_vectors:
            raise ValueError(
                "accepted IQN step snapshot is missing trial-vector fields: "
                f"{missing_trial_vectors}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        summary = save_solver_npz_from_flow_snapshot(
            temporary,
            snapshot,
            span_reduction=span_reduction,
            streamwise_velocity_sign=float(streamwise_velocity_sign),
            reverse_streamwise_axis=bool(reverse_streamwise_axis),
            physical_solid_bounds=PHYSICAL_SOLID_BOUNDS,
        )
        _append_npz_arrays(
            temporary,
            {
                **_step_structure_arrays(
                    snapshot,
                    streamwise_velocity_sign=streamwise_velocity_sign,
                    reverse_streamwise_axis=reverse_streamwise_axis,
                    streamwise_length_m=streamwise_length_m,
                ),
                **{
                    key: np.asarray(snapshot[key])
                    for key in STEP_FRAME_DIAGNOSTIC_KEYS
                    if key in snapshot
                },
                **{
                    key: np.asarray(snapshot[key])
                    for key in IQN_TRIAL_VECTOR_STEP_KEYS
                    if save_iqn_trial_vectors
                },
            },
        )
        _publish_create_or_semantic_same(
            temporary=temporary,
            destination=path,
            matches_existing=_npz_semantically_equal,
            artifact_name="step frame",
        )
        return {**summary, "path": str(path)}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_step_artifacts(
    output_dir: Path,
    *,
    expected_steps: int,
    require_iqn_trial_vectors: bool = False,
    allow_partial_terminal_step: int | None = None,
) -> dict[str, Any]:
    expected_frame_names = [f"step_{step:04d}.npz" for step in range(1, expected_steps + 1)]
    expected_history_names = [
        f"step_{step:04d}.json" for step in range(1, expected_steps + 1)
    ]
    fields_dir = output_dir / "step_fields"
    history_dir = output_dir / "step_history"
    observed_frames = [
        f"step_{step:04d}.npz"
        for step in sorted(_resume_artifact_steps(fields_dir, ".npz"))
    ]
    observed_histories = [
        f"step_{step:04d}.json"
        for step in sorted(_resume_artifact_steps(history_dir, ".json"))
    ]
    if allow_partial_terminal_step is not None:
        terminal_step = int(allow_partial_terminal_step)
        if terminal_step <= expected_steps:
            raise ValueError("partial terminal step must follow the validated prefix")
        observed_frames = [
            name for name in observed_frames
            if name != f"step_{terminal_step:04d}.npz"
        ]
        observed_histories = [
            name for name in observed_histories
            if name != f"step_{terminal_step:04d}.json"
        ]
    if observed_frames != expected_frame_names or observed_histories != expected_history_names:
        raise RuntimeError(
            "step artifact sequence mismatch: "
            f"frames={observed_frames}, histories={observed_histories}, "
            f"expected_steps={expected_steps}"
        )

    for step, (frame_name, history_name) in enumerate(
        zip(expected_frame_names, expected_history_names),
        start=1,
    ):
        try:
            with np.load(fields_dir / frame_name, allow_pickle=False) as frame:
                missing = sorted(
                    (
                        set(STEP_FRAME_STRUCTURE_KEYS)
                        | set(STEP_FRAME_DIAGNOSTIC_KEYS)
                        | (
                            set(IQN_TRIAL_VECTOR_STEP_KEYS)
                            if require_iqn_trial_vectors
                            else set()
                        )
                    )
                    - set(frame.files)
                )
                if missing:
                    raise ValueError(f"missing required step fields: {missing}")
                for key in frame.files:
                    np.asarray(frame[key])
                if require_iqn_trial_vectors:
                    _validate_iqn_trial_vector_frame(frame, expected_step=step)
        except Exception as exc:
            raise RuntimeError(f"unreadable step frame {frame_name}: {exc}") from exc
        try:
            history = json.loads((history_dir / history_name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable step history {history_name}: {exc}") from exc
        if history.get("step_index") != step or not isinstance(history.get("history"), dict):
            raise RuntimeError(f"invalid step history payload: {history_name}")
    return {
        "status": "passed",
        "expected_steps": int(expected_steps),
        "frame_count": len(observed_frames),
        "history_count": len(observed_histories),
    }


def _validate_iqn_trial_vector_frame(frame: Any, *, expected_step: int) -> None:
    guess = np.asarray(frame["iqn_trial_guess_mps"], dtype=np.float64)
    candidate = np.asarray(frame["iqn_trial_candidate_mps"], dtype=np.float64)
    residual = np.asarray(frame["iqn_trial_residual_mps"], dtype=np.float64)
    if (
        guess.ndim != 3
        or guess.shape[0] <= 0
        or guess.shape[1] <= 0
        or guess.shape[2] != 3
        or candidate.shape != guess.shape
        or residual.shape != guess.shape
    ):
        raise ValueError("IQN trial vectors must share shape (T, M, 3)")
    if not all(np.all(np.isfinite(values)) for values in (guess, candidate, residual)):
        raise ValueError("IQN trial vectors must be finite")
    if not np.array_equal(residual, candidate - guess):
        raise ValueError("IQN trial residual must equal candidate - guess")
    trial_index = np.asarray(frame["iqn_trial_index"], dtype=np.int64)
    if not np.array_equal(
        trial_index,
        np.arange(guess.shape[0], dtype=np.int64),
    ):
        raise ValueError("IQN trial index must be contiguous and zero-based")
    layout = str(np.asarray(frame["iqn_trial_layout_sha256"]).item())
    if len(layout) != 64 or any(character not in "0123456789abcdef" for character in layout):
        raise ValueError("IQN trial layout identity must be lowercase SHA-256")
    if int(np.asarray(frame["iqn_trial_step"]).item()) != int(expected_step):
        raise ValueError("IQN trial step identity does not match its frame")
    time_s = float(np.asarray(frame["iqn_trial_time_s"]).item())
    dt_s = float(np.asarray(frame["iqn_trial_dt_s"]).item())
    if not math.isfinite(time_s) or time_s <= 0.0:
        raise ValueError("IQN trial physical time must be positive finite")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("IQN trial dt must be positive finite")


def _make_step_observer(
    *,
    output_dir: Path,
    progress_dir: Path | None = None,
    span_reduction: str,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float = 0.1,
    save_iqn_trial_vectors: bool = False,
):
    progress_path = (output_dir if progress_dir is None else progress_dir) / "progress.json"

    def observe(
        step_index: int,
        time_s: float,
        history_row: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> None:
        real_fields_dir = _real_artifact_directory(output_dir, "step_fields")
        real_history_dir = _real_artifact_directory(output_dir, "step_history")
        frame_path = real_fields_dir / f"step_{int(step_index):04d}.npz"
        history_path = real_history_dir / f"step_{int(step_index):04d}.json"
        _require_real_artifact_file(frame_path, output_dir)
        _require_real_artifact_file(history_path, output_dir)
        history_payload = {
            "step_index": int(step_index),
            "time_s": float(time_s),
            "history": history_row,
        }
        _require_existing_history_semantic_match(history_path, history_payload)
        frame_summary = _save_step_frame_atomic(
            frame_path,
            snapshot,
            span_reduction=span_reduction,
            streamwise_velocity_sign=float(streamwise_velocity_sign),
            reverse_streamwise_axis=bool(reverse_streamwise_axis),
            streamwise_length_m=float(streamwise_length_m),
            save_iqn_trial_vectors=bool(save_iqn_trial_vectors),
        )
        _write_step_history_create_or_semantic_same(
            history_path,
            history_payload,
        )
        progress = {
            "status": "running",
            "phase": "fsi_step",
            "step_completed": int(step_index),
            "time_s": float(time_s),
            "max_displacement_m": history_row.get("max_displacement_m"),
            "tip_mean_displacement_m": history_row.get(
                "tip_mean_displacement_m"
            ),
            "local_velocity_peak_mps": history_row.get(
                "local_velocity_peak_mps"
            ),
            "frame": frame_summary,
        }
        try:
            existing = json.loads(progress_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            existing = {}
        next_progress = _merge_progress_event(existing, progress)
        _write_json_atomic(progress_path, next_progress)
        print(json.dumps(_json_safe(next_progress), sort_keys=True), flush=True)

    observe.record_iqn_trial_vectors = bool(save_iqn_trial_vectors)
    observe.checkpoint_replay_safe = True
    observe.checkpoint_identity = _checkpoint_observer_identity(
        output_dir=output_dir,
        span_reduction=span_reduction,
        streamwise_velocity_sign=streamwise_velocity_sign,
        reverse_streamwise_axis=reverse_streamwise_axis,
        streamwise_length_m=streamwise_length_m,
        save_iqn_trial_vectors=save_iqn_trial_vectors,
    )
    return observe


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    history = [dict(_json_safe(row)) for row in history]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else _json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def _snapshot_meta(snapshot: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in snapshot.items():
        array = np.asarray(value)
        meta[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if array.size and np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            if finite.size:
                meta[key].update(
                    {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                    }
                )
    return meta


def _grid_summary(config: VerticalFlapFsiConfig) -> dict[str, Any]:
    nx, ny, nz = [int(v) for v in config.grid_nodes]
    modeled_height = float(
        ANSYS_VERTICAL_FLAP_CASE_METADATA["geometry"]["modeled_height_m"]
    )
    physical_full_height = float(config.duct_height_m)
    full_height_ratio = physical_full_height / modeled_height
    mirrored_full_height_cells = int(round(ny * full_height_ratio))
    return {
        "grid_nodes": [nx, ny, nz],
        "span_cells": nx,
        "wall_normal_cells_modeled_half_height": ny,
        "wall_normal_cells_mirrored_full_height": mirrored_full_height_cells,
        "streamwise_cells": nz,
        "total_3d_cells": nx * ny * nz,
        "modeled_half_height_2d_equivalent_cells": ny * nz,
        "mirrored_full_height_2d_equivalent_cells": (
            mirrored_full_height_cells * nz
        ),
        "cell_size_m": {
            "span": float(config.span_m) / nx,
            "wall_normal_modeled_half_height": modeled_height / ny,
            "streamwise": float(config.duct_length_m) / nz,
        },
        "domain_m": {
            "streamwise_length": float(config.duct_length_m),
            "numerical_modeled_height": modeled_height,
            "physical_full_height": physical_full_height,
            "span": float(config.span_m),
        },
    }


def _optional_kalman_config(
    args: argparse.Namespace,
    *,
    owner: str,
    dt_s: float,
) -> InterfaceKalmanConfig | None:
    process_noise = getattr(args, f"kalman_{owner}_q", None)
    measurement_variance = getattr(args, f"kalman_{owner}_r", None)
    if process_noise is None and measurement_variance is None:
        return None
    if process_noise is None or measurement_variance is None:
        raise ValueError(
            f"--kalman-{owner}-q and --kalman-{owner}-r must be provided together"
        )
    measurement_variance = float(measurement_variance)
    return InterfaceKalmanConfig(
        rate_process_noise_spectral_density=float(process_noise),
        measurement_variance=measurement_variance,
        initial_value_variance=measurement_variance,
        initial_rate_variance=measurement_variance / float(dt_s) ** 2,
        warmup_accepted_states=int(
            getattr(args, "kalman_warmup_accepted_states", 6)
        ),
    )


def _modified_physics_kalman_configs(
    args: argparse.Namespace,
    *,
    dt_s: float,
) -> tuple[
    str,
    InterfaceKalmanConfig | None,
    InterfaceKalmanConfig | None,
    InterfaceKalmanConfig | None,
]:
    mode = str(getattr(args, "kalman_mode", "off"))
    configs = {
        owner: _optional_kalman_config(args, owner=owner, dt_s=dt_s)
        for owner in ("interface", "fluid", "solid")
    }
    required_owners = {
        "off": (),
        "interface": ("interface",),
        "fluid": ("fluid",),
        "solid": ("solid",),
        "global": ("interface", "fluid", "solid"),
    }[mode]
    missing = [owner for owner in required_owners if configs[owner] is None]
    if missing:
        flags = ", ".join(
            f"--kalman-{owner}-q/--kalman-{owner}-r" for owner in missing
        )
        raise ValueError(f"Kalman mode {mode!r} requires {flags}")
    unexpected = [
        owner
        for owner, owner_config in configs.items()
        if owner not in required_owners and owner_config is not None
    ]
    if unexpected:
        flags = ", ".join(
            f"--kalman-{owner}-q/--kalman-{owner}-r"
            for owner in unexpected
        )
        raise ValueError(f"Kalman mode {mode!r} does not use {flags}")
    return mode, configs["interface"], configs["fluid"], configs["solid"]


def _initial_guess_inputs(
    args: argparse.Namespace,
    *,
    dt_s: float,
) -> tuple[InterfaceKalmanConfig | None, str | None]:
    """Build exactly one first-guess input without enabling writeback."""

    mode = str(getattr(args, "initial_guess_mode", "carry_forward"))
    process_noise = getattr(args, "initial_guess_kalman_q", None)
    measurement_variance = getattr(args, "initial_guess_kalman_r", None)
    process_noise_xyz = getattr(args, "initial_guess_kalman_q_xyz", None)
    measurement_variance_xyz = getattr(
        args, "initial_guess_kalman_r_xyz", None
    )
    warmup = getattr(args, "initial_guess_kalman_warmup_accepted_states", None)
    oracle_path = getattr(args, "initial_guess_oracle_path", None)
    has_scalar_values = (
        process_noise is not None or measurement_variance is not None
    )
    has_xyz_values = (
        process_noise_xyz is not None or measurement_variance_xyz is not None
    )
    has_kalman_values = has_scalar_values or has_xyz_values

    if mode == "kalman":
        if has_scalar_values and has_xyz_values:
            raise ValueError(
                "initial-guess Kalman scalar and xyz Q/R cannot be mixed"
            )
        if has_xyz_values:
            if process_noise_xyz is None or measurement_variance_xyz is None:
                raise ValueError(
                    "initial_guess_mode='kalman' requires both "
                    "--initial-guess-kalman-q-xyz and "
                    "--initial-guess-kalman-r-xyz"
                )
            selected_process_noise = tuple(
                float(value) for value in process_noise_xyz
            )
            selected_measurement_variance = tuple(
                float(value) for value in measurement_variance_xyz
            )
            initial_rate_variance = tuple(
                value / float(dt_s) ** 2
                for value in selected_measurement_variance
            )
        elif process_noise is None or measurement_variance is None:
            raise ValueError(
                "initial_guess_mode='kalman' requires "
                "either scalar --initial-guess-kalman-q/r or xyz "
                "--initial-guess-kalman-q-xyz/r-xyz"
            )
        else:
            selected_process_noise = float(process_noise)
            selected_measurement_variance = float(measurement_variance)
            initial_rate_variance = (
                selected_measurement_variance / float(dt_s) ** 2
            )
        if oracle_path is not None:
            raise ValueError(
                "initial_guess_mode='kalman' does not use "
                "--initial-guess-oracle-path"
            )
        return (
            InterfaceKalmanConfig(
                rate_process_noise_spectral_density=selected_process_noise,
                measurement_variance=selected_measurement_variance,
                initial_value_variance=selected_measurement_variance,
                initial_rate_variance=initial_rate_variance,
                warmup_accepted_states=(
                    6 if warmup is None else int(warmup)
                ),
            ),
            None,
        )

    if mode == "oracle_replay":
        if oracle_path is None:
            raise ValueError(
                "initial_guess_mode='oracle_replay' requires "
                "--initial-guess-oracle-path"
            )
        if has_kalman_values or warmup is not None:
            raise ValueError(
                "initial_guess_mode='oracle_replay' does not use "
                "initial-guess Kalman parameters"
            )
        return None, str(oracle_path)

    if has_kalman_values or warmup is not None:
        raise ValueError(
            f"initial_guess_mode={mode!r} does not use initial-guess Kalman "
            "parameters"
        )
    if oracle_path is not None:
        raise ValueError(
            f"initial_guess_mode={mode!r} does not use "
            "--initial-guess-oracle-path"
        )
    return None, None


def _validate_initial_guess_oracle_producer(
    *,
    producer_output: str | Path,
    consumer_output: str | Path,
    consumer_config_payload: dict[str, Any],
    current_source_sha256: dict[str, str],
    require_full_lineage: bool = False,
) -> dict[str, Any]:
    """Validate and seal a completed Q0 trajectory before Q3 starts."""

    producer = Path(producer_output).expanduser().resolve()
    consumer = Path(consumer_output).expanduser().resolve()
    if producer == consumer:
        raise ValueError("oracle producer and consumer outputs must differ")

    def read_mapping(name: str) -> dict[str, Any]:
        path = producer / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"oracle producer has invalid {name}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"oracle producer {name} must contain an object")
        return payload

    manifest = read_mapping("run_manifest.json")
    progress = read_mapping("progress.json")
    summary = read_mapping("our_solver_summary.json")
    producer_config = manifest.get("config")
    producer_sources = manifest.get("source_sha256")
    if not isinstance(producer_config, dict):
        raise ValueError("oracle producer manifest is missing config")
    if not isinstance(producer_sources, dict) or not _json_values_equal(
        producer_sources,
        current_source_sha256,
    ):
        raise ValueError("oracle producer source identity does not match consumer")
    if not bool(manifest.get("save_step_fields", False)):
        raise ValueError("oracle producer did not save accepted step fields")
    progress, summary = require_completed_output(
        producer,
        read_mapping=read_mapping,
        error_message="oracle producer is not terminal-complete",
    )
    if producer_config.get("coupling_mode") != "iqn_ils":
        raise ValueError("oracle producer must use iqn_ils")
    if producer_config.get("initial_guess_mode") != "carry_forward":
        raise ValueError("oracle producer must be the Q0 carry-forward baseline")
    if producer_config.get("kalman_writeback_mode") != "off":
        raise ValueError("oracle producer must disable modified-physics writeback")

    expected_steps = int(consumer_config_payload.get("step_count", 0))
    if expected_steps <= 0:
        raise ValueError("oracle consumer step_count must be positive")
    if int(summary.get("step_count_completed", -1)) != expected_steps:
        raise ValueError("oracle producer completed-step count does not match consumer")
    if int(progress.get("step_completed", -1)) != expected_steps:
        raise ValueError("oracle producer progress-step count does not match consumer")

    comparable_producer = dict(producer_config)
    comparable_consumer = dict(consumer_config_payload)
    for field in (
        "initial_guess_mode",
        "initial_guess_oracle_path",
        "initial_guess_kalman_config",
        "iqn_kalman_oracle_interpolation_target_step",
        "iqn_kalman_oracle_interpolation_oracle_path",
        "iqn_kalman_oracle_interpolation_alphas",
    ):
        comparable_producer.pop(field, None)
        comparable_consumer.pop(field, None)
    if not _json_values_equal(comparable_producer, comparable_consumer):
        differing = sorted(
            key
            for key in set(comparable_producer) | set(comparable_consumer)
            if not _json_values_equal(
                comparable_producer.get(key),
                comparable_consumer.get(key),
            )
        )
        raise ValueError(
            "oracle producer config does not match consumer: "
            f"differing_fields={differing}"
        )

    fields_dir = producer / "step_fields"
    histories_dir = producer / "step_history"
    expected_names = [
        f"step_{step:04d}.npz" for step in range(1, expected_steps + 1)
    ]
    expected_history_names = [
        f"step_{step:04d}.json" for step in range(1, expected_steps + 1)
    ]
    observed_names = sorted(path.name for path in fields_dir.glob("step_*.npz"))
    observed_history_names = sorted(
        path.name for path in histories_dir.glob("step_*.json")
    )
    if observed_names != expected_names:
        raise ValueError(
            "oracle producer step-field sequence mismatch: "
            f"observed={observed_names}, expected={expected_names}"
        )
    if observed_history_names != expected_history_names:
        raise ValueError(
            "oracle producer step-history sequence mismatch: "
            f"observed={observed_history_names}, expected={expected_history_names}"
        )
    trajectory_hasher = hashlib.sha256()
    history_trajectory_hasher = hashlib.sha256()
    frame_hashes: dict[str, str] = {}
    history_hashes: dict[str, str] = {}
    for frame_name in expected_names:
        payload = (fields_dir / frame_name).read_bytes()
        frame_digest = hashlib.sha256(payload).hexdigest()
        frame_hashes[frame_name] = frame_digest
        trajectory_hasher.update(frame_name.encode("utf-8"))
        trajectory_hasher.update(bytes.fromhex(frame_digest))
    for history_name in expected_history_names:
        payload = (histories_dir / history_name).read_bytes()
        history_digest = hashlib.sha256(payload).hexdigest()
        history_hashes[history_name] = history_digest
        history_trajectory_hasher.update(history_name.encode("utf-8"))
        history_trajectory_hasher.update(bytes.fromhex(history_digest))

    base_identity = {
        "offline_oracle": True,
        "deployable": False,
        "producer_output": str(producer),
        "producer_run_label": str(manifest.get("run_label", "")),
        "source_sha256": dict(producer_sources),
        "frame_sha256": frame_hashes,
        "trajectory_sha256": trajectory_hasher.hexdigest(),
        "history_sha256": history_hashes,
        "history_trajectory_sha256": history_trajectory_hasher.hexdigest(),
        "step_count": expected_steps,
    }
    if not require_full_lineage:
        return base_identity

    try:
        q0_run = _load_oracle_headroom_run(
            producer,
            expected_mode="carry_forward",
        )
        full_identity = _q0_probe_identity(q0_run)
    except Exception as exc:
        raise ValueError(
            f"oracle producer full lineage validation failed: {exc}"
        ) from exc
    for field, expected in base_identity.items():
        if not _json_values_equal(full_identity.get(field), expected):
            raise ValueError(f"oracle producer {field} identity mismatch")
    return dict(full_identity)


def _build_config(args: argparse.Namespace) -> VerticalFlapFsiConfig:
    config = selected_formulation_solver_config(step_count=int(args.steps))
    config = replace(
        config,
        dt_s=float(getattr(args, "dt_s", config.dt_s)),
    )
    (
        kalman_mode,
        kalman_interface_config,
        kalman_fluid_config,
        kalman_solid_config,
    ) = _modified_physics_kalman_configs(args, dt_s=float(config.dt_s))
    initial_guess_kalman_config, initial_guess_oracle_path = _initial_guess_inputs(
        args,
        dt_s=float(config.dt_s),
    )
    config = replace(
        config,
        grid_nodes=tuple(int(v) for v in args.grid_nodes),
        solid_particle_counts=tuple(int(v) for v in args.solid_particle_counts),
        marker_count=int(args.marker_count),
        flow_projection_iterations=int(args.flow_projection_iterations),
        flow_post_dirichlet_consistency_projection_iterations=int(
            getattr(args, "flow_post_dirichlet_consistency_projections", 1)
        ),
        flow_reprojection_iterations=(
            int(getattr(args, "flow_reprojection_iterations", None))
            if getattr(args, "flow_reprojection_iterations", None) is not None
            else None
        ),
        flow_reprojection_cg_tolerance=(
            float(getattr(args, "flow_reprojection_cg_tolerance", None))
            if getattr(args, "flow_reprojection_cg_tolerance", None) is not None
            else None
        ),
        flow_cg_preconditioner=str(args.flow_cg_preconditioner),
        flow_pressure_solve_failure_policy=str(
            args.flow_pressure_solve_failure_policy
        ),
        solid_substeps=(
            int(args.solid_substeps)
            if args.solid_substeps is not None
            else None
        ),
        coupling_mode=str(
            getattr(args, "coupling_mode", config.coupling_mode)
        ),
        fsi_coupling_max_iterations=int(
            getattr(
                args,
                "fsi_max_iterations",
                config.fsi_coupling_max_iterations,
            )
        ),
        fsi_coupling_absolute_tolerance_mps=float(
            getattr(
                args,
                "fsi_absolute_tolerance_mps",
                config.fsi_coupling_absolute_tolerance_mps,
            )
        ),
        fsi_coupling_relative_tolerance=float(
            getattr(
                args,
                "fsi_relative_tolerance",
                config.fsi_coupling_relative_tolerance,
            )
        ),
        iqn_history_limit=int(
            getattr(args, "iqn_history_limit", config.iqn_history_limit)
        ),
        iqn_initial_picard_relaxation=float(
            getattr(
                args,
                "iqn_initial_picard_relaxation",
                config.iqn_initial_picard_relaxation,
            )
        ),
        iqn_svd_relative_cutoff=float(
            getattr(
                args,
                "iqn_svd_relative_cutoff",
                config.iqn_svd_relative_cutoff,
            )
        ),
        iqn_reuse_previous_step_history=bool(
            getattr(
                args,
                "iqn_reuse_previous_step_history",
                config.iqn_reuse_previous_step_history,
            )
        ),
        initial_guess_mode=str(
            getattr(args, "initial_guess_mode", config.initial_guess_mode)
        ),
        initial_guess_kalman_config=initial_guess_kalman_config,
        initial_guess_oracle_path=initial_guess_oracle_path,
        iqn_kalman_oracle_interpolation_target_step=(
            int(getattr(args, "research_iqn_kalman_oracle_interpolation_target_step", None))
            if getattr(args, "research_iqn_kalman_oracle_interpolation_target_step", None) is not None
            else None
        ),
        iqn_kalman_oracle_interpolation_oracle_path=(
            str(
                Path(
                    str(
                        getattr(
                            args,
                            "research_iqn_kalman_oracle_interpolation_oracle_path",
                            None,
                        )
                    )
                )
                .expanduser()
                .resolve()
            )
            if getattr(args, "research_iqn_kalman_oracle_interpolation_oracle_path", None) is not None
            else None
        ),
        iqn_kalman_oracle_interpolation_alphas=(
            tuple(float(value) for value in getattr(args, "research_iqn_kalman_oracle_interpolation_alphas", None))
            if getattr(args, "research_iqn_kalman_oracle_interpolation_alphas", None) is not None
            else VerticalFlapFsiConfig.iqn_kalman_oracle_interpolation_alphas
        ),
        research_initial_guess_candidate_matrix_path=(
            str(
                Path(
                    str(
                        getattr(
                            args,
                            "research_initial_guess_candidate_matrix_path",
                            None,
                        )
                    )
                ).expanduser().resolve()
            )
            if getattr(args, "research_initial_guess_candidate_matrix_path", None)
            is not None
            else None
        ),
        kalman_writeback_mode=kalman_mode,
        kalman_interface_config=kalman_interface_config,
        kalman_fluid_config=kalman_fluid_config,
        kalman_solid_config=kalman_solid_config,
        flow_driver_mode="sustained_boundary_predictor",
        # The exact external velocity-boundary ledger owns inlet topology.
        # A volume-source preflow driver would still double-drive this
        # boundary-driven field, so retain the boundary predictor here.
        preflow_flow_driver_mode="sustained_boundary_predictor",
        preflow_steps=int(args.preflow_steps),
        preflow_convergence_mode=str(
            getattr(args, "preflow_convergence_mode", "single_step_legacy")
        ),
        preflow_stationary_min_steps=int(
            getattr(args, "preflow_stationary_min_steps", 20)
        ),
        preflow_stationary_window_steps=int(
            getattr(args, "preflow_stationary_window_steps", 10)
        ),
        preflow_stationary_consecutive_windows=int(
            getattr(args, "preflow_stationary_consecutive_windows", 3)
        ),
        preflow_stationary_tolerance=float(
            getattr(args, "preflow_stationary_tolerance", 0.05)
        ),
        preflow_stationary_divergence_tolerance=float(
            getattr(args, "preflow_stationary_divergence_tolerance", 0.05)
        ),
        preflow_stationary_no_slip_tolerance_fraction=float(
            getattr(
                args,
                "preflow_stationary_no_slip_tolerance_fraction",
                0.05,
            )
        ),
        detailed_preflow_stage_progress=bool(
            getattr(args, "detailed_preflow_stage_progress", False)
        ),
        preflow_snapshot_input_path=getattr(args, "preflow_snapshot_in", None),
        preflow_snapshot_output_path=getattr(args, "preflow_snapshot_out", None),
        fsi_checkpoint_input_path=getattr(args, "fsi_checkpoint_in", None),
        fsi_checkpoint_output_path=getattr(args, "fsi_checkpoint_out", None),
        flow_hibm_sharp_search_radius_m=1.7e-3,
        flow_hibm_marker_compatibility_closure_tolerance_mps=float(
            getattr(
                args,
                "flow_hibm_marker_compatibility_closure_tolerance_mps",
                config.flow_hibm_marker_compatibility_closure_tolerance_mps,
            )
        ),
        flow_report_include_percentiles=bool(
            getattr(args, "flow_report_percentiles", False)
        ),
        # The physical flap core is now represented independently by the MPM
        # volume mask.  HIBM only needs a narrow mesh-scaled interface band.
        flow_hibm_sharp_search_radius_xyz_m=(
            0.5 * float(config.span_m)
            - 0.4 * (float(config.span_m) / float(args.grid_nodes[0])),
            5.0
            * (
                0.5
                * float(config.duct_height_m)
                / float(args.grid_nodes[1])
            ),
            1.5
            * (float(config.duct_length_m) / float(args.grid_nodes[2])),
        ),
        flow_hibm_sharp_interior_probe_distance_m=(
            1.5
            * max(
                float(config.span_m) / float(args.grid_nodes[0]),
                0.5
                * float(config.duct_height_m)
                / float(args.grid_nodes[1]),
                float(config.duct_length_m) / float(args.grid_nodes[2]),
            )
        ),
        flow_hibm_dynamic_solid_volume_enabled=True,
        # Remove only small outlet-disconnected row-cloud fragments before
        # pressure-row assembly.  They are discrete grid artifacts without an
        # outlet pressure path; each conversion invalidates topology and
        # rebuilds the canonical velocity rows before projection.
        flow_hibm_tiny_unreached_cleanup_component_cells=128,
        update_fluid_obstacle_from_solid=True,
        export_final_flow_snapshot=True,
        # Campaign runs fail loud on under-seeded solids (2026-07-03 audit:
        # counts (1, 64, 12) on grid 4x256x320 leave ~2 cells between
        # wall-normal particle layers, the root clamp fractures, and the
        # flap free-falls until a particle leaves the background grid).
        enforce_solid_seeding_limit=True,
    )
    if args.flow_predictor_substeps is not None:
        config = replace(
            config,
            flow_predictor_substeps=int(args.flow_predictor_substeps),
        )
    if args.young_modulus_pa is not None:
        config = replace(
            config,
            young_modulus_pa=float(args.young_modulus_pa),
        )
    if args.hibm_search_radius_m is not None:
        config = replace(
            config,
            flow_hibm_sharp_search_radius_m=float(args.hibm_search_radius_m),
        )
    if getattr(args, "hibm_search_radius_xyz_m", None) is not None:
        config = replace(
            config,
            flow_hibm_sharp_search_radius_xyz_m=tuple(
                float(value)
                for value in getattr(args, "hibm_search_radius_xyz_m")
            ),
        )
    if bool(getattr(args, "disable_hibm_anisotropic_search", False)):
        if getattr(args, "hibm_search_radius_xyz_m", None) is not None:
            raise ValueError(
                "--disable-hibm-anisotropic-search conflicts with "
                "--hibm-search-radius-xyz-m"
            )
        config = replace(
            config,
            flow_hibm_sharp_search_radius_xyz_m=None,
        )
    if getattr(args, "hibm_interior_probe_distance_m", None) is not None:
        config = replace(
            config,
            flow_hibm_sharp_interior_probe_distance_m=float(
                getattr(args, "hibm_interior_probe_distance_m")
            ),
            # An explicit legacy scalar override must retain its historical
            # isotropic semantics instead of being shadowed by the fine-grid
            # tuple above.
            flow_hibm_sharp_interior_probe_distance_xyz_m=None,
        )
    return with_local_surface_force_support(config)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "final_flow_field_snapshot"
    }


def _summary_from_report(
    *,
    report: dict[str, Any],
    config: VerticalFlapFsiConfig,
    output_dir: Path,
    elapsed_s: float,
    solver_npz_summary: dict[str, Any] | None,
    run_label: str,
    artifact_output_dir: Path | None = None,
    solver_elapsed_s: float | None = None,
    post_solver_artifact_export_wall_time_s: float = 0.0,
    pre_summary_artifact_elapsed_s: float | None = None,
    require_runtime_identity: bool = False,
) -> dict[str, Any]:
    resolved_artifact_output_dir = (
        output_dir if artifact_output_dir is None else artifact_output_dir
    )
    history = list(report.get("history", []))
    final_history = history[-1] if history else {}
    status = "completed" if len(history) == int(config.step_count) else "blocked"
    resolved_solver_elapsed_s = (
        float(elapsed_s)
        if solver_elapsed_s is None
        else float(solver_elapsed_s)
    )
    resolved_pre_summary_artifact_elapsed_s = (
        float(elapsed_s)
        if pre_summary_artifact_elapsed_s is None
        else float(pre_summary_artifact_elapsed_s)
    )
    profile_wall_time_enabled = report.get("profile_wall_time_enabled")
    taichi_runtime_identity = report.get("taichi_runtime_identity")
    if require_runtime_identity and status == "completed":
        if not isinstance(profile_wall_time_enabled, bool):
            raise ValueError(
                "completed formal run report is missing profile wall-time identity"
            )
        if not isinstance(taichi_runtime_identity, dict) or not taichi_runtime_identity:
            raise ValueError(
                "completed formal run report is missing runtime identity"
            )
    profile_field_names = (
        "flow_wall_time_s_total",
        "solid_wall_time_s_total",
        "hibm_pre_predictor_wall_time_s_total",
        "hibm_projection_cycle_wall_time_s_total",
        "hibm_post_solid_observer_wall_time_s_total",
        "hibm_wall_time_s_total",
        "snapshot_capture_wall_time_s_total",
        "step_artifact_export_wall_time_s_total",
        "kalman_filter_overhead_s_total",
        "kalman_state_transfer_overhead_s_total",
        "kalman_total_overhead_s_total",
    )
    profile_totals = {
        field: float(report[field])
        for field in profile_field_names
        if field in report
    }
    experiment_metric_field_names = (
        "initial_guess_mode",
        "initial_guess_summary",
        "hibm_coupling_scheme",
        "hibm_fsi_accepted_macro_step_count",
        "hibm_fsi_coupling_iterations_total",
        "hibm_fsi_coupling_iterations_min",
        "hibm_fsi_coupling_iterations_max",
        "hibm_fsi_coupling_iterations_mean",
        "hibm_fsi_coupling_iterations_median",
        "hibm_fsi_coupling_iterations_p95",
        "hibm_fsi_coupling_rejected_trial_count_total",
        "hibm_fsi_coupling_fluid_solve_count",
        "hibm_fsi_coupling_solid_macro_solve_count",
        "hibm_fsi_coupling_converged_step_count",
        "hibm_fsi_trial_work_report",
        "hibm_fsi_trial_cg_iterations_total",
        "hibm_fsi_trial_flow_momentum_advection_substeps_total",
        "hibm_fsi_trial_flow_sst_transport_substeps_total",
        "hibm_fsi_trial_solid_substeps_executed_total",
        "hibm_fsi_trial_flow_wall_time_s_total",
        "hibm_fsi_trial_hibm_wall_time_s_total",
        "hibm_fsi_trial_solid_wall_time_s_total",
        "fluid_projection_consumed_feedback_count",
        "fluid_projection_consumed_feedback_trial_count",
        "solid_trial_substeps_executed_total",
    )
    experiment_metrics = {
        field: report[field]
        for field in experiment_metric_field_names
        if field in report
    }
    material_evidence_fields = (
        "material_transfer_configuration",
        "material_transfer_verified",
        "material_binding_identity",
        "scatter_action_reaction_residual_n",
        "force_roundoff_bound_n",
        "torque_residual_n_m",
        "torque_roundoff_bound_n_m",
        "material_power_residual_w",
        "material_power_roundoff_bound_w",
        "mpm_direct_fixed_external_force_n",
        "mpm_support_reaction_impulse_n_s",
        "mpm_support_reaction_angular_impulse_n_m_s",
        "mpm_damping_impulse_n_s",
        "mpm_damping_angular_impulse_n_m_s",
    )
    material_evidence = {
        field: report[field] for field in material_evidence_fields if field in report
    }
    return {
        "run_label": run_label,
        "status": status,
        "elapsed_s": float(elapsed_s),
        "solver_elapsed_s": resolved_solver_elapsed_s,
        "post_solver_artifact_export_wall_time_s": float(
            post_solver_artifact_export_wall_time_s
        ),
        "pre_summary_artifact_elapsed_s": (
            resolved_pre_summary_artifact_elapsed_s
        ),
        "profile_wall_time_enabled": profile_wall_time_enabled,
        "taichi_runtime_identity": taichi_runtime_identity,
        "output_dir": str(output_dir),
        "artifact_root": str(resolved_artifact_output_dir),
        "step_count_requested": int(config.step_count),
        "step_count_completed": len(history),
        "final_time_s": float(config.dt_s) * len(history),
        "dt_s": float(config.dt_s),
        "grid": _grid_summary(config),
        "solid_particle_counts": list(config.solid_particle_counts),
        "marker_count": int(config.marker_count),
        "flow_projection_iterations": int(config.flow_projection_iterations),
        "solid_substeps": config.solid_substeps,
        "solid_substeps_mode": (
            "adaptive" if config.solid_substeps is None else "fixed_override"
        ),
        "kalman_writeback_mode": str(config.kalman_writeback_mode),
        "kalman_modified_physics": bool(
            report.get("kalman_modified_physics", False)
        ),
        "kalman_summary": report.get("kalman_summary", {}),
        "preflow_snapshot_loaded": report.get("preflow_snapshot_loaded"),
        "preflow_snapshot_identity": report.get("preflow_snapshot_identity"),
        "preflow_snapshot_artifact_identity": report.get(
            "preflow_snapshot_artifact_identity"
        ),
        "max_displacement_m": report.get("max_displacement_m"),
        "max_displacement_relative_error": report.get(
            "max_displacement_relative_error"
        ),
        "local_velocity_peak_mps": report.get("local_velocity_peak_mps"),
        "max_abs_traction_pa": report.get("max_abs_traction_pa"),
        "final_history": final_history,
        "solver_npz_summary": solver_npz_summary or {},
        "step_field_frame_count": len(list((output_dir / "step_fields").glob("step_*.npz"))),
        **experiment_metrics,
        **profile_totals,
        **material_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", default="our_solver")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--dt-s",
        type=float,
        default=VerticalFlapFsiConfig.dt_s,
        help="Physical FSI macro-step duration used by the formal experiment.",
    )
    parser.add_argument(
        "--preflow-steps",
        type=int,
        default=40,
        help=(
            "Override fixed-solid preflow steps; use --steps 0 "
            "--preflow-steps 1 for the bounded pressure gate."
        ),
    )
    parser.add_argument(
        "--preflow-convergence-mode",
        default="single_step_legacy",
        choices=("single_step_legacy", "windowed_stationary"),
    )
    parser.add_argument("--preflow-stationary-min-steps", type=int, default=20)
    parser.add_argument("--preflow-stationary-window-steps", type=int, default=10)
    parser.add_argument(
        "--preflow-stationary-consecutive-windows",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--preflow-stationary-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preflow-stationary-divergence-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preflow-stationary-no-slip-tolerance-fraction",
        type=float,
        default=0.05,
    )
    preflow_snapshot_group = parser.add_mutually_exclusive_group()
    preflow_snapshot_group.add_argument(
        "--preflow-snapshot-in",
        help="Load a strict post-preflow snapshot and skip fixed-solid preflow.",
    )
    preflow_snapshot_group.add_argument(
        "--preflow-snapshot-out",
        help="Atomically save the validated post-preflow state under this prefix.",
    )
    parser.add_argument(
        "--fsi-checkpoint-in",
        help="Resume from a complete accepted FSI checkpoint prefix.",
    )
    parser.add_argument(
        "--fsi-checkpoint-out",
        help="Save every accepted complete FSI state under this prefix.",
    )
    parser.add_argument(
        "--resume-run-dir",
        help=(
            "Canonical accepted-state root to resume. It owns checkpoint and "
            "accepted step artifacts; --output-dir becomes a fresh attempt root."
        ),
    )
    parser.add_argument("--grid-nodes", type=int, nargs=3, default=(4, 32, 64))
    parser.add_argument(
        "--solid-particle-counts", type=int, nargs=3, default=(1, 12, 4)
    )
    parser.add_argument("--marker-count", type=int, default=12)
    parser.add_argument("--flow-projection-iterations", type=int, default=1080)
    parser.add_argument(
        "--flow-hibm-marker-compatibility-closure-tolerance-mps",
        type=float,
        default=(
            VerticalFlapFsiConfig
            .flow_hibm_marker_compatibility_closure_tolerance_mps
        ),
        help=(
            "Explicit research override for the hard HIBM marker compatibility "
            "closure gate; it must remain positive and no larger than the "
            "marker MAC absolute tolerance."
        ),
    )
    parser.add_argument(
        "--flow-post-dirichlet-consistency-projections",
        type=int,
        default=1,
        help=(
            "Sharp-HIBM row reassembly + accumulating pressure-correction "
            "passes after the main projection."
        ),
    )
    parser.add_argument(
        "--flow-reprojection-iterations",
        type=int,
        default=None,
        help="Optional CG budget for each consistency projection.",
    )
    parser.add_argument(
        "--flow-reprojection-cg-tolerance",
        type=float,
        default=None,
        help="Optional relative CG tolerance for each consistency projection.",
    )
    parser.add_argument(
        "--flow-cg-preconditioner",
        default="fv_multigrid_light",
        choices=("auto", "jacobi", "fv_multigrid", "fv_multigrid_light"),
    )
    parser.add_argument(
        "--flow-pressure-solve-failure-policy",
        default="raise",
        choices=("raise", "report"),
    )
    parser.add_argument(
        "--solid-substeps",
        type=int,
        default=None,
        help=(
            "Optional fixed MPM substep count for controlled A/B reference "
            "runs. Omit it for the production per-macro adaptive selector."
        ),
    )
    parser.add_argument(
        "--coupling-mode",
        default=VerticalFlapFsiConfig.coupling_mode,
        choices=("direct_explicit", "iqn_ils"),
        help="Physical-step coupling route; direct_explicit remains the default.",
    )
    parser.add_argument(
        "--initial-guess-mode",
        default=VerticalFlapFsiConfig.initial_guess_mode,
        choices=(
            "carry_forward",
            "linear_extrapolation",
            "kalman",
            "oracle_replay",
        ),
        help="Iteration-0 marker-velocity guess used only by IQN-ILS.",
    )
    parser.add_argument(
        "--initial-guess-kalman-q",
        type=float,
        default=None,
        help="First-guess Kalman rate-process-noise spectral density.",
    )
    parser.add_argument(
        "--initial-guess-kalman-r",
        type=float,
        default=None,
        help="First-guess Kalman measurement variance.",
    )
    parser.add_argument(
        "--initial-guess-kalman-q-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("QX", "QY", "QZ"),
        help="Per-axis first-guess Kalman rate-process spectral densities.",
    )
    parser.add_argument(
        "--initial-guess-kalman-r-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Per-axis first-guess Kalman measurement variances.",
    )
    parser.add_argument(
        "--initial-guess-kalman-warmup-accepted-states",
        type=int,
        default=None,
        help="Accepted-state warmup before the Kalman first-guess is active.",
    )
    parser.add_argument(
        "--initial-guess-oracle-path",
        type=str,
        default=None,
        help="Oracle first-guess replay artifact for upper-bound experiments.",
    )
    parser.add_argument(
        "--research-iqn-kalman-oracle-interpolation-target-step",
        type=int,
        default=None,
        help="Offline no-commit Kalman-Oracle alpha sweep target macro step.",
    )
    parser.add_argument(
        "--research-iqn-kalman-oracle-interpolation-oracle-path",
        type=str,
        default=None,
        help="Source-matched completed Q0 producer used only by the offline sweep.",
    )
    parser.add_argument(
        "--research-iqn-kalman-oracle-interpolation-alphas",
        type=float,
        nargs="+",
        default=None,
        help="Strictly increasing alpha values in [0, 1] for the offline sweep.",
    )
    parser.add_argument(
        "--research-initial-guess-candidate-matrix-path",
        type=str,
        default=None,
        help="Frozen R25B candidate manifest for a no-commit target-step sweep.",
    )
    parser.add_argument(
        "--fsi-max-iterations",
        type=int,
        default=VerticalFlapFsiConfig.fsi_coupling_max_iterations,
    )
    parser.add_argument(
        "--fsi-absolute-tolerance-mps",
        type=float,
        default=VerticalFlapFsiConfig.fsi_coupling_absolute_tolerance_mps,
    )
    parser.add_argument(
        "--fsi-relative-tolerance",
        type=float,
        default=VerticalFlapFsiConfig.fsi_coupling_relative_tolerance,
    )
    parser.add_argument(
        "--iqn-history-limit",
        type=int,
        default=VerticalFlapFsiConfig.iqn_history_limit,
    )
    parser.add_argument(
        "--iqn-initial-picard-relaxation",
        type=float,
        default=VerticalFlapFsiConfig.iqn_initial_picard_relaxation,
    )
    parser.add_argument(
        "--iqn-svd-relative-cutoff",
        type=float,
        default=VerticalFlapFsiConfig.iqn_svd_relative_cutoff,
    )
    parser.add_argument(
        "--iqn-reuse-previous-step-history",
        action="store_true",
        help="Research-only: reuse immutable IQN secants from the prior accepted step.",
    )
    parser.add_argument(
        "--kalman-mode",
        default="off",
        choices=("off", "interface", "fluid", "solid", "global"),
        help=(
            "Modified-physics posterior writeback placement. 'off' constructs "
            "no filter and preserves the unfiltered numerical path."
        ),
    )
    parser.add_argument(
        "--kalman-warmup-accepted-states",
        type=int,
        default=6,
        help=(
            "Accepted-state count required before writeback. Initialization "
            "counts as one, so 6 means FSI steps 1-5 assimilate only and step "
            "6 is the first active writeback."
        ),
    )
    for owner in ("interface", "fluid", "solid"):
        parser.add_argument(
            f"--kalman-{owner}-q",
            type=float,
            default=None,
            help=f"Frozen {owner} rate-process-noise spectral density.",
        )
        parser.add_argument(
            f"--kalman-{owner}-r",
            type=float,
            default=None,
            help=f"Frozen {owner} measurement variance.",
        )
    parser.add_argument(
        "--flow-predictor-substeps",
        type=int,
        default=None,
        help=(
            "Diagnostic override for outer predictor slices; production "
            "inherits one outer step and lets the core MUSCL transport own "
            "its adaptive CFL substeps."
        ),
    )
    parser.add_argument(
        "--hibm-search-radius-m",
        type=float,
        default=None,
        help=(
            "Override the sharp-IB row search radius; should scale with the "
            "grid (roughly 1.5x the finest in-plane spacing) so the Dirichlet "
            "row halo does not fatten the effective flap."
        ),
    )
    parser.add_argument(
        "--hibm-search-radius-xyz-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help=(
            "Override the anisotropic sharp-IB interface envelope in solver "
            "(span, wall-normal, streamwise) coordinates."
        ),
    )
    parser.add_argument(
        "--hibm-interior-probe-distance-m",
        type=float,
        default=None,
        help="Override the marker pressure/velocity interior probe distance.",
    )
    parser.add_argument(
        "--disable-hibm-anisotropic-search",
        action="store_true",
        help=(
            "Use the legacy scalar spherical HIBM search. Intended only for "
            "controlled A/B validation of the optimized interface envelope."
        ),
    )
    parser.add_argument(
        "--young-modulus-pa",
        type=float,
        default=None,
        help=(
            "Override the flap Young's modulus; a stiff value gives a "
            "rigid-proxy flap for flow-only parity runs while the structural "
            "load calibration is still open."
        ),
    )
    parser.add_argument("--span-reduction", default="mean", choices=("mean", "center"))
    parser.add_argument("--streamwise-velocity-sign", type=float, default=-1.0)
    parser.add_argument(
        "--no-reverse-streamwise-axis",
        action="store_true",
        help="Disable solver-to-Fluent streamwise axis reversal.",
    )
    parser.add_argument(
        "--taichi-offline-cache-dir",
        type=Path,
        default=DEFAULT_TAICHI_OFFLINE_CACHE_DIR,
        help=(
            "Reusable isolated Taichi kernel-cache directory. This affects "
            "compilation latency only, never solver numerics."
        ),
    )
    parser.add_argument(
        "--disable-taichi-offline-cache",
        action="store_true",
        help="Force cold Taichi JIT compilation for diagnostic A/B runs.",
    )
    parser.add_argument(
        "--detailed-preflow-stage-progress",
        action="store_true",
        help=(
            "Persist fine-grained preflow stage progress. This diagnostic mode "
            "adds durable filesystem writes; production runs keep only "
            "step-level progress. Combine with --profile-wall-time for "
            "synchronized per-stage timing."
        ),
    )
    parser.add_argument(
        "--profile-wall-time",
        action="store_true",
        help=(
            "Enable synchronized GPU phase timing. This adds host/device "
            "barriers and is diagnostic-only."
        ),
    )
    parser.add_argument(
        "--flow-report-percentiles",
        action="store_true",
        help="Include p99 and p99.9 flow-speed diagnostics in reports.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-step-fields",
        action="store_true",
        help=(
            "Save one span-reduced parity NPZ and atomic progress record after "
            "each completed FSI step."
        ),
    )
    parser.add_argument(
        "--save-iqn-trial-vectors",
        action="store_true",
        help=(
            "Research-only accepted-step export of IQN trial guess, candidate, "
            "and residual vectors into each step NPZ."
        ),
    )
    args = parser.parse_args()
    if args.save_iqn_trial_vectors and not args.save_step_fields:
        parser.error("--save-iqn-trial-vectors requires --save-step-fields")
    if args.save_iqn_trial_vectors and args.coupling_mode != "iqn_ils":
        parser.error("--save-iqn-trial-vectors requires --coupling-mode iqn_ils")
    if args.resume_run_dir is not None:
        if args.fsi_checkpoint_in is not None or args.fsi_checkpoint_out is not None:
            parser.error(
                "--fsi-checkpoint-in/--fsi-checkpoint-out cannot be combined "
                "with --resume-run-dir"
            )
        canonical_output_dir = Path(args.resume_run_dir).expanduser().absolute()
        checkpoint_prefix = canonical_output_dir / "checkpoint"
        args.fsi_checkpoint_in = str(checkpoint_prefix)
        args.fsi_checkpoint_out = str(checkpoint_prefix)
    else:
        canonical_output_dir = None
        if args.fsi_checkpoint_in is not None:
            parser.error(
                "--fsi-checkpoint-in now requires --resume-run-dir so accepted "
                "evidence is never rewritten; use --fsi-checkpoint-out alone "
                "only for a fresh run"
            )

    output_dir = Path(args.output_dir).expanduser().absolute()
    artifact_output_dir = (
        output_dir if canonical_output_dir is None else canonical_output_dir
    )
    config = _build_config(args)
    config_payload = asdict(config)
    checkpoint_input_path, _ = _fsi_checkpoint_paths(config)
    step_observer = (
        _make_step_observer(
            output_dir=artifact_output_dir,
            progress_dir=output_dir,
            span_reduction=str(args.span_reduction),
            streamwise_velocity_sign=float(args.streamwise_velocity_sign),
            reverse_streamwise_axis=not bool(args.no_reverse_streamwise_axis),
            streamwise_length_m=float(config.duct_length_m),
            save_iqn_trial_vectors=bool(args.save_iqn_trial_vectors),
        )
        if args.save_step_fields
        else None
    )
    resume_source_sha256: dict[str, str] | None = None
    resume_provenance: dict[str, Any] | None = None
    if canonical_output_dir is not None:
        if checkpoint_input_path is None:
            raise RuntimeError("resume run directory did not configure a checkpoint")
        checkpoint_head = _preflight_checkpoint_resume(
            output_dir=canonical_output_dir,
            config=config,
            checkpoint_input_path=checkpoint_input_path,
            step_observer=step_observer,
        )
        config = replace(
            config, fsi_checkpoint_expected_generation=checkpoint_head.generation
        )
        config_payload = asdict(config)
        resume_source_sha256 = _source_hashes()
        _prepare_output_dir(
            output_dir,
            resume=True,
            canonical_output_dir=canonical_output_dir,
        )
        attempt_id = "resume-" + hashlib.sha256(
            f"{output_dir}:{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:24]
        prepare_resume_attempt(
            canonical_root=canonical_output_dir,
            attempt_root=output_dir,
            checkpoint_generation=checkpoint_head.generation,
            checkpoint_identity=checkpoint_head.metadata["identity"],
            accepted_step=checkpoint_head.accepted_step,
            source_hashes=resume_source_sha256,
            target_step=int(config.step_count),
            attempt_id=attempt_id,
            attempt_role="resume",
        )
        resume_provenance = {
            "canonical_root": str(canonical_output_dir),
            "checkpoint_generation": checkpoint_head.generation,
            "checkpoint_identity": dict(checkpoint_head.metadata["identity"]),
            "accepted_step": int(checkpoint_head.accepted_step),
            "artifact_root": str(artifact_output_dir),
        }
    else:
        _prepare_output_dir(output_dir, resume=False)
    start = time.perf_counter()
    progress_observer = _make_run_progress_observer(output_dir=output_dir)
    progress_observer(
        {
            "status": "running",
            "phase": "setup",
            "elapsed_s": 0.0,
        }
    )
    oracle_replay_identity: dict[str, Any] | None = None
    research_probe_requested = (
        config.iqn_kalman_oracle_interpolation_target_step is not None
    )
    candidate_probe_requested = (
        config.research_initial_guess_candidate_matrix_path is not None
    )
    oracle_requested = bool(
        config.initial_guess_mode == "oracle_replay"
        or research_probe_requested
    )
    validate_oracle_identity = bool(
        research_probe_requested
        or (
            config.initial_guess_mode == "oracle_replay"
            and not bool(args.dry_run)
        )
    )
    try:
        taichi_runtime = _configure_taichi_offline_cache(
            enabled=not bool(args.disable_taichi_offline_cache),
            cache_dir=Path(args.taichi_offline_cache_dir),
        )
        taichi_runtime = {
            **taichi_runtime,
            "requested_arch": "cuda",
            "default_fp": "f32",
            "random_seed": 0,
            "strict_arch": True,
        }
        progress_observer(
            {
                "phase": "configuring",
                "taichi_runtime": taichi_runtime,
            }
        )
        source_sha256 = (
            _source_hashes()
            if resume_source_sha256 is None
            else resume_source_sha256
        )
        if validate_oracle_identity:
            oracle_replay_identity = _validate_initial_guess_oracle_producer(
                producer_output=str(
                    config.initial_guess_oracle_path
                    if config.initial_guess_mode == "oracle_replay"
                    else config.iqn_kalman_oracle_interpolation_oracle_path
                ),
                consumer_output=output_dir,
                consumer_config_payload=config_payload,
                current_source_sha256=source_sha256,
                require_full_lineage=research_probe_requested,
            )
        manifest = {
            "offline_oracle": oracle_requested,
            "offline_candidates": candidate_probe_requested,
            "deployable": bool(
                not oracle_requested
                and not candidate_probe_requested
                and not bool(args.dry_run)
            ),
            "run_label": args.run_label,
            "repo_root": str(REPO_ROOT),
            "script": str(Path(__file__).resolve()),
            "case": "official Fluent fsi_2way vertical flap",
            "solver_entry": (
                "cases.ansys_vertical_flap_fsi."
                "run_ansys_vertical_flap_benchmark"
            ),
            "numerical_solver_entry": (
                "benchmarks.official.solid_mpm_fsi_runner."
                "run_hibm_mpm_fsi"
            ),
            "selected_formulation": "selected_formulation_solver_config",
            "physical_solid_bounds": PHYSICAL_SOLID_BOUNDS,
            "config": config_payload,
            "grid": _grid_summary(config),
            "dry_run": bool(args.dry_run),
            "save_step_fields": bool(args.save_step_fields),
            "save_iqn_trial_vectors": bool(args.save_iqn_trial_vectors),
            "profile_wall_time": bool(args.profile_wall_time),
            "taichi_runtime": taichi_runtime,
            "source_sha256": source_sha256,
            "initial_guess_oracle_identity": oracle_replay_identity,
            "resume_provenance": resume_provenance,
            "artifact_root": str(artifact_output_dir),
        }
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
        _write_json_atomic(output_dir / "our_solver_config.json", config_payload)
        if args.dry_run:
            progress_observer(
                {
                    "status": "dry_run",
                    "phase": "dry_run",
                    "elapsed_s": time.perf_counter() - start,
                }
            )
            print(json.dumps({"status": "dry_run", "output_dir": str(output_dir)}))
            return 0
    except (Exception, KeyboardInterrupt) as exc:
        _record_failure_artifacts(
            output_dir=output_dir,
            exc=exc,
            elapsed_s=time.perf_counter() - start,
            config_payload=config_payload,
            config=config,
        )
        raise

    progress_observer(
        {
            "status": "running",
            "phase": "initializing",
            "elapsed_s": time.perf_counter() - start,
            "taichi_runtime": taichi_runtime,
        }
    )
    try:
        solver_started_s = time.perf_counter()
        report = run_ansys_vertical_flap_benchmark(
            config,
            step_observer=step_observer,
            progress_observer=progress_observer,
            profile_wall_time=bool(args.profile_wall_time),
        )
        solver_elapsed_s = time.perf_counter() - solver_started_s
        elapsed_s = time.perf_counter() - start
        report = dict(report)
        terminal_status = str(report.get("status", ""))
        if terminal_status in {
            "research_probe_terminal",
            "research_candidate_probe_terminal",
        }:
            if oracle_replay_identity is not None:
                report["initial_guess_oracle_identity"] = dict(
                    oracle_replay_identity
                )
            _write_history_csv(output_dir / "our_solver_history.csv", list(report["history"]))
            if step_observer is not None:
                _validate_step_artifacts(
                    artifact_output_dir,
                    expected_steps=int(report["accepted_step_count"]),
                    require_iqn_trial_vectors=bool(args.save_iqn_trial_vectors),
                )
            research_probe_elapsed_s = time.perf_counter() - start
            progress_observer(
                {
                    "status": terminal_status,
                    "phase": terminal_status,
                    "elapsed_s": research_probe_elapsed_s,
                }
            )
            _write_json(output_dir / "our_solver_report_compact.json", report)
            _write_json(
                output_dir / "our_solver_summary.json",
                {
                    "status": terminal_status,
                    "output_dir": str(output_dir),
                    "artifact_root": str(artifact_output_dir),
                    "resume_provenance": resume_provenance,
                    "offline_oracle": bool(
                        report.get("offline_oracle", False)
                    ),
                    "offline_candidates": bool(
                        report.get("offline_candidates", False)
                    ),
                    "deployable": False,
                    "accepted_step_count": int(report["accepted_step_count"]),
                    "accepted_time_s": float(report["accepted_time_s"]),
                    "research_probe_wall_time_s": float(report["research_probe_wall_time_s"]),
                    "raw_elapsed_s_including_probe": research_probe_elapsed_s,
                    "profile_wall_time_enabled": report.get(
                        "profile_wall_time_enabled"
                    ),
                    "taichi_runtime_identity": report.get(
                        "taichi_runtime_identity"
                    ),
                    "grid": _grid_summary(config),
                    "hibm_coupling_scheme": (
                        "iterative_marker_velocity_iqn_ils"
                    ),
                    "kalman_modified_physics": bool(
                        report.get("kalman_modified_physics", False)
                    ),
                    "kalman_writeback_mode": str(config.kalman_writeback_mode),
                    "marker_count": int(config.marker_count),
                    "solid_particle_counts": list(config.solid_particle_counts),
                    "solid_substeps": config.solid_substeps,
                    "solid_substeps_mode": (
                        "adaptive"
                        if config.solid_substeps is None
                        else "fixed_override"
                    ),
                    "preflow_snapshot_loaded": report.get(
                        "preflow_snapshot_loaded"
                    ),
                    "preflow_snapshot_identity": report.get(
                        "preflow_snapshot_identity"
                    ),
                    "preflow_snapshot_artifact_identity": report.get(
                        "preflow_snapshot_artifact_identity"
                    ),
                    "initial_guess_oracle_identity": report.get("initial_guess_oracle_identity"),
                },
            )
            return 0
        if oracle_replay_identity is not None:
            report["initial_guess_oracle_identity"] = dict(
                oracle_replay_identity
            )
        history = list(report.get("history", []))
        post_solver_artifact_export_started_s = time.perf_counter()
        _write_history_csv(output_dir / "our_solver_history.csv", history)
        _write_json(
            output_dir / "our_solver_report_compact.json",
            _compact_report(report),
        )

        snapshot = report.get("final_flow_field_snapshot")
        solver_npz_summary = None
        if isinstance(snapshot, dict) and snapshot:
            _write_json(
                output_dir / "final_flow_snapshot_meta.json",
                _snapshot_meta(snapshot),
            )
            solver_npz_summary = save_solver_npz_from_flow_snapshot(
                output_dir / "our_solver_final_fields.npz",
                snapshot,
                span_reduction=args.span_reduction,
                streamwise_velocity_sign=float(args.streamwise_velocity_sign),
                reverse_streamwise_axis=not bool(args.no_reverse_streamwise_axis),
                physical_solid_bounds=PHYSICAL_SOLID_BOUNDS,
            )

        step_artifact_validation = None
        if step_observer is not None:
            step_artifact_validation = _validate_step_artifacts(
                artifact_output_dir,
                expected_steps=int(config.step_count),
                require_iqn_trial_vectors=bool(args.save_iqn_trial_vectors),
            )
        post_solver_artifact_export_wall_time_s = (
            time.perf_counter()
            - post_solver_artifact_export_started_s
        )
        pre_summary_artifact_elapsed_s = time.perf_counter() - start
        summary = _summary_from_report(
            report=report,
            config=config,
            output_dir=output_dir,
            artifact_output_dir=artifact_output_dir,
            elapsed_s=elapsed_s,
            solver_npz_summary=solver_npz_summary,
            run_label=args.run_label,
            solver_elapsed_s=solver_elapsed_s,
            post_solver_artifact_export_wall_time_s=(
                post_solver_artifact_export_wall_time_s
            ),
            pre_summary_artifact_elapsed_s=pre_summary_artifact_elapsed_s,
            require_runtime_identity=True,
        )
        if step_artifact_validation is not None:
            summary["step_artifact_validation"] = step_artifact_validation
        if oracle_replay_identity is not None:
            summary["initial_guess_oracle_identity"] = dict(
                oracle_replay_identity
            )
        summary["artifact_root"] = str(artifact_output_dir)
        summary["step_field_frame_count"] = len(
            list((artifact_output_dir / "step_fields").glob("step_*.npz"))
        )
        summary["resume_provenance"] = resume_provenance
        _write_json_atomic(output_dir / "our_solver_summary.json", summary)
        terminal_elapsed_s = time.perf_counter() - start
        terminal_status = (
            "completed" if summary.get("status") == "completed" else "blocked"
        )
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _write_json_atomic(
            progress_path,
            {
                **progress,
                "status": terminal_status,
                "phase": terminal_status,
                "elapsed_s": terminal_elapsed_s,
            },
        )
        print(json.dumps(_json_safe(summary), sort_keys=True))
        return 0 if terminal_status == "completed" else 1
    except (Exception, KeyboardInterrupt) as exc:
        _record_failure_artifacts(
            output_dir=output_dir,
            exc=exc,
            elapsed_s=time.perf_counter() - start,
            config_payload=config_payload,
            config=config,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
