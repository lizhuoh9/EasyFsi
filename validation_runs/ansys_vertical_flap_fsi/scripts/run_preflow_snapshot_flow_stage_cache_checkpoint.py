"""Persist production-shaped Taichi compilation at fixed FSI stage boundaries.

This diagnostic-only tool restores the validated h preflow snapshot, injects a
process-local observer into FSI step zero, and stops only after one supported
``*_after`` stage has returned.  It then synchronizes and resets Taichi so
Taichi 1.7.4 can finalize its offline cache before the process exits.  No FSI
step is completed and none of the emitted artifacts are formal/parity evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation_runs.ansys_vertical_flap_fsi.scripts._preflow_snapshot_diagnostic_contracts import (
    DiagnosticReplayError,
    json_safe,
    read_json_object,
    sha256_file,
    write_json_exclusive,
)
from validation_runs.ansys_vertical_flap_fsi.scripts.run_preflow_snapshot_one_step_diagnostic import (
    DEFAULT_ALLOWED_SOURCE_DIFFS,
    METADATA_FILENAME,
    run_diagnostic_replay,
)


FLOW_FUNCTION_NAME = "_flow_advance_current_step"
TRACTION_GATE_FUNCTION_NAME = "_require_fresh_external_force_for_solid_step"
SOLID_UPDATE_FUNCTION_NAME = "_select_and_advance_solid_macro_step"
HIBM_HEALTH_FUNCTION_NAME = "_require_hibm_velocity_dirichlet_health"
CHECKPOINT_FILENAME = "flow_stage_cache_checkpoint.json"
PROGRESS_FILENAME = "flow_stage_cache_progress.json"
FSI_FLOW_ADVANCE_AFTER_STAGE = "fsi_flow_advance_after"
TRACTION_SCATTER_GATE_AFTER_STAGE = "traction_scatter_gate_after"
SOLID_UPDATE_AFTER_STAGE = "solid_update_after"
POST_SOLID_HIBM_HEALTH_AFTER_STAGE = "post_solid_hibm_health_after"
POST_SOLID_HIBM_HEALTH_CONTEXT = "FSI step 1 post-solid observer assembly"
SUPPORTED_TARGET_STAGES = (
    "sst_wall_distance_after",
    "sst_transport_after",
    "momentum_predictor_after",
    "projection_hibm_after",
    "main_pressure_projection_after",
    FSI_FLOW_ADVANCE_AFTER_STAGE,
    TRACTION_SCATTER_GATE_AFTER_STAGE,
    SOLID_UPDATE_AFTER_STAGE,
    POST_SOLID_HIBM_HEALTH_AFTER_STAGE,
)


class _FlowStageCheckpointReached(Exception):
    """Private identity-checked marker raised after one fixed stage returns."""

    def __init__(self, *, target_stage: str, observed_stages: Sequence[str]) -> None:
        self.target_stage = target_stage
        self.observed_stages = tuple(observed_stages)
        super().__init__(
            "diagnostic flow-stage cache checkpoint reached: "
            f"target_stage={target_stage!r}"
        )


class _TaichiControl(NamedTuple):
    requested_cache_identity: Callable[[], tuple[bool | None, str | None]]
    initialized_cache_identity: Callable[[], tuple[bool | None, str | None]]
    program_present: Callable[[], bool]
    sync: Callable[[], None]
    reset: Callable[[], None]


def _load_runner_module() -> Any:
    from benchmarks.official import solid_mpm_fsi_runner

    return solid_mpm_fsi_runner


def _load_taichi_control() -> _TaichiControl:
    import taichi as ti
    from taichi.lang import impl

    from simulation_core.diagnostics import runtime as runtime_module

    def requested() -> tuple[bool | None, str | None]:
        return runtime_module._requested_offline_cache(
            runtime_module.TaichiRuntimeConfig()
        )

    def initialized() -> tuple[bool | None, str | None]:
        return (
            runtime_module._INITIALIZED_OFFLINE_CACHE,
            runtime_module._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
        )

    return _TaichiControl(
        requested_cache_identity=requested,
        initialized_cache_identity=initialized,
        program_present=lambda: impl.get_runtime().prog is not None,
        sync=ti.sync,
        reset=ti.reset,
    )


def run_flow_stage_cache_checkpoint(
    *,
    snapshot_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    target_stage: str,
    allowed_source_diffs: Sequence[str] = DEFAULT_ALLOWED_SOURCE_DIFFS,
) -> dict[str, Any]:
    """Run to one fixed FSI stage, finalize Taichi, and emit zero-step evidence."""

    _require_supported_target_stage(target_stage)
    output_path, cache_path = _validated_paths(output_dir, cache_dir)
    control = _load_taichi_control()
    _require_cache_identity(
        control.requested_cache_identity(),
        cache_path,
        label="requested",
    )
    cache_before = _cache_inventory(cache_path)
    runner = _load_runner_module()
    original_flow = getattr(runner, FLOW_FUNCTION_NAME)
    if not callable(original_flow):
        raise DiagnosticReplayError(f"runner {FLOW_FUNCTION_NAME} is not callable")
    original_traction_gate: Callable[..., Any] | None = None
    if target_stage == TRACTION_SCATTER_GATE_AFTER_STAGE:
        original_traction_gate = getattr(runner, TRACTION_GATE_FUNCTION_NAME)
        if not callable(original_traction_gate):
            raise DiagnosticReplayError(
                f"runner {TRACTION_GATE_FUNCTION_NAME} is not callable"
            )
    original_solid_update: Callable[..., Any] | None = None
    if target_stage == SOLID_UPDATE_AFTER_STAGE:
        original_solid_update = getattr(runner, SOLID_UPDATE_FUNCTION_NAME)
        if not callable(original_solid_update):
            raise DiagnosticReplayError(
                f"runner {SOLID_UPDATE_FUNCTION_NAME} is not callable"
            )
    original_hibm_health: Callable[..., Any] | None = None
    if target_stage == POST_SOLID_HIBM_HEALTH_AFTER_STAGE:
        original_hibm_health = getattr(runner, HIBM_HEALTH_FUNCTION_NAME)
        if not callable(original_hibm_health):
            raise DiagnosticReplayError(
                f"runner {HIBM_HEALTH_FUNCTION_NAME} is not callable"
            )

    observed_stages: tuple[str, ...] = ()
    expected_sentinel: _FlowStageCheckpointReached | None = None
    observer_token = object()

    def checkpoint_observer(stage_name: str) -> None:
        nonlocal observed_stages, expected_sentinel
        stage = _validated_observed_stage(stage_name)
        observed_stages = (*observed_stages, stage)
        _write_progress(
            output_path,
            target_stage=target_stage,
            observed_stages=observed_stages,
            status="running",
            target_reached=stage == target_stage,
        )
        if stage == target_stage:
            sentinel = _FlowStageCheckpointReached(
                target_stage=target_stage,
                observed_stages=observed_stages,
            )
            expected_sentinel = sentinel
            raise sentinel

    setattr(checkpoint_observer, "_flow_stage_checkpoint_token", observer_token)

    def instrumented_flow(*args: Any, **kwargs: Any) -> Any:
        existing = kwargs.get("preflow_stage_observer")
        if getattr(existing, "_flow_stage_checkpoint_token", None) is observer_token:
            return original_flow(*args, **kwargs)
        if not _is_target_fsi_invocation(kwargs):
            return original_flow(*args, **kwargs)

        def combined_observer(stage_name: str) -> None:
            if existing is not None:
                existing(stage_name)
            checkpoint_observer(stage_name)

        setattr(combined_observer, "_flow_stage_checkpoint_token", observer_token)
        forwarded = {**kwargs, "preflow_stage_observer": combined_observer}
        result = original_flow(*args, **forwarded)
        if target_stage == FSI_FLOW_ADVANCE_AFTER_STAGE:
            checkpoint_observer(FSI_FLOW_ADVANCE_AFTER_STAGE)
        return result

    def instrumented_traction_gate(*args: Any, **kwargs: Any) -> Any:
        assert original_traction_gate is not None
        result = original_traction_gate(*args, **kwargs)
        checkpoint_observer(TRACTION_SCATTER_GATE_AFTER_STAGE)
        return result

    def instrumented_solid_update(*args: Any, **kwargs: Any) -> Any:
        assert original_solid_update is not None
        result = original_solid_update(*args, **kwargs)
        checkpoint_observer(SOLID_UPDATE_AFTER_STAGE)
        return result

    def instrumented_hibm_health(*args: Any, **kwargs: Any) -> Any:
        assert original_hibm_health is not None
        result = original_hibm_health(*args, **kwargs)
        if kwargs.get("context") == POST_SOLID_HIBM_HEALTH_CONTEXT:
            checkpoint_observer(POST_SOLID_HIBM_HEALTH_AFTER_STAGE)
        return result

    setattr(runner, FLOW_FUNCTION_NAME, instrumented_flow)
    traction_gate_patched = False
    solid_update_patched = False
    hibm_health_patched = False
    if original_traction_gate is not None:
        try:
            setattr(
                runner,
                TRACTION_GATE_FUNCTION_NAME,
                instrumented_traction_gate,
            )
            traction_gate_patched = True
        except BaseException:
            try:
                setattr(runner, FLOW_FUNCTION_NAME, original_flow)
            finally:
                setattr(
                    runner,
                    TRACTION_GATE_FUNCTION_NAME,
                    original_traction_gate,
                )
            raise
    if original_solid_update is not None:
        try:
            setattr(
                runner,
                SOLID_UPDATE_FUNCTION_NAME,
                instrumented_solid_update,
            )
            solid_update_patched = True
        except BaseException:
            try:
                setattr(runner, FLOW_FUNCTION_NAME, original_flow)
            finally:
                setattr(
                    runner,
                    SOLID_UPDATE_FUNCTION_NAME,
                    original_solid_update,
                )
            raise
    if original_hibm_health is not None:
        try:
            setattr(
                runner,
                HIBM_HEALTH_FUNCTION_NAME,
                instrumented_hibm_health,
            )
            hibm_health_patched = True
        except BaseException:
            try:
                setattr(runner, FLOW_FUNCTION_NAME, original_flow)
            finally:
                setattr(
                    runner,
                    HIBM_HEALTH_FUNCTION_NAME,
                    original_hibm_health,
                )
            raise
    replay_error: DiagnosticReplayError | None = None
    unexpected_error: Exception | None = None
    try:
        try:
            run_diagnostic_replay(
                snapshot_path=snapshot_path,
                config_path=config_path,
                source_manifest_path=source_manifest_path,
                output_dir=output_path,
                allowed_source_diffs=allowed_source_diffs,
            )
        except DiagnosticReplayError as exc:
            replay_error = exc
        except Exception as exc:  # fail closed if the base contract changes.
            unexpected_error = exc
    finally:
        if traction_gate_patched:
            try:
                setattr(runner, FLOW_FUNCTION_NAME, original_flow)
            finally:
                assert original_traction_gate is not None
                setattr(
                    runner,
                    TRACTION_GATE_FUNCTION_NAME,
                    original_traction_gate,
                )
        elif solid_update_patched:
            try:
                setattr(runner, FLOW_FUNCTION_NAME, original_flow)
            finally:
                assert original_solid_update is not None
                setattr(
                    runner,
                    SOLID_UPDATE_FUNCTION_NAME,
                    original_solid_update,
                )
        elif hibm_health_patched:
            try:
                setattr(runner, FLOW_FUNCTION_NAME, original_flow)
            finally:
                assert original_hibm_health is not None
                setattr(
                    runner,
                    HIBM_HEALTH_FUNCTION_NAME,
                    original_hibm_health,
                )
        else:
            setattr(runner, FLOW_FUNCTION_NAME, original_flow)

    finalize: Mapping[str, Any] | None = None
    finalize_error: DiagnosticReplayError | None = None
    try:
        finalize = _finalize_taichi_cache(control, cache_path)
    except DiagnosticReplayError as exc:
        finalize_error = exc
    cache_after = _cache_inventory(cache_path)
    if unexpected_error is not None:
        primary = DiagnosticReplayError(
            f"diagnostic replay escaped its error contract: {unexpected_error}"
        )
        _attach_secondary_failure(primary, finalize_error)
        raise primary from unexpected_error
    if replay_error is None:
        primary = DiagnosticReplayError(
            "diagnostic replay completed before the requested flow-stage checkpoint"
        )
        _attach_secondary_failure(primary, finalize_error)
        raise primary
    if expected_sentinel is None or replay_error.__cause__ is not expected_sentinel:
        _attach_secondary_failure(replay_error, finalize_error)
        raise replay_error
    if finalize_error is not None:
        raise finalize_error
    assert finalize is not None

    return _classify_checkpoint(
        output_path=output_path,
        replay_error=replay_error,
        sentinel=expected_sentinel,
        cache_path=cache_path,
        cache_before=cache_before,
        cache_after=cache_after,
        finalize=finalize,
    )


def _require_supported_target_stage(target_stage: str) -> None:
    if target_stage not in SUPPORTED_TARGET_STAGES:
        raise DiagnosticReplayError(
            "target flow stage is not a supported canonical after-stage: "
            f"{target_stage!r}"
        )


def _validated_paths(
    output_dir: str | Path,
    cache_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    cache_path = Path(cache_dir).resolve()
    if not cache_path.is_dir():
        raise DiagnosticReplayError(
            f"Taichi cache directory does not exist: {cache_path}"
        )
    resolved_output = output_path.resolve()
    if resolved_output == cache_path or cache_path in resolved_output.parents:
        raise DiagnosticReplayError("diagnostic output must not be inside the cache")
    return output_path, cache_path


def _normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _require_cache_identity(
    identity: tuple[bool | None, str | None],
    cache_path: Path,
    *,
    label: str,
) -> None:
    enabled, configured_path = identity
    if enabled is not True or configured_path is None:
        raise DiagnosticReplayError(
            f"{label} Taichi offline-cache identity is not enabled with a path"
        )
    if _normalized_path(configured_path) != _normalized_path(cache_path):
        raise DiagnosticReplayError(
            f"{label} Taichi cache identity differs from the requested directory: "
            f"configured={configured_path!r}, requested={str(cache_path)!r}"
        )


def _validated_observed_stage(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise DiagnosticReplayError("observed flow stage name is invalid")
    return value


def _is_target_fsi_invocation(kwargs: Mapping[str, Any]) -> bool:
    return (
        kwargs.get("flow_phase") == "fsi"
        and int(kwargs.get("step_index_local", -1)) == 0
    )


def _cache_inventory(cache_path: Path) -> dict[str, Any]:
    files = tuple(sorted(path for path in cache_path.rglob("*") if path.is_file()))
    sizes = tuple(path.stat().st_size for path in files)
    mtimes = tuple(path.stat().st_mtime_ns for path in files)
    index_path = cache_path / "ticache.tcb"
    return {
        "cache_dir": str(cache_path),
        "file_count": len(files),
        "total_bytes": int(sum(sizes)),
        "latest_mtime_ns": int(max(mtimes, default=0)),
        "index_present": index_path.is_file(),
        "index_bytes": index_path.stat().st_size if index_path.is_file() else 0,
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
    }


def _finalize_taichi_cache(
    control: Any,
    cache_path: Path,
) -> dict[str, Any]:
    _require_cache_identity(
        control.initialized_cache_identity(),
        cache_path,
        label="initialized",
    )
    present_before = bool(control.program_present())
    if not present_before:
        raise DiagnosticReplayError(
            "Taichi program is unavailable before cache finalization"
        )
    try:
        control.sync()
        control.reset()
    except Exception as exc:
        raise DiagnosticReplayError(
            f"Taichi cache finalization failed: {exc}"
        ) from exc
    present_after = bool(control.program_present())
    if present_after:
        raise DiagnosticReplayError("Taichi program remains live after reset")
    return {
        "taichi_sync_succeeded": True,
        "taichi_reset_succeeded": True,
        "program_present_before_reset": present_before,
        "program_present_after_reset": present_after,
    }


def _attach_secondary_failure(
    primary: BaseException,
    secondary: BaseException | None,
) -> None:
    if secondary is None:
        return
    note = f"secondary Taichi cache-finalization failure: {secondary}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    existing = tuple(getattr(primary, "__notes__", ()))
    primary.__notes__ = [*existing, note]


def _classify_checkpoint(
    *,
    output_path: Path,
    replay_error: DiagnosticReplayError,
    sentinel: _FlowStageCheckpointReached,
    cache_path: Path,
    cache_before: Mapping[str, Any],
    cache_after: Mapping[str, Any],
    finalize: Mapping[str, Any],
) -> dict[str, Any]:
    _require_usable_cache_artifact(cache_after)
    base = read_json_object(
        output_path / METADATA_FILENAME,
        label="base diagnostic replay metadata",
    )
    invalid = _invalid_base_evidence(base, sentinel)
    if invalid:
        raise DiagnosticReplayError(
            "cannot classify flow-stage cache checkpoint: " + ", ".join(invalid)
        ) from replay_error
    payload = {
        "flow_stage_cache_checkpoint": True,
        "diagnostic_replay": True,
        "status": "cache_checkpoint_completed",
        "target_stage": sentinel.target_stage,
        "target_after_stage_reached": True,
        "observed_stages": list(sentinel.observed_stages),
        "requested_fsi_steps": 1,
        "completed_fsi_steps": 0,
        "full_fsi_step_completed": False,
        "one_step_completed": False,
        "interrupted_before_step_completion": True,
        **dict(finalize),
        "cache_dir": str(cache_path),
        "cache_inventory_before": dict(cache_before),
        "cache_inventory_after": dict(cache_after),
        "cache_inventory_changed": dict(cache_before) != dict(cache_after),
        "loader_invocation_count": 1,
        "canonical_snapshot_loaded": True,
        "preflow_snapshot_loaded": True,
        "evidence_class": "diagnostic_only",
        "formal_validation_eligible": False,
        "fresh_preflow": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "production_identity_valid": False,
        "snapshot_hashes_before": json_safe(base["snapshot_hashes_before"]),
        "snapshot_hashes_after": json_safe(base["snapshot_hashes_after"]),
        "snapshot_artifacts_unchanged": True,
        "snapshot_mutation_detected": False,
        "allowed_source_diffs": json_safe(base.get("allowed_source_diffs")),
        "stored_identity": json_safe(base.get("stored_identity")),
        "current_identity": json_safe(base.get("current_identity")),
        "base_error_type": type(sentinel).__name__,
        "base_diagnostic_metadata": METADATA_FILENAME,
        "output_dir": str(output_path),
        "checkpoint_tool": _tool_identity(Path(__file__).resolve()),
    }
    write_json_exclusive(output_path / CHECKPOINT_FILENAME, payload)
    _write_progress(
        output_path,
        target_stage=sentinel.target_stage,
        observed_stages=sentinel.observed_stages,
        status="cache_checkpoint_completed",
        target_reached=True,
        extra={
            **dict(finalize),
            "cache_inventory_changed": payload["cache_inventory_changed"],
        },
    )
    return payload


def _require_usable_cache_artifact(cache_after: Mapping[str, Any]) -> None:
    if (
        cache_after.get("index_present") is not True
        or int(cache_after.get("index_bytes", 0)) <= 0
        or not cache_after.get("index_sha256")
        or int(cache_after.get("file_count", 0)) <= 0
        or int(cache_after.get("total_bytes", 0)) <= 0
    ):
        raise DiagnosticReplayError(
            "Taichi reset returned without a usable offline cache artifact"
        )


def _invalid_base_evidence(
    base: Mapping[str, Any],
    sentinel: _FlowStageCheckpointReached,
) -> list[str]:
    expected = {
        "status": "failed",
        "diagnostic_replay": True,
        "completed_fsi_steps": 0,
        "loader_invocation_count": 1,
        "canonical_snapshot_loaded": True,
        "preflow_snapshot_loaded": True,
        "snapshot_artifacts_unchanged": True,
        "snapshot_mutation_detected": False,
        "formal_validation_eligible": False,
        "fresh_preflow": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "production_identity_valid": False,
        "error_type": type(sentinel).__name__,
        "error": str(sentinel),
    }
    invalid = [key for key, value in expected.items() if base.get(key) != value]
    if base.get("snapshot_hashes_before") != base.get("snapshot_hashes_after"):
        invalid.append("snapshot_hashes")
    if "history" in base or "runtime_report" in base:
        invalid.append("completed_step_payload")
    return invalid


def _write_progress(
    output_path: Path,
    *,
    target_stage: str,
    observed_stages: Sequence[str],
    status: str,
    target_reached: bool,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "flow_stage_cache_checkpoint": True,
        "diagnostic_replay": True,
        "status": status,
        "target_stage": target_stage,
        "target_after_stage_reached": bool(target_reached),
        "observed_stages": list(observed_stages),
        "last_stage": observed_stages[-1] if observed_stages else None,
        "completed_fsi_steps": 0,
        "formal_validation_eligible": False,
        "fresh_preflow": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "production_identity_valid": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **dict(extra or {}),
    }
    path = output_path / PROGRESS_FILENAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tool_identity(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one fixed production-shaped FSI flow stage, then sync and "
            "reset Taichi so its offline cache is persisted."
        )
    )
    parser.add_argument(
        "--snapshot",
        "--snapshot-prefix",
        dest="snapshot",
        required=True,
        type=Path,
    )
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--source-manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--target-stage",
        required=True,
        choices=SUPPORTED_TARGET_STAGES,
    )
    parser.add_argument("--allow-source-diff", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    allowed = tuple(args.allow_source_diff or DEFAULT_ALLOWED_SOURCE_DIFFS)
    try:
        payload = run_flow_stage_cache_checkpoint(
            snapshot_path=args.snapshot,
            config_path=args.config_json,
            source_manifest_path=args.source_manifest_json,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            target_stage=args.target_stage,
            allowed_source_diffs=allowed,
        )
    except Exception as exc:  # pragma: no cover - CLI failure path.
        print(f"[flow_stage_cache_checkpoint] ERROR: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"[flow_stage_cache_checkpoint] NOTE: {note}", file=sys.stderr)
        return 1
    print(
        "[flow_stage_cache_checkpoint] "
        f"status={payload['status']}, target_stage={payload['target_stage']}, "
        f"completed_fsi_steps={payload['completed_fsi_steps']}, "
        f"cache_inventory_changed={payload['cache_inventory_changed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
