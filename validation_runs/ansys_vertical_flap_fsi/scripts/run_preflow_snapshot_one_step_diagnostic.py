"""Run one non-authoritative FSI step from a source-stale preflow snapshot."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation_runs.ansys_vertical_flap_fsi.scripts._preflow_snapshot_diagnostic_contracts import (
    IDENTITY_FIELDS,
    DiagnosticReplayError,
    SnapshotArtifacts,
    SourceDiffEvidence,
    identity_to_mapping,
    inspect_snapshot,
    json_safe,
    read_json_object,
    sha256_file,
    snapshot_hashes,
    stored_preflow_provenance,
    validate_source_manifest_diff,
    validated_identity_mapping,
    validated_sha256,
    write_json_exclusive,
)


DEFAULT_ALLOWED_SOURCE_DIFFS = ("simulation_core/coupling/hibm_mpm/core.py",)
REQUIRED_PREFLOW_STEPS = 200
CONFIG_OVERRIDE_FIELDS = frozenset(
    {
        "step_count",
        "preflow_snapshot_input_path",
        "preflow_snapshot_output_path",
        "export_final_flow_snapshot",
    }
)
REQUIRED_STATIONARY_CONFIG_FIELDS = (
    "preflow_convergence_mode",
    "preflow_stationary_min_steps",
    "preflow_stationary_window_steps",
    "preflow_stationary_consecutive_windows",
    "preflow_stationary_tolerance",
)
TAICHI_CACHE_ENVIRONMENT_FIELDS = (
    "SIMULATION_TAICHI_OFFLINE_CACHE",
    "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
    "TI_OFFLINE_CACHE",
    "TI_OFFLINE_CACHE_FILE_PATH",
)
METADATA_FILENAME = "diagnostic_replay.json"
HELPER_PATH = Path(__file__).with_name("_preflow_snapshot_diagnostic_contracts.py")


@dataclass(frozen=True)
class RuntimeBindings:
    snapshot_format: str
    snapshot_schema_version: int
    identity_type: type[Any]
    canonical_source_sha256: Callable[[Mapping[str, bytes]], str]
    canonical_config_sha256: Callable[[Mapping[str, Any]], str]
    public_loader: Callable[..., Any]
    runner_module: Any
    current_source_payload: Callable[[], Mapping[str, bytes]]
    snapshot_config_payload: Callable[[Any], Mapping[str, Any]]
    config_type: type[Any]
    run_case: Callable[[Any], Mapping[str, Any]]


def _load_runtime() -> RuntimeBindings:
    from benchmarks.official import solid_mpm_fsi_runner as runner_module
    from cases.ansys_vertical_flap_fsi import (
        VerticalFlapFsiConfig,
        run_vertical_flap_fsi_smoke,
    )
    from simulation_core.fluids.preflow_snapshot import (
        PREFLOW_SNAPSHOT_FORMAT,
        PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        PreflowSnapshotIdentity,
        canonical_config_sha256,
        canonical_source_sha256,
        load_preflow_snapshot,
    )

    return RuntimeBindings(
        snapshot_format=PREFLOW_SNAPSHOT_FORMAT,
        snapshot_schema_version=PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        identity_type=PreflowSnapshotIdentity,
        canonical_source_sha256=canonical_source_sha256,
        canonical_config_sha256=canonical_config_sha256,
        public_loader=load_preflow_snapshot,
        runner_module=runner_module,
        current_source_payload=runner_module._preflow_snapshot_source_payload,
        snapshot_config_payload=runner_module._preflow_snapshot_config_payload,
        config_type=VerticalFlapFsiConfig,
        run_case=run_vertical_flap_fsi_smoke,
    )


def run_diagnostic_replay(
    *,
    snapshot_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    allowed_source_diffs: Sequence[str] = DEFAULT_ALLOWED_SOURCE_DIFFS,
) -> dict[str, Any]:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    runtime = _load_runtime()
    artifacts = inspect_snapshot(
        snapshot_path,
        snapshot_format=runtime.snapshot_format,
        snapshot_schema_version=runtime.snapshot_schema_version,
    )
    stored_config = read_json_object(Path(config_path), label="solver config")
    _require_paired_snapshot_config(
        stored_config=stored_config,
        snapshot_path=artifacts.snapshot_path,
    )
    source_evidence = validate_source_manifest_diff(
        source_manifest_path=source_manifest_path,
        expected_config=stored_config,
        current_payload=runtime.current_source_payload(),
        canonical_source_sha256=runtime.canonical_source_sha256,
        stored_source_sha256=artifacts.stored_identity["source_sha256"],
        allowed_source_diffs=allowed_source_diffs,
    )
    config = _load_one_step_config(
        stored_config=stored_config,
        snapshot_path=artifacts.snapshot_path,
        config_type=runtime.config_type,
    )
    current_config_sha256 = _validated_current_config_sha256(
        runtime=runtime,
        config=config,
        stored_config_sha256=artifacts.stored_identity["config_sha256"],
    )

    runtime.public_loader(
        artifacts.snapshot_path,
        expected_identity=runtime.identity_type(**artifacts.stored_identity),
        expected_velocity_dirichlet_boundary_authority=artifacts.stored_authority,
    )
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DiagnosticReplayError(
            f"diagnostic output directory already exists: {output_path}"
        ) from exc
    metadata_path = output_path / METADATA_FILENAME
    observations: dict[str, Any] = {}
    base_payload = _base_metadata(
        artifacts=artifacts,
        config_path=Path(config_path),
        source_manifest_path=Path(source_manifest_path),
        output_dir=output_path,
        source_evidence=source_evidence,
        current_config_sha256=current_config_sha256,
        started_at_utc=started_at_utc,
    )

    report: Mapping[str, Any] | None = None
    sanitized_report: Mapping[str, Any] | None = None
    runner_requested_identity: Mapping[str, str] | None = None
    execution_error: Exception | None = None
    execution_traceback = ""
    try:
        report = _run_with_temporary_source_only_loader(
            runtime=runtime,
            artifacts=artifacts,
            source_evidence=source_evidence,
            config=config,
            observations=observations,
        )
        _require_exactly_one_completed_step(report)
        sanitized_report, runner_requested_identity = _sanitize_runtime_report(
            report=report,
            current_identity=observations.get("current_identity"),
        )
    except Exception as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    try:
        hashes_after: Mapping[str, str] = snapshot_hashes(
            artifacts.metadata_path,
            artifacts.npz_path,
        )
    except Exception as exc:
        hashes_after = {
            "metadata_sha256": "unavailable",
            "npz_sha256": "unavailable",
        }
        if execution_error is None:
            execution_error = DiagnosticReplayError(
                f"snapshot artifacts could not be hashed after replay: {exc}"
            )
            execution_traceback = traceback.format_exc()
    artifacts_unchanged = dict(hashes_after) == dict(artifacts.hashes_before)
    if not artifacts_unchanged and execution_error is None:
        execution_error = DiagnosticReplayError(
            "snapshot JSON or NPZ changed during the diagnostic replay"
        )
        execution_traceback = ""

    payload = {
        **base_payload,
        **observations,
        "snapshot_hashes_after": dict(hashes_after),
        "snapshot_artifacts_unchanged": artifacts_unchanged,
        "snapshot_mutation_detected": not artifacts_unchanged,
    }
    if execution_error is None:
        assert sanitized_report is not None and runner_requested_identity is not None
        payload.update(
            {
                "status": "completed",
                "completed_fsi_steps": 1,
                "preflow_snapshot_loaded": True,
                "runner_requested_identity": dict(runner_requested_identity),
                "validated_loader_identity": dict(artifacts.stored_identity),
                "runtime_report": json_safe(sanitized_report),
            }
        )
        write_json_exclusive(metadata_path, payload)
        return payload

    payload.update(
        {
            "status": "failed",
            "completed_fsi_steps": 0,
            "preflow_snapshot_loaded": bool(
                observations.get("canonical_snapshot_loaded", False)
            ),
            "error_type": type(execution_error).__name__,
            "error": str(execution_error),
            "traceback": execution_traceback,
        }
    )
    write_json_exclusive(metadata_path, payload)
    raise DiagnosticReplayError(
        f"diagnostic one-step replay failed: {execution_error}"
    ) from execution_error


def _require_paired_snapshot_config(
    *,
    stored_config: Mapping[str, Any],
    snapshot_path: Path,
) -> None:
    input_path = stored_config.get("preflow_snapshot_input_path")
    output_path = stored_config.get("preflow_snapshot_output_path")
    if not isinstance(input_path, str) or not input_path:
        raise DiagnosticReplayError(
            "paired config snapshot input path is missing"
        )
    if output_path is not None:
        raise DiagnosticReplayError(
            "paired config snapshot output path must be null for replay"
        )
    if _resolve_repo_path(input_path) != _resolve_repo_path(snapshot_path):
        raise DiagnosticReplayError(
            "paired config snapshot input does not identify the requested stable prefix"
        )


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _load_one_step_config(
    *,
    stored_config: Mapping[str, Any],
    snapshot_path: Path,
    config_type: type[Any],
) -> Any:
    one_step_payload = {
        **stored_config,
        "step_count": 1,
        "preflow_snapshot_input_path": str(snapshot_path),
        "preflow_snapshot_output_path": None,
        "export_final_flow_snapshot": False,
    }
    try:
        config = config_type(**one_step_payload)
    except (TypeError, ValueError) as exc:
        raise DiagnosticReplayError(f"solver config cannot be reconstructed: {exc}") from exc
    if getattr(config, "preflow_steps", None) != REQUIRED_PREFLOW_STEPS:
        raise DiagnosticReplayError(
            "diagnostic replay requires the paired 200-step preflow configuration"
        )
    missing_stationary = [
        field_name
        for field_name in REQUIRED_STATIONARY_CONFIG_FIELDS
        if field_name not in stored_config
    ]
    if missing_stationary:
        raise DiagnosticReplayError(
            f"solver config is missing stationary preflow fields: {missing_stationary}"
        )
    changed_non_overrides = [
        field_name
        for field_name, stored_value in stored_config.items()
        if field_name not in CONFIG_OVERRIDE_FIELDS
        and getattr(config, field_name, object()) != stored_value
    ]
    if changed_non_overrides:
        raise DiagnosticReplayError(
            "diagnostic config changed non-FSI/preflow identity fields: "
            f"{changed_non_overrides}"
        )
    if (
        getattr(config, "step_count", None) != 1
        or getattr(config, "preflow_snapshot_output_path", "invalid") is not None
        or getattr(config, "export_final_flow_snapshot", None) is not False
        or Path(str(getattr(config, "preflow_snapshot_input_path", "")))
        != snapshot_path
    ):
        raise DiagnosticReplayError(
            "diagnostic config did not preserve the exactly-one-step/no-output contract"
        )
    return config


def _validated_current_config_sha256(
    *,
    runtime: RuntimeBindings,
    config: Any,
    stored_config_sha256: str,
) -> str:
    try:
        current_sha256 = runtime.canonical_config_sha256(
            runtime.snapshot_config_payload(config)
        )
    except (TypeError, ValueError) as exc:
        raise DiagnosticReplayError(
            f"current preflow config identity cannot be computed: {exc}"
        ) from exc
    validated_sha256(current_sha256, field_name="current preflow config aggregate")
    if current_sha256 != stored_config_sha256:
        raise DiagnosticReplayError(
            "current preflow config does not match the stored snapshot: "
            f"stored={stored_config_sha256}, current={current_sha256}"
        )
    return current_sha256


def _run_with_temporary_source_only_loader(
    *,
    runtime: RuntimeBindings,
    artifacts: SnapshotArtifacts,
    source_evidence: SourceDiffEvidence,
    config: Any,
    observations: dict[str, Any],
) -> Mapping[str, Any]:
    original_loader = runtime.runner_module.load_preflow_snapshot
    if original_loader is not runtime.public_loader:
        raise DiagnosticReplayError(
            "runner loader reference is already patched or differs from the public loader"
        )
    invocation_count = 0

    def diagnostic_loader(
        path: str | Path,
        *,
        expected_identity: Any,
        expected_velocity_dirichlet_boundary_authority: str = "legacy",
    ) -> Any:
        nonlocal invocation_count
        invocation_count += 1
        observations["loader_invocation_count"] = invocation_count
        if invocation_count != 1:
            raise DiagnosticReplayError(
                "diagnostic replay attempted to load the preflow snapshot more than once"
            )
        if Path(path) != artifacts.snapshot_path:
            raise DiagnosticReplayError(
                f"runner requested an unexpected snapshot path: {path}"
            )
        current_identity = identity_to_mapping(expected_identity)
        mismatch_fields = [
            field_name
            for field_name in IDENTITY_FIELDS
            if current_identity[field_name] != artifacts.stored_identity[field_name]
        ]
        observations.update(
            {
                "current_identity": current_identity,
                "identity_mismatch_fields": mismatch_fields,
            }
        )
        if mismatch_fields != ["source_sha256"]:
            raise DiagnosticReplayError(
                "source_sha256 must be the sole snapshot identity mismatch; "
                f"mismatch_fields={mismatch_fields}"
            )
        if current_identity["source_sha256"] != source_evidence.current_source_sha256:
            raise DiagnosticReplayError(
                "runner current source_sha256 does not match the independently "
                "computed source payload aggregate"
            )
        if expected_velocity_dirichlet_boundary_authority != artifacts.stored_authority:
            raise DiagnosticReplayError(
                "snapshot authority does not match the current solver: "
                f"stored={artifacts.stored_authority!r}, "
                f"current={expected_velocity_dirichlet_boundary_authority!r}"
            )
        loader_identity = runtime.identity_type(
            config_sha256=current_identity["config_sha256"],
            source_sha256=artifacts.stored_identity["source_sha256"],
            geometry_sha256=current_identity["geometry_sha256"],
        )
        snapshot = runtime.public_loader(
            path,
            expected_identity=loader_identity,
            expected_velocity_dirichlet_boundary_authority=(
                expected_velocity_dirichlet_boundary_authority
            ),
        )
        observations["canonical_snapshot_loaded"] = True
        return snapshot

    runtime.runner_module.load_preflow_snapshot = diagnostic_loader
    try:
        report = runtime.run_case(config)
    finally:
        runtime.runner_module.load_preflow_snapshot = original_loader
    if invocation_count != 1:
        raise DiagnosticReplayError(
            "diagnostic runner did not restore exactly one preflow snapshot"
        )
    if not isinstance(report, Mapping):
        raise DiagnosticReplayError("diagnostic runner report is not a mapping")
    return report


def _require_exactly_one_completed_step(report: Mapping[str, Any]) -> None:
    history = report.get("history")
    if (
        not isinstance(history, Sequence)
        or isinstance(history, (str, bytes, bytearray))
        or len(history) != 1
    ):
        count = len(history) if isinstance(history, Sequence) else "invalid"
        raise DiagnosticReplayError(
            f"diagnostic runner did not complete exactly one FSI step: {count}"
        )
    if report.get("preflow_snapshot_loaded") is not True:
        raise DiagnosticReplayError(
            "diagnostic runner did not report the preflow snapshot as loaded"
        )
    row = history[0]
    if not isinstance(row, Mapping) or row.get("step") != 1:
        raise DiagnosticReplayError(
            "diagnostic runner history row is not labeled as step 1"
        )


def _sanitize_runtime_report(
    *,
    report: Mapping[str, Any],
    current_identity: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    runner_requested_identity = validated_identity_mapping(
        report.get("preflow_snapshot_identity")
    )
    if runner_requested_identity != current_identity:
        raise DiagnosticReplayError(
            "runtime report snapshot identity does not match the runner request"
        )
    sanitized = {
        key: value
        for key, value in report.items()
        if key != "preflow_snapshot_identity"
    }
    return sanitized, runner_requested_identity


def _base_metadata(
    *,
    artifacts: SnapshotArtifacts,
    config_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    source_evidence: SourceDiffEvidence,
    current_config_sha256: str,
    started_at_utc: str,
) -> dict[str, Any]:
    return {
        "started_at_utc": started_at_utc,
        "diagnostic_replay": True,
        "evidence_class": "diagnostic_only",
        "formal_validation_eligible": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "fresh_preflow": False,
        "production_identity_valid": False,
        "production_identity_reason": "source_sha256_mismatch_diagnostic_only",
        "requested_fsi_steps": 1,
        "preflow_snapshot_output_path": None,
        "snapshot_path": str(artifacts.snapshot_path),
        "snapshot_metadata_path": str(artifacts.metadata_path),
        "snapshot_npz_path": str(artifacts.npz_path),
        "config_path": str(config_path),
        "source_manifest_path": str(source_manifest_path),
        "output_dir": str(output_dir),
        "snapshot_schema_version": artifacts.manifest["schema_version"],
        "snapshot_authority": artifacts.stored_authority,
        "stored_preflow": stored_preflow_provenance(artifacts.manifest),
        "stored_identity": dict(artifacts.stored_identity),
        "stored_source_sha256": artifacts.stored_identity["source_sha256"],
        "current_source_sha256": source_evidence.current_source_sha256,
        "validated_current_config_sha256": current_config_sha256,
        "allowed_source_diffs": list(source_evidence.allowed_paths),
        "source_file_diff": [dict(row) for row in source_evidence.rows],
        "snapshot_hashes_before": dict(artifacts.hashes_before),
        "taichi_cache_environment": {
            name: os.environ.get(name) for name in TAICHI_CACHE_ENVIRONMENT_FIELDS
        },
        "taichi_cache_identity_authoritative": False,
        "diagnostic_tool_files": _diagnostic_tool_files(),
    }


def _diagnostic_tool_files() -> list[dict[str, str]]:
    paths = (Path(__file__).resolve(), HELPER_PATH.resolve())
    return [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one diagnostic-only ANSYS vertical-flap FSI step from a "
            "source-stale but config/geometry-identical preflow snapshot."
        )
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--source-manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-source-diff",
        action="append",
        default=None,
        help=(
            "Exact changed source path allowed for this diagnostic; repeat for "
            "multiple paths. Defaults to the current core.py repair only."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    allowed = tuple(args.allow_source_diff or DEFAULT_ALLOWED_SOURCE_DIFFS)
    try:
        payload = run_diagnostic_replay(
            snapshot_path=args.snapshot,
            config_path=args.config_json,
            source_manifest_path=args.source_manifest_json,
            output_dir=args.output_dir,
            allowed_source_diffs=allowed,
        )
    except Exception as exc:  # pragma: no cover - command-line failure path.
        print(f"[preflow_snapshot_one_step_diagnostic] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[preflow_snapshot_one_step_diagnostic] wrote "
        f"{payload['status']} diagnostic-only evidence to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
