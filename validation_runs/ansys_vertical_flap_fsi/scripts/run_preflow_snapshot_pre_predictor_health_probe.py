"""Stop a diagnostic replay immediately after pre-predictor health succeeds.

This process-local probe is deliberately non-authoritative.  It delegates
snapshot loading, source-diff validation, output isolation, and snapshot
integrity checks to the existing replay, then recognizes only the exact
sentinel raised after the production health validator accepts the target
pre-predictor report.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
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


PROBE_FILENAME = "pre_predictor_health_probe.json"
TARGET_HEALTH_CONTEXT = "fsi step 0 pre-predictor assembly"
PROJECTION_ASSEMBLY_PROBE_FILENAME = "projection_assembly_health_probe.json"
PROJECTION_ASSEMBLY_HEALTH_CONTEXT = "fsi step 0 projection assembly"
HEALTH_FUNCTION_NAME = "_require_hibm_velocity_dirichlet_health"


class _HibmHealthGatePassed(Exception):
    """Base for a private, identity-checked health-gate control-flow marker."""

    def __init__(
        self,
        *,
        context: str,
        report: Mapping[str, Any],
        message: str,
    ) -> None:
        self.context = context
        self.report = dict(report)
        super().__init__(message)


class _PrePredictorHealthGatePassed(_HibmHealthGatePassed):
    """Private control-flow marker for one exact validated health call."""

    def __init__(self, *, context: str, report: Mapping[str, Any]) -> None:
        super().__init__(
            context=context,
            report=report,
            message=(
                "diagnostic pre-predictor health gate passed: "
                f"context={context!r}"
            ),
        )


class _ProjectionAssemblyHealthGatePassed(_HibmHealthGatePassed):
    """Private marker for the fixed projection-assembly health call."""

    def __init__(self, *, context: str, report: Mapping[str, Any]) -> None:
        super().__init__(
            context=context,
            report=report,
            message=(
                "diagnostic projection-assembly health gate passed: "
                f"context={context!r}"
            ),
        )


class HibmHealthGateProbeContract(NamedTuple):
    """Fixed internal contract for one supported production health gate."""

    target_context: str
    probe_filename: str
    probe_marker_field: str
    stage_health_check_passed_field: str | None
    stopped_before_field: str
    classification_label: str
    sentinel_type: type[_HibmHealthGatePassed]
    probe_tool_path: Path


_PRE_PREDICTOR_CONTRACT = HibmHealthGateProbeContract(
    target_context=TARGET_HEALTH_CONTEXT,
    probe_filename=PROBE_FILENAME,
    probe_marker_field="pre_predictor_health_probe",
    stage_health_check_passed_field=None,
    stopped_before_field="stopped_before_projection_hibm_assembly",
    classification_label="pre-predictor",
    sentinel_type=_PrePredictorHealthGatePassed,
    probe_tool_path=Path(__file__).resolve(),
)
_PROJECTION_ASSEMBLY_CONTRACT = HibmHealthGateProbeContract(
    target_context=PROJECTION_ASSEMBLY_HEALTH_CONTEXT,
    probe_filename=PROJECTION_ASSEMBLY_PROBE_FILENAME,
    probe_marker_field="projection_assembly_health_probe",
    stage_health_check_passed_field="projection_assembly_health_check_passed",
    stopped_before_field="stopped_before_main_pressure_projection",
    classification_label="projection-assembly",
    sentinel_type=_ProjectionAssemblyHealthGatePassed,
    probe_tool_path=Path(__file__).with_name(
        "run_preflow_snapshot_projection_assembly_health_probe.py"
    ).resolve(),
)
SUPPORTED_CONTRACTS = (
    _PRE_PREDICTOR_CONTRACT,
    _PROJECTION_ASSEMBLY_CONTRACT,
)
_RESERVED_PROBE_PAYLOAD_FIELDS = frozenset(
    {
        "diagnostic_replay",
        "status",
        "diagnostic_gate_passed",
        "target_context",
        "requested_fsi_steps",
        "completed_fsi_steps",
        "original_health_check_passed",
        "full_fsi_step_completed",
        "one_step_completed",
        "interrupted_before_step_completion",
        "loader_invocation_count",
        "canonical_snapshot_loaded",
        "preflow_snapshot_loaded",
        "base_error_type",
        "base_diagnostic_metadata",
        "evidence_class",
        "formal_validation_eligible",
        "parity_claimed",
        "fluent_parity_claimed",
        "fresh_preflow",
        "production_identity_valid",
        "snapshot_hashes_before",
        "snapshot_hashes_after",
        "snapshot_artifacts_unchanged",
        "snapshot_mutation_detected",
        "allowed_source_diffs",
        "stored_identity",
        "current_identity",
        "health_report",
        "output_dir",
        "probe_tool",
        "probe_tools",
    }
)


def _require_supported_contract(
    contract: HibmHealthGateProbeContract,
) -> None:
    if not isinstance(contract, HibmHealthGateProbeContract):
        raise DiagnosticReplayError("health gate probe contract type is invalid")
    filename = Path(contract.probe_filename)
    if filename.is_absolute() or filename.name != contract.probe_filename:
        raise DiagnosticReplayError(
            "health gate probe filename must be a single basename"
        )
    dynamic_fields = (
        contract.probe_marker_field,
        contract.stopped_before_field,
        contract.stage_health_check_passed_field,
    )
    field_names = tuple(value for value in dynamic_fields if value is not None)
    if any(not isinstance(value, str) or not value for value in field_names):
        raise DiagnosticReplayError(
            "health gate probe dynamic fields must be non-empty strings"
        )
    if len(set(field_names)) != len(field_names):
        raise DiagnosticReplayError(
            "health gate probe dynamic fields must be distinct"
        )
    reserved = sorted(set(field_names) & _RESERVED_PROBE_PAYLOAD_FIELDS)
    if reserved:
        raise DiagnosticReplayError(
            f"health gate probe dynamic fields overlap reserved fields: {reserved}"
        )
    if not any(contract is supported for supported in SUPPORTED_CONTRACTS):
        raise DiagnosticReplayError(
            "health gate probe contract is not a supported canonical object"
        )


def _load_runner_module() -> Any:
    from benchmarks.official import solid_mpm_fsi_runner

    return solid_mpm_fsi_runner


def run_pre_predictor_health_probe(
    *,
    snapshot_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    allowed_source_diffs: Sequence[str] = DEFAULT_ALLOWED_SOURCE_DIFFS,
) -> dict[str, Any]:
    """Run until the target production health validator succeeds, then stop."""

    return run_hibm_health_gate_probe(
        snapshot_path=snapshot_path,
        config_path=config_path,
        source_manifest_path=source_manifest_path,
        output_dir=output_dir,
        contract=_PRE_PREDICTOR_CONTRACT,
        allowed_source_diffs=allowed_source_diffs,
    )


def run_hibm_health_gate_probe(
    *,
    snapshot_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    contract: HibmHealthGateProbeContract,
    allowed_source_diffs: Sequence[str] = DEFAULT_ALLOWED_SOURCE_DIFFS,
) -> dict[str, Any]:
    """Stop after one fixed production HIBM health gate accepts its report."""

    _require_supported_contract(contract)
    runner = _load_runner_module()
    original_health = getattr(runner, HEALTH_FUNCTION_NAME)
    if not callable(original_health):
        raise DiagnosticReplayError(
            f"runner {HEALTH_FUNCTION_NAME} is not callable"
        )
    expected_sentinel: _HibmHealthGatePassed | None = None

    def instrumented_health(
        report: Mapping[str, Any],
        *,
        context: str,
    ) -> Any:
        nonlocal expected_sentinel
        result = original_health(report, context=context)
        if context != contract.target_context:
            return result
        if result is not None:
            raise DiagnosticReplayError(
                f"target {contract.classification_label} health validator "
                "returned a non-None value"
            )
        captured = json_safe(report)
        if not isinstance(captured, Mapping):
            raise DiagnosticReplayError(
                f"target {contract.classification_label} health report is "
                "not a mapping"
            )
        sentinel = contract.sentinel_type(
            context=context,
            report=dict(captured),
        )
        expected_sentinel = sentinel
        raise sentinel

    setattr(runner, HEALTH_FUNCTION_NAME, instrumented_health)
    try:
        try:
            run_diagnostic_replay(
                snapshot_path=snapshot_path,
                config_path=config_path,
                source_manifest_path=source_manifest_path,
                output_dir=output_dir,
                allowed_source_diffs=allowed_source_diffs,
            )
        except DiagnosticReplayError as replay_error:
            if (
                expected_sentinel is None
                or replay_error.__cause__ is not expected_sentinel
            ):
                raise
            return _classify_and_write_gate_pass(
                output_dir=Path(output_dir),
                sentinel=expected_sentinel,
                replay_error=replay_error,
                contract=contract,
            )
        raise DiagnosticReplayError(
            "diagnostic replay completed before the target "
            f"{contract.classification_label} health gate could be isolated"
        )
    finally:
        setattr(runner, HEALTH_FUNCTION_NAME, original_health)


def _classify_and_write_gate_pass(
    *,
    output_dir: Path,
    sentinel: _HibmHealthGatePassed,
    replay_error: DiagnosticReplayError,
    contract: HibmHealthGateProbeContract,
) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    try:
        base = read_json_object(
            metadata_path,
            label="base diagnostic replay metadata",
        )
    except DiagnosticReplayError as exc:
        raise DiagnosticReplayError(
            f"cannot classify {contract.classification_label} health gate pass: "
            "base diagnostic metadata is unavailable"
        ) from exc

    invalid = _invalid_base_evidence(
        base=base,
        sentinel=sentinel,
        target_context=contract.target_context,
    )
    if invalid:
        raise DiagnosticReplayError(
            f"cannot classify {contract.classification_label} health gate pass: "
            + ", ".join(invalid)
        ) from replay_error

    hashes_before = dict(base["snapshot_hashes_before"])
    hashes_after = dict(base["snapshot_hashes_after"])
    probe_tools = _probe_tool_identities(contract)
    payload = {
        contract.probe_marker_field: True,
        "diagnostic_replay": True,
        "status": "diagnostic_gate_passed",
        "diagnostic_gate_passed": True,
        "target_context": sentinel.context,
        "requested_fsi_steps": 1,
        "completed_fsi_steps": 0,
        "original_health_check_passed": True,
        contract.stopped_before_field: True,
        "full_fsi_step_completed": False,
        "one_step_completed": False,
        "interrupted_before_step_completion": True,
        "loader_invocation_count": 1,
        "canonical_snapshot_loaded": True,
        "preflow_snapshot_loaded": True,
        "base_error_type": type(sentinel).__name__,
        "base_diagnostic_metadata": METADATA_FILENAME,
        "evidence_class": "diagnostic_only",
        "formal_validation_eligible": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "fresh_preflow": False,
        "production_identity_valid": False,
        "snapshot_hashes_before": hashes_before,
        "snapshot_hashes_after": hashes_after,
        "snapshot_artifacts_unchanged": True,
        "snapshot_mutation_detected": False,
        "allowed_source_diffs": json_safe(base.get("allowed_source_diffs")),
        "stored_identity": json_safe(base.get("stored_identity")),
        "current_identity": json_safe(base.get("current_identity")),
        "health_report": json_safe(sentinel.report),
        "output_dir": str(output_dir),
        "probe_tool": probe_tools[0],
        "probe_tools": probe_tools,
    }
    if contract.stage_health_check_passed_field is not None:
        payload[contract.stage_health_check_passed_field] = True
    write_json_exclusive(output_dir / contract.probe_filename, payload)
    return payload


def _invalid_base_evidence(
    *,
    base: Mapping[str, Any],
    sentinel: _HibmHealthGatePassed,
    target_context: str,
) -> list[str]:
    expected = {
        "status": "failed",
        "diagnostic_replay": True,
        "evidence_class": "diagnostic_only",
        "formal_validation_eligible": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "fresh_preflow": False,
        "production_identity_valid": False,
        "requested_fsi_steps": 1,
        "completed_fsi_steps": 0,
        "loader_invocation_count": 1,
        "canonical_snapshot_loaded": True,
        "preflow_snapshot_loaded": True,
        "error_type": type(sentinel).__name__,
        "error": str(sentinel),
        "snapshot_artifacts_unchanged": True,
        "snapshot_mutation_detected": False,
    }
    invalid = [
        key for key, value in expected.items() if base.get(key) != value
    ]
    if sentinel.context != target_context:
        invalid.append("target_context")
    hashes_before = base.get("snapshot_hashes_before")
    hashes_after = base.get("snapshot_hashes_after")
    if not isinstance(hashes_before, Mapping):
        invalid.append("snapshot_hashes_before")
    if not isinstance(hashes_after, Mapping):
        invalid.append("snapshot_hashes_after")
    if (
        isinstance(hashes_before, Mapping)
        and isinstance(hashes_after, Mapping)
        and dict(hashes_before) != dict(hashes_after)
    ):
        invalid.append("snapshot_hashes_equal")
    return invalid


def _probe_tool_identity(tool_path: Path) -> dict[str, str]:
    tool_path = tool_path.resolve()
    return {
        "path": tool_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(tool_path),
    }


def _probe_tool_identities(
    contract: HibmHealthGateProbeContract,
) -> list[dict[str, str]]:
    paths: list[Path] = []
    for candidate in (contract.probe_tool_path, Path(__file__).resolve()):
        resolved = candidate.resolve()
        if resolved not in paths:
            paths.append(resolved)
    return [_probe_tool_identity(path) for path in paths]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostic-only ANSYS vertical-flap replay until the "
            "production pre-predictor health gate succeeds; complete no FSI step."
        )
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--source-manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-source-diff", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    allowed = tuple(args.allow_source_diff or DEFAULT_ALLOWED_SOURCE_DIFFS)
    try:
        payload = run_pre_predictor_health_probe(
            snapshot_path=args.snapshot,
            config_path=args.config_json,
            source_manifest_path=args.source_manifest_json,
            output_dir=args.output_dir,
            allowed_source_diffs=allowed,
        )
    except Exception as exc:  # pragma: no cover - command-line failure path.
        print(
            f"[preflow_snapshot_pre_predictor_health_probe] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[preflow_snapshot_pre_predictor_health_probe] wrote diagnostic-only "
        f"gate evidence with status {payload['status']!r}, "
        f"completed_fsi_steps={payload['completed_fsi_steps']}, "
        "original_health_check_passed="
        f"{payload['original_health_check_passed']}, "
        "stopped_before_projection_hibm_assembly="
        f"{payload['stopped_before_projection_hibm_assembly']}, "
        "full_fsi_step_completed="
        f"{payload['full_fsi_step_completed']} to "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
