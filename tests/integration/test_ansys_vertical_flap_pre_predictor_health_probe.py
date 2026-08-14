from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_preflow_snapshot_pre_predictor_health_probe.py"
)
CORE_SOURCE = "simulation_core/coupling/hibm_mpm/core.py"
RUNNER_SOURCE = "benchmarks/official/solid_mpm_fsi_runner.py"
HASHES = {
    "metadata_sha256": "a" * 64,
    "npz_sha256": "b" * 64,
}


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_pre_predictor_health_probe",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_failure_payload(
    cause: BaseException,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "error_type": type(cause).__name__,
        "error": str(cause),
        "snapshot_hashes_before": dict(HASHES),
        "snapshot_hashes_after": dict(HASHES),
        "snapshot_artifacts_unchanged": True,
        "snapshot_mutation_detected": False,
        "allowed_source_diffs": [CORE_SOURCE, RUNNER_SOURCE],
        "stored_identity": {
            "config_sha256": "c" * 64,
            "source_sha256": "d" * 64,
            "geometry_sha256": "e" * 64,
        },
        "current_identity": {
            "config_sha256": "c" * 64,
            "source_sha256": "f" * 64,
            "geometry_sha256": "e" * 64,
        },
    }
    if overrides:
        payload.update(overrides)
    return payload


def _wrapping_replay(
    module: Any,
    runner: Any,
    health_report: dict[str, Any],
    seen: dict[str, Any],
    *,
    context: str | None = None,
    base_overrides: dict[str, Any] | None = None,
) -> Any:
    def replay(**kwargs: Any) -> dict[str, Any]:
        assert tuple(kwargs["allowed_source_diffs"]) == (CORE_SOURCE, RUNNER_SOURCE)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            runner._require_hibm_velocity_dirichlet_health(
                health_report,
                context=context or module.TARGET_HEALTH_CONTEXT,
            )
        except Exception as cause:
            seen["cause"] = cause
            payload = _base_failure_payload(cause, overrides=base_overrides)
            (output_dir / "diagnostic_replay.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            wrapped = module.DiagnosticReplayError("base diagnostic failed")
            seen["wrapped"] = wrapped
            raise wrapped from cause
        raise AssertionError("target health call did not interrupt the replay")

    return replay


def test_target_gate_passes_after_original_and_writes_zero_step_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    events: list[tuple[str, str]] = []

    def original(report: dict[str, Any], *, context: str) -> None:
        events.append(("original", context))
        report["observed_after_original"] = True
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    health_report = {
        "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": 10,
        "canonical_velocity_dirichlet_report": {
            "duplicate_claim_component_count": 10,
            "direct_geometry_reconstructed_component_count": 0,
        },
    }
    seen: dict[str, Any] = {}
    output_dir = tmp_path / "new-health-probe"
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(
        module,
        "run_diagnostic_replay",
        _wrapping_replay(module, runner, health_report, seen),
    )

    payload = module.run_pre_predictor_health_probe(
        snapshot_path=tmp_path / "preflow_state",
        config_path=tmp_path / "config.json",
        source_manifest_path=tmp_path / "run_manifest.json",
        output_dir=output_dir,
        allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
    )

    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert events == [("original", module.TARGET_HEALTH_CONTEXT)]
    assert seen["wrapped"].__cause__ is seen["cause"]
    assert payload["status"] == "diagnostic_gate_passed"
    assert payload["diagnostic_gate_passed"] is True
    assert payload["target_context"] == module.TARGET_HEALTH_CONTEXT
    assert payload["requested_fsi_steps"] == 1
    assert payload["completed_fsi_steps"] == 0
    assert payload["original_health_check_passed"] is True
    assert payload["stopped_before_projection_hibm_assembly"] is True
    assert payload["full_fsi_step_completed"] is False
    assert payload["one_step_completed"] is False
    assert payload["interrupted_before_step_completion"] is True
    assert payload["evidence_class"] == "diagnostic_only"
    assert payload["formal_validation_eligible"] is False
    assert payload["fresh_preflow"] is False
    assert payload["parity_claimed"] is False
    assert payload["fluent_parity_claimed"] is False
    assert payload["snapshot_hashes_before"] == HASHES
    assert payload["snapshot_hashes_after"] == HASHES
    assert payload["snapshot_artifacts_unchanged"] is True
    assert payload["health_report"]["observed_after_original"] is True
    assert payload["probe_tools"] == [
        {
            "path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        }
    ]
    assert payload["probe_tool"] == payload["probe_tools"][0]
    assert "history" not in payload
    assert "runtime_report" not in payload
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "diagnostic_replay.json",
        module.PROBE_FILENAME,
    ]
    written = json.loads(
        (output_dir / module.PROBE_FILENAME).read_text(encoding="utf-8")
    )
    assert written == payload


def test_original_health_runtime_error_remains_failure_and_restores_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    original_error = RuntimeError("original health failure")

    def original(_report: dict[str, Any], *, context: str) -> None:
        assert context == module.TARGET_HEALTH_CONTEXT
        raise original_error

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    seen: dict[str, Any] = {}
    output_dir = tmp_path / "original-health-failure"
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(
        module,
        "run_diagnostic_replay",
        _wrapping_replay(module, runner, {}, seen),
    )

    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        module.run_pre_predictor_health_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
            allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
        )

    assert exc_info.value is seen["wrapped"]
    assert exc_info.value.__cause__ is original_error
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_non_target_context_passes_through_but_completed_base_is_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    marker = object()
    observed: dict[str, Any] = {}

    def original(report: dict[str, Any], *, context: str) -> object:
        observed["original"] = (report, context)
        return marker

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    output_dir = tmp_path / "non-target"

    def replay(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=False)
        observed["returned"] = runner._require_hibm_velocity_dirichlet_health(
            {"stage": "other"},
            context="preflow step 0 pre-predictor assembly",
        )
        return {"status": "completed", "completed_fsi_steps": 1}

    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)

    with pytest.raises(module.DiagnosticReplayError, match="completed"):
        module.run_pre_predictor_health_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
        )

    assert observed["returned"] is marker
    assert observed["original"][1] != module.TARGET_HEALTH_CONTEXT
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_same_sentinel_type_with_different_instance_is_not_classified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def original(_report: dict[str, Any], *, context: str) -> None:
        raise AssertionError(f"original must not be called: {context}")

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    output_dir = tmp_path / "spoof"
    seen: dict[str, Any] = {}

    def replay(**kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs["output_dir"])
        path.mkdir(parents=True, exist_ok=False)
        spoof = module._PrePredictorHealthGatePassed(
            context=module.TARGET_HEALTH_CONTEXT,
            report={"spoof": True},
        )
        (path / "diagnostic_replay.json").write_text(
            json.dumps(_base_failure_payload(spoof)),
            encoding="utf-8",
        )
        wrapped = module.DiagnosticReplayError("spoof wrapper")
        seen["wrapped"] = wrapped
        raise wrapped from spoof

    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)

    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        module.run_pre_predictor_health_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
        )

    assert exc_info.value is seen["wrapped"]
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


@pytest.mark.parametrize(
    "overrides",
    (
        {"loader_invocation_count": 0},
        {"canonical_snapshot_loaded": False},
        {"preflow_snapshot_loaded": False},
        {"error_type": "RuntimeError"},
        {"status": "completed"},
        {"completed_fsi_steps": 1},
        {
            "snapshot_artifacts_unchanged": False,
            "snapshot_mutation_detected": True,
            "snapshot_hashes_after": {
                "metadata_sha256": "9" * 64,
                "npz_sha256": "b" * 64,
            },
        },
    ),
)
def test_invalid_base_evidence_cannot_be_reclassified_as_gate_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    module = _load_script()

    def original(_report: dict[str, Any], *, context: str) -> None:
        assert context == module.TARGET_HEALTH_CONTEXT
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    seen: dict[str, Any] = {}
    output_dir = tmp_path / "invalid-base"
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(
        module,
        "run_diagnostic_replay",
        _wrapping_replay(
            module,
            runner,
            {},
            seen,
            base_overrides=overrides,
        ),
    )

    with pytest.raises(module.DiagnosticReplayError, match="classif"):
        module.run_pre_predictor_health_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
            allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
        )

    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_existing_output_is_preserved_and_function_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def original(_report: dict[str, Any], *, context: str) -> None:
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    def replay(**_kwargs: Any) -> dict[str, Any]:
        raise module.DiagnosticReplayError("output directory already exists")

    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)

    with pytest.raises(module.DiagnosticReplayError, match="already exists"):
        module.run_pre_predictor_health_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=output_dir,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_cli_preserves_repeated_source_diffs_and_reports_zero_step_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    args = module._build_parser().parse_args(
        [
            "--snapshot",
            "snapshot",
            "--config-json",
            "config.json",
            "--source-manifest-json",
            "manifest.json",
            "--output-dir",
            "new-output",
            "--allow-source-diff",
            CORE_SOURCE,
            "--allow-source-diff",
            RUNNER_SOURCE,
        ]
    )
    assert args.allow_source_diff == [CORE_SOURCE, RUNNER_SOURCE]

    observed: dict[str, Any] = {}

    def passed(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "status": "diagnostic_gate_passed",
            "completed_fsi_steps": 0,
            "diagnostic_gate_passed": True,
            "original_health_check_passed": True,
            "stopped_before_projection_hibm_assembly": True,
            "full_fsi_step_completed": False,
        }

    monkeypatch.setattr(module, "run_pre_predictor_health_probe", passed)
    exit_code = module.main(
        [
            "--snapshot",
            "snapshot",
            "--config-json",
            "config.json",
            "--source-manifest-json",
            "manifest.json",
            "--output-dir",
            "new-output",
            "--allow-source-diff",
            CORE_SOURCE,
            "--allow-source-diff",
            RUNNER_SOURCE,
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert tuple(observed["allowed_source_diffs"]) == (CORE_SOURCE, RUNNER_SOURCE)
    assert "diagnostic-only" in stdout
    assert "completed_fsi_steps=0" in stdout
    assert "original_health_check_passed=True" in stdout
    assert "stopped_before_projection_hibm_assembly=True" in stdout
    assert "full_fsi_step_completed=False" in stdout
    assert "one-step completed" not in stdout
