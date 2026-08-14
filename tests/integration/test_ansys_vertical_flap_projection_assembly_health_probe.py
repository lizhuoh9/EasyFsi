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
    / "run_preflow_snapshot_projection_assembly_health_probe.py"
)
CORE_SOURCE = "simulation_core/coupling/hibm_mpm/core.py"
RUNNER_SOURCE = "benchmarks/official/solid_mpm_fsi_runner.py"
HASHES = {
    "metadata_sha256": "a" * 64,
    "npz_sha256": "b" * 64,
}
SHARED_PROBE_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_preflow_snapshot_pre_predictor_health_probe.py"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_projection_assembly_health_probe",
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
    base_overrides: dict[str, Any] | None = None,
) -> Any:
    def replay(**kwargs: Any) -> dict[str, Any]:
        assert tuple(kwargs["allowed_source_diffs"]) == (CORE_SOURCE, RUNNER_SOURCE)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        seen["pre_predictor_return"] = (
            runner._require_hibm_velocity_dirichlet_health(
                {"stage": "pre-predictor"},
                context=module.PRE_PREDICTOR_HEALTH_CONTEXT,
            )
        )
        try:
            runner._require_hibm_velocity_dirichlet_health(
                health_report,
                context=module.TARGET_HEALTH_CONTEXT,
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
        raise AssertionError("projection health call did not interrupt the replay")

    return replay


def _run_probe(module: Any, tmp_path: Path, output_dir: Path) -> dict[str, Any]:
    return module.run_projection_assembly_health_probe(
        snapshot_path=tmp_path / "preflow_state",
        config_path=tmp_path / "config.json",
        source_manifest_path=tmp_path / "run_manifest.json",
        output_dir=output_dir,
        allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
    )


def test_projection_gate_passes_after_original_and_writes_stage_specific_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    events: list[tuple[str, str]] = []

    def original(report: dict[str, Any], *, context: str) -> None:
        events.append((str(report.get("stage", "target")), context))
        report["observed_after_original"] = True
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    health_report = {"stage": "projection", "target_conflict_count": 0}
    seen: dict[str, Any] = {}
    output_dir = tmp_path / "projection-gate"
    monkeypatch.setattr(
        module._health_probe_support,
        "_load_runner_module",
        lambda: runner,
    )
    monkeypatch.setattr(
        module._health_probe_support,
        "run_diagnostic_replay",
        _wrapping_replay(module, runner, health_report, seen),
    )

    payload = _run_probe(module, tmp_path, output_dir)

    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert seen["pre_predictor_return"] is None
    assert events == [
        ("pre-predictor", module.PRE_PREDICTOR_HEALTH_CONTEXT),
        ("projection", module.TARGET_HEALTH_CONTEXT),
    ]
    assert seen["wrapped"].__cause__ is seen["cause"]
    assert payload["projection_assembly_health_probe"] is True
    assert payload["status"] == "diagnostic_gate_passed"
    assert payload["diagnostic_gate_passed"] is True
    assert payload["target_context"] == module.TARGET_HEALTH_CONTEXT
    assert payload["completed_fsi_steps"] == 0
    assert payload["original_health_check_passed"] is True
    assert payload["projection_assembly_health_check_passed"] is True
    assert payload["stopped_before_main_pressure_projection"] is True
    assert "stopped_before_projection_hibm_assembly" not in payload
    assert payload["full_fsi_step_completed"] is False
    assert payload["one_step_completed"] is False
    assert payload["formal_validation_eligible"] is False
    assert payload["fresh_preflow"] is False
    assert payload["parity_claimed"] is False
    assert payload["snapshot_hashes_before"] == HASHES
    assert payload["snapshot_hashes_after"] == HASHES
    assert payload["snapshot_artifacts_unchanged"] is True
    assert payload["health_report"]["observed_after_original"] is True
    assert payload["probe_tools"] == [
        {
            "path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        },
        {
            "path": SHARED_PROBE_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(SHARED_PROBE_PATH.read_bytes()).hexdigest(),
        },
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


def test_original_projection_health_error_remains_failure_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    original_error = RuntimeError("projection health failure")

    def original(_report: dict[str, Any], *, context: str) -> None:
        if context == module.TARGET_HEALTH_CONTEXT:
            raise original_error
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    seen: dict[str, Any] = {}
    output_dir = tmp_path / "original-health-failure"
    monkeypatch.setattr(
        module._health_probe_support,
        "_load_runner_module",
        lambda: runner,
    )
    monkeypatch.setattr(
        module._health_probe_support,
        "run_diagnostic_replay",
        _wrapping_replay(module, runner, {}, seen),
    )

    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        _run_probe(module, tmp_path, output_dir)

    assert exc_info.value is seen["wrapped"]
    assert exc_info.value.__cause__ is original_error
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"probe_filename": "../escape.json"}, "basename"),
        ({"probe_marker_field": "status"}, "reserved"),
        ({"stopped_before_field": "diagnostic_replay"}, "reserved"),
    ),
)
def test_generic_rejects_unsafe_dynamic_contract_fields_before_loading_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, str],
    match: str,
) -> None:
    module = _load_script()
    support = module._health_probe_support
    malicious = module._PROJECTION_ASSEMBLY_CONTRACT._replace(**mutation)
    monkeypatch.setattr(
        support,
        "_load_runner_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("runner must not load for an unsafe contract")
        ),
    )

    with pytest.raises(module.DiagnosticReplayError, match=match):
        support.run_hibm_health_gate_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=tmp_path / "must-not-exist",
            contract=malicious,
            allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_generic_rejects_equal_but_noncanonical_contract_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    support = module._health_probe_support
    supported = module._PROJECTION_ASSEMBLY_CONTRACT
    clone = support.HibmHealthGateProbeContract(*supported)
    assert clone == supported
    assert clone is not supported
    monkeypatch.setattr(
        support,
        "_load_runner_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("runner must not load for a cloned contract")
        ),
    )

    with pytest.raises(module.DiagnosticReplayError, match="supported"):
        support.run_hibm_health_gate_probe(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "run_manifest.json",
            output_dir=tmp_path / "must-not-exist",
            contract=clone,
            allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_same_projection_sentinel_type_with_different_instance_is_rejected(
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
        spoof = module._ProjectionAssemblyHealthGatePassed(
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

    monkeypatch.setattr(
        module._health_probe_support,
        "_load_runner_module",
        lambda: runner,
    )
    monkeypatch.setattr(
        module._health_probe_support,
        "run_diagnostic_replay",
        replay,
    )

    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        _run_probe(module, tmp_path, output_dir)

    assert exc_info.value is seen["wrapped"]
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_projection_gate_hash_drift_cannot_be_reclassified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def original(_report: dict[str, Any], *, context: str) -> None:
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    output_dir = tmp_path / "hash-drift"
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        module._health_probe_support,
        "_load_runner_module",
        lambda: runner,
    )
    monkeypatch.setattr(
        module._health_probe_support,
        "run_diagnostic_replay",
        _wrapping_replay(
            module,
            runner,
            {},
            seen,
            base_overrides={
                "snapshot_artifacts_unchanged": False,
                "snapshot_mutation_detected": True,
                "snapshot_hashes_after": {
                    "metadata_sha256": "9" * 64,
                    "npz_sha256": "b" * 64,
                },
            },
        ),
    )

    with pytest.raises(module.DiagnosticReplayError, match="classif"):
        _run_probe(module, tmp_path, output_dir)

    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_projection_probe_write_failure_restores_health_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def original(_report: dict[str, Any], *, context: str) -> None:
        return None

    runner = SimpleNamespace(_require_hibm_velocity_dirichlet_health=original)
    original_identity = runner._require_hibm_velocity_dirichlet_health
    output_dir = tmp_path / "write-failure"
    seen: dict[str, Any] = {}
    write_error = module.DiagnosticReplayError("projection probe write failed")
    monkeypatch.setattr(
        module._health_probe_support,
        "_load_runner_module",
        lambda: runner,
    )
    monkeypatch.setattr(
        module._health_probe_support,
        "run_diagnostic_replay",
        _wrapping_replay(module, runner, {}, seen),
    )
    monkeypatch.setattr(
        module._health_probe_support,
        "write_json_exclusive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(write_error),
    )

    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        _run_probe(module, tmp_path, output_dir)

    assert exc_info.value is write_error
    assert runner._require_hibm_velocity_dirichlet_health is original_identity
    assert not (output_dir / module.PROBE_FILENAME).exists()


def test_projection_cli_is_fixed_context_and_reports_zero_step_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    parser = module._build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--target-context" not in option_strings
    args = parser.parse_args(
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
            "projection_assembly_health_check_passed": True,
            "stopped_before_main_pressure_projection": True,
            "full_fsi_step_completed": False,
        }

    monkeypatch.setattr(module, "run_projection_assembly_health_probe", passed)
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
    assert "projection_assembly_health_check_passed=True" in stdout
    assert "stopped_before_main_pressure_projection=True" in stdout
    assert "full_fsi_step_completed=False" in stdout
    assert "one-step completed" not in stdout
