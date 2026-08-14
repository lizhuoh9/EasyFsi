from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_preflow_snapshot_one_step_diagnostic.py"
)
CORE_SOURCE = "simulation_core/coupling/hibm_mpm/core.py"
RUNNER_SOURCE = "benchmarks/official/solid_mpm_fsi_runner.py"


def _load_script() -> Any:
    module_name = "run_preflow_snapshot_one_step_diagnostic_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Identity:
    config_sha256: str
    source_sha256: str
    geometry_sha256: str


@dataclass(frozen=True)
class _Config:
    step_count: int = 50
    preflow_steps: int = 200
    preflow_convergence_mode: str = "windowed_stationary"
    preflow_stationary_min_steps: int = 20
    preflow_stationary_window_steps: int = 10
    preflow_stationary_consecutive_windows: int = 3
    preflow_stationary_tolerance: float = 0.01
    preflow_snapshot_input_path: str | None = None
    preflow_snapshot_output_path: str | None = "must-be-disabled"
    export_final_flow_snapshot: bool = True


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _aggregate_source(sources: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path, payload in sorted(sources.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_config(payload: Mapping[str, object]) -> str:
    return _sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def _snapshot_config_payload(config: _Config) -> dict[str, object]:
    return {
        "preflow_convergence_mode": config.preflow_convergence_mode,
        "preflow_stationary_consecutive_windows": (
            config.preflow_stationary_consecutive_windows
        ),
        "preflow_stationary_min_steps": config.preflow_stationary_min_steps,
        "preflow_stationary_tolerance": config.preflow_stationary_tolerance,
        "preflow_stationary_window_steps": config.preflow_stationary_window_steps,
        "preflow_steps": config.preflow_steps,
    }


def _write_inputs(
    tmp_path: Path,
    *,
    schema_version: int = 8,
    authority: str = "canonical",
    stored_sources: Mapping[str, bytes] | None = None,
) -> tuple[Path, Path, Path, dict[str, str]]:
    snapshot_path = tmp_path / "preflow_state"
    npz_path = tmp_path / "preflow_state.0123456789abcdef0123456789abcdef.npz"
    npz_path.write_bytes(b"immutable snapshot payload")
    config_payload = {
        "step_count": 2,
        "preflow_steps": 200,
        "preflow_convergence_mode": "windowed_stationary",
        "preflow_stationary_min_steps": 20,
        "preflow_stationary_window_steps": 10,
        "preflow_stationary_consecutive_windows": 3,
        "preflow_stationary_tolerance": 0.01,
        "preflow_snapshot_input_path": str(snapshot_path),
        "preflow_snapshot_output_path": None,
        "export_final_flow_snapshot": True,
    }
    stored_identity = {
        "config_sha256": _canonical_config(
            _snapshot_config_payload(_Config(**config_payload))
        ),
        "source_sha256": "2" * 64,
        "geometry_sha256": "3" * 64,
    }
    manifest = {
        "format": "simulation_core.preflow_snapshot",
        "schema_version": schema_version,
        "identity": stored_identity,
        "velocity_dirichlet_boundary_authority": authority,
        "npz_file": npz_path.name,
        "npz_sha256": _sha256(npz_path.read_bytes()),
        "history": {
            "preflow_status": "windowed_stationary",
            "preflow_stop_reason": "windowed_stationary",
            "preflow_steps_completed": 77,
            "preflow_steps_requested": 200,
            "preflow_converged": True,
        },
    }
    snapshot_path.with_suffix(".json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    config_path = tmp_path / "our_solver_config.json"
    config_path.write_text(
        json.dumps(config_payload, sort_keys=True),
        encoding="utf-8",
    )

    source_manifest_path = tmp_path / "run_manifest.json"
    old_sources = dict(
        stored_sources
        or {
            RUNNER_SOURCE: b"same runner",
            CORE_SOURCE: b"old core",
            "simulation_core/fluids/helper.py": b"same helper",
        }
    )
    source_manifest_path.write_text(
        json.dumps(
            {
                "config": config_payload,
                "source_sha256": {
                    **{path: _sha256(payload) for path, payload in old_sources.items()},
                    "cases/ansys_vertical_flap_fsi.py": _sha256(b"outside surface"),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return snapshot_path, config_path, source_manifest_path, stored_identity


def _runtime_for(
    module: Any,
    stored_identity: Mapping[str, str],
    *,
    current_sources: Mapping[str, bytes] | None = None,
    current_config_sha256: str | None = None,
    current_geometry_sha256: str | None = None,
    current_authority: str = "canonical",
    mutate_npz: Path | None = None,
    runtime_error: Exception | None = None,
    history_step: int = 1,
    loader_invocations: int = 1,
) -> tuple[Any, Any, list[tuple[_Identity, str]], list[_Config]]:
    sources = dict(
        current_sources
        or {
            RUNNER_SOURCE: b"same runner",
            CORE_SOURCE: b"new core",
            "simulation_core/fluids/helper.py": b"same helper",
        }
    )
    public_loader_calls: list[tuple[_Identity, str]] = []
    configs: list[_Config] = []

    def public_loader(
        _path: str | Path,
        *,
        expected_identity: _Identity,
        expected_velocity_dirichlet_boundary_authority: str,
    ) -> object:
        public_loader_calls.append(
            (expected_identity, expected_velocity_dirichlet_boundary_authority)
        )
        return object()

    runner_module = SimpleNamespace(load_preflow_snapshot=public_loader)

    def run_case(config: _Config) -> dict[str, object]:
        configs.append(config)
        assert config.step_count == 1
        assert config.preflow_snapshot_output_path is None
        assert config.export_final_flow_snapshot is False
        assert config.preflow_steps == 200
        assert config.preflow_convergence_mode == "windowed_stationary"
        assert config.preflow_stationary_min_steps == 20
        assert config.preflow_stationary_window_steps == 10
        assert config.preflow_stationary_consecutive_windows == 3
        assert config.preflow_stationary_tolerance == 0.01
        for _index in range(loader_invocations):
            runner_module.load_preflow_snapshot(
                str(config.preflow_snapshot_input_path),
                expected_identity=_Identity(
                    config_sha256=(
                        current_config_sha256 or stored_identity["config_sha256"]
                    ),
                    source_sha256=_aggregate_source(sources),
                    geometry_sha256=(
                        current_geometry_sha256 or stored_identity["geometry_sha256"]
                    ),
                ),
                expected_velocity_dirichlet_boundary_authority=current_authority,
            )
        if mutate_npz is not None:
            mutate_npz.write_bytes(b"mutated during diagnostic")
        if runtime_error is not None:
            raise runtime_error
        return {
            "history": [{"step": history_step, "diagnostic": "mock-only"}],
            "preflow_snapshot_loaded": True,
            "preflow_snapshot_identity": {
                "config_sha256": (
                    current_config_sha256 or stored_identity["config_sha256"]
                ),
                "source_sha256": _aggregate_source(sources),
                "geometry_sha256": (
                    current_geometry_sha256 or stored_identity["geometry_sha256"]
                ),
            },
            "preflow_status": "snapshot_loaded",
        }

    runtime = module.RuntimeBindings(
        snapshot_format="simulation_core.preflow_snapshot",
        snapshot_schema_version=8,
        identity_type=_Identity,
        canonical_source_sha256=_aggregate_source,
        canonical_config_sha256=_canonical_config,
        public_loader=public_loader,
        runner_module=runner_module,
        current_source_payload=lambda: sources,
        snapshot_config_payload=_snapshot_config_payload,
        config_type=_Config,
        run_case=run_case,
    )
    return runtime, runner_module, public_loader_calls, configs


def test_source_only_replay_is_one_step_isolated_and_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, runner_module, loader_calls, configs = _runtime_for(
        module,
        stored_identity,
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "1")
    monkeypatch.setenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        str(tmp_path / "warm-cache"),
    )
    monkeypatch.delenv("TI_OFFLINE_CACHE", raising=False)
    monkeypatch.delenv("TI_OFFLINE_CACHE_FILE_PATH", raising=False)

    output_dir = tmp_path / "diagnostic-output"
    payload = module.run_diagnostic_replay(
        snapshot_path=snapshot,
        config_path=config,
        source_manifest_path=source_manifest,
        output_dir=output_dir,
        allowed_source_diffs=(CORE_SOURCE,),
    )

    assert runner_module.load_preflow_snapshot is original_loader
    assert len(configs) == 1
    assert len(loader_calls) == 2
    assert [call[0].source_sha256 for call in loader_calls] == [
        stored_identity["source_sha256"],
        stored_identity["source_sha256"],
    ]
    assert payload["status"] == "completed"
    assert payload["diagnostic_replay"] is True
    assert payload["evidence_class"] == "diagnostic_only"
    assert payload["formal_validation_eligible"] is False
    assert payload["parity_claimed"] is False
    assert payload["fluent_parity_claimed"] is False
    assert payload["fresh_preflow"] is False
    assert payload["started_at_utc"].endswith("+00:00")
    assert payload["stored_preflow"] == {
        "preflow_status": "windowed_stationary",
        "preflow_stop_reason": "windowed_stationary",
        "preflow_steps_completed": 77,
        "preflow_steps_requested": 200,
        "preflow_converged": True,
    }
    assert payload["taichi_cache_environment"] == {
        "SIMULATION_TAICHI_OFFLINE_CACHE": "1",
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH": str(tmp_path / "warm-cache"),
        "TI_OFFLINE_CACHE": None,
        "TI_OFFLINE_CACHE_FILE_PATH": None,
    }
    assert payload["taichi_cache_identity_authoritative"] is False
    assert {row["path"] for row in payload["diagnostic_tool_files"]} == {
        (
            "validation_runs/ansys_vertical_flap_fsi/scripts/"
            "run_preflow_snapshot_one_step_diagnostic.py"
        ),
        (
            "validation_runs/ansys_vertical_flap_fsi/scripts/"
            "_preflow_snapshot_diagnostic_contracts.py"
        ),
    }
    assert all(
        len(row["sha256"]) == 64 for row in payload["diagnostic_tool_files"]
    )
    assert payload["production_identity_valid"] is False
    assert payload["requested_fsi_steps"] == 1
    assert payload["completed_fsi_steps"] == 1
    assert payload["preflow_snapshot_loaded"] is True
    assert payload["preflow_snapshot_output_path"] is None
    assert payload["snapshot_artifacts_unchanged"] is True
    assert payload["stored_source_sha256"] == stored_identity["source_sha256"]
    assert payload["current_source_sha256"] == _aggregate_source(
        runtime.current_source_payload()
    )
    assert payload["identity_mismatch_fields"] == ["source_sha256"]
    assert "preflow_snapshot_identity" not in payload["runtime_report"]
    assert payload["runner_requested_identity"] == {
        "config_sha256": stored_identity["config_sha256"],
        "source_sha256": _aggregate_source(runtime.current_source_payload()),
        "geometry_sha256": stored_identity["geometry_sha256"],
    }
    assert payload["validated_loader_identity"] == stored_identity
    assert payload["validated_current_config_sha256"] == stored_identity[
        "config_sha256"
    ]
    assert payload["source_file_diff"] == [
        {
            "path": CORE_SOURCE,
            "status": "changed",
            "stored_sha256": _sha256(b"old core"),
            "current_sha256": _sha256(b"new core"),
        }
    ]
    assert [path.name for path in output_dir.iterdir()] == [
        "diagnostic_replay.json"
    ]
    assert json.loads(
        (output_dir / "diagnostic_replay.json").read_text(encoding="utf-8")
    ) == payload


@pytest.mark.parametrize(
    ("identity_field", "replacement"),
    (("config_sha256", "4" * 64), ("geometry_sha256", "5" * 64)),
)
def test_replay_rejects_config_or_geometry_mismatch_and_restores_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
    replacement: str,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, runner_module, loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
        current_config_sha256=(replacement if identity_field == "config_sha256" else None),
        current_geometry_sha256=(
            replacement if identity_field == "geometry_sha256" else None
        ),
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)
    output_dir = tmp_path / f"diagnostic-{identity_field}"

    with pytest.raises(module.DiagnosticReplayError, match=identity_field):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=output_dir,
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert runner_module.load_preflow_snapshot is original_loader
    assert len(loader_calls) == 1  # Integrity preflight only; no bypassed load.
    failure = json.loads(
        (output_dir / "diagnostic_replay.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["formal_validation_eligible"] is False
    assert failure["parity_claimed"] is False


def test_replay_rejects_stale_schema_before_invoking_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(
        tmp_path,
        schema_version=7,
    )
    runtime, _runner_module, loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
    )
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="schema"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / "diagnostic-stale-schema",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert loader_calls == []


def test_replay_rejects_source_manifest_paired_with_another_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["config"]["preflow_stationary_tolerance"] = 0.05
    source_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    runtime, _runner_module, loader_calls, configs = _runtime_for(
        module,
        stored_identity,
    )
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="config"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / "diagnostic-wrong-manifest",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert loader_calls == []
    assert configs == []


@pytest.mark.parametrize("pairing_error", ("missing_input", "wrong_input", "output_set"))
def test_replay_rejects_config_not_bound_to_requested_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pairing_error: str,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if pairing_error == "missing_input":
        config_payload["preflow_snapshot_input_path"] = None
    elif pairing_error == "wrong_input":
        config_payload["preflow_snapshot_input_path"] = str(tmp_path / "other")
    else:
        config_payload["preflow_snapshot_output_path"] = str(tmp_path / "new-output")
    config.write_text(json.dumps(config_payload, sort_keys=True), encoding="utf-8")
    run_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    run_manifest["config"] = config_payload
    source_manifest.write_text(
        json.dumps(run_manifest, sort_keys=True),
        encoding="utf-8",
    )
    runtime, _runner_module, loader_calls, configs = _runtime_for(
        module,
        stored_identity,
    )
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="snapshot"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / f"diagnostic-unpaired-{pairing_error}",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert loader_calls == []
    assert configs == []


def test_replay_rejects_authority_mismatch_and_restores_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, runner_module, loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
        current_authority="legacy",
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="authority"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / "diagnostic-authority",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert runner_module.load_preflow_snapshot is original_loader
    assert len(loader_calls) == 1


@pytest.mark.parametrize("source_change", ("unexpected", "added", "removed"))
def test_source_manifest_diff_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_change: str,
) -> None:
    module = _load_script()
    stored_sources = {
        RUNNER_SOURCE: b"same runner",
        CORE_SOURCE: b"old core",
        "simulation_core/fluids/helper.py": b"same helper",
    }
    current_sources = {
        RUNNER_SOURCE: b"same runner",
        CORE_SOURCE: b"new core",
        "simulation_core/fluids/helper.py": b"same helper",
    }
    if source_change == "unexpected":
        current_sources["simulation_core/fluids/helper.py"] = b"changed helper"
    elif source_change == "added":
        current_sources["simulation_core/fluids/new_module.py"] = b"new"
    else:
        stored_sources["simulation_core/fluids/removed_module.py"] = b"old"
    snapshot, config, source_manifest, stored_identity = _write_inputs(
        tmp_path,
        stored_sources=stored_sources,
    )
    runtime, _runner_module, loader_calls, configs = _runtime_for(
        module,
        stored_identity,
        current_sources=current_sources,
    )
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="source"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / f"diagnostic-source-{source_change}",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert loader_calls == []
    assert configs == []


def test_snapshot_mutation_is_detected_after_replay_and_loader_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    manifest = json.loads(snapshot.with_suffix(".json").read_text(encoding="utf-8"))
    npz_path = snapshot.parent / manifest["npz_file"]
    runtime, runner_module, _loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
        mutate_npz=npz_path,
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)
    output_dir = tmp_path / "diagnostic-mutation"

    with pytest.raises(module.DiagnosticReplayError, match="changed"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=output_dir,
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert runner_module.load_preflow_snapshot is original_loader
    failure = json.loads(
        (output_dir / "diagnostic_replay.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["snapshot_artifacts_unchanged"] is False


def test_runtime_failure_is_recorded_without_becoming_validation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, runner_module, _loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
        runtime_error=RuntimeError("target conflict exposed"),
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)
    output_dir = tmp_path / "diagnostic-runtime-failure"

    with pytest.raises(module.DiagnosticReplayError, match="target conflict exposed"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=output_dir,
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert runner_module.load_preflow_snapshot is original_loader
    failure = json.loads(
        (output_dir / "diagnostic_replay.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "target conflict exposed"
    assert failure["evidence_class"] == "diagnostic_only"
    assert failure["formal_validation_eligible"] is False
    assert failure["parity_claimed"] is False


@pytest.mark.parametrize("loader_invocations", (0, 2))
def test_replay_requires_exactly_one_runner_loader_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader_invocations: int,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, runner_module, _loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
        loader_invocations=loader_invocations,
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="snapshot"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / f"diagnostic-load-count-{loader_invocations}",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert runner_module.load_preflow_snapshot is original_loader


def test_replay_rejects_existing_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, _runner_module, _loader_calls, configs = _runtime_for(
        module,
        stored_identity,
    )
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)
    output_dir = tmp_path / "diagnostic-existing"
    output_dir.mkdir()
    (output_dir / "do-not-touch.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(module.DiagnosticReplayError, match="already exists"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=output_dir,
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert configs == []
    assert (output_dir / "do-not-touch.txt").read_text(encoding="utf-8") == "preserve"


def test_replay_rejects_history_not_labeled_as_step_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    snapshot, config, source_manifest, stored_identity = _write_inputs(tmp_path)
    runtime, runner_module, _loader_calls, _configs = _runtime_for(
        module,
        stored_identity,
        history_step=0,
    )
    original_loader = runner_module.load_preflow_snapshot
    monkeypatch.setattr(module, "_load_runtime", lambda: runtime)

    with pytest.raises(module.DiagnosticReplayError, match="step 1"):
        module.run_diagnostic_replay(
            snapshot_path=snapshot,
            config_path=config,
            source_manifest_path=source_manifest,
            output_dir=tmp_path / "diagnostic-wrong-step",
            allowed_source_diffs=(CORE_SOURCE,),
        )

    assert runner_module.load_preflow_snapshot is original_loader
