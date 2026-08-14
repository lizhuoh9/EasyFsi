from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "scripts"
    / "run_preflow_snapshot_flow_stage_cache_checkpoint.py"
)
CORE_SOURCE = "simulation_core/coupling/hibm_mpm/core.py"
RUNNER_SOURCE = "benchmarks/official/solid_mpm_fsi_runner.py"
POST_SOLID_STAGE = "post_solid_hibm_health_after"
TARGET_CONTEXT = "FSI step 1 post-solid observer assembly"
HASHES = {"metadata_sha256": "a" * 64, "npz_sha256": "b" * 64}


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_post_solid_cache_checkpoint", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTaichi:
    def __init__(
        self,
        cache_dir: Path,
        *,
        sync_error: Exception | None = None,
        reset_error: Exception | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.program_present_value = True
        self.sync_error = sync_error
        self.reset_error = reset_error
        self.calls: list[str] = []

    def requested(self) -> tuple[bool, str]:
        return True, str(self.cache_dir.resolve())

    def initialized(self) -> tuple[bool, str]:
        return True, str(self.cache_dir.resolve())

    def program_present(self) -> bool:
        self.calls.append("program_present")
        return self.program_present_value

    def sync(self) -> None:
        self.calls.append("sync")
        if self.sync_error is not None:
            raise self.sync_error

    def reset(self) -> None:
        self.calls.append("reset")
        if self.reset_error is not None:
            raise self.reset_error
        (self.cache_dir / "ticache.tcb").write_bytes(b"cached")
        self.program_present_value = False

    def control(self) -> Any:
        return SimpleNamespace(
            requested_cache_identity=self.requested,
            initialized_cache_identity=self.initialized,
            program_present=self.program_present,
            sync=self.sync,
            reset=self.reset,
        )


def _base_payload(cause: BaseException) -> dict[str, Any]:
    return {
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
    }


def _wrapping_replay(
    module: Any,
    invoke: Callable[[], Any],
    seen: dict[str, Any],
) -> Callable[..., dict[str, Any]]:
    def replay(**kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        try:
            seen["return"] = invoke()
        except Exception as cause:
            seen["cause"] = cause
            (output / "diagnostic_replay.json").write_text(
                json.dumps(_base_payload(cause)), encoding="utf-8"
            )
            wrapped = module.DiagnosticReplayError("base diagnostic failed")
            seen["wrapped"] = wrapped
            raise wrapped from cause
        return {"status": "completed", "completed_fsi_steps": 1}

    return replay


def _run(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    invoke: Callable[[], Any],
    fake: _FakeTaichi,
    *,
    target_stage: str = POST_SOLID_STAGE,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake.control)
    monkeypatch.setattr(
        module, "run_diagnostic_replay", _wrapping_replay(module, invoke, seen)
    )
    payload = module.run_flow_stage_cache_checkpoint(
        snapshot_path=tmp_path / "preflow_state",
        config_path=tmp_path / "config.json",
        source_manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "checkpoint-output",
        cache_dir=fake.cache_dir,
        target_stage=target_stage,
        allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
    )
    seen["payload"] = payload
    return seen


def test_exact_post_solid_context_calls_original_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    events: list[str] = []

    def flow(*_args: Any, **_kwargs: Any) -> None:
        return None

    def health(_report: Any, *, context: str) -> None:
        assert context == TARGET_CONTEXT
        events.append("original_health_return")

    runner = SimpleNamespace(
        _flow_advance_current_step=flow,
        _require_hibm_velocity_dirichlet_health=health,
    )
    identities = (flow, health)

    def invoke() -> Any:
        try:
            return runner._require_hibm_velocity_dirichlet_health(
                {"healthy": True}, context=TARGET_CONTEXT
            )
        except Exception:
            events.append("post_solid_sentinel")
            raise

    seen = _run(module, tmp_path, monkeypatch, runner, invoke, _FakeTaichi(cache))
    assert events == ["original_health_return", "post_solid_sentinel"]
    assert seen["wrapped"].__cause__ is seen["cause"]
    assert seen["payload"]["target_stage"] == POST_SOLID_STAGE
    assert seen["payload"]["completed_fsi_steps"] == 0
    assert (
        runner._flow_advance_current_step,
        runner._require_hibm_velocity_dirichlet_health,
    ) == identities


def test_earlier_health_contexts_pass_through_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = object()
    contexts: list[str] = []

    def health(_report: Any, *, context: str) -> object:
        contexts.append(context)
        return marker

    runner = SimpleNamespace(
        _flow_advance_current_step=lambda *_args, **_kwargs: None,
        _require_hibm_velocity_dirichlet_health=health,
    )
    earlier = (
        "fsi step 0 pre-predictor assembly",
        "fsi step 0 projection assembly",
        "fsi step 0 consistency projection 1 assembly",
    )

    def invoke() -> None:
        for context in earlier:
            assert runner._require_hibm_velocity_dirichlet_health(
                {}, context=context
            ) is marker
        runner._require_hibm_velocity_dirichlet_health({}, context=TARGET_CONTEXT)

    _run(module, tmp_path, monkeypatch, runner, invoke, _FakeTaichi(cache))
    assert contexts == [*earlier, TARGET_CONTEXT]


def test_exact_context_original_failure_is_primary_and_emits_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    primary = RuntimeError("post-solid health failed")

    def health(_report: Any, *, context: str) -> None:
        assert context == TARGET_CONTEXT
        raise primary

    flow = lambda *_args, **_kwargs: None
    runner = SimpleNamespace(
        _flow_advance_current_step=flow,
        _require_hibm_velocity_dirichlet_health=health,
    )
    seen: dict[str, Any] = {}
    fake = _FakeTaichi(cache, reset_error=RuntimeError("secondary reset"))
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake.control)
    monkeypatch.setattr(
        module,
        "run_diagnostic_replay",
        _wrapping_replay(
            module,
            lambda: runner._require_hibm_velocity_dirichlet_health(
                {}, context=TARGET_CONTEXT
            ),
            seen,
        ),
    )
    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output",
            cache_dir=cache,
            target_stage=POST_SOLID_STAGE,
        )
    assert exc_info.value is seen["wrapped"]
    assert exc_info.value.__cause__ is primary
    assert any("secondary reset" in note for note in exc_info.value.__notes__)
    assert runner._flow_advance_current_step is flow
    assert runner._require_hibm_velocity_dirichlet_health is health
    assert not (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).exists()


@pytest.mark.parametrize(
    ("sync_error", "reset_error"),
    ((RuntimeError("sync failed"), None), (None, RuntimeError("reset failed"))),
    ids=("sync-failure", "reset-failure"),
)
def test_flow_and_health_restore_on_finalization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sync_error: Exception | None,
    reset_error: Exception | None,
) -> None:
    module = _load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    flow = lambda *_args, **_kwargs: None
    health = lambda _report, *, context: None
    runner = SimpleNamespace(
        _flow_advance_current_step=flow,
        _require_hibm_velocity_dirichlet_health=health,
    )
    seen: dict[str, Any] = {}
    fake = _FakeTaichi(cache, sync_error=sync_error, reset_error=reset_error)
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake.control)
    monkeypatch.setattr(
        module,
        "run_diagnostic_replay",
        _wrapping_replay(
            module,
            lambda: runner._require_hibm_velocity_dirichlet_health(
                {}, context=TARGET_CONTEXT
            ),
            seen,
        ),
    )
    with pytest.raises(module.DiagnosticReplayError, match="finalization failed"):
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output",
            cache_dir=cache,
            target_stage=POST_SOLID_STAGE,
        )
    assert runner._flow_advance_current_step is flow
    assert runner._require_hibm_velocity_dirichlet_health is health


def test_existing_solid_target_does_not_require_health_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    runner = SimpleNamespace(
        _flow_advance_current_step=lambda *_args, **_kwargs: None,
        _advance_solid_substeps_batched=lambda *_args, **_kwargs: None,
    )
    seen = _run(
        module,
        tmp_path,
        monkeypatch,
        runner,
        lambda: runner._advance_solid_substeps_batched(object(), object()),
        _FakeTaichi(cache),
        target_stage="solid_update_after",
    )
    assert seen["payload"]["target_stage"] == "solid_update_after"
    assert not hasattr(runner, "_require_hibm_velocity_dirichlet_health")


def test_cli_preserves_eight_choices_and_adds_post_solid_health_choice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    common = [
        "--snapshot", "snapshot", "--config-json", "config.json",
        "--source-manifest-json", "manifest.json", "--output-dir", "output",
        "--cache-dir", "cache", "--target-stage",
    ]
    prior = (
        "sst_wall_distance_after", "sst_transport_after",
        "momentum_predictor_after", "projection_hibm_after",
        "main_pressure_projection_after", "fsi_flow_advance_after",
        "traction_scatter_gate_after", "solid_update_after",
    )
    for stage in (*prior, POST_SOLID_STAGE):
        assert module._build_parser().parse_args(common + [stage]).target_stage == stage
    with pytest.raises(SystemExit):
        module._build_parser().parse_args(common + ["unknown_after"])

    observed: dict[str, Any] = {}

    def passed(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "status": "cache_checkpoint_completed",
            "target_stage": POST_SOLID_STAGE,
            "completed_fsi_steps": 0,
            "cache_inventory_changed": False,
        }

    monkeypatch.setattr(module, "run_flow_stage_cache_checkpoint", passed)
    assert module.main(common + [POST_SOLID_STAGE]) == 0
    assert observed["target_stage"] == POST_SOLID_STAGE
    assert "target_stage=post_solid_hibm_health_after" in capsys.readouterr().out
