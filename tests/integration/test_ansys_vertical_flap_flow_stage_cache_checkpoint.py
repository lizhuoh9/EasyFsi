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
HASHES = {"metadata_sha256": "a" * 64, "npz_sha256": "b" * 64}
TRACTION_STAGE = "traction_scatter_gate_after"
SOLID_STAGE = "solid_update_after"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_flow_stage_cache_checkpoint", SCRIPT_PATH
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
    ):
        self.cache_dir = cache_dir
        self.program = SimpleNamespace(prog=object())
        self.calls: list[str] = []
        self.sync_error = sync_error
        self.reset_error = reset_error

    def get_runtime(self) -> Any:
        self.calls.append("get_runtime")
        return self.program

    def sync(self) -> None:
        self.calls.append("sync")
        if self.sync_error is not None:
            raise self.sync_error

    def reset(self) -> None:
        self.calls.append("reset")
        if self.reset_error is not None:
            raise self.reset_error
        (self.cache_dir / "ticache.tcb").write_bytes(b"cached")
        self.program.prog = None

    def control(self) -> Any:
        return SimpleNamespace(
            requested_cache_identity=lambda: (True, str(self.cache_dir.resolve())),
            initialized_cache_identity=lambda: (True, str(self.cache_dir.resolve())),
            program_present=self.program_present,
            sync=self.sync,
            reset=self.reset,
        )

    def program_present(self) -> bool:
        self.calls.append("get_runtime")
        return self.program.prog is not None


def _base_payload(cause: BaseException, **overrides: Any) -> dict[str, Any]:
    payload = {
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
    payload.update(overrides)
    return payload


def _wrapped_replay(
    module: Any,
    invoke: Callable[[], Any],
    seen: dict[str, Any],
    *,
    base_overrides: dict[str, Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    def replay(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            seen["return"] = invoke()
        except Exception as cause:
            seen["cause"] = cause
            payload = _base_payload(cause, **(base_overrides or {}))
            (output_dir / "diagnostic_replay.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            wrapped = module.DiagnosticReplayError("base diagnostic failed")
            seen["wrapped"] = wrapped
            raise wrapped from cause
        payload = {"status": "completed", "completed_fsi_steps": 1}
        (output_dir / "diagnostic_replay.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    return replay


def _run(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    replay: Callable[..., dict[str, Any]],
    fake_taichi: _FakeTaichi,
    *,
    target_stage: str | None = None,
) -> dict[str, Any]:
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake_taichi.control)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)
    return module.run_flow_stage_cache_checkpoint(
        snapshot_path=tmp_path / "preflow_state",
        config_path=tmp_path / "config.json",
        source_manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "checkpoint-output",
        cache_dir=fake_taichi.cache_dir,
        target_stage=target_stage or module.FSI_FLOW_ADVANCE_AFTER_STAGE,
        allowed_source_diffs=(CORE_SOURCE, RUNNER_SOURCE),
    )


def test_target_return_precedes_synthetic_sentinel_and_writes_zero_step_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    events: list[str] = []
    observer = lambda _stage: None

    def original(*_args: Any, **kwargs: Any) -> object:
        events.append("original_return")
        return object()

    runner = SimpleNamespace(_flow_advance_current_step=original)
    original_identity = runner._flow_advance_current_step
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        try:
            return runner._flow_advance_current_step(
                object(), object(), flow_phase="fsi", step_index_local=0,
                step_index_global=0, preflow_history=[], reset_pressure=False,
                preflow_stage_observer=observer,
            )
        except Exception:
            events.append("synthetic_sentinel")
            raise

    fake_taichi = _FakeTaichi(cache_dir)
    payload = _run(
        module, tmp_path, monkeypatch, runner,
        _wrapped_replay(module, invoke, seen), fake_taichi,
    )

    assert events == ["original_return", "synthetic_sentinel"]
    assert runner._flow_advance_current_step is original_identity
    assert seen["wrapped"].__cause__ is seen["cause"]
    assert payload["status"] == "cache_checkpoint_completed"
    assert payload["target_stage"] == module.FSI_FLOW_ADVANCE_AFTER_STAGE
    assert payload["completed_fsi_steps"] == 0
    assert payload["full_fsi_step_completed"] is False
    assert payload["one_step_completed"] is False
    assert payload["evidence_class"] == "diagnostic_only"
    assert payload["formal_validation_eligible"] is False
    assert payload["fresh_preflow"] is False
    assert payload["parity_claimed"] is False
    assert payload["program_present_before_reset"] is True
    assert payload["program_present_after_reset"] is False
    assert payload["cache_inventory_changed"] is True
    assert fake_taichi.calls == ["get_runtime", "sync", "reset", "get_runtime"]
    assert "history" not in payload and "runtime_report" not in payload
    written = json.loads(
        (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).read_text()
    )
    assert written == payload


def test_original_exception_emits_no_sentinel_and_remains_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    primary = RuntimeError("flow advance failed")

    def original(*_args: Any, **_kwargs: Any) -> Any:
        raise primary

    runner = SimpleNamespace(_flow_advance_current_step=original)
    original_identity = runner._flow_advance_current_step
    seen: dict[str, Any] = {}
    replay = _wrapped_replay(
        module,
        lambda: runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
        ),
        seen,
    )
    fake_taichi = _FakeTaichi(cache_dir, reset_error=RuntimeError("reset failed"))
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake_taichi.control)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)

    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output",
            cache_dir=cache_dir,
            target_stage=module.FSI_FLOW_ADVANCE_AFTER_STAGE,
        )

    assert exc_info.value is seen["wrapped"]
    assert exc_info.value.__cause__ is primary
    assert any("reset failed" in note for note in exc_info.value.__notes__)
    assert runner._flow_advance_current_step is original_identity
    assert not (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).exists()


def test_non_target_and_preflow_calls_are_unmodified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    token = object()
    marker = object()
    calls: list[tuple[str, int, Any]] = []

    def original(*_args: Any, **kwargs: Any) -> object:
        calls.append(
            (kwargs["flow_phase"], kwargs["step_index_local"],
             kwargs.get("preflow_stage_observer"))
        )
        return marker

    runner = SimpleNamespace(_flow_advance_current_step=original)
    seen: dict[str, Any] = {}

    def invoke() -> None:
        assert runner._flow_advance_current_step(
            object(), object(), flow_phase="preflow", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
            preflow_stage_observer=token,
        ) is marker
        assert runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=1,
            step_index_global=1, preflow_history=[], reset_pressure=False,
            preflow_stage_observer=token,
        ) is marker
        runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
            preflow_stage_observer=token,
        )

    payload = _run(
        module, tmp_path, monkeypatch, runner,
        _wrapped_replay(module, invoke, seen), _FakeTaichi(cache_dir),
    )
    assert payload["target_stage"] == module.FSI_FLOW_ADVANCE_AFTER_STAGE
    assert calls[:2] == [("preflow", 0, token), ("fsi", 1, token)]
    assert calls[2][:2] == ("fsi", 0)
    assert callable(calls[2][2])


def test_recursive_predictor_preserves_explicit_observer_and_emits_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    explicit_observer = lambda _stage: None
    observed: list[Any] = []
    depths: list[int] = []
    runner = SimpleNamespace()

    def original(*args: Any, **kwargs: Any) -> str:
        observed.append(kwargs.get("preflow_stage_observer"))
        depth = int(getattr(args[1], "depth", 0))
        depths.append(depth)
        if depth == 0:
            nested = dict(kwargs)
            return runner._flow_advance_current_step(
                args[0], SimpleNamespace(depth=1), **nested
            )
        return "inner-return"

    runner._flow_advance_current_step = original
    seen: dict[str, Any] = {}
    sentinel_count = 0

    def invoke() -> Any:
        nonlocal sentinel_count
        try:
            return runner._flow_advance_current_step(
                object(), SimpleNamespace(depth=0), flow_phase="fsi",
                step_index_local=0, step_index_global=0, preflow_history=[],
                reset_pressure=False, preflow_stage_observer=explicit_observer,
            )
        except Exception:
            sentinel_count += 1
            raise

    _run(
        module, tmp_path, monkeypatch, runner,
        _wrapped_replay(module, invoke, seen), _FakeTaichi(cache_dir),
    )
    assert depths == [0, 1]
    assert observed[0] is observed[1]
    assert observed[0] is not explicit_observer
    assert sentinel_count == 1


def test_existing_explicit_after_stage_still_interrupts_before_flow_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    events: list[str] = []

    def explicit_observer(stage: str) -> None:
        events.append(f"existing:{stage}")

    def original(*_args: Any, **kwargs: Any) -> None:
        kwargs["preflow_stage_observer"]("sst_transport_after")
        events.append("original_return")

    runner = SimpleNamespace(_flow_advance_current_step=original)
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        try:
            return runner._flow_advance_current_step(
                object(), object(), flow_phase="fsi", step_index_local=0,
                step_index_global=0, preflow_history=[], reset_pressure=False,
                preflow_stage_observer=explicit_observer,
            )
        except Exception:
            events.append("explicit_sentinel")
            raise

    payload = _run(
        module, tmp_path, monkeypatch, runner,
        _wrapped_replay(module, invoke, seen), _FakeTaichi(cache_dir),
        target_stage="sst_transport_after",
    )
    assert events == ["existing:sst_transport_after", "explicit_sentinel"]
    assert payload["target_stage"] == "sst_transport_after"


def test_traction_checkpoint_follows_normal_gate_return_and_restores_both_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    events: list[str] = []

    def original_flow(*_args: Any, **_kwargs: Any) -> str:
        events.append("flow_return")
        return "flow-report"

    def original_traction(**_kwargs: Any) -> None:
        events.append("traction_return")

    runner = SimpleNamespace(
        _flow_advance_current_step=original_flow,
        _require_fresh_external_force_for_solid_step=original_traction,
    )
    flow_identity = runner._flow_advance_current_step
    traction_identity = runner._require_fresh_external_force_for_solid_step
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        assert runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
        ) == "flow-report"
        try:
            return runner._require_fresh_external_force_for_solid_step(
                clear={}, scatter={}, marker_forces={}, stress={}, no_slip={},
                projection={},
            )
        except Exception:
            events.append("traction_sentinel")
            raise

    payload = _run(
        module, tmp_path, monkeypatch, runner,
        _wrapped_replay(module, invoke, seen), _FakeTaichi(cache_dir),
        target_stage=TRACTION_STAGE,
    )
    assert events == ["flow_return", "traction_return", "traction_sentinel"]
    assert seen["wrapped"].__cause__ is seen["cause"]
    assert payload["target_stage"] == TRACTION_STAGE
    assert payload["completed_fsi_steps"] == 0
    assert payload["full_fsi_step_completed"] is False
    assert runner._flow_advance_current_step is flow_identity
    assert runner._require_fresh_external_force_for_solid_step is traction_identity


def test_traction_gate_exception_emits_no_checkpoint_and_remains_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    primary = RuntimeError("traction freshness failed")

    def original_flow(*_args: Any, **_kwargs: Any) -> None:
        return None

    def original_traction(**_kwargs: Any) -> None:
        raise primary

    runner = SimpleNamespace(
        _flow_advance_current_step=original_flow,
        _require_fresh_external_force_for_solid_step=original_traction,
    )
    flow_identity = runner._flow_advance_current_step
    traction_identity = runner._require_fresh_external_force_for_solid_step
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
        )
        return runner._require_fresh_external_force_for_solid_step(
            clear={}, scatter={}, marker_forces={}, stress={}, no_slip={},
            projection={},
        )

    replay = _wrapped_replay(module, invoke, seen)
    fake_taichi = _FakeTaichi(
        cache_dir, reset_error=RuntimeError("secondary reset failed")
    )
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake_taichi.control)
    monkeypatch.setattr(module, "run_diagnostic_replay", replay)
    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output",
            cache_dir=cache_dir,
            target_stage=TRACTION_STAGE,
        )
    assert exc_info.value is seen["wrapped"]
    assert exc_info.value.__cause__ is primary
    assert any(
        "secondary reset failed" in note for note in exc_info.value.__notes__
    )
    assert runner._flow_advance_current_step is flow_identity
    assert runner._require_fresh_external_force_for_solid_step is traction_identity
    assert not (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).exists()


@pytest.mark.parametrize(
    ("sync_error", "reset_error"),
    (
        (RuntimeError("sync failed"), None),
        (None, RuntimeError("reset failed")),
    ),
    ids=("sync-failure", "reset-failure"),
)
def test_traction_wrappers_restore_when_cache_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sync_error: Exception | None,
    reset_error: Exception | None,
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def original_flow(*_args: Any, **_kwargs: Any) -> None:
        return None

    def original_traction(**_kwargs: Any) -> None:
        return None

    runner = SimpleNamespace(
        _flow_advance_current_step=original_flow,
        _require_fresh_external_force_for_solid_step=original_traction,
    )
    flow_identity = runner._flow_advance_current_step
    traction_identity = runner._require_fresh_external_force_for_solid_step
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
        )
        return runner._require_fresh_external_force_for_solid_step(
            clear={}, scatter={}, marker_forces={}, stress={}, no_slip={},
            projection={},
        )

    fake_taichi = _FakeTaichi(
        cache_dir,
        sync_error=sync_error,
        reset_error=reset_error,
    )
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake_taichi.control)
    monkeypatch.setattr(
        module, "run_diagnostic_replay", _wrapped_replay(module, invoke, seen)
    )
    with pytest.raises(module.DiagnosticReplayError, match="finalization failed"):
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output",
            cache_dir=cache_dir,
            target_stage=TRACTION_STAGE,
        )
    assert runner._flow_advance_current_step is flow_identity
    assert runner._require_fresh_external_force_for_solid_step is traction_identity
    assert not (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).exists()


def test_solid_checkpoint_follows_guarded_return_and_preserves_prior_callables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    report = object()
    events: list[tuple[str, Any]] = []

    def flow(*_args: Any, **_kwargs: Any) -> None:
        events.append(("flow_return", None))

    def traction(**_kwargs: Any) -> None:
        events.append(("traction_return", None))

    def solid(*_args: Any, **_kwargs: Any) -> object:
        events.append(("solid_return", report))
        return report

    runner = SimpleNamespace(
        _flow_advance_current_step=flow,
        _require_fresh_external_force_for_solid_step=traction,
        _select_and_advance_solid_macro_step=solid,
    )
    identities = (flow, traction, solid)
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        runner._flow_advance_current_step(
            object(), object(), flow_phase="fsi", step_index_local=0,
            step_index_global=0, preflow_history=[], reset_pressure=False,
        )
        runner._require_fresh_external_force_for_solid_step(
            clear={}, scatter={}, marker_forces={}, stress={}, no_slip={},
            projection={},
        )
        try:
            return runner._select_and_advance_solid_macro_step(
                object(),
                object(),
                mu_pa=1.0,
                lambda_pa=2.0,
                retry_prepare=lambda: None,
            )
        except Exception:
            events.append(("solid_sentinel", None))
            raise

    payload = _run(
        module, tmp_path, monkeypatch, runner,
        _wrapped_replay(module, invoke, seen), _FakeTaichi(cache_dir),
        target_stage=SOLID_STAGE,
    )
    assert events == [
        ("flow_return", None), ("traction_return", None),
        ("solid_return", report), ("solid_sentinel", None),
    ]
    assert seen["wrapped"].__cause__ is seen["cause"]
    assert payload["target_stage"] == SOLID_STAGE
    assert (
        runner._flow_advance_current_step,
        runner._require_fresh_external_force_for_solid_step,
        runner._select_and_advance_solid_macro_step,
    ) == identities


def test_solid_exception_remains_primary_and_restores_all_callables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    primary = RuntimeError("solid guard failed")

    def flow(*_args: Any, **_kwargs: Any) -> None:
        return None

    def traction(**_kwargs: Any) -> None:
        return None

    def solid(*_args: Any, **_kwargs: Any) -> Any:
        raise primary

    runner = SimpleNamespace(
        _flow_advance_current_step=flow,
        _require_fresh_external_force_for_solid_step=traction,
        _select_and_advance_solid_macro_step=solid,
    )
    seen: dict[str, Any] = {}

    def invoke() -> Any:
        return runner._select_and_advance_solid_macro_step(
            object(),
            object(),
            mu_pa=1.0,
            lambda_pa=2.0,
            retry_prepare=lambda: None,
        )
    fake = _FakeTaichi(cache_dir, reset_error=RuntimeError("secondary reset"))
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake.control)
    monkeypatch.setattr(
        module, "run_diagnostic_replay", _wrapped_replay(module, invoke, seen)
    )
    with pytest.raises(module.DiagnosticReplayError) as exc_info:
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output", cache_dir=cache_dir,
            target_stage=SOLID_STAGE,
        )
    assert exc_info.value is seen["wrapped"]
    assert exc_info.value.__cause__ is primary
    assert any("secondary reset" in note for note in exc_info.value.__notes__)
    assert (
        runner._flow_advance_current_step,
        runner._require_fresh_external_force_for_solid_step,
        runner._select_and_advance_solid_macro_step,
    ) == (flow, traction, solid)
    assert not (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).exists()


@pytest.mark.parametrize(
    ("sync_error", "reset_error"),
    ((RuntimeError("sync failed"), None), (None, RuntimeError("reset failed"))),
    ids=("sync-failure", "reset-failure"),
)
def test_solid_wrappers_restore_on_cache_finalization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sync_error: Exception | None,
    reset_error: Exception | None,
) -> None:
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def passthrough(*_args: Any, **_kwargs: Any) -> None:
        return None

    runner = SimpleNamespace(
        _flow_advance_current_step=passthrough,
        _require_fresh_external_force_for_solid_step=passthrough,
        _select_and_advance_solid_macro_step=passthrough,
    )
    seen: dict[str, Any] = {}
    invoke = lambda: runner._select_and_advance_solid_macro_step(object(), object())
    fake = _FakeTaichi(
        cache_dir, sync_error=sync_error, reset_error=reset_error
    )
    monkeypatch.setattr(module, "_load_runner_module", lambda: runner)
    monkeypatch.setattr(module, "_load_taichi_control", fake.control)
    monkeypatch.setattr(
        module, "run_diagnostic_replay", _wrapped_replay(module, invoke, seen)
    )
    with pytest.raises(module.DiagnosticReplayError, match="finalization failed"):
        module.run_flow_stage_cache_checkpoint(
            snapshot_path=tmp_path / "preflow_state",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "checkpoint-output", cache_dir=cache_dir,
            target_stage=SOLID_STAGE,
        )
    assert (
        runner._flow_advance_current_step,
        runner._require_fresh_external_force_for_solid_step,
        runner._select_and_advance_solid_macro_step,
    ) == (passthrough, passthrough, passthrough)
    assert not (tmp_path / "checkpoint-output" / module.CHECKPOINT_FILENAME).exists()


def test_cli_preserves_seven_choices_and_adds_solid_checkpoint_choice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    common = [
        "--snapshot", "snapshot", "--config-json", "config.json",
        "--source-manifest-json", "manifest.json", "--output-dir", "output",
        "--cache-dir", "cache", "--target-stage",
    ]
    for stage in (
        "sst_wall_distance_after",
        "sst_transport_after",
        "momentum_predictor_after",
        "projection_hibm_after",
        "main_pressure_projection_after",
        module.FSI_FLOW_ADVANCE_AFTER_STAGE,
        TRACTION_STAGE,
        SOLID_STAGE,
    ):
        assert module._build_parser().parse_args(common + [stage]).target_stage == stage
    with pytest.raises(SystemExit):
        module._build_parser().parse_args(common + ["unknown_after"])

    observed: dict[str, Any] = {}

    def passed(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "status": "cache_checkpoint_completed",
            "target_stage": SOLID_STAGE,
            "completed_fsi_steps": 0,
            "full_fsi_step_completed": False,
            "cache_inventory_changed": False,
        }

    monkeypatch.setattr(module, "run_flow_stage_cache_checkpoint", passed)
    code = module.main(common + [SOLID_STAGE])
    assert code == 0
    assert observed["target_stage"] == SOLID_STAGE
    stdout = capsys.readouterr().out
    assert "status=cache_checkpoint_completed" in stdout
    assert "target_stage=solid_update_after" in stdout
    assert "completed_fsi_steps=0" in stdout
