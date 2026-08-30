from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import cases.ansys_vertical_flap_fsi as case_module
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig
from simulation_core.diagnostics.run_attempt import require_completed_output


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CLI = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)


def test_fsi_checkpoint_config_defaults_and_case_cli_forwarding(
    monkeypatch,
) -> None:
    assert VerticalFlapFsiConfig().fsi_checkpoint_input_path is None
    assert VerticalFlapFsiConfig().fsi_checkpoint_output_path is None
    captured: dict[str, VerticalFlapFsiConfig] = {}

    def fake_benchmark(config: VerticalFlapFsiConfig) -> dict[str, object]:
        captured["config"] = config
        return {"status": "mocked"}

    monkeypatch.setattr(
        case_module,
        "run_ansys_vertical_flap_benchmark",
        fake_benchmark,
    )
    report = case_module.main(
        [
            "--steps",
            "17",
            "--fsi-checkpoint-in",
            "accepted/fsi_in",
            "--fsi-checkpoint-out",
            "accepted/fsi_out",
            "--json",
        ]
    )

    assert report == {"status": "mocked"}
    assert captured["config"].step_count == 17
    assert captured["config"].fsi_checkpoint_input_path == "accepted/fsi_in"
    assert captured["config"].fsi_checkpoint_output_path == "accepted/fsi_out"


def test_checkpoint_generation_pin_is_not_part_of_physics_identity(
    validation_cli_module,
) -> None:
    unpinned = VerticalFlapFsiConfig()
    pinned = replace(
        unpinned,
        fsi_checkpoint_expected_generation="d" * 32,
    )
    assert validation_cli_module._fsi_checkpoint_config_payload(
        pinned
    ) == validation_cli_module._fsi_checkpoint_config_payload(unpinned)


def test_validation_cli_help_exposes_fsi_checkpoint_paths() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATION_CLI), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--fsi-checkpoint-in" in completed.stdout
    assert "--fsi-checkpoint-out" in completed.stdout


@pytest.fixture(scope="module")
def validation_cli_module():
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_checkpoint_cli_under_test",
        VALIDATION_CLI,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_identity(module) -> dict[str, str]:
    return {
        "config_sha256": "a" * 64,
        "source_sha256": module.canonical_source_sha256(
            module._preflow_snapshot_source_payload()
        ),
        "geometry_sha256": "c" * 64,
    }


def _resume_config(**changes):
    values = {
        "step_count": 3,
        "dt_s": 0.001,
        "young_modulus_pa": 1.0e6,
        "fsi_checkpoint_input_path": "checkpoint",
        "fsi_checkpoint_output_path": "checkpoint",
        "preflow_snapshot_input_path": None,
        "preflow_snapshot_output_path": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _write_existing_config(output_dir: Path, config: SimpleNamespace) -> None:
    output_dir.mkdir()
    (output_dir / "our_solver_config.json").write_text(
        json.dumps(vars(config)), encoding="utf-8"
    )


def _loaded_checkpoint(*, step: int, observer_identity: str, generation: str):
    return SimpleNamespace(
        generation=generation,
        state=SimpleNamespace(
            macro_state=SimpleNamespace(accepted_step_index=step),
            runner_state={
                "observer_identity": observer_identity,
                "observer_outbox": {"step": step},
            },
        )
    )


def _resume_preflight(
    module,
    *,
    output_dir: Path,
    config: SimpleNamespace,
    observer,
    accepted_step: int = 2,
    saved_observer_identity: str | None = None,
):
    loaded = _loaded_checkpoint(
        step=accepted_step,
        observer_identity=(
            observer.checkpoint_identity
            if saved_observer_identity is None
            else saved_observer_identity
        ),
        generation="d" * 32,
    )
    calls: list[object] = []

    identity = _checkpoint_identity(module)
    head = SimpleNamespace(
        metadata=MappingProxyType({"identity": identity}),
        generation="d" * 32,
        accepted_step=accepted_step,
    )

    def read_head(path):
        calls.append(("head", path))
        return head

    def load(
        path,
        *,
        expected_identity,
        target_step_count,
        expected_generation,
    ):
        calls.append(
            (
                "load",
                path,
                expected_identity,
                target_step_count,
                expected_generation,
            )
        )
        assert expected_identity == identity
        assert expected_generation == head.generation
        return loaded

    returned_head = module._preflight_checkpoint_resume(
        output_dir=output_dir,
        config=config,
        checkpoint_input_path=Path("checkpoint.json"),
        step_observer=observer,
        checkpoint_head_reader=read_head,
        checkpoint_loader=load,
        artifact_validator=lambda *_args, **_kwargs: calls.append("artifacts"),
    )
    return calls, returned_head, head


def test_resume_preflight_accepts_matching_nonempty_destination(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "existing"
    config = _resume_config()
    _write_existing_config(output_dir, config)
    (output_dir / "step_fields").mkdir()
    (output_dir / "step_history").mkdir()
    (output_dir / "step_fields" / "step_0001.npz").write_bytes(b"frame")
    (output_dir / "step_history" / "step_0001.json").write_text("{}", encoding="utf-8")
    observer = validation_cli_module._make_step_observer(
        output_dir=output_dir,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
        save_iqn_trial_vectors=False,
    )

    calls, returned_head, head = _resume_preflight(
        validation_cli_module,
        output_dir=output_dir,
        config=config,
        observer=observer,
    )

    assert calls[0][0] == "head"
    assert calls[1][0] == "load"
    assert calls[-1] == "artifacts"
    assert returned_head is head


def test_resume_preflight_rejects_current_source_mismatch_before_loading(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "existing"
    config = _resume_config()
    _write_existing_config(output_dir, config)
    wrong_identity = {
        "config_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "geometry_sha256": "c" * 64,
    }
    calls: list[str] = []

    def read_head(_path):
        calls.append("head")
        return SimpleNamespace(
            metadata=MappingProxyType({"identity": wrong_identity}),
            generation="d" * 32,
            accepted_step=2,
        )

    def must_not_load(*_args, **_kwargs):
        calls.append("load")
        raise AssertionError("source mismatch must reject before loading")

    with pytest.raises(ValueError, match="source identity does not match current source"):
        validation_cli_module._preflight_checkpoint_resume(
            output_dir=output_dir,
            config=config,
            checkpoint_input_path=tmp_path / "checkpoint",
            step_observer=None,
            checkpoint_head_reader=read_head,
            checkpoint_loader=must_not_load,
        )

    assert calls == ["head"]


def test_resume_source_rejection_leaves_active_failure_noncompleted(
    tmp_path: Path,
) -> None:
    (tmp_path / "progress.json").write_text('{"status":"completed"}', "utf-8")
    (tmp_path / "our_solver_summary.json").write_text(
        '{"status":"completed"}', "utf-8"
    )
    (tmp_path / "failure.json").write_text('{"status":"failed"}', "utf-8")

    with pytest.raises(ValueError, match="terminal-complete"):
        require_completed_output(tmp_path)


def test_resume_preflight_rejects_loaded_generation_mismatch(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "existing"
    config = _resume_config()
    _write_existing_config(output_dir, config)
    identity = _checkpoint_identity(validation_cli_module)
    head = SimpleNamespace(
        metadata=MappingProxyType({"identity": identity}),
        generation="d" * 32,
        accepted_step=2,
    )
    loaded = _loaded_checkpoint(
        step=2,
        observer_identity=None,
        generation="e" * 32,
    )

    with pytest.raises(ValueError, match="head generation differs from loaded state"):
        validation_cli_module._preflight_checkpoint_resume(
            output_dir=output_dir,
            config=config,
            checkpoint_input_path=tmp_path / "checkpoint",
            step_observer=None,
            checkpoint_head_reader=lambda _path: head,
            checkpoint_loader=lambda *_args, **_kwargs: loaded,
        )


@pytest.mark.parametrize("damage", ["different_dir", "old_hole", "forward_frame", "physics"])
def test_resume_preflight_rejects_mismatched_or_noncontiguous_artifacts_before_writes(
    tmp_path: Path,
    validation_cli_module,
    damage: str,
) -> None:
    output_dir = tmp_path / "existing"
    config = _resume_config()
    _write_existing_config(output_dir, config)
    observer = validation_cli_module._make_step_observer(
        output_dir=(tmp_path / "different") if damage == "different_dir" else output_dir,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
        save_iqn_trial_vectors=False,
    )
    if damage == "old_hole":
        (output_dir / "step_fields").mkdir()
        (output_dir / "step_history").mkdir()
        (output_dir / "step_fields" / "step_0001.npz").write_bytes(b"frame")
        (output_dir / "step_history" / "step_0001.json").write_text("{}", encoding="utf-8")
    elif damage == "forward_frame":
        (output_dir / "step_fields").mkdir()
        (output_dir / "step_history").mkdir()
        (output_dir / "step_fields" / "step_0003.npz").write_bytes(b"frame")
        (output_dir / "step_history" / "step_0003.json").write_text("{}", encoding="utf-8")
    elif damage == "physics":
        config = _resume_config(young_modulus_pa=2.0e6)

    before = sorted(path.relative_to(output_dir) for path in output_dir.rglob("*"))
    with pytest.raises(ValueError):
        _resume_preflight(
            validation_cli_module,
            output_dir=output_dir,
            config=config,
            observer=observer,
            accepted_step=3 if damage == "old_hole" else 2,
            saved_observer_identity=(
                validation_cli_module._checkpoint_observer_identity(
                    output_dir=output_dir,
                    span_reduction="mean",
                    streamwise_velocity_sign=1.0,
                    reverse_streamwise_axis=True,
                    streamwise_length_m=0.1,
                    save_iqn_trial_vectors=False,
                )
                if damage == "different_dir"
                else None
            ),
        )
    after = sorted(path.relative_to(output_dir) for path in output_dir.rglob("*"))
    assert after == before


def test_resume_prefix_allows_partial_terminal_step_for_outbox_repair(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "existing"
    (output_dir / "step_fields").mkdir(parents=True)
    (output_dir / "step_history").mkdir()
    (output_dir / "step_fields" / "step_0001.npz").write_bytes(b"frame")
    (output_dir / "step_history" / "step_0001.json").write_text("{}", encoding="utf-8")
    (output_dir / "step_fields" / "step_0002.npz").write_bytes(b"partial")

    calls: list[dict[str, object]] = []
    validation_cli_module._validate_resume_artifact_prefix(
        output_dir=output_dir,
        accepted_step=2,
        require_iqn_trial_vectors=False,
        artifact_validator=lambda *_args, **kwargs: calls.append(kwargs),
    )

    assert calls == [{
        "expected_steps": 1,
        "require_iqn_trial_vectors": False,
        "allow_partial_terminal_step": 2,
    }]


@pytest.mark.parametrize("layout", ["noncanonical_alias", "duplicate_alias"])
def test_resume_prefix_rejects_step_one_aliases(
    tmp_path: Path,
    validation_cli_module,
    layout: str,
) -> None:
    output_dir = tmp_path / "canonical"
    fields_dir = output_dir / "step_fields"
    histories_dir = output_dir / "step_history"
    fields_dir.mkdir(parents=True)
    histories_dir.mkdir()
    artifact_stems = ["step_00001"]
    if layout == "duplicate_alias":
        artifact_stems.append("step_0001")
    for stem in artifact_stems:
        (fields_dir / f"{stem}.npz").write_bytes(stem.encode("ascii"))
        (histories_dir / f"{stem}.json").write_text(
            json.dumps({"step_index": 1, "history": {}}),
            encoding="utf-8",
        )
    before = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(
        ValueError,
        match="invalid checkpoint step artifact",
    ):
        validation_cli_module._validate_resume_artifact_prefix(
            output_dir=output_dir,
            accepted_step=1,
            require_iqn_trial_vectors=False,
        )

    assert {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    } == before


def test_observer_replays_partial_final_step_atomically_without_duplicates(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    grid_shape = (1, 2, 2)
    snapshot: dict[str, object] = {
        "velocity": np.zeros((*grid_shape, 3), dtype=np.float32),
        "pressure": np.zeros(grid_shape, dtype=np.float32),
        "obstacle": np.zeros(grid_shape, dtype=np.int32),
        "cell_center_y_m": np.asarray([0.0, 0.01]),
        "cell_center_z_m": np.asarray([0.05, 0.053]),
        "solid_position_m": np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32),
        "solid_velocity_mps": np.zeros((1, 3), dtype=np.float32),
        "solid_rest_position_m": np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32),
        "solid_fixed_mask": np.asarray([True]),
        "solid_tip_mask": np.asarray([False]),
        "marker_position_m": np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32),
        "marker_velocity_mps": np.zeros((1, 3), dtype=np.float32),
        "marker_normal": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        "marker_area_m2": np.asarray([1.0e-6], dtype=np.float32),
        "marker_region_id": np.asarray([0], dtype=np.int32),
    }
    grid_diagnostics = {
        "velocity_dirichlet_boundary_active",
        "velocity_dirichlet_boundary_projection_weight",
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_owned_row",
        "velocity_dirichlet_boundary_marker_region_id",
    }
    for name in validation_cli_module.STEP_FRAME_DIAGNOSTIC_KEYS:
        snapshot[name] = (
            np.zeros(grid_shape, dtype=np.float32)
            if name in grid_diagnostics
            else np.asarray("host-test")
        )
    observer = validation_cli_module._make_step_observer(
        output_dir=output_dir,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
        streamwise_length_m=0.1,
        save_iqn_trial_vectors=False,
    )

    observer(1, 0.001, {"step": 1, "time_s": 0.001}, snapshot)
    validation_cli_module._save_step_frame_atomic(
        output_dir / "step_fields" / "step_0002.npz",
        snapshot,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
        streamwise_length_m=0.1,
    )
    validation_cli_module._validate_resume_artifact_prefix(
        output_dir=output_dir,
        accepted_step=2,
        require_iqn_trial_vectors=False,
    )
    observer(2, 0.002, {"step": 2, "time_s": 0.002}, snapshot)
    observer(2, 0.002, {"step": 2, "time_s": 0.002}, snapshot)

    assert sorted(path.name for path in (output_dir / "step_fields").iterdir()) == [
        "step_0001.npz", "step_0002.npz",
    ]
    assert sorted(path.name for path in (output_dir / "step_history").iterdir()) == [
        "step_0001.json", "step_0002.json",
    ]
    assert not list((output_dir / "step_fields").glob(".step_*.npz"))
    assert not list((output_dir / "step_history").glob(".step_*.tmp"))
    with np.load(output_dir / "step_fields" / "step_0002.npz", allow_pickle=False) as frame:
        assert set(validation_cli_module.STEP_FRAME_STRUCTURE_KEYS).issubset(frame.files)
    assert json.loads((output_dir / "step_history" / "step_0002.json").read_text(encoding="utf-8")) == {
        "history": {"step": 2, "time_s": 0.002},
        "step_index": 2,
        "time_s": 0.002,
    }
    assert validation_cli_module._validate_step_artifacts(output_dir, expected_steps=2)["status"] == "passed"


def test_fresh_nonempty_output_still_refuses_and_observer_is_replay_safe(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "prior.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        validation_cli_module._prepare_output_dir(output_dir, resume=False)

    observer = validation_cli_module._make_step_observer(
        output_dir=output_dir,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
        save_iqn_trial_vectors=True,
    )
    assert observer.checkpoint_replay_safe is True
    assert observer.checkpoint_identity


def test_running_progress_merge_clears_stale_diagnostics_without_mutating_existing(
    validation_cli_module,
) -> None:
    existing = {
        "status": "error",
        "phase": "error",
        "step_completed": 31,
        "time_s": 0.031,
        "frame": {"npz": "step_0031.npz"},
        "error": "prior geometry audit failure",
        "error_type": "RuntimeError",
        "pressure_solve_diagnostics": {"iterations": 17},
        "reporting_errors": ["prior reporter failure"],
        "traceback": "old traceback",
    }
    original = json.loads(json.dumps(existing))

    merged = validation_cli_module._merge_progress_event(
        existing,
        {
            "status": "running",
            "phase": "fsi_checkpoint_restored",
            "step_completed": 31,
            "time_s": 0.031,
        },
    )

    assert existing == original
    assert merged["status"] == "running"
    assert merged["phase"] == "fsi_checkpoint_restored"
    assert merged["step_completed"] == 31
    assert merged["frame"] == {"npz": "step_0031.npz"}
    assert not set(validation_cli_module._PROGRESS_FAILURE_DIAGNOSTIC_FIELDS) & set(merged)


def test_running_progress_merge_preserves_current_event_diagnostics(
    validation_cli_module,
) -> None:
    merged = validation_cli_module._merge_progress_event(
        {"status": "error", "error": "old", "error_type": "OldError"},
        {
            "status": "running",
            "phase": "diagnostic_live_event",
            "error": "current observer diagnostic",
            "error_type": "CurrentError",
        },
    )

    assert merged == {
        "status": "running",
        "phase": "diagnostic_live_event",
        "error": "current observer diagnostic",
        "error_type": "CurrentError",
    }


def test_nonrunning_progress_merge_retains_failure_diagnostics(
    validation_cli_module,
) -> None:
    merged = validation_cli_module._merge_progress_event(
        {
            "status": "running",
            "step_completed": 31,
            "error": "prior failure detail",
            "reporting_errors": ["prior reporter failure"],
        },
        {
            "status": "error",
            "phase": "error",
            "error": "current failure detail",
            "error_type": "RuntimeError",
            "pressure_solve_diagnostics": {"reason": "strict support"},
        },
    )

    assert merged["status"] == "error"
    assert merged["error"] == "current failure detail"
    assert merged["error_type"] == "RuntimeError"
    assert merged["pressure_solve_diagnostics"] == {"reason": "strict support"}
    assert merged["reporting_errors"] == ["prior reporter failure"]


def test_run_and_step_progress_observers_clear_stale_resume_diagnostics(
    tmp_path: Path,
    monkeypatch,
    validation_cli_module,
) -> None:
    output_dir = tmp_path / "resume"
    output_dir.mkdir()
    progress_path = output_dir / "progress.json"
    stale = {
        "status": "error",
        "phase": "error",
        "step_completed": 31,
        "time_s": 0.031,
        "frame": {"npz": "step_0031.npz"},
        "error": "prior geometry audit failure",
        "error_type": "RuntimeError",
        "pressure_solve_diagnostics": {"reason": "strict support"},
        "reporting_errors": ["prior reporter failure"],
    }
    progress_path.write_text(json.dumps(stale), encoding="utf-8")

    run_observer = validation_cli_module._make_run_progress_observer(
        output_dir=output_dir,
    )
    run_observer(
        {
            "status": "running",
            "phase": "fsi_checkpoint_restored",
            "step_completed": 31,
            "time_s": 0.031,
        }
    )
    restored = json.loads(progress_path.read_text(encoding="utf-8"))
    assert restored["frame"] == {"npz": "step_0031.npz"}
    assert not set(validation_cli_module._PROGRESS_FAILURE_DIAGNOSTIC_FIELDS) & set(restored)

    progress_path.write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(
        validation_cli_module,
        "_save_step_frame_atomic",
        lambda *_args, **_kwargs: {"npz": "step_0032.npz"},
    )
    step_observer = validation_cli_module._make_step_observer(
        output_dir=output_dir,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
    )
    step_observer(32, 0.032, {"step": 32, "time_s": 0.032}, {})

    stepped = json.loads(progress_path.read_text(encoding="utf-8"))
    assert stepped["status"] == "running"
    assert stepped["phase"] == "fsi_step"
    assert stepped["step_completed"] == 32
    assert stepped["frame"] == {"npz": "step_0032.npz"}
    assert not set(validation_cli_module._PROGRESS_FAILURE_DIAGNOSTIC_FIELDS) & set(stepped)

def _canonical_step_snapshot(module) -> dict[str, object]:
    grid_shape = (1, 2, 2)
    snapshot: dict[str, object] = {
        "velocity": np.zeros((*grid_shape, 3), dtype=np.float32),
        "pressure": np.zeros(grid_shape, dtype=np.float32),
        "obstacle": np.zeros(grid_shape, dtype=np.int32),
        "cell_center_y_m": np.asarray([0.0, 0.01]),
        "cell_center_z_m": np.asarray([0.05, 0.053]),
        "solid_position_m": np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32),
        "solid_velocity_mps": np.zeros((1, 3), dtype=np.float32),
        "solid_rest_position_m": np.asarray(
            [[0.0, 0.0, 0.05]], dtype=np.float32
        ),
        "solid_fixed_mask": np.asarray([True]),
        "solid_tip_mask": np.asarray([False]),
        "marker_position_m": np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32),
        "marker_velocity_mps": np.zeros((1, 3), dtype=np.float32),
        "marker_normal": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        "marker_area_m2": np.asarray([1.0e-6], dtype=np.float32),
        "marker_region_id": np.asarray([0], dtype=np.int32),
    }
    grid_diagnostics = {
        "velocity_dirichlet_boundary_active",
        "velocity_dirichlet_boundary_projection_weight",
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_owned_row",
        "velocity_dirichlet_boundary_marker_region_id",
    }
    for name in module.STEP_FRAME_DIAGNOSTIC_KEYS:
        snapshot[name] = (
            np.zeros(grid_shape, dtype=np.float32)
            if name in grid_diagnostics
            else np.asarray("host-test")
        )
    return snapshot


def _canonical_resume_observer(module, *, canonical_root: Path, attempt_dir: Path):
    return module._make_step_observer(
        output_dir=canonical_root,
        progress_dir=attempt_dir,
        span_reduction="mean",
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=True,
        streamwise_length_m=0.1,
        save_iqn_trial_vectors=False,
    )


def test_validation_cli_help_exposes_canonical_resume_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATION_CLI), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--resume-run-dir" in completed.stdout


@pytest.mark.parametrize("checkpoint_flag", ["--fsi-checkpoint-in", "--fsi-checkpoint-out"])
def test_resume_run_dir_rejects_explicit_checkpoint_paths(
    tmp_path: Path,
    checkpoint_flag: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_CLI),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "attempt"),
            "--resume-run-dir",
            str(tmp_path / "canonical"),
            checkpoint_flag,
            str(tmp_path / "explicit-checkpoint"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 2
    assert "cannot be combined with --resume-run-dir" in completed.stderr


def test_standalone_checkpoint_input_requires_canonical_resume_root(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_CLI),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "attempt"),
            "--fsi-checkpoint-in",
            str(tmp_path / "checkpoint"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 2
    assert "--fsi-checkpoint-in now requires --resume-run-dir" in completed.stderr


def test_resume_attempt_dir_must_be_fresh_empty_and_distinct_from_canonical_root(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    attempt_dir = tmp_path / "attempt"

    validation_cli_module._prepare_output_dir(
        attempt_dir,
        resume=True,
        canonical_output_dir=canonical_root,
    )
    assert attempt_dir.is_dir()
    assert not any(attempt_dir.iterdir())

    with pytest.raises(ValueError, match="distinct"):
        validation_cli_module._prepare_output_dir(
            canonical_root,
            resume=True,
            canonical_output_dir=canonical_root,
        )

    occupied_attempt = tmp_path / "occupied-attempt"
    occupied_attempt.mkdir()
    (occupied_attempt / "previous-attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fresh|empty|reuse"):
        validation_cli_module._prepare_output_dir(
            occupied_attempt,
            resume=True,
            canonical_output_dir=canonical_root,
        )


def test_resume_step_observer_separates_canonical_artifacts_from_attempt_progress(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    attempt_dir = tmp_path / "attempt"
    observer = _canonical_resume_observer(
        validation_cli_module,
        canonical_root=canonical_root,
        attempt_dir=attempt_dir,
    )

    observer(1, 0.001, {"step": 1, "time_s": 0.001}, _canonical_step_snapshot(validation_cli_module))

    assert (canonical_root / "step_fields" / "step_0001.npz").is_file()
    assert (canonical_root / "step_history" / "step_0001.json").is_file()
    assert not (canonical_root / "progress.json").exists()
    assert (attempt_dir / "progress.json").is_file()
    assert not (attempt_dir / "step_fields").exists()
    assert not (attempt_dir / "step_history").exists()


def test_resume_replay_semantic_match_preserves_canonical_artifact_bytes(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    observer = _canonical_resume_observer(
        validation_cli_module,
        canonical_root=canonical_root,
        attempt_dir=tmp_path / "attempt",
    )
    snapshot = _canonical_step_snapshot(validation_cli_module)
    history = {"step": 1, "time_s": 0.001}
    observer(1, 0.001, history, snapshot)
    frame_path = canonical_root / "step_fields" / "step_0001.npz"
    history_path = canonical_root / "step_history" / "step_0001.json"
    frame_before = frame_path.read_bytes()
    history_before = history_path.read_bytes()
    frame_inode = frame_path.stat().st_ino
    history_inode = history_path.stat().st_ino

    observer(1, 0.001, history, snapshot)

    assert frame_path.read_bytes() == frame_before
    assert history_path.read_bytes() == history_before
    assert frame_path.stat().st_ino == frame_inode
    assert history_path.stat().st_ino == history_inode


def test_resume_replay_repairs_missing_history_peer_only(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    observer = _canonical_resume_observer(
        validation_cli_module,
        canonical_root=canonical_root,
        attempt_dir=tmp_path / "attempt",
    )
    snapshot = _canonical_step_snapshot(validation_cli_module)
    history = {"step": 1, "time_s": 0.001}
    observer(1, 0.001, history, snapshot)
    frame_path = canonical_root / "step_fields" / "step_0001.npz"
    history_path = canonical_root / "step_history" / "step_0001.json"
    frame_before = frame_path.read_bytes()
    frame_inode = frame_path.stat().st_ino
    history_path.unlink()

    observer(1, 0.001, history, snapshot)

    assert frame_path.read_bytes() == frame_before
    assert frame_path.stat().st_ino == frame_inode
    assert json.loads(history_path.read_text(encoding="utf-8")) == {
        "history": history,
        "step_index": 1,
        "time_s": 0.001,
    }
    assert not list((canonical_root / "step_fields").glob(".step_*.npz"))
    assert not list((canonical_root / "step_history").glob(".step_*.tmp"))


def test_resume_replay_repairs_missing_frame_peer_only(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    observer = _canonical_resume_observer(
        validation_cli_module,
        canonical_root=canonical_root,
        attempt_dir=tmp_path / "attempt",
    )
    snapshot = _canonical_step_snapshot(validation_cli_module)
    history = {"step": 1, "time_s": 0.001}
    observer(1, 0.001, history, snapshot)
    frame_path = canonical_root / "step_fields" / "step_0001.npz"
    history_path = canonical_root / "step_history" / "step_0001.json"
    history_before = history_path.read_bytes()
    history_inode = history_path.stat().st_ino
    frame_path.unlink()

    observer(1, 0.001, history, snapshot)

    assert frame_path.is_file()
    assert history_path.read_bytes() == history_before
    assert history_path.stat().st_ino == history_inode
    with np.load(frame_path, allow_pickle=False) as frame:
        assert set(validation_cli_module.STEP_FRAME_STRUCTURE_KEYS).issubset(
            frame.files
        )
    assert not list((canonical_root / "step_fields").glob(".step_*.npz"))
    assert not list((canonical_root / "step_history").glob(".step_*.tmp"))


def test_resume_history_mismatch_does_not_create_missing_frame(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    observer = _canonical_resume_observer(
        validation_cli_module,
        canonical_root=canonical_root,
        attempt_dir=tmp_path / "attempt",
    )
    snapshot = _canonical_step_snapshot(validation_cli_module)
    history = {"step": 1, "time_s": 0.001}
    observer(1, 0.001, history, snapshot)
    frame_path = canonical_root / "step_fields" / "step_0001.npz"
    history_path = canonical_root / "step_history" / "step_0001.json"
    frame_path.unlink()
    history_path.write_text(
        json.dumps({"step_index": 1, "time_s": 0.001, "history": {"step": 99}}),
        encoding="utf-8",
    )
    history_before = history_path.read_bytes()

    with pytest.raises(ValueError, match="existing accepted step history replay mismatch"):
        observer(1, 0.001, history, snapshot)

    assert not frame_path.exists()
    assert history_path.read_bytes() == history_before
    assert not list((canonical_root / "step_fields").glob(".step_*.npz"))
    assert not list((canonical_root / "step_history").glob(".step_*.tmp"))


def test_resume_replay_mismatch_rejects_without_overwriting_canonical_artifacts(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    canonical_root = tmp_path / "canonical"
    observer = _canonical_resume_observer(
        validation_cli_module,
        canonical_root=canonical_root,
        attempt_dir=tmp_path / "attempt",
    )
    history = {"step": 1, "time_s": 0.001}
    snapshot = _canonical_step_snapshot(validation_cli_module)
    observer(1, 0.001, history, snapshot)
    frame_path = canonical_root / "step_fields" / "step_0001.npz"
    history_path = canonical_root / "step_history" / "step_0001.json"
    frame_before = frame_path.read_bytes()
    history_before = history_path.read_bytes()
    changed_snapshot = _canonical_step_snapshot(validation_cli_module)
    changed_velocity = np.asarray(changed_snapshot["velocity"]).copy()
    changed_velocity[0, 0, 0, 2] = 1.0
    changed_snapshot["velocity"] = changed_velocity

    with pytest.raises(ValueError, match="replay|existing|mismatch"):
        observer(1, 0.001, history, changed_snapshot)

    assert frame_path.read_bytes() == frame_before
    assert history_path.read_bytes() == history_before
