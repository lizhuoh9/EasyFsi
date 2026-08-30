from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from simulation_core.coupling.accepted_fsi_checkpoint import (
    load_accepted_fsi_checkpoint,
    write_accepted_fsi_checkpoint,
)
from tests.coupling.test_accepted_fsi_checkpoint import IDENTITY, _record, _state


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CLI = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)
GENERATION = "d" * 32


@pytest.fixture(scope="module")
def validation_cli_module():
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_resume_hardening_under_test",
        VALIDATION_CLI,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_checkpoint_input_is_rejected_by_validation_cli(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_CLI),
            "--output-dir",
            str(tmp_path / "attempt"),
            "--fsi-checkpoint-in",
            str(tmp_path / "checkpoint"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 2
    assert "requires --resume-run-dir" in completed.stderr
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize(
    "names",
    [
        ("step_00001.npz",),
        ("step_0001.npz", "step_00001.npz"),
    ],
)
def test_k1_resume_rejects_noncanonical_or_duplicate_step_aliases(
    tmp_path: Path,
    validation_cli_module,
    names: tuple[str, ...],
) -> None:
    fields = tmp_path / "step_fields"
    history = tmp_path / "step_history"
    fields.mkdir()
    history.mkdir()
    for name in names:
        (fields / name).write_bytes(b"frame")

    with pytest.raises(ValueError, match="invalid checkpoint step artifact"):
        validation_cli_module._validate_resume_artifact_prefix(
            output_dir=tmp_path,
            accepted_step=1,
            require_iqn_trial_vectors=False,
            artifact_validator=lambda *_args, **_kwargs: None,
        )


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"host does not permit symlink creation: {exc}")


def test_resume_rejects_symlink_artifact_directory(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-fields"
    outside.mkdir()
    _symlink_or_skip(tmp_path / "step_fields", outside, directory=True)
    (tmp_path / "step_history").mkdir()

    with pytest.raises(ValueError, match="real directory"):
        validation_cli_module._validate_resume_artifact_prefix(
            output_dir=tmp_path,
            accepted_step=1,
            require_iqn_trial_vectors=False,
            artifact_validator=lambda *_args, **_kwargs: None,
        )


def test_resume_rejects_symlink_artifact_file(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    fields = tmp_path / "step_fields"
    history = tmp_path / "step_history"
    fields.mkdir()
    history.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-frame.npz"
    outside.write_bytes(b"outside")
    _symlink_or_skip(fields / "step_0001.npz", outside, directory=False)

    with pytest.raises(ValueError, match="invalid checkpoint step artifact"):
        validation_cli_module._validate_resume_artifact_prefix(
            output_dir=tmp_path,
            accepted_step=1,
            require_iqn_trial_vectors=False,
            artifact_validator=lambda *_args, **_kwargs: None,
        )


def test_resume_rejects_simulated_windows_reparse_artifact_directory(
    tmp_path: Path,
    validation_cli_module,
    monkeypatch,
) -> None:
    fields = tmp_path / "step_fields"
    (tmp_path / "step_history").mkdir()
    fields.mkdir()
    original = validation_cli_module._lstat_or_none

    def fake_lstat(path: Path):
        entry = original(path)
        if path == fields:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR,
                st_file_attributes=0x0400,
            )
        return entry

    monkeypatch.setattr(validation_cli_module, "_lstat_or_none", fake_lstat)
    with pytest.raises(ValueError, match="real directory"):
        validation_cli_module._validate_resume_artifact_prefix(
            output_dir=tmp_path,
            accepted_step=1,
            require_iqn_trial_vectors=False,
            artifact_validator=lambda *_args, **_kwargs: None,
        )


def test_create_only_npz_publish_repairs_missing_frame_peer(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    fields = tmp_path / "step_fields"
    history = tmp_path / "step_history"
    fields.mkdir()
    history.mkdir()
    history_path = history / "step_0001.json"
    history_payload = {"history": {"step": 1}, "step_index": 1, "time_s": 0.001}
    history_path.write_text(json.dumps(history_payload), encoding="utf-8")
    history_before = history_path.read_bytes()
    candidate = fields / ".step_0001.candidate.npz"
    destination = fields / "step_0001.npz"
    np.savez(candidate, velocity=np.asarray([1.0], dtype=np.float32))

    validation_cli_module._require_existing_history_semantic_match(
        history_path, history_payload
    )
    validation_cli_module._publish_create_or_semantic_same(
        temporary=candidate,
        destination=destination,
        matches_existing=validation_cli_module._npz_semantically_equal,
        artifact_name="step frame",
    )

    assert destination.is_file()
    assert not candidate.exists()
    assert history_path.read_bytes() == history_before


@pytest.mark.parametrize("damage", ["key", "dtype", "shape", "value"])
def test_npz_replay_mismatch_never_overwrites_or_creates_peer(
    tmp_path: Path,
    validation_cli_module,
    damage: str,
) -> None:
    fields = tmp_path / "step_fields"
    history = tmp_path / "step_history"
    fields.mkdir()
    history.mkdir()
    destination = fields / "step_0001.npz"
    candidate = fields / ".step_0001.candidate.npz"
    base = np.asarray([1.0, 2.0], dtype=np.float32)
    np.savez(destination, velocity=base)
    candidate_arrays: dict[str, np.ndarray] = {"velocity": base.copy()}
    if damage == "key":
        candidate_arrays["extra"] = np.asarray([0], dtype=np.int32)
    elif damage == "dtype":
        candidate_arrays["velocity"] = base.astype(np.float64)
    elif damage == "shape":
        candidate_arrays["velocity"] = base.reshape(1, 2)
    elif damage == "value":
        candidate_arrays["velocity"] = np.asarray([1.0, 3.0], dtype=np.float32)
    np.savez(candidate, **candidate_arrays)
    before = destination.read_bytes()
    inode = destination.stat().st_ino

    with pytest.raises(ValueError, match="replay mismatch"):
        validation_cli_module._publish_create_or_semantic_same(
            temporary=candidate,
            destination=destination,
            matches_existing=validation_cli_module._npz_semantically_equal,
            artifact_name="step frame",
        )

    assert destination.read_bytes() == before
    assert destination.stat().st_ino == inode
    assert not candidate.exists()
    assert not (history / "step_0001.json").exists()


def test_resume_preflight_pins_head_generation_before_checkpoint_decode(
    tmp_path: Path,
    validation_cli_module,
) -> None:
    config = SimpleNamespace(
        step_count=1,
        dt_s=0.001,
        fsi_checkpoint_input_path="checkpoint",
        fsi_checkpoint_output_path="checkpoint",
        fsi_checkpoint_expected_generation=None,
        preflow_snapshot_input_path=None,
        preflow_snapshot_output_path=None,
    )
    tmp_path.joinpath("our_solver_config.json").write_text(
        json.dumps(vars(config)), encoding="utf-8"
    )
    identity = {
        "config_sha256": "a" * 64,
        "source_sha256": validation_cli_module.canonical_source_sha256(
            validation_cli_module._preflow_snapshot_source_payload()
        ),
        "geometry_sha256": "c" * 64,
    }
    head = SimpleNamespace(
        metadata=MappingProxyType({"identity": identity}),
        generation=GENERATION,
        accepted_step=0,
    )
    loaded = SimpleNamespace(
        generation=GENERATION,
        state=SimpleNamespace(
            macro_state=SimpleNamespace(accepted_step_index=0),
            runner_state={"observer_identity": None},
        ),
    )
    captured: dict[str, object] = {}

    def load_checkpoint(_path, **kwargs):
        captured.update(kwargs)
        return loaded

    validation_cli_module._preflight_checkpoint_resume(
        output_dir=tmp_path,
        config=config,
        checkpoint_input_path=tmp_path / "checkpoint",
        step_observer=None,
        checkpoint_head_reader=lambda _path: head,
        checkpoint_loader=load_checkpoint,
    )

    assert captured["expected_generation"] == GENERATION


def test_generation_pin_is_operational_not_checkpoint_identity(
    validation_cli_module,
) -> None:
    baseline = SimpleNamespace(
        step_count=2,
        fsi_checkpoint_expected_generation=None,
        fsi_checkpoint_input_path="a",
        fsi_checkpoint_output_path="b",
        preflow_snapshot_input_path=None,
        preflow_snapshot_output_path=None,
        dt_s=0.001,
    )
    pinned = SimpleNamespace(**vars(baseline))
    pinned.fsi_checkpoint_expected_generation = GENERATION

    assert validation_cli_module._fsi_checkpoint_config_payload(
        baseline
    ) == validation_cli_module._fsi_checkpoint_config_payload(pinned)


@pytest.mark.parametrize(
    "identity_field",
    ("config_sha256", "source_sha256", "geometry_sha256"),
)
def test_accepted_checkpoint_identity_mismatch_names_changed_field(
    tmp_path: Path,
    identity_field: str,
) -> None:
    checkpoint_prefix = tmp_path / "accepted"
    write_accepted_fsi_checkpoint(
        checkpoint_prefix,
        state=_state(),
        identity=IDENTITY,
        record=_record(1),
    )
    expected_identity = {
        **IDENTITY,
        identity_field: "d" * 64,
    }

    with pytest.raises(
        ValueError,
        match="accepted FSI checkpoint identity mismatch",
    ) as exc_info:
        load_accepted_fsi_checkpoint(
            checkpoint_prefix,
            expected_identity=expected_identity,
            target_step_count=1,
        )

    assert identity_field in str(exc_info.value)
