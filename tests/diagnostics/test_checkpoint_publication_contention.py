"""Publication contention must never re-run physics or damage the old head."""

from __future__ import annotations

import errno
import importlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from simulation_core.diagnostics import checkpoint_store


def _windows_error(code: int) -> PermissionError:
    error = PermissionError(errno.EACCES, "injected Windows publication conflict")
    error.winerror = code
    return error


def _save(prefix: Path, step: int, *, previous=None, generation=None):
    tail = checkpoint_store.append_history(
        prefix, step=step, record={"step": step}, previous=previous,
    )
    saved = checkpoint_store.save_checkpoint(
        prefix, accepted_step=step, metadata={"step": step},
        arrays={"pressure": np.asarray([float(step)], dtype=np.float32)},
        history_tail=tail, expected_generation=generation,
    )
    return saved, tail


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_committed_head_survives_transient_manifest_contention(tmp_path, monkeypatch, winerror):
    prefix = tmp_path / "checkpoint"
    generation, tail = _save(prefix, 1)
    manifest = prefix.with_suffix(".json")
    old_bytes = manifest.read_bytes()
    real_replace = os.replace
    manifest_calls = []
    npz_calls = []

    def contended_replace(source, destination):
        if Path(destination) == manifest:
            manifest_calls.append((Path(source), Path(destination)))
            assert manifest.read_bytes() == old_bytes
            assert checkpoint_store.load_checkpoint(prefix).generation == generation
            if len(manifest_calls) < 4:
                raise _windows_error(winerror)
        elif str(destination).endswith(".npz"):
            npz_calls.append(Path(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", contended_replace)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    new_generation, new_tail = _save(prefix, 2, previous=tail, generation=generation)
    assert len(manifest_calls) == 4
    assert len(set(manifest_calls)) == 1
    assert len(npz_calls) == 1
    loaded = checkpoint_store.load_checkpoint(prefix)
    assert loaded.generation == new_generation
    assert loaded.accepted_step == 2 and loaded.history_tail == new_tail
    assert len(loaded.history) == 2
    assert np.array_equal(loaded.arrays["pressure"], [2.0])


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_persistent_windows_conflict_is_bounded_and_preserves_last_head(tmp_path, monkeypatch, winerror):
    prefix = tmp_path / "checkpoint"
    generation, tail = _save(prefix, 1)
    manifest = prefix.with_suffix(".json")
    before = manifest.read_bytes()
    real_replace = os.replace
    attempts = []
    delays = []
    failure = _windows_error(winerror)

    def blocked_replace(source, destination):
        if Path(destination) == manifest:
            attempts.append(Path(source))
            raise failure
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", blocked_replace)
    monkeypatch.setattr(time, "sleep", delays.append)
    with pytest.raises(PermissionError) as caught:
        _save(prefix, 2, previous=tail, generation=generation)
    assert caught.value is failure
    assert 1 < len(attempts) <= 8
    assert len(set(attempts)) == 1
    assert len(delays) == len(attempts) - 1
    assert 0.0 < sum(delays) <= 1.0
    assert manifest.read_bytes() == before
    assert checkpoint_store.load_checkpoint(prefix).generation == generation


@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "POSIX access denied"),
    OSError(errno.ENOSPC, "disk full"),
    OSError(errno.EIO, "I/O error"),
    OSError(errno.EROFS, "read-only filesystem"),
    FileNotFoundError(errno.ENOENT, "missing path"),
    _windows_error(87),
])
def test_nonretryable_publication_errors_are_immediate(tmp_path, monkeypatch, error):
    prefix = tmp_path / "checkpoint"
    generation, tail = _save(prefix, 1)
    manifest = prefix.with_suffix(".json")
    before = manifest.read_bytes()
    real_replace = os.replace
    attempts = []

    def failed_replace(source, destination):
        if Path(destination) == manifest:
            attempts.append(Path(source))
            raise error
        return real_replace(source, destination)

    def forbidden_sleep(_seconds):
        raise AssertionError("permanent/non-Windows I/O failure must not be retried")

    monkeypatch.setattr(os, "replace", failed_replace)
    monkeypatch.setattr(time, "sleep", forbidden_sleep)
    with pytest.raises(OSError) as caught:
        _save(prefix, 2, previous=tail, generation=generation)
    assert caught.value is error and len(attempts) == 1
    assert manifest.read_bytes() == before
    assert checkpoint_store.load_checkpoint(prefix).generation == generation


def test_atomic_replace_success_never_sleeps(tmp_path, monkeypatch):
    atomic = importlib.import_module("simulation_core.diagnostics.atomic_file")
    source, destination = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("uncontended write must not sleep"))
    atomic.replace_file_atomically(source, destination)
    assert destination.read_bytes() == b"new"
    assert not source.exists()


@pytest.mark.parametrize("target_kind", ["history", "npz", "manifest"])
def test_all_checkpoint_publications_retry_the_same_prepared_file(tmp_path, monkeypatch, target_kind):
    prefix = tmp_path / "checkpoint"
    generation, tail = _save(prefix, 1)
    real_replace = os.replace
    attempted = []
    delays = []

    def publish(source, destination):
        name = Path(destination).name
        selected = {
            "history": ".history." in name,
            "npz": name.endswith(".npz"),
            "manifest": name == "checkpoint.json",
        }[target_kind]
        if selected:
            attempted.append((Path(source), Path(destination)))
            if len(attempted) <= 3:
                raise _windows_error(5)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", publish)
    monkeypatch.setattr(time, "sleep", delays.append)
    _save(prefix, 2, previous=tail, generation=generation)
    assert len(attempted) == 4 and len(set(attempted)) == 1
    assert delays == [0.01, 0.02, 0.04]
    assert checkpoint_store.load_checkpoint(prefix).accepted_step == 2


def _load_validation_cli():
    repository = Path(__file__).resolve().parents[2]
    path = repository / (
        "validation_runs/ansys_vertical_flap_fsi/"
        "our_solver_fine_vs_fluent_2026-07-02/scripts/run_our_solver_vertical_flap.py"
    )
    spec = importlib.util.spec_from_file_location("checkpoint_contention_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_mutable_json_publication_retries_without_regenerating_payload(
    tmp_path,
    monkeypatch,
):
    cli = _load_validation_cli()
    destination = tmp_path / "progress.json"
    destination.write_bytes(b"old")
    real_replace = os.replace
    attempted = []

    def publish(source, target):
        attempted.append((Path(source), Path(target)))
        assert destination.read_bytes() == b"old"
        if len(attempted) < 3:
            raise _windows_error(5)
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", publish)
    monkeypatch.setattr(Path, "replace", lambda source, target: publish(source, target))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    cli._write_json_atomic(destination, {"step": 2})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"step": 2}
    assert len(attempted) == 3 and len(set(attempted)) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows create-only rename retries")
def test_cli_immutable_npz_publication_retries_without_regenerating_payload(
    tmp_path,
    monkeypatch,
):
    cli = _load_validation_cli()
    destination = tmp_path / "step_0001.npz"
    rendered = []
    real_rename = os.rename
    attempted = []

    def render(path, _snapshot, **_kwargs):
        rendered.append(Path(path))
        np.savez(path, value=np.asarray([2.0]))
        return {"path": str(path)}

    def publish(source, target):
        attempted.append((Path(source), Path(target)))
        assert not destination.exists()
        if len(attempted) < 3:
            raise _windows_error(5)
        return real_rename(source, target)

    monkeypatch.setattr(os, "rename", publish)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(cli, "save_solver_npz_from_flow_snapshot", render)
    monkeypatch.setattr(cli, "_step_structure_arrays", lambda *_args, **_kwargs: {})
    cli._save_step_frame_atomic(
        destination, {}, span_reduction="mean", streamwise_velocity_sign=-1.0,
        reverse_streamwise_axis=True, streamwise_length_m=0.1,
    )
    assert len(rendered) == 1
    with np.load(destination, allow_pickle=False) as loaded:
        assert np.array_equal(loaded["value"], [2.0])
    assert len(attempted) == 3 and len(set(attempted)) == 1


def test_cli_non_windows_permission_error_is_not_retried(tmp_path, monkeypatch):
    cli = _load_validation_cli()
    attempted = []

    def publish(*_args):
        attempted.append(True)
        raise PermissionError(errno.EACCES, "not a Windows sharing error")

    monkeypatch.setattr(os, "replace", publish)
    monkeypatch.setattr(Path, "replace", lambda source, target: publish(source, target))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        cli._write_json_atomic(tmp_path / "progress.json", {"step": 1})
    assert len(attempted) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows read sharing semantics")
def test_real_windows_read_handle_can_release_during_atomic_publication(tmp_path):
    atomic = importlib.import_module("simulation_core.diagnostics.atomic_file")
    source, destination = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    handle = destination.open("rb")
    # A normal Python read handle does not grant delete/rename sharing.
    try:
        with pytest.raises(PermissionError) as caught:
            os.replace(source, destination)
        assert caught.value.winerror in (5, 32, 33)
        release = threading.Timer(0.05, handle.close)
        release.start()
        try:
            atomic.replace_file_atomically(source, destination)
        finally:
            release.join(timeout=1.0)
    finally:
        handle.close()
    assert destination.read_bytes() == b"new" and not source.exists()
