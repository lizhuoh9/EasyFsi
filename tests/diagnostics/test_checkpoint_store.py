"""Host-only contract tests for accepted-step checkpoint persistence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from simulation_core.diagnostics import checkpoint_store


def _manifest_path(prefix: Path) -> Path:
    return prefix.with_suffix(".json")


def _history_tail(prefix: Path, step: int, previous=None):
    return checkpoint_store.append_history(
        prefix,
        step=step,
        record={"accepted_step": step, "residual": float(step) / 10.0},
        previous=previous,
    )


def _save_step(
    prefix: Path,
    *,
    step: int,
    previous=None,
    expected_generation: str | None = None,
):
    tail = _history_tail(prefix, step, previous)
    return checkpoint_store.save_checkpoint(
        prefix,
        accepted_step=step,
        metadata={
            "identity": {"case": "host-only", "step": step},
            "flags": [True, False],
        },
        arrays={
            "pressure": np.asarray([[step, step + 1]], dtype=np.float32),
            "fluid_mask": np.asarray([[True, False]], dtype=np.bool_),
        },
        history_tail=tail,
        expected_generation=expected_generation,
    ), tail


def _generation_npz_paths(prefix: Path) -> set[Path]:
    return set(prefix.parent.glob(f"{prefix.name}.generation.*.npz"))


def test_read_checkpoint_head_reads_only_the_strict_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "state"
    generation, tail = _save_step(prefix, step=1)

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("head read must not load NPZ or walk history")

    monkeypatch.setattr(checkpoint_store, "_load_arrays", fail_if_called)
    monkeypatch.setattr(checkpoint_store, "_load_history", fail_if_called)

    head = checkpoint_store.read_checkpoint_head(prefix)

    assert head is not None
    assert head.accepted_step == 1
    assert head.generation == generation
    assert head.history_tail == tail
    assert head.metadata == {"identity": {"case": "host-only", "step": 1}, "flags": (True, False)}


def test_read_checkpoint_head_returns_none_only_for_absent_manifest(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    assert checkpoint_store.read_checkpoint_head(prefix) is None

    _generation, _tail = _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest_path.write_text("{not JSON", encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.read_checkpoint_head(prefix)


def test_retention_keeps_only_current_and_previous_generation_after_many_saves(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    previous = None
    for step in range(1, 10):
        _generation, previous = _save_step(prefix, step=step, previous=previous)

    stored = checkpoint_store.load_checkpoint(prefix)

    assert stored.accepted_step == 9
    assert stored.history_tail == previous
    assert len(_generation_npz_paths(prefix)) <= 2
    assert len(list(tmp_path.glob("state.history.*.json"))) == 9


def test_manifest_replace_failure_never_prunes_last_committed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "state"
    old_generation, old_tail = _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest_before = manifest_path.read_bytes()
    old_npz = tmp_path / json.loads(manifest_before)["npz_file"]
    next_tail = _history_tail(prefix, 2, old_tail)
    real_replace = checkpoint_store.os.replace

    def fail_manifest_replace(source, destination) -> None:
        if Path(destination) == manifest_path:
            raise OSError("injected manifest replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_store.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest replacement"):
        checkpoint_store.save_checkpoint(
            prefix,
            accepted_step=2,
            metadata={"identity": {"case": "fault"}},
            arrays={"pressure": np.asarray([2.0])},
            history_tail=next_tail,
            expected_generation=old_generation,
        )

    assert manifest_path.read_bytes() == manifest_before
    assert old_npz.is_file()
    assert checkpoint_store.load_checkpoint(prefix).generation == old_generation


def test_tampered_previous_manifest_fails_closed_without_generation_cleanup(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    _old_generation, old_tail = _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_npz = tmp_path / manifest["npz_file"]
    manifest["npz_file"] = "../outside.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered_manifest = manifest_path.read_bytes()

    with pytest.raises((TypeError, ValueError)):
        _save_step(prefix, step=2, previous=old_tail)

    assert manifest_path.read_bytes() == tampered_manifest
    assert old_npz.is_file()


def test_retention_preserves_other_prefix_lookalikes_history_and_symlinks(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    other_prefix = tmp_path / "state_other"
    _other_generation, _other_tail = _save_step(other_prefix, step=1)
    first_generation, first_tail = _save_step(prefix, step=1)
    first_npz = tmp_path / f"{prefix.name}.generation.{first_generation}.npz"
    lookalike = tmp_path / f"{prefix.name}.generation.not-a-generation.npz"
    lookalike.write_bytes(b"lookalike")
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    link = tmp_path / f"{prefix.name}.generation.ffffffffffffffffffffffffffffffff.npz"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    _second_generation, second_tail = _save_step(prefix, step=2, previous=first_tail)
    _third_generation, _third_tail = _save_step(prefix, step=3, previous=second_tail)

    assert not first_npz.exists()
    assert checkpoint_store.load_checkpoint(other_prefix).accepted_step == 1
    assert lookalike.read_bytes() == b"lookalike"
    assert link.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert len(list(tmp_path.glob("state.history.*.json"))) == 3


def test_retention_cleanup_failure_warns_after_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "state"
    first_generation, first_tail = _save_step(prefix, step=1)
    second_generation, second_tail = _save_step(
        prefix,
        step=2,
        previous=first_tail,
        expected_generation=first_generation,
    )

    def fail_cleanup(*_args, **_kwargs) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        checkpoint_store,
        "_cleanup_obsolete_generations",
        fail_cleanup,
        raising=False,
    )
    with pytest.warns(RuntimeWarning, match="retention"):
        third_generation, _third_tail = _save_step(
            prefix,
            step=3,
            previous=second_tail,
            expected_generation=second_generation,
        )

    assert third_generation != second_generation
    assert checkpoint_store.load_checkpoint(prefix).accepted_step == 3


def test_retention_cleanup_warning_cannot_escape_warnings_as_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "state"
    first_generation, first_tail = _save_step(prefix, step=1)
    second_generation, second_tail = _save_step(
        prefix,
        step=2,
        previous=first_tail,
        expected_generation=first_generation,
    )
    first_path = tmp_path / f"{prefix.name}.generation.{first_generation}.npz"
    real_unlink = Path.unlink

    def fail_obsolete_unlink(path: Path, *args, **kwargs) -> None:
        if path == first_path:
            raise PermissionError("injected generation retention failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_obsolete_unlink)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        third_generation, _third_tail = _save_step(
            prefix,
            step=3,
            previous=second_tail,
            expected_generation=second_generation,
        )

    assert checkpoint_store.load_checkpoint(prefix).generation == third_generation


def test_retention_cleanup_uses_manifest_pointer_without_directory_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "state"
    first_generation, first_tail = _save_step(prefix, step=1)
    second_generation, second_tail = _save_step(
        prefix,
        step=2,
        previous=first_tail,
        expected_generation=first_generation,
    )

    def fail_directory_enumeration(*_args, **_kwargs) -> None:
        raise AssertionError("generation retention must not enumerate the store")

    monkeypatch.setattr(Path, "iterdir", fail_directory_enumeration)
    third_generation, _third_tail = _save_step(
        prefix,
        step=3,
        previous=second_tail,
        expected_generation=second_generation,
    )

    assert third_generation != second_generation


def test_manifest_previous_generation_pointer_retains_two_generations(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    first_generation, first_tail = _save_step(prefix, step=1)
    second_generation, second_tail = _save_step(
        prefix,
        step=2,
        previous=first_tail,
        expected_generation=first_generation,
    )
    third_generation, _third_tail = _save_step(
        prefix,
        step=3,
        previous=second_tail,
        expected_generation=second_generation,
    )

    manifest = json.loads(_manifest_path(prefix).read_text(encoding="utf-8"))
    first_path = tmp_path / f"{prefix.name}.generation.{first_generation}.npz"
    second_path = tmp_path / f"{prefix.name}.generation.{second_generation}.npz"

    assert manifest["generation"] == third_generation
    assert manifest["previous_generation"] == second_generation
    assert not first_path.exists()
    assert second_path.is_file()


def test_head_rejects_invalid_previous_generation_pointer(tmp_path: Path) -> None:
    prefix = tmp_path / "state"
    _generation, _tail = _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["previous_generation"] = "not-a-generation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.read_checkpoint_head(prefix)


def test_pointer_cleanup_skips_external_symlink_replacing_obsolete_generation(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    first_generation, first_tail = _save_step(prefix, step=1)
    second_generation, second_tail = _save_step(
        prefix,
        step=2,
        previous=first_tail,
        expected_generation=first_generation,
    )
    first_path = tmp_path / f"{prefix.name}.generation.{first_generation}.npz"
    outside = tmp_path.parent / "external-retention-target.npz"
    outside.write_bytes(b"outside")
    first_path.unlink()
    try:
        first_path.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    third_generation, _third_tail = _save_step(
        prefix,
        step=3,
        previous=second_tail,
        expected_generation=second_generation,
    )

    assert checkpoint_store.load_checkpoint(prefix).generation == third_generation
    assert first_path.is_symlink()
    assert outside.read_bytes() == b"outside"


def test_round_trip_returns_immutable_defensive_payloads(tmp_path: Path) -> None:
    prefix = tmp_path / "state"
    arrays = {
        "pressure": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "fluid_mask": np.asarray([[True, False]], dtype=np.bool_),
    }
    tail = _history_tail(prefix, 1)
    generation = checkpoint_store.save_checkpoint(
        prefix,
        accepted_step=1,
        metadata={"identity": {"case": "roundtrip"}, "enabled": True},
        arrays=arrays,
        history_tail=tail,
    )
    arrays["pressure"][0, 0] = 99.0

    stored = checkpoint_store.load_checkpoint(prefix)

    assert generation == stored.generation
    assert _manifest_path(prefix).is_file()
    assert stored.accepted_step == 1
    assert stored.history_tail == tail
    assert stored.history == ({"accepted_step": 1, "residual": 0.1},)
    assert stored.metadata == {"identity": {"case": "roundtrip"}, "enabled": True}
    assert stored.arrays["pressure"].flags.writeable is False
    assert stored.arrays["pressure"][0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stored.arrays["pressure"][0, 0] = 5.0
    with pytest.raises(TypeError):
        stored.metadata["enabled"] = False


def test_history_is_incremental_and_loads_exact_contiguous_prefix(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    first = _history_tail(prefix, 1)
    second = _history_tail(prefix, 2, first)
    generation = checkpoint_store.save_checkpoint(
        prefix,
        accepted_step=2,
        metadata={"identity": {"case": "history"}},
        arrays={"pressure": np.asarray([2.0], dtype=np.float64)},
        history_tail=second,
    )

    manifest = json.loads(_manifest_path(prefix).read_text(encoding="utf-8"))
    stored = checkpoint_store.load_checkpoint(prefix)

    assert manifest["generation"] == generation
    assert "history" not in manifest
    assert manifest["history_tail"]["step"] == 2
    assert [entry["accepted_step"] for entry in stored.history] == [1, 2]
    assert len({entry["accepted_step"] for entry in stored.history}) == 2
    assert len(list(tmp_path.glob("state.history.*.json"))) == 2
    assert len(_manifest_path(prefix).read_bytes()) < 1024


@pytest.mark.parametrize(
    ("metadata", "arrays"),
    [
        ({"bad": float("nan")}, {"pressure": np.asarray([1.0])}),
        ({"bad": {"nested": float("inf")}}, {"pressure": np.asarray([1.0])}),
        ({}, {"object": np.asarray([{"bad": 1}], dtype=object)}),
        ({}, {"text": np.asarray(["not physical"])}),
        ({}, {"pressure": np.asarray([np.inf], dtype=np.float32)}),
        ({}, {"pressure": np.asarray([np.nan], dtype=np.float64)}),
    ],
)
def test_save_rejects_non_json_metadata_and_nonphysical_arrays(
    tmp_path: Path,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.save_checkpoint(
            tmp_path / "state",
            accepted_step=0,
            metadata=metadata,
            arrays=arrays,
            history_tail=None,
        )


def test_history_and_path_contracts_fail_closed(tmp_path: Path) -> None:
    prefix = tmp_path / "state"
    first = _history_tail(prefix, 1)

    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.append_history(
            prefix,
            step=3,
            record={"accepted_step": 3},
            previous=first,
        )
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.save_checkpoint(
            prefix,
            accepted_step=2,
            metadata={},
            arrays={"pressure": np.asarray([2.0])},
            history_tail=first,
        )
    with pytest.raises(ValueError):
        checkpoint_store.save_checkpoint(
            prefix.with_suffix(".npz"),
            accepted_step=1,
            metadata={},
            arrays={"pressure": np.asarray([1.0])},
            history_tail=first,
        )

    _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_file"] = "../outside.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)


def test_load_rejects_duplicate_manifest_keys_and_broken_history(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "state"
    _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest_path.write_text(
        '{"format":"checkpoint-store","format":"checkpoint-store"}',
        encoding="utf-8",
    )
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)

    manifest_path.unlink()

    _save_step(prefix, step=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["history_tail"]["step"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)


def test_load_rejects_unknown_schema_array_manifest_and_bad_hash(tmp_path: Path) -> None:
    prefix = tmp_path / "state"
    _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)

    manifest_path.unlink()

    _save_step(prefix, step=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"]["pressure"]["dtype"] = "<f8"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)

    manifest_path.unlink()

    _save_step(prefix, step=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"]["pressure"]["shape"] = [2]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)

    manifest_path.unlink()

    _save_step(prefix, step=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"]["pressure"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)

    manifest_path.unlink()

    _save_step(prefix, step=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (tmp_path / manifest["npz_file"]).unlink()
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)

    _save_step(prefix, step=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.load_checkpoint(prefix)


def test_expected_generation_rejects_stale_writer(tmp_path: Path) -> None:
    prefix = tmp_path / "state"
    first_generation, first_tail = _save_step(prefix, step=1)
    second_generation, second_tail = _save_step(
        prefix,
        step=2,
        previous=first_tail,
        expected_generation=first_generation,
    )

    assert second_generation != first_generation
    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.save_checkpoint(
            prefix,
            accepted_step=2,
            metadata={"identity": {"case": "stale"}},
            arrays={"pressure": np.asarray([2.0])},
            history_tail=second_tail,
            expected_generation=first_generation,
        )


@pytest.mark.parametrize(
    "boundary",
    ("npz_write", "npz_publish", "manifest_replace"),
)
def test_failed_save_never_replaces_last_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    prefix = tmp_path / "state"
    old_generation, old_tail = _save_step(prefix, step=1)
    manifest_path = _manifest_path(prefix)
    manifest_before = manifest_path.read_bytes()
    next_tail = _history_tail(prefix, 2, old_tail)

    if boundary == "npz_write":
        def fail_write(*_args, **_kwargs) -> None:
            raise OSError("injected npz write failure")

        monkeypatch.setattr(checkpoint_store, "_write_npz", fail_write)
    else:
        real_replace = checkpoint_store.os.replace

        def fail_replace(source, destination) -> None:
            destination_path = Path(destination)
            if (
                boundary == "npz_publish"
                and destination_path.suffix == ".npz"
            ) or (
                boundary == "manifest_replace"
                and destination_path == manifest_path
            ):
                raise OSError(f"injected {boundary} failure")
            real_replace(source, destination)

        monkeypatch.setattr(checkpoint_store.os, "replace", fail_replace)

    with pytest.raises(OSError):
        checkpoint_store.save_checkpoint(
            prefix,
            accepted_step=2,
            metadata={"identity": {"case": "fault"}},
            arrays={"pressure": np.asarray([2.0])},
            history_tail=next_tail,
            expected_generation=old_generation,
        )

    assert manifest_path.read_bytes() == manifest_before
    restored = checkpoint_store.load_checkpoint(prefix)
    assert restored.accepted_step == 1
    assert restored.generation == old_generation


def test_history_reference_hash_binds_filename_and_content(tmp_path: Path) -> None:
    prefix = tmp_path / "state"
    tail = _history_tail(prefix, 1)
    history_path = tmp_path / tail.filename
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload["record"]["accepted_step"] = 2
    history_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        checkpoint_store.save_checkpoint(
            prefix,
            accepted_step=1,
            metadata={},
            arrays={"pressure": np.asarray([1.0])},
            history_tail=tail,
        )


@pytest.mark.parametrize("operation", ["save", "append"])
@pytest.mark.parametrize("bad_step", [True, 1.0])
def test_invalid_tail_step_cannot_publish_an_unloadable_checkpoint(
    tmp_path: Path, operation: str, bad_step: object,
) -> None:
    prefix = tmp_path / "state"
    generation, tail = _save_step(prefix, step=1)
    before = _manifest_path(prefix).read_bytes()
    invalid_tail = replace(tail, step=bad_step)
    with pytest.raises((TypeError, ValueError)):
        if operation == "append":
            checkpoint_store.append_history(
                prefix, step=2, record={"accepted_step": 2}, previous=invalid_tail,
            )
        else:
            checkpoint_store.save_checkpoint(
                prefix, accepted_step=1, metadata={},
                arrays={"pressure": np.asarray([1.0])},
                history_tail=invalid_tail, expected_generation=generation,
            )
    assert _manifest_path(prefix).read_bytes() == before
    assert checkpoint_store.load_checkpoint(prefix).generation == generation
