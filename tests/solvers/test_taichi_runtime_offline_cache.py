from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from simulation_core.diagnostics import runtime


def test_init_taichi_forwards_explicit_offline_cache_configuration(
    tmp_path: Path,
) -> None:
    original_state = (
        runtime._INITIALIZED,
        runtime._INITIALIZED_ARCH,
        runtime._INITIALIZED_FP,
        runtime._INITIALIZED_OFFLINE_CACHE,
        runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
    )
    runtime._INITIALIZED = False
    runtime._INITIALIZED_ARCH = None
    runtime._INITIALIZED_FP = None
    runtime._INITIALIZED_OFFLINE_CACHE = None
    runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = None
    cache_dir = tmp_path / "taichi-cache"

    try:
        with patch.object(runtime.ti, "init") as taichi_init:
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(
                    arch="cuda",
                    offline_cache=True,
                    offline_cache_file_path=str(cache_dir),
                )
            )
    finally:
        (
            runtime._INITIALIZED,
            runtime._INITIALIZED_ARCH,
            runtime._INITIALIZED_FP,
            runtime._INITIALIZED_OFFLINE_CACHE,
            runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
        ) = original_state

    taichi_init.assert_called_once_with(
        arch=runtime.ti.cuda,
        default_fp=runtime.ti.f32,
        random_seed=0,
        offline_cache=True,
        offline_cache_file_path=str(cache_dir),
    )


def test_init_taichi_rejects_conflicting_cache_reconfiguration(
    tmp_path: Path,
) -> None:
    state_names = (
        "_INITIALIZED",
        "_INITIALIZED_ARCH",
        "_INITIALIZED_FP",
        "_INITIALIZED_OFFLINE_CACHE",
        "_INITIALIZED_OFFLINE_CACHE_FILE_PATH",
    )
    missing = object()
    original_state = {
        name: getattr(runtime, name, missing)
        for name in state_names
    }
    runtime._INITIALIZED = False
    runtime._INITIALIZED_ARCH = None
    runtime._INITIALIZED_FP = None
    runtime._INITIALIZED_OFFLINE_CACHE = None
    runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = None
    cache_dir = tmp_path / "taichi-cache"

    try:
        with patch.object(runtime.ti, "init"):
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(
                    arch="cuda",
                    offline_cache=True,
                    offline_cache_file_path=str(cache_dir),
                )
            )
            with pytest.raises(
                RuntimeError,
                match="offline-cache configuration",
            ):
                runtime.init_taichi(
                    runtime.TaichiRuntimeConfig(
                        arch="cuda",
                        offline_cache=False,
                    )
                )
    finally:
        for name, value in original_state.items():
            if value is missing:
                delattr(runtime, name)
            else:
                setattr(runtime, name, value)


def test_init_taichi_uses_simulation_cache_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_state = (
        runtime._INITIALIZED,
        runtime._INITIALIZED_ARCH,
        runtime._INITIALIZED_FP,
        runtime._INITIALIZED_OFFLINE_CACHE,
        runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
    )
    runtime._INITIALIZED = False
    runtime._INITIALIZED_ARCH = None
    runtime._INITIALIZED_FP = None
    runtime._INITIALIZED_OFFLINE_CACHE = None
    runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = None
    cache_dir = tmp_path / "environment-cache"
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "true")
    monkeypatch.setenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        str(cache_dir),
    )

    try:
        with patch.object(runtime.ti, "init") as taichi_init:
            runtime.init_taichi(runtime.TaichiRuntimeConfig())
    finally:
        (
            runtime._INITIALIZED,
            runtime._INITIALIZED_ARCH,
            runtime._INITIALIZED_FP,
            runtime._INITIALIZED_OFFLINE_CACHE,
            runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
        ) = original_state

    assert taichi_init.call_args.kwargs["offline_cache"] is True
    assert (
        taichi_init.call_args.kwargs["offline_cache_file_path"]
        == str(cache_dir)
    )


def test_init_taichi_rejects_invalid_cache_environment(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "sometimes")
    with pytest.raises(ValueError, match="SIMULATION_TAICHI_OFFLINE_CACHE"):
        runtime._environment_flag("SIMULATION_TAICHI_OFFLINE_CACHE")


def test_concurrent_matching_initialization_calls_taichi_once(
    tmp_path: Path,
) -> None:
    original_state = (
        runtime._INITIALIZED,
        runtime._INITIALIZED_ARCH,
        runtime._INITIALIZED_FP,
        runtime._INITIALIZED_OFFLINE_CACHE,
        runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
    )
    runtime._INITIALIZED = False
    runtime._INITIALIZED_ARCH = None
    runtime._INITIALIZED_FP = None
    runtime._INITIALIZED_OFFLINE_CACHE = None
    runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = None
    cache_dir = tmp_path / "concurrent-cache"
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    call_count = 0

    def fake_init(**kwargs) -> None:
        nonlocal call_count
        del kwargs
        call_count += 1
        entered.set()
        assert release.wait(timeout=5.0)

    def initialize() -> None:
        try:
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(
                    offline_cache=True,
                    offline_cache_file_path=str(cache_dir),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    try:
        with patch.object(runtime.ti, "init", side_effect=fake_init):
            first = threading.Thread(target=initialize)
            second = threading.Thread(target=initialize)
            first.start()
            assert entered.wait(timeout=5.0)
            second.start()
            release.set()
            first.join(timeout=5.0)
            second.join(timeout=5.0)
    finally:
        (
            runtime._INITIALIZED,
            runtime._INITIALIZED_ARCH,
            runtime._INITIALIZED_FP,
            runtime._INITIALIZED_OFFLINE_CACHE,
            runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
        ) = original_state

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert call_count == 1


def test_failed_taichi_initialization_does_not_publish_runtime_state() -> None:
    original_state = (
        runtime._INITIALIZED,
        runtime._INITIALIZED_ARCH,
        runtime._INITIALIZED_FP,
        runtime._INITIALIZED_OFFLINE_CACHE,
        runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
    )
    runtime._INITIALIZED = False
    runtime._INITIALIZED_ARCH = None
    runtime._INITIALIZED_FP = None
    runtime._INITIALIZED_OFFLINE_CACHE = None
    runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = None

    try:
        with (
            patch.object(
                runtime.ti,
                "init",
                side_effect=RuntimeError("synthetic ti.init failure"),
            ),
            pytest.raises(RuntimeError, match="synthetic ti.init failure"),
        ):
            runtime.init_taichi(runtime.TaichiRuntimeConfig())
        assert runtime._INITIALIZED is False
        assert runtime._INITIALIZED_OFFLINE_CACHE is None
        assert runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH is None
    finally:
        (
            runtime._INITIALIZED,
            runtime._INITIALIZED_ARCH,
            runtime._INITIALIZED_FP,
            runtime._INITIALIZED_OFFLINE_CACHE,
            runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH,
        ) = original_state
