from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from simulation_core.diagnostics import runtime


_RUNTIME_ENVIRONMENT_NAMES = (
    "SIMULATION_TAICHI_OFFLINE_CACHE",
    "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
    "TI_OFFLINE_CACHE",
    "TI_OFFLINE_CACHE_FILE_PATH",
)


@pytest.fixture(autouse=True)
def _reset_runtime_state(monkeypatch) -> None:
    for name, value in {
        "_INITIALIZED": False,
        "_INITIALIZED_ARCH": None,
        "_INITIALIZED_FP": None,
        "_INITIALIZED_RANDOM_SEED": None,
        "_INITIALIZED_OFFLINE_CACHE": None,
        "_INITIALIZED_OFFLINE_CACHE_FILE_PATH": None,
    }.items():
        monkeypatch.setattr(runtime, name, value, raising=False)
    for name in _RUNTIME_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_runtime_config_defaults_preserve_legacy_non_strict_api() -> None:
    config = runtime.TaichiRuntimeConfig()

    assert config.arch == "cuda"
    assert config.default_fp == "f32"
    assert config.random_seed == 0
    assert config.offline_cache is None
    assert config.offline_cache_file_path is None
    assert config.strict_arch is False


def _publish_preinitialized_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_INITIALIZED", True)
    monkeypatch.setattr(runtime, "_INITIALIZED_ARCH", "cpu")
    monkeypatch.setattr(runtime, "_INITIALIZED_FP", "f32")


def test_implicit_default_reuses_preinitialized_cpu_runtime(monkeypatch) -> None:
    _publish_preinitialized_cpu_runtime(monkeypatch)

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi()

    taichi_init.assert_not_called()


def test_implicit_default_does_not_resolve_cache_environment(monkeypatch) -> None:
    _publish_preinitialized_cpu_runtime(monkeypatch)
    monkeypatch.setenv("TI_OFFLINE_CACHE", "invalid-but-unused")

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi()

    taichi_init.assert_not_called()


def test_implicit_default_reuses_explicit_nondefault_identity(
    tmp_path: Path,
) -> None:
    explicit_config = runtime.TaichiRuntimeConfig(
        arch="gpu",
        random_seed=7,
        offline_cache=False,
        offline_cache_file_path=str(tmp_path / "cache"),
    )

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(explicit_config)
        runtime.init_taichi()

    assert taichi_init.call_count == 1


def test_implicit_default_still_rejects_float_mode_conflict(
    monkeypatch,
) -> None:
    _publish_preinitialized_cpu_runtime(monkeypatch)
    monkeypatch.setattr(runtime, "_INITIALIZED_FP", "f64")

    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(
            ValueError,
            match=(
                "already initialized with default_fp='f64'; cannot "
                "re-initialize with default_fp='f32'"
            ),
        ),
    ):
        runtime.init_taichi()

    taichi_init.assert_not_called()


def test_explicit_non_strict_cuda_rejects_preinitialized_cpu_runtime(
    monkeypatch,
) -> None:
    _publish_preinitialized_cpu_runtime(monkeypatch)

    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(
            ValueError,
            match=(
                "already initialized with arch='cpu'; cannot re-initialize "
                "with arch='cuda'"
            ),
        ),
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig(arch="cuda"))

    taichi_init.assert_not_called()


def test_strict_cuda_rejects_preinitialized_cpu_runtime(monkeypatch) -> None:
    _publish_preinitialized_cpu_runtime(monkeypatch)
    actual_config = SimpleNamespace(arch=runtime.ti.cpu)

    with (
        patch.object(runtime.ti, "cfg", actual_config),
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(RuntimeError, match="actual runtime arch"),
    ):
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(arch="cuda", strict_arch=True)
        )

    taichi_init.assert_not_called()


def test_strict_cuda_disables_fallback_and_checks_actual_backend() -> None:
    actual_config = SimpleNamespace(arch=runtime.ti.cuda)

    with (
        patch.object(runtime.ti, "cfg", actual_config),
        patch.object(runtime.ti, "init") as taichi_init,
    ):
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(arch="cuda", strict_arch=True)
        )

    taichi_init.assert_called_once_with(
        arch=runtime.ti.cuda,
        default_fp=runtime.ti.f32,
        random_seed=0,
        offline_cache=True,
        enable_fallback=False,
    )


def test_strict_cuda_rejects_fallback_without_publishing_state() -> None:
    actual_config = SimpleNamespace(arch=runtime.ti.cpu)

    with (
        patch.object(runtime.ti, "cfg", actual_config),
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(RuntimeError, match="actual runtime arch"),
    ):
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(arch="cuda", strict_arch=True)
        )

    taichi_init.assert_called_once()
    assert runtime._INITIALIZED is False
    assert runtime._INITIALIZED_ARCH is None


def test_strict_fast_path_rechecks_actual_backend() -> None:
    actual_config = SimpleNamespace(arch=runtime.ti.cpu)

    with (
        patch.object(runtime.ti, "cfg", actual_config),
        patch.object(runtime.ti, "init") as taichi_init,
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig(arch="cuda"))
        with pytest.raises(RuntimeError, match="actual runtime arch"):
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(arch="cuda", strict_arch=True)
            )

    assert taichi_init.call_count == 1


@pytest.mark.parametrize("value", [None, "true", 0, 1])
def test_strict_arch_requires_boolean(value: object) -> None:
    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match="strict_arch must be a bool"),
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig(strict_arch=value))

    taichi_init.assert_not_called()


def test_explicit_offline_cache_configuration_is_forwarded(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "taichi-cache"

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(
                offline_cache=False,
                offline_cache_file_path=str(cache_dir),
            )
        )

    taichi_init.assert_called_once_with(
        arch=runtime.ti.cuda,
        default_fp=runtime.ti.f32,
        random_seed=0,
        offline_cache=False,
        offline_cache_file_path=os.path.normcase(str(cache_dir)),
    )


def test_validation_environment_disables_offline_cache(monkeypatch) -> None:
    monkeypatch.setenv("TI_OFFLINE_CACHE", "0")

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(runtime.TaichiRuntimeConfig())

    assert taichi_init.call_args.kwargs["offline_cache"] is False
    assert "offline_cache_file_path" not in taichi_init.call_args.kwargs


def test_simulation_cache_environment_has_priority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    simulation_cache = tmp_path / "simulation-cache"
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "false")
    monkeypatch.setenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        str(simulation_cache),
    )
    monkeypatch.setenv("TI_OFFLINE_CACHE", "true")
    monkeypatch.setenv(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(tmp_path / "taichi-cache"),
    )

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(runtime.TaichiRuntimeConfig())

    assert taichi_init.call_args.kwargs["offline_cache"] is False
    assert taichi_init.call_args.kwargs["offline_cache_file_path"] == (
        os.path.normcase(str(simulation_cache))
    )


@pytest.mark.parametrize(
    "name",
    ["SIMULATION_TAICHI_OFFLINE_CACHE", "TI_OFFLINE_CACHE"],
)
def test_invalid_cache_environment_fails_before_init(
    monkeypatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "sometimes")

    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match=name),
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig())

    taichi_init.assert_not_called()


def test_reinitialization_rejects_arch_seed_and_cache_conflicts(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    base = runtime.TaichiRuntimeConfig(
        arch="cuda",
        random_seed=7,
        offline_cache=True,
        offline_cache_file_path=str(cache_dir),
    )

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(base)
        with pytest.raises(ValueError, match="arch='cuda'"):
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(
                    arch="gpu",
                    random_seed=7,
                    offline_cache=True,
                    offline_cache_file_path=str(cache_dir),
                )
            )
        with pytest.raises(ValueError, match="random_seed=7"):
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(
                    random_seed=8,
                    offline_cache=True,
                    offline_cache_file_path=str(cache_dir),
                )
            )
        with pytest.raises(RuntimeError, match="offline-cache configuration"):
            runtime.init_taichi(
                runtime.TaichiRuntimeConfig(
                    random_seed=7,
                    offline_cache=False,
                )
            )

    assert taichi_init.call_count == 1


@pytest.mark.parametrize("value", [True, "7", 1.5])
def test_random_seed_requires_integer(value: object) -> None:
    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match="random_seed must be an integer"),
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig(random_seed=value))

    taichi_init.assert_not_called()


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_offline_cache_requires_boolean(value: object) -> None:
    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match="offline_cache must be a bool"),
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig(offline_cache=value))

    taichi_init.assert_not_called()


@pytest.mark.parametrize("value", ["", "   "])
def test_cache_path_rejects_blank_values(value: str) -> None:
    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match="offline_cache_file_path"),
    ):
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(offline_cache_file_path=value)
        )

    taichi_init.assert_not_called()


def test_concurrent_matching_initialization_calls_taichi_once(
    tmp_path: Path,
) -> None:
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

    config = runtime.TaichiRuntimeConfig(
        offline_cache_file_path=str(tmp_path / "cache")
    )

    def initialize() -> None:
        try:
            runtime.init_taichi(config)
        except BaseException as error:
            errors.append(error)

    with patch.object(runtime.ti, "init", side_effect=fake_init):
        first = threading.Thread(target=initialize)
        second = threading.Thread(target=initialize)
        first.start()
        assert entered.wait(timeout=5.0)
        second.start()
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert call_count == 1


def test_failed_initialization_does_not_publish_runtime_state() -> None:
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
    assert runtime._INITIALIZED_ARCH is None
    assert runtime._INITIALIZED_RANDOM_SEED is None
    assert runtime._INITIALIZED_OFFLINE_CACHE is None
    assert runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH is None
