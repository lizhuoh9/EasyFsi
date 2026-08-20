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
def test_init_taichi_forwards_explicit_offline_cache_configuration(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "taichi-cache"

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(
                arch="cuda",
                offline_cache=True,
                offline_cache_file_path=str(cache_dir),
            )
        )

    taichi_init.assert_called_once_with(
        arch=runtime.ti.cuda,
        default_fp=runtime.ti.f32,
        random_seed=0,
        offline_cache=True,
        offline_cache_file_path=os.path.normcase(str(cache_dir)),
    )


def test_init_taichi_rejects_conflicting_cache_reconfiguration(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "taichi-cache"

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


def test_init_taichi_uses_simulation_cache_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "environment-cache"
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "true")
    monkeypatch.setenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        str(cache_dir),
    )

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(runtime.TaichiRuntimeConfig())

    assert taichi_init.call_args.kwargs["offline_cache"] is True
    assert (
        taichi_init.call_args.kwargs["offline_cache_file_path"]
        == os.path.normcase(str(cache_dir))
    )


@pytest.mark.parametrize(
    "name",
    ["SIMULATION_TAICHI_OFFLINE_CACHE", "TI_OFFLINE_CACHE"],
)
def test_init_taichi_rejects_invalid_cache_environment(
    monkeypatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "sometimes")
    with pytest.raises(ValueError, match=name):
        runtime._environment_flag(name)


def test_concurrent_matching_initialization_calls_taichi_once(
    tmp_path: Path,
) -> None:
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


def test_failed_taichi_initialization_does_not_publish_runtime_state() -> None:
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        (" on ", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        (" off ", False),
    ],
)
def test_environment_flag_accepts_supported_spellings(
    monkeypatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("TI_OFFLINE_CACHE", value)

    assert runtime._environment_flag("TI_OFFLINE_CACHE") is expected


def test_init_taichi_forwards_effective_taichi_cache_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIMULATION_TAICHI_OFFLINE_CACHE", raising=False)
    monkeypatch.delenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        raising=False,
    )
    monkeypatch.setenv("TI_OFFLINE_CACHE", "yes")
    monkeypatch.setenv("TI_OFFLINE_CACHE_FILE_PATH", "ambient-cache")

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(runtime.TaichiRuntimeConfig())

    taichi_init.assert_called_once_with(
        arch=runtime.ti.cuda,
        default_fp=runtime.ti.f32,
        random_seed=0,
        offline_cache=True,
        offline_cache_file_path=os.path.normcase(
            str(tmp_path / "ambient-cache")
        ),
    )


def test_explicit_cache_configuration_overrides_cache_environments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "true")
    monkeypatch.setenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        str(tmp_path / "simulation-cache"),
    )
    monkeypatch.setenv("TI_OFFLINE_CACHE", "true")
    monkeypatch.setenv(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(tmp_path / "taichi-cache"),
    )
    explicit_path = tmp_path / "explicit-cache"

    assert runtime._requested_offline_cache(
        runtime.TaichiRuntimeConfig(
            offline_cache=False,
            offline_cache_file_path=str(explicit_path),
        )
    ) == (False, os.path.normcase(str(explicit_path)))


def test_simulation_cache_environment_overrides_taichi_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    simulation_path = tmp_path / "simulation-cache"
    monkeypatch.setenv("SIMULATION_TAICHI_OFFLINE_CACHE", "false")
    monkeypatch.setenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        str(simulation_path),
    )
    monkeypatch.setenv("TI_OFFLINE_CACHE", "true")
    monkeypatch.setenv(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(tmp_path / "taichi-cache"),
    )

    assert runtime._requested_offline_cache(
        runtime.TaichiRuntimeConfig()
    ) == (False, os.path.normcase(str(simulation_path)))


def test_later_taichi_cache_environment_change_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_path = tmp_path / "first-cache"
    second_path = tmp_path / "second-cache"
    monkeypatch.delenv("SIMULATION_TAICHI_OFFLINE_CACHE", raising=False)
    monkeypatch.delenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        raising=False,
    )
    monkeypatch.setenv("TI_OFFLINE_CACHE", "1")
    monkeypatch.setenv("TI_OFFLINE_CACHE_FILE_PATH", str(first_path))

    with patch.object(runtime.ti, "init"):
        runtime.init_taichi(runtime.TaichiRuntimeConfig())
        monkeypatch.setenv(
            "TI_OFFLINE_CACHE_FILE_PATH",
            str(second_path),
        )
        with pytest.raises(
            RuntimeError,
            match="offline-cache configuration",
        ):
            runtime.init_taichi(runtime.TaichiRuntimeConfig())


@pytest.mark.parametrize(
    "name",
    ["TI_OFFLINE_CACHE", "TI_OFFLINE_CACHE_FILE_PATH"],
)
def test_whitespace_taichi_cache_environment_is_rejected_before_init(
    monkeypatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "   ")

    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match=name),
    ):
        runtime.init_taichi(runtime.TaichiRuntimeConfig())

    taichi_init.assert_not_called()


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_explicit_offline_cache_requires_a_boolean(value) -> None:
    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match="offline_cache must be a bool"),
    ):
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(offline_cache=value)
        )

    taichi_init.assert_not_called()


@pytest.mark.parametrize("value", ["", "   "])
def test_explicit_cache_path_rejects_blank_values(value: str) -> None:
    with (
        patch.object(runtime.ti, "init") as taichi_init,
        pytest.raises(ValueError, match="offline_cache_file_path"),
    ):
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(
                offline_cache_file_path=value,
            )
        )

    taichi_init.assert_not_called()


def test_default_cache_flag_matches_explicit_true(monkeypatch) -> None:
    for name in (
        "SIMULATION_TAICHI_OFFLINE_CACHE",
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        "TI_OFFLINE_CACHE",
        "TI_OFFLINE_CACHE_FILE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(runtime.TaichiRuntimeConfig())
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(offline_cache=True)
        )

    assert runtime._INITIALIZED_OFFLINE_CACHE is True
    assert taichi_init.call_count == 1
    assert taichi_init.call_args.kwargs["offline_cache"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity")
def test_cache_path_identity_is_case_insensitive(tmp_path: Path) -> None:
    upper_path = str(tmp_path / "Cache-Dir").upper()
    lower_path = upper_path.lower()

    with patch.object(runtime.ti, "init") as taichi_init:
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(
                offline_cache_file_path=upper_path,
            )
        )
        runtime.init_taichi(
            runtime.TaichiRuntimeConfig(
                offline_cache_file_path=lower_path,
            )
        )

    assert runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH == lower_path
    assert taichi_init.call_count == 1
