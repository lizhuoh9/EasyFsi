from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from numbers import Integral

import taichi as ti


_SUPPORTED_ARCHS = ("cuda", "gpu")
_SUPPORTED_FPS = ("f32", "f64")


@dataclass(frozen=True)
class TaichiRuntimeConfig:
    arch: str = "cuda"
    default_fp: str = "f32"
    random_seed: int = 0
    offline_cache: bool | None = None
    offline_cache_file_path: str | None = None


_INITIALIZED = False
_INITIALIZED_ARCH: str | None = None
_INITIALIZED_FP: str | None = None
_INITIALIZED_RANDOM_SEED: int | None = None
_INITIALIZED_OFFLINE_CACHE: bool | None = None
_INITIALIZED_OFFLINE_CACHE_FILE_PATH: str | None = None
_INIT_LOCK = threading.RLock()


def _environment_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _normalized_cache_path(value: object, source: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{source} must be a filesystem path") from exc
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{source} must be a non-blank filesystem path")
    return os.path.normcase(
        os.path.abspath(os.path.expanduser(path))
    )


def _requested_offline_cache(
    config: TaichiRuntimeConfig,
) -> tuple[bool, str | None]:
    offline_cache = config.offline_cache
    if offline_cache is not None and not isinstance(offline_cache, bool):
        raise ValueError("offline_cache must be a bool or None")
    if offline_cache is None:
        offline_cache = _environment_flag(
            "SIMULATION_TAICHI_OFFLINE_CACHE"
        )
    if offline_cache is None:
        offline_cache = _environment_flag("TI_OFFLINE_CACHE")
    if offline_cache is None:
        # Match Taichi's frontend default and make it explicit to ti.init.
        offline_cache = True

    if config.offline_cache_file_path is not None:
        normalized_path = _normalized_cache_path(
            config.offline_cache_file_path,
            "offline_cache_file_path",
        )
    else:
        offline_cache_file_path = os.environ.get(
            "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH"
        )
        path_source = "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH"
        if offline_cache_file_path in (None, ""):
            offline_cache_file_path = os.environ.get(
                "TI_OFFLINE_CACHE_FILE_PATH"
            )
            path_source = "TI_OFFLINE_CACHE_FILE_PATH"
        normalized_path = None
        if offline_cache_file_path not in (None, ""):
            normalized_path = _normalized_cache_path(
                offline_cache_file_path,
                path_source,
            )
    return offline_cache, normalized_path


def init_taichi(config: TaichiRuntimeConfig | None = None) -> None:
    """Initialize Taichi once for the simulation core.

    The first call wins; later calls are no-ops only when they request the
    same architecture, floating-point mode, random seed, and offline-cache
    identity.
    Conflicting requests fail instead of producing misleading provenance.
    """

    cfg = config or TaichiRuntimeConfig()
    requested_arch = cfg.arch.lower()
    if requested_arch == "cpu":
        raise ValueError("simulation_core is GPU-only; use arch='cuda' or arch='gpu'")
    if requested_arch not in _SUPPORTED_ARCHS:
        raise ValueError(f"unsupported Taichi arch: {cfg.arch!r}")
    requested_fp = str(cfg.default_fp)
    if requested_fp not in _SUPPORTED_FPS:
        raise ValueError(
            f"unsupported Taichi default_fp: {cfg.default_fp!r}; expected one of {_SUPPORTED_FPS}"
        )
    if isinstance(cfg.random_seed, bool) or not isinstance(
        cfg.random_seed, Integral
    ):
        raise ValueError("random_seed must be an integer")
    requested_random_seed = int(cfg.random_seed)

    offline_cache, offline_cache_file_path = _requested_offline_cache(cfg)

    global _INITIALIZED, _INITIALIZED_ARCH, _INITIALIZED_FP
    global _INITIALIZED_RANDOM_SEED
    global _INITIALIZED_OFFLINE_CACHE
    global _INITIALIZED_OFFLINE_CACHE_FILE_PATH
    with _INIT_LOCK:
        if _INITIALIZED:
            if _INITIALIZED_ARCH is not None and requested_arch != _INITIALIZED_ARCH:
                raise ValueError(
                    "Taichi is already initialized with "
                    f"arch={_INITIALIZED_ARCH!r}; cannot re-initialize with "
                    f"arch={requested_arch!r}"
                )
            if _INITIALIZED_FP is not None and requested_fp != _INITIALIZED_FP:
                raise ValueError(
                    "Taichi is already initialized with "
                    f"default_fp={_INITIALIZED_FP!r}; cannot re-initialize with "
                    f"default_fp={requested_fp!r}"
                )
            if requested_random_seed != _INITIALIZED_RANDOM_SEED:
                raise ValueError(
                    "Taichi is already initialized with "
                    f"random_seed={_INITIALIZED_RANDOM_SEED!r}; cannot "
                    f"re-initialize with random_seed={requested_random_seed!r}"
                )
            initialized_cache_identity = (
                _INITIALIZED_OFFLINE_CACHE,
                _INITIALIZED_OFFLINE_CACHE_FILE_PATH,
            )
            requested_cache_identity = (
                offline_cache,
                offline_cache_file_path,
            )
            if requested_cache_identity != initialized_cache_identity:
                raise RuntimeError(
                    "Taichi is already initialized with offline-cache "
                    f"configuration={initialized_cache_identity!r}; cannot "
                    f"reconfigure it as {requested_cache_identity!r}"
                )
            return

        arch = ti.cuda if requested_arch == "cuda" else ti.gpu
        default_fp = ti.f32 if requested_fp == "f32" else ti.f64
        taichi_kwargs: dict[str, object] = {}
        if offline_cache is not None:
            taichi_kwargs["offline_cache"] = bool(offline_cache)
        if offline_cache_file_path:
            taichi_kwargs["offline_cache_file_path"] = (
                offline_cache_file_path
            )
        ti.init(
            arch=arch,
            default_fp=default_fp,
            random_seed=requested_random_seed,
            **taichi_kwargs,
        )
        _INITIALIZED = True
        _INITIALIZED_ARCH = requested_arch
        _INITIALIZED_FP = requested_fp
        _INITIALIZED_RANDOM_SEED = requested_random_seed
        _INITIALIZED_OFFLINE_CACHE = offline_cache
        _INITIALIZED_OFFLINE_CACHE_FILE_PATH = offline_cache_file_path
