from __future__ import annotations

from dataclasses import dataclass

import taichi as ti


_SUPPORTED_ARCHS = ("cuda", "gpu")
_SUPPORTED_FPS = ("f32", "f64")


@dataclass(frozen=True)
class TaichiRuntimeConfig:
    arch: str = "cuda"
    default_fp: str = "f32"
    random_seed: int = 0


_INITIALIZED = False
_INITIALIZED_ARCH: str | None = None
_INITIALIZED_FP: str | None = None


def init_taichi(config: TaichiRuntimeConfig | None = None) -> None:
    """Initialize Taichi once for the simulation core.

    The first call wins; later calls are no-ops when they request the same
    floating-point mode. Requesting a different ``default_fp`` after
    initialization raises instead of being silently ignored.
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

    global _INITIALIZED, _INITIALIZED_ARCH, _INITIALIZED_FP
    if _INITIALIZED:
        if _INITIALIZED_FP is not None and requested_fp != _INITIALIZED_FP:
            raise ValueError(
                "Taichi is already initialized with "
                f"default_fp={_INITIALIZED_FP!r}; cannot re-initialize with "
                f"default_fp={requested_fp!r}"
            )
        return

    arch = ti.cuda if requested_arch == "cuda" else ti.gpu
    default_fp = ti.f32 if requested_fp == "f32" else ti.f64
    ti.init(arch=arch, default_fp=default_fp, random_seed=cfg.random_seed)
    _INITIALIZED = True
    _INITIALIZED_ARCH = requested_arch
    _INITIALIZED_FP = requested_fp
