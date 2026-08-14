from __future__ import annotations

from unittest.mock import patch

import pytest

from simulation_core.diagnostics import runtime


def test_repeated_taichi_initialization_rejects_a_different_random_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (
        ("_INITIALIZED", False),
        ("_INITIALIZED_ARCH", None),
        ("_INITIALIZED_FP", None),
        ("_INITIALIZED_RANDOM_SEED", None),
        ("_INITIALIZED_OFFLINE_CACHE", None),
        ("_INITIALIZED_OFFLINE_CACHE_FILE_PATH", None),
    ):
        monkeypatch.setattr(runtime, name, value, raising=False)

    with patch.object(runtime.ti, "init"):
        runtime.init_taichi(runtime.TaichiRuntimeConfig(random_seed=7))
        with pytest.raises(ValueError, match="random_seed"):
            runtime.init_taichi(runtime.TaichiRuntimeConfig(random_seed=8))
