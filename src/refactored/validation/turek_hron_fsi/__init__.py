"""Offline validation and rendering helpers for the Turek-Hron FSI cases."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FlowSnapshotContractError",
    "FlowSnapshotFrame",
    "TurekHronRenderResult",
    "discover_flow_snapshot_paths",
    "load_flow_snapshot",
    "render_turek_hron_flow_gif",
]


def __getattr__(name: str) -> Any:
    """Load public rendering symbols lazily so ``python -m ...rendering`` is clean."""

    if name not in __all__:
        raise AttributeError(name)
    module = import_module(f"{__name__}.rendering")
    return getattr(module, name)
