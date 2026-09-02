"""Offline validation and rendering helpers for the Turek-Hron FSI cases."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_MODULES = {
    "FlowSnapshotContractError": "rendering",
    "FlowSnapshotFrame": "rendering",
    "TurekHronRenderResult": "rendering",
    "discover_flow_snapshot_paths": "rendering",
    "load_flow_snapshot": "rendering",
    "render_turek_hron_flow_gif": "rendering",
    "FeatflowManifest": "featflow",
    "FeatflowSeries": "featflow",
    "FeatflowValidationError": "featflow",
    "load_featflow_series": "featflow",
    "CompleteCycle": "limit_cycle",
    "LimitCycleReport": "limit_cycle",
    "LimitCycleStability": "limit_cycle",
    "LimitCycleValidationError": "limit_cycle",
    "SignalExtrema": "limit_cycle",
    "analyze_featflow_limit_cycle": "limit_cycle",
    "analyze_limit_cycle": "limit_cycle",
    "CaseReferenceContract": "references",
    "RawSeriesIdentity": "references",
    "ReferenceCatalog": "references",
    "ReferenceContractError": "references",
    "ReferenceMetric": "references",
    "ReferenceSource": "references",
    "REFERENCE_CATALOG": "references",
    "case_reference_contract": "references",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name: str) -> Any:
    """Load public rendering symbols lazily so ``python -m ...rendering`` is clean."""

    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
