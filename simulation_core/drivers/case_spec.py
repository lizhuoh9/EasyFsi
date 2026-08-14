from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FsiCaseSpec:
    """Reference-case metadata; it contains no solver implementation."""

    case_id: str
    source_url: str
    coordinate_model: str
    geometry: Mapping[str, Any]
    fluid: Mapping[str, Any]
    solid: Mapping[str, Any]
    boundary_conditions: Mapping[str, Any]
    reference_results: Mapping[str, float]
    acceptance_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.source_url:
            raise ValueError("source_url must be non-empty")
        if not 0.0 < float(self.acceptance_tolerance) < 1.0:
            raise ValueError("acceptance_tolerance must be in (0, 1)")
