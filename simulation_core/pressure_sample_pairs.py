from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


CellIndex = tuple[int, int, int]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class PressureSamplePair:
    marker_index: int
    region_id: str
    inside_cell: CellIndex
    outside_cell: CellIndex
    sample_status: str
    fallback_status: str
    diagnostic_reason: str

    def __post_init__(self) -> None:
        if int(self.marker_index) < 0:
            raise ValueError("marker_index must be non-negative")
        _require_non_empty(self.region_id, name="region_id")
        _cell_index(self.inside_cell, name="inside_cell")
        _cell_index(self.outside_cell, name="outside_cell")
        _require_non_empty(self.sample_status, name="sample_status")
        _require_non_empty(self.fallback_status, name="fallback_status")
        _require_non_empty(self.diagnostic_reason, name="diagnostic_reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "marker_index": int(self.marker_index),
            "region_id": self.region_id,
            "inside_cell": list(self.inside_cell),
            "outside_cell": list(self.outside_cell),
            "sample_status": self.sample_status,
            "fallback_status": self.fallback_status,
            "diagnostic_reason": self.diagnostic_reason,
        }


@dataclass(frozen=True)
class PressureSamplePairMap:
    pairs: tuple[PressureSamplePair, ...]
    pair_map_sha256: str
    provider_mode: str
    fallback_count: int
    selected_count: int

    def __post_init__(self) -> None:
        _require_non_empty(self.provider_mode, name="provider_mode")
        if int(self.fallback_count) < 0:
            raise ValueError("fallback_count must be non-negative")
        if int(self.selected_count) < 0:
            raise ValueError("selected_count must be non-negative")
        expected_sha = pressure_sample_pair_map_sha256(self.pairs)
        if self.pair_map_sha256 != expected_sha:
            raise ValueError("pair_map_sha256 does not match pairs")

    @property
    def inside_cells(self) -> tuple[CellIndex, ...]:
        return tuple(pair.inside_cell for pair in self.pairs)

    @property
    def outside_cells(self) -> tuple[CellIndex, ...]:
        return tuple(pair.outside_cell for pair in self.pairs)

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "provider_mode": self.provider_mode,
            "pair_map_sha256": self.pair_map_sha256,
            "fallback_count": int(self.fallback_count),
            "selected_count": int(self.selected_count),
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


class PressureSamplePairProviderProtocol(Protocol):
    def compute_pairs(
        self,
        markers: Any,
        fluid_state: Any,
        interface_surface: Any,
    ) -> PressureSamplePairMap:
        ...


def pressure_sample_pair_map_sha256(
    pairs: Sequence[PressureSamplePair],
) -> str:
    payload = [pair.as_dict() for pair in pairs]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def pressure_sample_pair_map_from_pairs(
    pairs: Sequence[PressureSamplePair],
    *,
    provider_mode: str,
) -> PressureSamplePairMap:
    pair_tuple = tuple(pairs)
    fallback_count = sum(
        1 for pair in pair_tuple if pair.fallback_status != "no_fallback"
    )
    selected_count = sum(
        1 for pair in pair_tuple if pair.sample_status == "runtime_generated"
    )
    return PressureSamplePairMap(
        pairs=pair_tuple,
        pair_map_sha256=pressure_sample_pair_map_sha256(pair_tuple),
        provider_mode=provider_mode,
        fallback_count=fallback_count,
        selected_count=selected_count,
    )


def compute_runtime_anchored_cell_pair_map(
    *,
    marker_positions_m: Sequence[Sequence[float]],
    marker_normals: Sequence[Sequence[float]],
    marker_region_ids: Sequence[int | str],
    domain_bounds_m: tuple[Sequence[float], Sequence[float]],
    grid_nodes: Sequence[int],
    anchor_axis: int,
    inside_axis_position_m: float,
    outside_axis_offset_cells: int = 1,
) -> PressureSamplePairMap:
    marker_count = len(marker_positions_m)
    if marker_count == 0:
        raise ValueError("at least one marker is required")
    if not (len(marker_normals) == len(marker_region_ids) == marker_count):
        raise ValueError("marker positions, normals, and region IDs must match")
    axis = int(anchor_axis)
    if axis not in (0, 1, 2):
        raise ValueError("anchor_axis must be 0, 1, or 2")
    grid = _grid_nodes(grid_nodes)
    bounds_min = _point3(domain_bounds_m[0], name="domain_bounds_m min")
    bounds_max = _point3(domain_bounds_m[1], name="domain_bounds_m max")
    spacing = tuple(
        (bounds_max[index] - bounds_min[index]) / float(grid[index])
        for index in range(3)
    )
    if any(value <= 0.0 or not math.isfinite(value) for value in spacing):
        raise ValueError("domain bounds must define positive finite cell spacing")
    inside_axis_cell = _coordinate_to_cell(
        float(inside_axis_position_m),
        bounds_min[axis],
        spacing[axis],
        grid[axis],
    )
    offset = int(outside_axis_offset_cells)
    if offset <= 0:
        raise ValueError("outside_axis_offset_cells must be positive")

    pairs: list[PressureSamplePair] = []
    for marker_index, (position_value, normal_value, region_id) in enumerate(
        zip(marker_positions_m, marker_normals, marker_region_ids)
    ):
        position = _point3(position_value, name="marker_positions_m")
        normal = _point3(normal_value, name="marker_normals")
        normal_axis_value = normal[axis]
        if abs(normal_axis_value) <= 1.0e-12:
            raise ValueError("marker normal must have nonzero anchor-axis component")
        base_cell = tuple(
            _coordinate_to_cell(
                position[index],
                bounds_min[index],
                spacing[index],
                grid[index],
            )
            for index in range(3)
        )
        direction = 1 if normal_axis_value > 0.0 else -1
        inside_cell = list(base_cell)
        outside_cell = list(base_cell)
        inside_cell[axis] = inside_axis_cell
        outside_cell[axis] = _clamp_cell(
            base_cell[axis] + direction * offset,
            grid[axis],
        )
        pairs.append(
            PressureSamplePair(
                marker_index=marker_index,
                region_id=str(region_id),
                inside_cell=tuple(inside_cell),  # type: ignore[arg-type]
                outside_cell=tuple(outside_cell),  # type: ignore[arg-type]
                sample_status="runtime_generated",
                fallback_status="no_fallback",
                diagnostic_reason="runtime_anchored_cell_pair",
            )
        )
    return pressure_sample_pair_map_from_pairs(
        pairs,
        provider_mode="runtime_anchored_cell_pair",
    )


def _coordinate_to_cell(
    coordinate: float,
    lower_bound: float,
    spacing: float,
    cell_count: int,
) -> int:
    if not math.isfinite(float(coordinate)):
        raise ValueError("marker coordinate must be finite")
    raw = math.floor((float(coordinate) - float(lower_bound)) / float(spacing))
    return _clamp_cell(raw, cell_count)


def _clamp_cell(value: int, cell_count: int) -> int:
    return max(0, min(int(cell_count) - 1, int(value)))


def _cell_index(value: Sequence[int], *, name: str) -> CellIndex:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three indices")
    cell = tuple(int(component) for component in value)
    if any(component < 0 for component in cell):
        raise ValueError(f"{name} must contain non-negative indices")
    return (cell[0], cell[1], cell[2])


def _point3(value: Sequence[float], *, name: str) -> Point3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    point = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in point):
        raise ValueError(f"{name} must contain finite values")
    return (point[0], point[1], point[2])


def _grid_nodes(value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError("grid_nodes must contain exactly three values")
    grid = tuple(int(component) for component in value)
    if any(component <= 0 for component in grid):
        raise ValueError("grid_nodes must contain positive values")
    return (grid[0], grid[1], grid[2])


def _require_non_empty(value: str, *, name: str) -> None:
    if not str(value):
        raise ValueError(f"{name} must be non-empty")
