from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


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
    marker_geometry_sha256: str = ""
    marker_geometry_revision: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.provider_mode, name="provider_mode")
        if int(self.fallback_count) < 0:
            raise ValueError("fallback_count must be non-negative")
        if int(self.selected_count) < 0:
            raise ValueError("selected_count must be non-negative")
        if (
            self.marker_geometry_revision is not None
            and int(self.marker_geometry_revision) < 0
        ):
            raise ValueError("marker_geometry_revision must be non-negative")
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
            "marker_geometry_sha256": self.marker_geometry_sha256,
            "marker_geometry_revision": self.marker_geometry_revision,
            "fallback_count": int(self.fallback_count),
            "selected_count": int(self.selected_count),
            "pairs": [pair.as_dict() for pair in self.pairs],
        }

    def require_current_marker_geometry(self, markers: Any) -> None:
        """Reject a pair map once its marker geometry identity is stale."""

        current_revision = marker_geometry_revision(markers)
        if self.marker_geometry_revision is not None:
            if current_revision is None:
                raise ValueError(
                    "current marker geometry revision is unavailable for anchored pairs"
                )
            if int(current_revision) != int(self.marker_geometry_revision):
                raise ValueError(
                    "pressure sample pair marker geometry revision mismatch: "
                    f"map={self.marker_geometry_revision}, current={current_revision}"
                )
        positions, normals, region_ids = marker_geometry_rows(markers)
        current_sha256 = pressure_sample_marker_geometry_sha256(
            marker_positions_m=positions,
            marker_normals=normals,
            marker_region_ids=region_ids,
        )
        if self.marker_geometry_sha256 != current_sha256:
            raise ValueError(
                "pressure sample pair marker geometry hash mismatch: "
                f"map={self.marker_geometry_sha256}, current={current_sha256}"
            )


class PressureSamplePairProviderProtocol(Protocol):
    def compute_pairs(
        self,
        markers: Any,
        fluid_state: Any,
        interface_surface: Any,
    ) -> PressureSamplePairMap:
        ...


@dataclass(frozen=True)
class RuntimeAnchoredCellPairProvider:
    domain_bounds_m: tuple[Sequence[float], Sequence[float]]
    grid_nodes: Sequence[int]
    anchor_axis: int
    inside_axis_position_m: float
    outside_axis_offset_cells: int = 1
    normal_aware_rays: bool = False

    def compute_pairs(
        self,
        markers: Any,
        fluid_state: Any = None,
        interface_surface: Any = None,
    ) -> PressureSamplePairMap:
        del interface_surface
        positions, normals, region_ids = marker_geometry_rows(markers)
        return compute_runtime_anchored_cell_pair_map(
            marker_positions_m=positions,
            marker_normals=normals,
            marker_region_ids=region_ids,
            domain_bounds_m=self.domain_bounds_m,
            grid_nodes=self.grid_nodes,
            anchor_axis=self.anchor_axis,
            inside_axis_position_m=self.inside_axis_position_m,
            outside_axis_offset_cells=self.outside_axis_offset_cells,
            normal_aware_rays=self.normal_aware_rays,
            obstacle_cells=_fluid_obstacle_cells(fluid_state),
            marker_geometry_revision=marker_geometry_revision(markers),
        )


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
    marker_geometry_sha256: str = "",
    marker_geometry_revision: int | None = None,
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
        marker_geometry_sha256=marker_geometry_sha256,
        marker_geometry_revision=marker_geometry_revision,
    )


def pressure_sample_marker_geometry_sha256(
    *,
    marker_positions_m: Sequence[Sequence[float]],
    marker_normals: Sequence[Sequence[float]],
    marker_region_ids: Sequence[int | str],
) -> str:
    payload = {
        "marker_positions_m": [
            [float(component) for component in point]
            for point in marker_positions_m
        ],
        "marker_normals": [
            [float(component) for component in normal]
            for normal in marker_normals
        ],
        "marker_region_ids": [str(region_id) for region_id in marker_region_ids],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def marker_geometry_rows(
    markers: Any,
) -> tuple[tuple[Point3, ...], tuple[Point3, ...], tuple[str, ...]]:
    if isinstance(markers, Mapping):
        positions = _rows_from_value(
            _first_present(markers, "marker_positions_m", "positions_m"),
            row_name="marker_positions_m",
        )
        normals = _rows_from_value(
            _first_present(markers, "marker_normals", "normals"),
            row_name="marker_normals",
        )
        region_ids = _region_ids_from_value(
            _first_present(markers, "marker_region_ids", "region_ids"),
        )
        return positions, normals, region_ids

    marker_count = int(getattr(markers, "marker_count"))
    positions = _rows_from_value(
        getattr(markers, "x_gamma_m"),
        row_name="marker_positions_m",
        count=marker_count,
    )
    normals = _rows_from_value(
        getattr(markers, "n_gamma"),
        row_name="marker_normals",
        count=marker_count,
    )
    region_ids = _region_ids_from_value(
        getattr(markers, "region_id"),
        count=marker_count,
    )
    return positions, normals, region_ids


def marker_geometry_revision(markers: Any) -> int | None:
    if isinstance(markers, Mapping):
        value = markers.get("marker_geometry_revision")
    else:
        value = getattr(markers, "marker_geometry_revision", None)
    if value is None:
        return None
    revision = int(value)
    if revision < 0:
        raise ValueError("marker_geometry_revision must be non-negative")
    return revision


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
    normal_aware_rays: bool = False,
    obstacle_cells: Any = None,
    marker_geometry_revision: int | None = None,
) -> PressureSamplePairMap:
    marker_count = len(marker_positions_m)
    if marker_count == 0:
        raise ValueError("at least one marker is required")
    if not (len(marker_normals) == len(marker_region_ids) == marker_count):
        raise ValueError("marker positions, normals, and region IDs must match")
    axis = int(anchor_axis)
    if axis not in (0, 1, 2):
        raise ValueError("anchor_axis must be 0, 1, or 2")
    if type(normal_aware_rays) is not bool:
        raise TypeError("normal_aware_rays must be a bool")
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
    obstacle = _validated_obstacle_cells(obstacle_cells, grid=grid)

    pairs: list[PressureSamplePair] = []
    for marker_index, (position_value, normal_value, region_id) in enumerate(
        zip(marker_positions_m, marker_normals, marker_region_ids)
    ):
        position = _point3(position_value, name="marker_positions_m")
        normal = _point3(normal_value, name="marker_normals")
        normal_axis = _dominant_normal_axis(normal) if normal_aware_rays else axis
        normal_axis_value = normal[normal_axis]
        if not math.isfinite(normal_axis_value):
            raise ValueError("marker normal must contain finite values")
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
        diagnostic_reason = "runtime_anchored_cell_pair"
        if obstacle is None:
            inside_cell = list(base_cell)
            outside_cell = list(base_cell)
            if normal_axis == axis:
                # Preserve the original anchored behavior for the declared
                # axis, including the established z-normal runtime contract.
                inside_cell[axis] = inside_axis_cell
                outside_cell[axis] = _clamp_cell(
                    base_cell[axis] + direction * offset,
                    grid[axis],
                )
            else:
                inside_cell[normal_axis] = _clamp_cell(
                    base_cell[normal_axis] - direction * offset,
                    grid[normal_axis],
                )
                outside_cell[normal_axis] = _clamp_cell(
                    base_cell[normal_axis] + direction * offset,
                    grid[normal_axis],
                )
                diagnostic_reason = "runtime_normal_ray_cell_pair"
        else:
            inside_cell = list(
                _first_fluid_cell_on_axis_side(
                    base_cell,
                    axis=normal_axis,
                    side_direction=-direction,
                    start_offset=offset,
                    grid=grid,
                    obstacle=obstacle,
                    marker_index=marker_index,
                    side_name="inside",
                )
            )
            outside_cell = list(
                _first_fluid_cell_on_axis_side(
                    base_cell,
                    axis=normal_axis,
                    side_direction=direction,
                    start_offset=offset,
                    grid=grid,
                    obstacle=obstacle,
                    marker_index=marker_index,
                    side_name="outside",
                )
            )
            diagnostic_reason = "runtime_dynamic_fluid_side_cell_pair"
        if tuple(inside_cell) == tuple(outside_cell):
            raise ValueError("inside_cell and outside_cell must differ")
        pairs.append(
            PressureSamplePair(
                marker_index=marker_index,
                region_id=str(region_id),
                inside_cell=tuple(inside_cell),  # type: ignore[arg-type]
                outside_cell=tuple(outside_cell),  # type: ignore[arg-type]
                sample_status="runtime_generated",
                fallback_status="no_fallback",
                diagnostic_reason=diagnostic_reason,
            )
        )
    return pressure_sample_pair_map_from_pairs(
        pairs,
        provider_mode="runtime_anchored_cell_pair",
        marker_geometry_sha256=pressure_sample_marker_geometry_sha256(
            marker_positions_m=marker_positions_m,
            marker_normals=marker_normals,
            marker_region_ids=marker_region_ids,
        ),
        marker_geometry_revision=marker_geometry_revision,
    )


def _dominant_normal_axis(normal: Point3) -> int:
    magnitudes = tuple(abs(float(component)) for component in normal)
    if not all(math.isfinite(component) for component in magnitudes):
        raise ValueError("marker normal must contain finite values")
    largest = max(magnitudes)
    if largest <= 1.0e-12:
        raise ValueError("marker normal must be non-degenerate")
    relative_ambiguity_tolerance = 1.0e-6
    ambiguity_tolerance = max(
        1.0e-12,
        largest * relative_ambiguity_tolerance,
    )
    dominant_axes = tuple(
        axis
        for axis, magnitude in enumerate(magnitudes)
        if largest - magnitude <= ambiguity_tolerance
    )
    if len(dominant_axes) != 1:
        raise ValueError("marker normal dominant axis is ambiguous")
    return dominant_axes[0]


def _fluid_obstacle_cells(fluid_state: Any) -> Any:
    if fluid_state is None:
        return None
    if isinstance(fluid_state, Mapping):
        for name in ("obstacle", "obstacle_field", "fluid_obstacle"):
            if name in fluid_state:
                return fluid_state[name]
    else:
        for name in ("obstacle", "obstacle_field", "fluid_obstacle"):
            if hasattr(fluid_state, name):
                return getattr(fluid_state, name)
    raise ValueError("fluid_state must expose an obstacle field")


def _validated_obstacle_cells(
    obstacle_cells: Any,
    *,
    grid: tuple[int, int, int],
) -> np.ndarray | None:
    if obstacle_cells is None:
        return None
    values = (
        obstacle_cells.to_numpy()
        if hasattr(obstacle_cells, "to_numpy")
        else obstacle_cells
    )
    obstacle = np.asarray(values)
    if tuple(int(value) for value in obstacle.shape) != grid:
        raise ValueError(
            "obstacle field shape must match grid_nodes: "
            f"shape={tuple(obstacle.shape)}, grid_nodes={grid}"
        )
    if np.issubdtype(obstacle.dtype, np.floating) and not np.all(
        np.isfinite(obstacle)
    ):
        raise ValueError("obstacle field must contain finite values")
    return obstacle != 0


def _first_fluid_cell_on_axis_side(
    base_cell: CellIndex,
    *,
    axis: int,
    side_direction: int,
    start_offset: int,
    grid: tuple[int, int, int],
    obstacle: np.ndarray,
    marker_index: int,
    side_name: str,
) -> CellIndex:
    for distance in range(int(start_offset), int(grid[axis]) + 1):
        axis_cell = int(base_cell[axis]) + int(side_direction) * distance
        if axis_cell < 0 or axis_cell >= int(grid[axis]):
            break
        candidate = list(base_cell)
        candidate[axis] = axis_cell
        cell = (int(candidate[0]), int(candidate[1]), int(candidate[2]))
        if not bool(obstacle[cell]):
            return cell
    raise ValueError(
        "no non-obstacle fluid cell on declared "
        f"{side_name} side for marker {marker_index}"
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


def _first_present(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    joined = ", ".join(names)
    raise ValueError(f"marker geometry must contain one of: {joined}")


def _rows_from_value(
    value: Any,
    *,
    row_name: str,
    count: int | None = None,
) -> tuple[Point3, ...]:
    rows = value.to_numpy() if hasattr(value, "to_numpy") else value
    limited_rows = rows if count is None else rows[:count]
    return tuple(_point3(row, name=row_name) for row in limited_rows)


def _region_ids_from_value(
    value: Any,
    *,
    count: int | None = None,
) -> tuple[str, ...]:
    region_ids = value.to_numpy() if hasattr(value, "to_numpy") else value
    limited_region_ids = region_ids if count is None else region_ids[:count]
    return tuple(str(region_id) for region_id in limited_region_ids)
