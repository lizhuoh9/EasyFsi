from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .coordinate_models import CoordinateModel

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_VALID_SIDES = {"min", "max"}


@dataclass(frozen=True)
class BoundaryRegion:
    name: str
    kind: str
    selector: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("boundary region name must be non-empty")
        if not self.kind:
            raise ValueError("boundary region kind must be non-empty")
        if not self.selector:
            raise ValueError("boundary region selector must be non-empty")


@dataclass(frozen=True)
class AxisAlignedBoundary:
    name: str
    kind: str
    axis: str
    side: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("axis-aligned boundary name must be non-empty")
        if not self.kind:
            raise ValueError("axis-aligned boundary kind must be non-empty")
        axis = str(self.axis).lower()
        side = str(self.side).lower()
        if axis not in _AXIS_TO_INDEX:
            raise ValueError("axis must be one of x, y, z")
        if side not in _VALID_SIDES:
            raise ValueError("side must be 'min' or 'max'")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "side", side)

    @classmethod
    def pressure_outlet(
        cls,
        *,
        axis: str,
        side: str,
        name: str = "pressure_outlet",
    ) -> "AxisAlignedBoundary":
        return cls(name=name, kind="pressure-outlet", axis=axis, side=side)

    @classmethod
    def from_selector(
        cls,
        *,
        name: str,
        kind: str,
        selector: str,
    ) -> "AxisAlignedBoundary":
        normalized = str(selector).lower().replace("-", "_")
        if len(normalized) != 5 or normalized[1] != "_":
            raise ValueError("selector must use the form x_min, x_max, ..., z_max")
        axis, side = normalized.split("_", 1)
        return cls(name=name, kind=kind, axis=axis, side=side)

    @property
    def axis_index(self) -> int:
        return _AXIS_TO_INDEX[self.axis]

    @property
    def side_index(self) -> int:
        return 0 if self.side == "min" else 1

    @property
    def selector(self) -> str:
        return f"{self.axis}_{self.side}"

    @property
    def legacy_zmin_outlet(self) -> bool:
        return self.kind == "pressure-outlet" and self.axis == "z" and self.side == "min"

    def as_boundary_region(self) -> BoundaryRegion:
        return BoundaryRegion(name=self.name, kind=self.kind, selector=self.selector)


@dataclass(frozen=True)
class FluidDomain:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    grid_nodes: tuple[int, int, int]
    coordinate_model: CoordinateModel
    boundary_regions: tuple[BoundaryRegion, ...] = ()

    def __post_init__(self) -> None:
        min_bounds = _vector3(self.bounds_min_m, name="bounds_min_m")
        max_bounds = _vector3(self.bounds_max_m, name="bounds_max_m")
        nodes = _positive_int3(self.grid_nodes, name="grid_nodes")
        for lower, upper in zip(min_bounds, max_bounds):
            if lower >= upper:
                raise ValueError("bounds_max_m must be greater than bounds_min_m")
        object.__setattr__(self, "bounds_min_m", min_bounds)
        object.__setattr__(self, "bounds_max_m", max_bounds)
        object.__setattr__(self, "grid_nodes", nodes)
        object.__setattr__(self, "boundary_regions", tuple(self.boundary_regions))

    @property
    def dimension(self) -> int:
        return int(self.coordinate_model.dimension)

    def boundary_by_name(self, name: str) -> BoundaryRegion:
        for region in self.boundary_regions:
            if region.name == name:
                return region
        raise KeyError(name)


def _vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    return (vector[0], vector[1], vector[2])


def _positive_int3(values: Sequence[int], *, name: str) -> tuple[int, int, int]:
    vector = tuple(int(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if any(value <= 0 for value in vector):
        raise ValueError(f"{name} components must be positive")
    return (vector[0], vector[1], vector[2])
