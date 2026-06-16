from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cartesian2DCoordinateModel:
    name: str = "cartesian-2d"
    dimension: int = 2


@dataclass(frozen=True)
class Cartesian3DCoordinateModel:
    name: str = "cartesian-3d"
    dimension: int = 3


@dataclass(frozen=True)
class Axisymmetric2DCoordinateModel:
    radial_axis: str = "x"
    axial_axis: str = "z"
    name: str = "axisymmetric-2d"
    dimension: int = 2

    def __post_init__(self) -> None:
        if self.radial_axis == self.axial_axis:
            raise ValueError("radial_axis and axial_axis must be different")
        valid_axes = {"x", "y", "z"}
        if self.radial_axis not in valid_axes:
            raise ValueError("radial_axis must be one of x, y, z")
        if self.axial_axis not in valid_axes:
            raise ValueError("axial_axis must be one of x, y, z")


CoordinateModel = (
    Cartesian2DCoordinateModel | Cartesian3DCoordinateModel | Axisymmetric2DCoordinateModel
)
