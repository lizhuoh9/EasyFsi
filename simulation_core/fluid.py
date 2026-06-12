from dataclasses import dataclass
import math
import os
from math import sqrt

import taichi as ti

from .runtime import TaichiRuntimeConfig, init_taichi


CG_PRECONDITIONER_CHOICES = ("auto", "jacobi", "fv_multigrid", "fv_multigrid_light")


@dataclass(frozen=True)
class CartesianGrid:
    bounds_min_m: tuple[float, float, float]
    cell_widths_x_m: tuple[float, ...]
    cell_widths_y_m: tuple[float, ...]
    cell_widths_z_m: tuple[float, ...]

    @classmethod
    def uniform(
        cls,
        *,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
    ) -> "CartesianGrid":
        if any(n <= 0 for n in grid_nodes):
            raise ValueError("grid_nodes must be positive in every dimension")
        widths = tuple(
            (hi - lo) / n
            for lo, hi, n in zip(bounds_min_m, bounds_max_m, grid_nodes, strict=True)
        )
        return cls(
            bounds_min_m=tuple(float(v) for v in bounds_min_m),
            cell_widths_x_m=(float(widths[0]),) * int(grid_nodes[0]),
            cell_widths_y_m=(float(widths[1]),) * int(grid_nodes[1]),
            cell_widths_z_m=(float(widths[2]),) * int(grid_nodes[2]),
        )

    def __post_init__(self) -> None:
        bounds_min = tuple(float(v) for v in self.bounds_min_m)
        widths_x = tuple(float(v) for v in self.cell_widths_x_m)
        widths_y = tuple(float(v) for v in self.cell_widths_y_m)
        widths_z = tuple(float(v) for v in self.cell_widths_z_m)
        if len(bounds_min) != 3:
            raise ValueError("bounds_min_m must have three coordinates")
        if not widths_x or not widths_y or not widths_z:
            raise ValueError("CartesianGrid axes must contain at least one cell")
        if any(width <= 0.0 for width in widths_x + widths_y + widths_z):
            raise ValueError("CartesianGrid cell widths must be positive")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "cell_widths_x_m", widths_x)
        object.__setattr__(self, "cell_widths_y_m", widths_y)
        object.__setattr__(self, "cell_widths_z_m", widths_z)

    @staticmethod
    def _axis_centers(origin: float, widths: tuple[float, ...]) -> tuple[float, ...]:
        centers: list[float] = []
        face = float(origin)
        for width in widths:
            centers.append(face + 0.5 * width)
            face += width
        return tuple(centers)

    @staticmethod
    def _axis_faces(origin: float, widths: tuple[float, ...]) -> tuple[float, ...]:
        faces = [float(origin)]
        for width in widths:
            faces.append(faces[-1] + width)
        return tuple(faces)

    @staticmethod
    def _axis_center_distances(centers: tuple[float, ...], widths: tuple[float, ...]) -> tuple[float, ...]:
        distances = [float(widths[0])]
        for index in range(1, len(centers)):
            distances.append(float(centers[index] - centers[index - 1]))
        return tuple(distances)

    @staticmethod
    def _axis_grid_coordinate(value: float, faces: tuple[float, ...], centers: tuple[float, ...]) -> float:
        coordinate = 0.0
        if value <= centers[0]:
            half_width = max(centers[0] - faces[0], 1.0e-18)
            return -0.5 * (centers[0] - value) / half_width
        if value >= centers[-1]:
            half_width = max(faces[-1] - centers[-1], 1.0e-18)
            return float(len(centers) - 1) + 0.5 * (value - centers[-1]) / half_width
        for index in range(len(centers) - 1):
            left = centers[index]
            right = centers[index + 1]
            if left <= value <= right:
                coordinate = float(index) + (value - left) / max(right - left, 1.0e-18)
                break
        return coordinate

    @property
    def grid_nodes(self) -> tuple[int, int, int]:
        return (len(self.cell_widths_x_m), len(self.cell_widths_y_m), len(self.cell_widths_z_m))

    @property
    def bounds_max_m(self) -> tuple[float, float, float]:
        return (
            self.bounds_min_m[0] + sum(self.cell_widths_x_m),
            self.bounds_min_m[1] + sum(self.cell_widths_y_m),
            self.bounds_min_m[2] + sum(self.cell_widths_z_m),
        )

    @property
    def cell_centers_x_m(self) -> tuple[float, ...]:
        return self._axis_centers(self.bounds_min_m[0], self.cell_widths_x_m)

    @property
    def cell_centers_y_m(self) -> tuple[float, ...]:
        return self._axis_centers(self.bounds_min_m[1], self.cell_widths_y_m)

    @property
    def cell_centers_z_m(self) -> tuple[float, ...]:
        return self._axis_centers(self.bounds_min_m[2], self.cell_widths_z_m)

    @property
    def cell_faces_x_m(self) -> tuple[float, ...]:
        return self._axis_faces(self.bounds_min_m[0], self.cell_widths_x_m)

    @property
    def cell_faces_y_m(self) -> tuple[float, ...]:
        return self._axis_faces(self.bounds_min_m[1], self.cell_widths_y_m)

    @property
    def cell_faces_z_m(self) -> tuple[float, ...]:
        return self._axis_faces(self.bounds_min_m[2], self.cell_widths_z_m)

    @property
    def center_distances_x_m(self) -> tuple[float, ...]:
        return self._axis_center_distances(self.cell_centers_x_m, self.cell_widths_x_m)

    @property
    def center_distances_y_m(self) -> tuple[float, ...]:
        return self._axis_center_distances(self.cell_centers_y_m, self.cell_widths_y_m)

    @property
    def center_distances_z_m(self) -> tuple[float, ...]:
        return self._axis_center_distances(self.cell_centers_z_m, self.cell_widths_z_m)

    @staticmethod
    def _axis_is_uniform(widths: tuple[float, ...]) -> bool:
        first = widths[0]
        return all(abs(width - first) <= max(abs(first), 1.0) * 1.0e-12 for width in widths)

    @property
    def is_uniform(self) -> bool:
        return (
            self._axis_is_uniform(self.cell_widths_x_m)
            and self._axis_is_uniform(self.cell_widths_y_m)
            and self._axis_is_uniform(self.cell_widths_z_m)
        )

    @property
    def uniform_spacing_m(self) -> tuple[float, float, float]:
        if not self.is_uniform:
            raise ValueError("uniform_spacing_m is only available for a uniform CartesianGrid")
        return (
            self.cell_widths_x_m[0],
            self.cell_widths_y_m[0],
            self.cell_widths_z_m[0],
        )

    def grid_coordinate_x(self, x_m: float) -> float:
        return self._axis_grid_coordinate(float(x_m), self.cell_faces_x_m, self.cell_centers_x_m)

    def grid_coordinate_y(self, y_m: float) -> float:
        return self._axis_grid_coordinate(float(y_m), self.cell_faces_y_m, self.cell_centers_y_m)

    def grid_coordinate_z(self, z_m: float) -> float:
        return self._axis_grid_coordinate(float(z_m), self.cell_faces_z_m, self.cell_centers_z_m)


@dataclass(frozen=True)
class RefinementRegion:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    target_spacing_m: float | tuple[float, float, float]

    def __post_init__(self) -> None:
        bounds_min = tuple(float(v) for v in self.bounds_min_m)
        bounds_max = tuple(float(v) for v in self.bounds_max_m)
        if len(bounds_min) != 3 or len(bounds_max) != 3:
            raise ValueError("RefinementRegion bounds must have three coordinates")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("RefinementRegion bounds_max_m must be greater than bounds_min_m")
        spacing = self.target_spacing_m
        if isinstance(spacing, tuple):
            target = tuple(float(v) for v in spacing)
            if len(target) != 3:
                raise ValueError("target_spacing_m tuple must have three values")
        else:
            target = (float(spacing),) * 3
        if any(value <= 0.0 for value in target):
            raise ValueError("target_spacing_m must be positive")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)
        object.__setattr__(self, "target_spacing_m", target)


@dataclass(frozen=True)
class GradedGridSpec:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    farfield_spacing_m: float | tuple[float, float, float]
    max_growth_ratio: float
    refinement_regions: tuple[RefinementRegion, ...] = ()
    max_cells: int | None = None

    def __post_init__(self) -> None:
        bounds_min = tuple(float(v) for v in self.bounds_min_m)
        bounds_max = tuple(float(v) for v in self.bounds_max_m)
        if len(bounds_min) != 3 or len(bounds_max) != 3:
            raise ValueError("GradedGridSpec bounds must have three coordinates")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        spacing = self.farfield_spacing_m
        if isinstance(spacing, tuple):
            farfield = tuple(float(v) for v in spacing)
            if len(farfield) != 3:
                raise ValueError("farfield_spacing_m tuple must have three values")
        else:
            farfield = (float(spacing),) * 3
        if any(value <= 0.0 for value in farfield):
            raise ValueError("farfield_spacing_m must be positive")
        if self.max_growth_ratio <= 1.0:
            raise ValueError("max_growth_ratio must be greater than 1")
        max_cells = None if self.max_cells is None else int(self.max_cells)
        if max_cells is not None and max_cells <= 0:
            raise ValueError("max_cells must be positive when provided")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)
        object.__setattr__(self, "farfield_spacing_m", farfield)
        object.__setattr__(self, "refinement_regions", tuple(self.refinement_regions))
        object.__setattr__(self, "max_cells", max_cells)


def _fill_axis_gap(left: float, right: float, spacing: float) -> list[float]:
    length = right - left
    if length <= 0.0:
        return []
    cells = max(1, int(round(length / spacing)))
    while length / cells > spacing * (1.0 + 1.0e-10):
        cells += 1
    width = length / cells
    return [left + width * index for index in range(1, cells)]


def _graded_side_widths(
    *,
    distance: float,
    inner_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
) -> tuple[float, ...]:
    if distance <= 0.0:
        return ()
    smallest_anchor = max(min(inner_width, farfield_spacing), 1.0e-12)
    max_cells = max(64, int(distance / smallest_anchor) * 8 + 64)
    growth = float(max_growth_ratio)
    for cells in range(1, max_cells + 1):
        lower: list[float] = []
        upper: list[float] = []
        for index in range(cells):
            steps = index + 1
            lower_width = float(inner_width) / (growth**steps)
            upper_width = min(float(farfield_spacing), float(inner_width) * (growth**steps))
            if lower_width > upper_width * (1.0 + 1.0e-10):
                break
            lower.append(lower_width)
            upper.append(upper_width)
        if len(lower) != cells:
            continue

        lower_sum = sum(lower)
        upper_sum = sum(upper)
        if distance < lower_sum * (1.0 - 1.0e-10) or distance > upper_sum * (1.0 + 1.0e-10):
            continue
        if upper_sum <= lower_sum:
            widths = tuple(lower)
        else:
            alpha = (float(distance) - lower_sum) / (upper_sum - lower_sum)
            widths = tuple(
                lower_width + alpha * (upper_width - lower_width)
                for lower_width, upper_width in zip(lower, upper, strict=True)
            )
        if _axis_widths_respect_growth(
            widths,
            max_growth_ratio=max_growth_ratio,
            left_boundary_width=inner_width,
        ):
            return widths
    raise ValueError("could not build graded side satisfying max_growth_ratio")


@dataclass(frozen=True)
class _AxisRefinementSegment:
    left: float
    right: float
    fine_width: float
    fine_cells: int


def _axis_widths_respect_growth(
    widths: tuple[float, ...],
    *,
    max_growth_ratio: float,
    left_boundary_width: float | None = None,
    right_boundary_width: float | None = None,
) -> bool:
    values: list[float] = []
    if left_boundary_width is not None:
        values.append(float(left_boundary_width))
    values.extend(float(width) for width in widths)
    if right_boundary_width is not None:
        values.append(float(right_boundary_width))
    tolerance = 1.0 + 1.0e-10
    for left, right in zip(values, values[1:]):
        if left <= 0.0 or right <= 0.0:
            return False
        ratio = max(left / right, right / left)
        if ratio > float(max_growth_ratio) * tolerance:
            return False
    return True


def _graded_bridge_widths(
    *,
    distance: float,
    left_width: float,
    right_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
) -> tuple[float, ...]:
    if distance <= 0.0:
        return ()
    smallest_anchor = max(min(left_width, right_width, farfield_spacing), 1.0e-12)
    max_cells = max(64, int(distance / smallest_anchor) * 8 + 64)
    for cells in range(1, max_cells + 1):
        left_upper: list[float] = []
        value = float(left_width)
        for _ in range(cells):
            value = min(float(farfield_spacing), value * float(max_growth_ratio))
            left_upper.append(value)
        right_upper: list[float] = []
        value = float(right_width)
        for _ in range(cells):
            value = min(float(farfield_spacing), value * float(max_growth_ratio))
            right_upper.append(value)

        lower: list[float] = []
        upper: list[float] = []
        feasible = True
        for index in range(cells):
            left_steps = index + 1
            right_steps = cells - index
            lower_width = max(
                float(left_width) / (float(max_growth_ratio) ** left_steps),
                float(right_width) / (float(max_growth_ratio) ** right_steps),
            )
            upper_width = min(
                float(farfield_spacing),
                left_upper[index],
                right_upper[cells - index - 1],
            )
            if lower_width > upper_width * (1.0 + 1.0e-10):
                feasible = False
                break
            lower.append(lower_width)
            upper.append(upper_width)
        if not feasible:
            continue
        lower_sum = sum(lower)
        upper_sum = sum(upper)
        if distance < lower_sum * (1.0 - 1.0e-10) or distance > upper_sum * (1.0 + 1.0e-10):
            continue
        if upper_sum <= lower_sum:
            widths = tuple(lower)
        else:
            alpha = (distance - lower_sum) / (upper_sum - lower_sum)
            widths = tuple(
                lower_width + alpha * (upper_width - lower_width)
                for lower_width, upper_width in zip(lower, upper, strict=True)
            )
        if _axis_widths_respect_growth(
            widths,
            max_growth_ratio=max_growth_ratio,
            left_boundary_width=left_width,
            right_boundary_width=right_width,
        ):
            return widths
    raise ValueError("could not build graded bridge satisfying max_growth_ratio")


def _axis_refinement_segments(
    *,
    bounds_min: float,
    bounds_max: float,
    farfield_spacing: float,
    max_growth_ratio: float,
    intervals: tuple[tuple[float, float, float], ...],
    tolerance: float,
) -> tuple[_AxisRefinementSegment, ...]:
    clipped: list[tuple[float, float, float]] = []
    for interval_min, interval_max, target_spacing in intervals:
        left = max(bounds_min, float(interval_min))
        right = min(bounds_max, float(interval_max))
        target = float(target_spacing)
        if right <= bounds_min or left >= bounds_max or right <= left:
            continue
        min_side_width = target / float(max_growth_ratio)
        if left - bounds_min < min_side_width:
            left = bounds_min
        if bounds_max - right < min_side_width:
            right = bounds_max
        clipped.append((left, right, target))
    if not clipped:
        return ()

    clipped.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = []
    for left, right, target in clipped:
        if not merged:
            merged.append([left, right, target])
            continue
        previous = merged[-1]
        gap = left - previous[1]
        merge_gap = min(previous[2], target) / float(max_growth_ratio)
        should_merge = left <= previous[1] + max(tolerance, merge_gap)
        if not should_merge and gap > 0.0:
            try:
                _graded_bridge_widths(
                    distance=gap,
                    left_width=previous[2],
                    right_width=target,
                    farfield_spacing=farfield_spacing,
                    max_growth_ratio=max_growth_ratio,
                )
            except ValueError:
                should_merge = True
        if should_merge:
            previous[1] = max(previous[1], right)
            previous[2] = min(previous[2], target)
        else:
            merged.append([left, right, target])

    segments: list[_AxisRefinementSegment] = []
    for left, right, target in merged:
        fine_cells = max(1, int(round((right - left) / target)))
        while (right - left) / fine_cells > target:
            fine_cells += 1
        fine_width = (right - left) / fine_cells
        segments.append(
            _AxisRefinementSegment(
                left=left,
                right=right,
                fine_width=fine_width,
                fine_cells=fine_cells,
            )
        )
    return tuple(segments)


def _build_graded_axis(
    *,
    bounds_min: float,
    bounds_max: float,
    farfield_spacing: float,
    max_growth_ratio: float,
    intervals: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    tolerance = max(abs(bounds_max - bounds_min), 1.0) * 1.0e-12
    if not intervals:
        faces = [bounds_min, bounds_max]
        faces[1:1] = _fill_axis_gap(bounds_min, bounds_max, farfield_spacing)
        return tuple(faces[index + 1] - faces[index] for index in range(len(faces) - 1))

    segments = _axis_refinement_segments(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        farfield_spacing=farfield_spacing,
        max_growth_ratio=max_growth_ratio,
        intervals=intervals,
        tolerance=tolerance,
    )
    if not segments:
        faces = [bounds_min, bounds_max]
        faces[1:1] = _fill_axis_gap(bounds_min, bounds_max, farfield_spacing)
        return tuple(faces[index + 1] - faces[index] for index in range(len(faces) - 1))

    widths: list[float] = []
    first = segments[0]
    widths.extend(
        reversed(
            _graded_side_widths(
                distance=first.left - bounds_min,
                inner_width=first.fine_width,
                farfield_spacing=farfield_spacing,
                max_growth_ratio=max_growth_ratio,
            )
        )
    )
    widths.extend([first.fine_width] * first.fine_cells)
    previous = first
    for segment in segments[1:]:
        widths.extend(
            _graded_bridge_widths(
                distance=segment.left - previous.right,
                left_width=previous.fine_width,
                right_width=segment.fine_width,
                farfield_spacing=farfield_spacing,
                max_growth_ratio=max_growth_ratio,
            )
        )
        widths.extend([segment.fine_width] * segment.fine_cells)
        previous = segment
    widths.extend(
        _graded_side_widths(
            distance=bounds_max - previous.right,
            inner_width=previous.fine_width,
            farfield_spacing=farfield_spacing,
            max_growth_ratio=max_growth_ratio,
        )
    )
    return tuple(widths)


def build_graded_grid(spec: GradedGridSpec) -> CartesianGrid:
    axis_intervals: list[list[tuple[float, float, float]]] = [[], [], []]
    for region in spec.refinement_regions:
        for axis in range(3):
            axis_intervals[axis].append(
                (
                    region.bounds_min_m[axis],
                    region.bounds_max_m[axis],
                    region.target_spacing_m[axis],
                )
            )
    widths = tuple(
        _build_graded_axis(
            bounds_min=spec.bounds_min_m[axis],
            bounds_max=spec.bounds_max_m[axis],
            farfield_spacing=spec.farfield_spacing_m[axis],
            max_growth_ratio=float(spec.max_growth_ratio),
            intervals=tuple(axis_intervals[axis]),
        )
        for axis in range(3)
    )
    cell_count = len(widths[0]) * len(widths[1]) * len(widths[2])
    if spec.max_cells is not None and cell_count > int(spec.max_cells):
        raise ValueError(
            f"graded grid cell count {cell_count} exceeds max_cells {int(spec.max_cells)}"
        )
    return CartesianGrid(
        bounds_min_m=spec.bounds_min_m,
        cell_widths_x_m=widths[0],
        cell_widths_y_m=widths[1],
        cell_widths_z_m=widths[2],
    )


@dataclass(frozen=True)
class FluidDomainSpec:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    grid_nodes: tuple[int, int, int] | None
    density_kgm3: float
    viscosity_pa_s: float
    dt_s: float
    cartesian_grid: CartesianGrid | None = None
    graded_grid: GradedGridSpec | None = None

    @classmethod
    def unit_box(
        cls,
        grid_nodes: tuple[int, int, int] = (32, 32, 32),
        density_kgm3: float = 1000.0,
        viscosity_pa_s: float = 1.0e-3,
        dt_s: float = 1.0e-3,
    ):
        return cls(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=grid_nodes,
            density_kgm3=density_kgm3,
            viscosity_pa_s=viscosity_pa_s,
            dt_s=dt_s,
        )

    def __post_init__(self) -> None:
        if any(hi <= lo for lo, hi in zip(self.bounds_min_m, self.bounds_max_m, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        if self.density_kgm3 <= 0.0:
            raise ValueError("density_kgm3 must be positive")
        if self.viscosity_pa_s < 0.0:
            raise ValueError("viscosity_pa_s must be non-negative")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.cartesian_grid is not None and self.graded_grid is not None:
            raise ValueError("cartesian_grid and graded_grid are mutually exclusive")

        if self.graded_grid is not None:
            grid = build_graded_grid(self.graded_grid)
        elif self.cartesian_grid is not None:
            grid = self.cartesian_grid
        else:
            if self.grid_nodes is None:
                raise ValueError("grid_nodes is required when no cartesian_grid or graded_grid is provided")
            grid = CartesianGrid.uniform(
                bounds_min_m=self.bounds_min_m,
                bounds_max_m=self.bounds_max_m,
                grid_nodes=self.grid_nodes,
            )

        grid_nodes = grid.grid_nodes if self.grid_nodes is None else tuple(int(n) for n in self.grid_nodes)
        if any(n < 4 for n in grid_nodes):
            raise ValueError("grid_nodes must be at least 4 in every dimension")
        if grid.grid_nodes != grid_nodes:
            raise ValueError("cartesian_grid grid_nodes must match FluidDomainSpec.grid_nodes")
        for actual, expected in zip(grid.bounds_min_m, self.bounds_min_m, strict=True):
            if abs(actual - expected) > max(abs(expected), 1.0) * 1.0e-12:
                raise ValueError("cartesian_grid bounds_min_m must match FluidDomainSpec.bounds_min_m")
        for actual, expected in zip(grid.bounds_max_m, self.bounds_max_m, strict=True):
            if abs(actual - expected) > max(abs(expected), 1.0) * 1.0e-12:
                raise ValueError("cartesian_grid bounds_max_m must match FluidDomainSpec.bounds_max_m")
        object.__setattr__(self, "grid_nodes", grid_nodes)
        object.__setattr__(self, "cartesian_grid", grid)

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        return self.cartesian_grid.uniform_spacing_m

    @property
    def cell_volume_m3(self) -> float:
        dx, dy, dz = self.spacing_m
        return dx * dy * dz


@dataclass(frozen=True)
class ForceSpreadingReport:
    surface_force_n: tuple[float, float, float]
    grid_force_n: tuple[float, float, float]
    action_reaction_relative_error: float
    active_grid_cells: int


@dataclass(frozen=True)
class FluidImpulseReport:
    grid_impulse_n_s: tuple[float, float, float]
    momentum_delta_n_s: tuple[float, float, float]
    impulse_relative_error: float
    active_velocity_cells: int


@dataclass(frozen=True)
class VelocityConstraintReport:
    active_cells: int
    max_delta_mps: float
    mean_delta_mps: float
    momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    primary_momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    secondary_momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VelocityDirichletBoundaryReport:
    active_cells: int
    max_delta_mps: float
    mean_delta_mps: float
    momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@ti.data_oriented
class CartesianFluidSolver:
    """Cell-centered 3D Cartesian fluid state for coupled FSI simulations."""

    DEFAULT_MULTIGRID_CYCLES = 12
    DEFAULT_NONUNIFORM_MULTIGRID_CYCLES = 24

    @staticmethod
    def _build_multigrid_shapes(shape: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
        shapes = [shape]
        current = shape
        while min(current) > 3:
            next_shape = tuple(max(2, (n + 1) // 2) for n in current)
            if next_shape == current:
                break
            shapes.append(next_shape)
            current = next_shape
        return tuple(shapes)

    @staticmethod
    def _coarsen_axis_widths(widths: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            sum(widths[index : min(index + 2, len(widths))])
            for index in range(0, len(widths), 2)
        )

    @staticmethod
    def _build_multigrid_axis_widths(
        grid: CartesianGrid,
        shapes: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]], ...]:
        levels: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []
        current = (
            grid.cell_widths_x_m,
            grid.cell_widths_y_m,
            grid.cell_widths_z_m,
        )
        for shape in shapes:
            if tuple(len(axis) for axis in current) != shape:
                raise ValueError("multigrid axis widths do not match multigrid shapes")
            levels.append(current)
            current = tuple(
                CartesianFluidSolver._coarsen_axis_widths(axis_widths)
                for axis_widths in current
            )
        return tuple(levels)

    def default_multigrid_cycles(self) -> int:
        if self.grid.is_uniform:
            return self.DEFAULT_MULTIGRID_CYCLES
        return self.DEFAULT_NONUNIFORM_MULTIGRID_CYCLES

    def __init__(self, spec: FluidDomainSpec, runtime: TaichiRuntimeConfig | None = None):
        init_taichi(runtime)
        self.spec = spec
        self.grid = spec.cartesian_grid
        self.nx, self.ny, self.nz = self.grid.grid_nodes
        if self.grid.is_uniform:
            self.dx, self.dy, self.dz = self.grid.uniform_spacing_m
        else:
            self.dx, self.dy, self.dz = (
                min(self.grid.cell_widths_x_m),
                min(self.grid.cell_widths_y_m),
                min(self.grid.cell_widths_z_m),
            )
        self.bounds_min = tuple(float(v) for v in self.grid.bounds_min_m)
        self.bounds_max = tuple(float(v) for v in self.grid.bounds_max_m)
        self.rho = float(spec.density_kgm3)
        self.mu = float(spec.viscosity_pa_s)
        self.dt = float(spec.dt_s)

        shape = self.grid.grid_nodes
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_prev = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.saved_velocity = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.pressure = ti.field(dtype=ti.f32, shape=shape)
        self.saved_pressure = ti.field(dtype=ti.f32, shape=shape)
        self.fsi_pressure = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_tmp = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_accum = ti.field(dtype=ti.f32, shape=shape)
        self.divergence = ti.field(dtype=ti.f32, shape=shape)
        self.volume_source_s = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_interface_matrix_diagonal = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_interface_matrix_rhs = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_interface_coupling_active = ti.field(dtype=ti.i32, shape=shape)
        self.pressure_interface_coupling_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_interface_coupling_coefficient = ti.field(
            dtype=ti.f32,
            shape=shape,
        )
        self.hibm_base_obstacle = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_pressure_outlet_reachable = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_pressure_outlet_reachable_next = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_pressure_unreached_component_label = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.cg_unreached_component_sum = ti.field(dtype=ti.f64, shape=32)
        self.cg_unreached_component_volume = ti.field(dtype=ti.f64, shape=32)
        self.cg_unreached_component_scan = ti.field(dtype=ti.i32, shape=())
        # R2-H1 observability: overlap between the flood-unreachable anchoring
        # set and rows touched by pressure-interface matrix terms. The cell-hit
        # grid deduplicates coupling-edge hits (a row can be the target of
        # several edges); the 32-slot array deduplicates per component.
        self.hibm_unreached_interface_cell_hit = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_unreached_interface_component_hit = ti.field(dtype=ti.i32, shape=32)
        self.report_hibm_unreached_interface_diagonal_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_hibm_unreached_interface_coupling_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_hibm_unreached_interface_component_hits = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.fv_diag = ti.field(dtype=ti.f32, shape=shape)
        self.cg_r = ti.field(dtype=ti.f32, shape=shape)
        self.cg_z = ti.field(dtype=ti.f32, shape=shape)
        self.cg_d = ti.field(dtype=ti.f32, shape=shape)
        self.cg_Ad = ti.field(dtype=ti.f32, shape=shape)
        self.cg_r_old = ti.field(dtype=ti.f32, shape=shape)
        self.cg_mg_rhs = ti.field(dtype=ti.f32, shape=shape)
        self.cg_mg_residual = ti.field(dtype=ti.f32, shape=shape)
        self.cg_rz = ti.field(dtype=ti.f64, shape=())
        self.cg_rz_new = ti.field(dtype=ti.f64, shape=())
        self.cg_dAd = ti.field(dtype=ti.f64, shape=())
        self.cg_rr = ti.field(dtype=ti.f64, shape=())
        self.cg_beta_numerator = ti.field(dtype=ti.f64, shape=())
        self.cg_alpha = ti.field(dtype=ti.f64, shape=())
        self.cg_beta = ti.field(dtype=ti.f64, shape=())
        self.cg_weighted_sum = ti.field(dtype=ti.f64, shape=())
        self.cg_free_volume = ti.field(dtype=ti.f64, shape=())
        self.cg_weighted_mean = ti.field(dtype=ti.f64, shape=())
        self.cg_breakdown_code = ti.field(dtype=ti.i32, shape=())
        self.cg_breakdown_dAd = ti.field(dtype=ti.f64, shape=())
        self._pcg_mg_rhs = [self.cg_mg_rhs]
        self._pcg_mg_residual = [self.cg_mg_residual]
        self._pcg_mg_pressure = [self.cg_z]
        self._pcg_mg_tmp = [self.cg_Ad]
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.obstacle = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_fresh_fluid_cell = ti.field(dtype=ti.i32, shape=shape)
        self.velocity_constraint_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_constraint_weight = ti.field(dtype=ti.f32, shape=shape)
        self.velocity_constraint_primary_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_constraint_primary_weight = ti.field(dtype=ti.f32, shape=shape)
        self.velocity_constraint_secondary_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_constraint_secondary_weight = ti.field(dtype=ti.f32, shape=shape)
        self.velocity_dirichlet_boundary_active = ti.field(dtype=ti.i32, shape=shape)
        self.velocity_dirichlet_boundary_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=shape,
        )
        self.velocity_dirichlet_boundary_projection_weight = ti.field(
            dtype=ti.f32,
            shape=shape,
        )
        self.reduction_sum = ti.field(dtype=ti.f32, shape=())
        self.reduction_max = ti.field(dtype=ti.f32, shape=())
        self.reduction_count = ti.field(dtype=ti.i32, shape=())
        self.divergence_report_snapshot = ti.Vector.field(3, dtype=ti.f64, shape=())
        # 18 slots: 16 partition slots + 2 anchored-unreached slots; must
        # match the static slot range of _divergence_final_report_kernel.
        self.divergence_combined_sum = ti.field(dtype=ti.f32, shape=18)
        self.divergence_combined_max = ti.field(dtype=ti.f32, shape=18)
        self.divergence_combined_count = ti.field(dtype=ti.i32, shape=18)
        self.divergence_final_report_snapshot = ti.Vector.field(
            24,
            dtype=ti.f64,
            shape=3,
        )
        self.divergence_dirichlet_partition_snapshot = ti.Vector.field(
            12,
            dtype=ti.f64,
            shape=(),
        )
        self.cleanup_target_l2_sq = ti.field(dtype=ti.f64, shape=())
        self.cleanup_required = ti.field(dtype=ti.i32, shape=())
        self.report_surface_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_grid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_force_spread_relative_error = ti.field(dtype=ti.f32, shape=())
        self.report_active_force_cells = ti.field(dtype=ti.i32, shape=())
        self.report_grid_impulse_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_impulse_relative_error = ti.field(dtype=ti.f32, shape=())
        self.report_active_velocity_cells = ti.field(dtype=ti.i32, shape=())
        self.report_velocity_constraint_cells = ti.field(dtype=ti.i32, shape=())
        self.report_velocity_constraint_delta_sum = ti.field(dtype=ti.f32, shape=())
        self.report_velocity_constraint_delta_max = ti.field(dtype=ti.f32, shape=())
        self.report_velocity_constraint_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_velocity_constraint_primary_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_velocity_constraint_secondary_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.velocity_constraint_primary_impulse_n_s = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.velocity_constraint_secondary_impulse_n_s = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_velocity_dirichlet_boundary_cells = ti.field(dtype=ti.i32, shape=())
        self.report_velocity_dirichlet_boundary_delta_sum = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_delta_max = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(),
        )
        self.report_source_volume_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_pressure_outlet_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_velocity_outlet_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_pressure_outlet_flux_ratio = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_velocity_outlet_flux_ratio = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.pressure_outlet_report_snapshot = ti.Vector.field(8, dtype=ti.f32, shape=())
        self.report_pressure_interface_matrix_diagonal_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_rhs_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_max_abs_diagonal = ti.field(dtype=ti.f32, shape=())
        self.report_pressure_interface_matrix_active_cells = ti.field(dtype=ti.i32, shape=())
        self.report_hibm_internal_obstacle_cells = ti.field(dtype=ti.i32, shape=())
        self.report_hibm_fresh_fluid_cells = ti.field(dtype=ti.i32, shape=())
        self.report_hibm_solid_band_nonprojectable_cells = ti.field(dtype=ti.i32, shape=())
        self.report_hibm_solid_band_interior_cells = ti.field(dtype=ti.i32, shape=())
        self.report_hibm_solid_band_enclosed_water_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_hibm_pressure_disconnected_nonprojectable_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # S2-A8'' post-projection pressure fill into hibm-converted cells
        # (obstacle != 0 and hibm_base_obstacle == 0): Jacobi double
        # buffers (filled flag + pressure) and the dedicated stress
        # sampling view (base geometry + classified row-cloud envelope
        # dry; converted sealed water samplable).
        self.hibm_pressure_filled = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_pressure_filled_next = ti.field(dtype=ti.i32, shape=shape)
        self.hibm_pressure_fill_next = ti.field(dtype=ti.f32, shape=shape)
        self.sampling_obstacle = ti.field(dtype=ti.i32, shape=shape)
        self.report_hibm_pressure_filled_cells = ti.field(dtype=ti.i32, shape=())
        self._hibm_base_obstacle_initialized = False
        self.cell_width_x_m = ti.field(dtype=ti.f32, shape=self.nx)
        self.cell_width_y_m = ti.field(dtype=ti.f32, shape=self.ny)
        self.cell_width_z_m = ti.field(dtype=ti.f32, shape=self.nz)
        self.cell_face_x_m = ti.field(dtype=ti.f32, shape=self.nx + 1)
        self.cell_face_y_m = ti.field(dtype=ti.f32, shape=self.ny + 1)
        self.cell_face_z_m = ti.field(dtype=ti.f32, shape=self.nz + 1)
        self.cell_center_x_m = ti.field(dtype=ti.f32, shape=self.nx)
        self.cell_center_y_m = ti.field(dtype=ti.f32, shape=self.ny)
        self.cell_center_z_m = ti.field(dtype=ti.f32, shape=self.nz)
        self.center_distance_x_m = ti.field(dtype=ti.f32, shape=self.nx)
        self.center_distance_y_m = ti.field(dtype=ti.f32, shape=self.ny)
        self.center_distance_z_m = ti.field(dtype=ti.f32, shape=self.nz)

        self._mg_shapes = self._build_multigrid_shapes(shape)
        self._mg_axis_widths_m = self._build_multigrid_axis_widths(self.grid, self._mg_shapes)
        self._mg_cell_width_x_m = [self.cell_width_x_m]
        self._mg_cell_width_y_m = [self.cell_width_y_m]
        self._mg_cell_width_z_m = [self.cell_width_z_m]
        self._mg_center_distance_x_m = [self.center_distance_x_m]
        self._mg_center_distance_y_m = [self.center_distance_y_m]
        self._mg_center_distance_z_m = [self.center_distance_z_m]
        for level_shape in self._mg_shapes[1:]:
            self._mg_cell_width_x_m.append(ti.field(dtype=ti.f32, shape=level_shape[0]))
            self._mg_cell_width_y_m.append(ti.field(dtype=ti.f32, shape=level_shape[1]))
            self._mg_cell_width_z_m.append(ti.field(dtype=ti.f32, shape=level_shape[2]))
            self._mg_center_distance_x_m.append(ti.field(dtype=ti.f32, shape=level_shape[0]))
            self._mg_center_distance_y_m.append(ti.field(dtype=ti.f32, shape=level_shape[1]))
            self._mg_center_distance_z_m.append(ti.field(dtype=ti.f32, shape=level_shape[2]))
        self._mg_pressure = [self.pressure]
        self._mg_tmp = [self.pressure_tmp]
        self._mg_obstacle = [self.obstacle]
        self._mg_velocity_dirichlet_boundary_active = [
            self.velocity_dirichlet_boundary_active
        ]
        self._mg_velocity_dirichlet_boundary_projection_weight = [
            self.velocity_dirichlet_boundary_projection_weight
        ]
        self._mg_pressure_interface_matrix_diagonal = [
            self.pressure_interface_matrix_diagonal
        ]
        self._mg_rhs = []
        self._mg_residual = []
        for level_shape in self._mg_shapes:
            self._mg_rhs.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._mg_residual.append(ti.field(dtype=ti.f32, shape=level_shape))
        for level_shape in self._mg_shapes[1:]:
            self._mg_pressure.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._mg_tmp.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._mg_obstacle.append(ti.field(dtype=ti.i32, shape=level_shape))
            self._mg_velocity_dirichlet_boundary_active.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_projection_weight.append(
                ti.field(dtype=ti.f32, shape=level_shape)
            )
            self._mg_pressure_interface_matrix_diagonal.append(
                ti.field(dtype=ti.f32, shape=level_shape)
            )
            self._pcg_mg_rhs.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._pcg_mg_residual.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._pcg_mg_pressure.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._pcg_mg_tmp.append(ti.field(dtype=ti.f32, shape=level_shape))

        self._load_cartesian_grid_fields()
        self._load_multigrid_axis_fields()
        self.last_cg_iterations = 0
        self.last_cg_initial_relative_residual = math.inf
        self.last_cg_relative_residual = math.inf
        self.last_cg_converged = False
        self.last_cg_breakdown = ""
        self.last_cg_host_residual_checks = 0
        self.last_cg_mean_host_reads = 0
        self.last_cg_mean_projection_count = 0
        self.last_cg_unreached_set_mean_projection_count = 0
        self.last_cg_restart_count = 0
        self.last_cg_restart_count_measured = False
        self.last_cg_restart_policy = ""
        self.last_cg_breakdown_dAd = 0.0
        self.last_hibm_pressure_unreached_cell_count = 0
        self.last_hibm_pressure_reachability_converged = True
        self.last_hibm_pressure_reachability_sweeps = 0
        self.last_hibm_pressure_reachability_reused = False
        self._hibm_pressure_unreached_count = 0
        self._hibm_reachability_checksum = None
        self.last_hibm_pressure_unreached_component_count = 0
        self._hibm_pressure_unreached_component_count = 0
        self.last_hibm_pressure_unreached_component_overflow = False
        self.last_hibm_pressure_component_labels_converged = True
        self.last_hibm_unreached_cells_with_interface_diagonal = 0
        self.last_hibm_unreached_cells_with_interface_coupling = 0
        self.last_hibm_unreached_components_with_interface_hits = 0
        self.last_hibm_solid_band_marked_increment = 0
        # -1 means "the last band sweep ran without a population split"
        # (legacy unclassified sweep); >= 0 are measured per-sweep
        # populations (S2-A8').
        self.last_hibm_solid_band_interior_cells = -1
        self.last_hibm_solid_band_enclosed_water_cells = -1
        # -1 means "the post-projection converted-cell pressure fill has
        # not run since construction / restore_state" (S2-A8''). The fill
        # is never mounted inside project(): the HIBM assemble calls it
        # explicitly after the projection returns and before stress
        # sampling, so project()'s report key mirrors the PREVIOUS fill.
        self.last_hibm_pressure_filled_cell_count = -1
        self.last_unreached_divergence_raw_stats = {
            "max_abs": 0.0,
            "l2": 0.0,
            "count": 0,
        }
        self.last_unreached_divergence_stats = {
            "max_abs": 0.0,
            "l2": 0.0,
            "count": 0,
        }
        self.last_project_cg_project_calls = 0
        self.last_project_cg_iterations_total = 0
        self.last_project_cg_iterations_max = 0
        self.last_project_cg_host_residual_checks = 0
        self.last_project_cg_mean_host_reads = 0
        self.last_project_cg_mean_projection_count = 0
        self.last_project_cg_unreached_set_mean_projection_count = 0
        self.last_project_cg_restart_count = 0
        self.last_project_cg_restart_count_measured = False
        self.last_project_cg_restart_policy = ""
        self.last_project_cg_initial_relative_residual_max = 0.0
        self.last_project_cg_relative_residual_max = 0.0
        self.last_project_cg_converged_all = True
        self.last_project_cg_breakdown_count = 0
        self.last_project_cg_breakdown_code = 0
        self.last_project_cg_breakdown_dAd = 0.0
        self.last_project_cg_breakdown = ""
        self.last_pressure_outlet_report_host_reads = 0
        self.last_divergence_report_host_reads = 0
        self.last_velocity_constraint_impulse_host_reads = 0
        self.clear()

    @staticmethod
    def _load_axis_field(field, values: tuple[float, ...]) -> None:
        for index, value in enumerate(values):
            field[index] = float(value)

    def _load_cartesian_grid_fields(self) -> None:
        self._load_axis_field(self.cell_width_x_m, self.grid.cell_widths_x_m)
        self._load_axis_field(self.cell_width_y_m, self.grid.cell_widths_y_m)
        self._load_axis_field(self.cell_width_z_m, self.grid.cell_widths_z_m)
        self._load_axis_field(self.cell_face_x_m, self.grid.cell_faces_x_m)
        self._load_axis_field(self.cell_face_y_m, self.grid.cell_faces_y_m)
        self._load_axis_field(self.cell_face_z_m, self.grid.cell_faces_z_m)
        self._load_axis_field(self.cell_center_x_m, self.grid.cell_centers_x_m)
        self._load_axis_field(self.cell_center_y_m, self.grid.cell_centers_y_m)
        self._load_axis_field(self.cell_center_z_m, self.grid.cell_centers_z_m)
        self._load_axis_field(self.center_distance_x_m, self.grid.center_distances_x_m)
        self._load_axis_field(self.center_distance_y_m, self.grid.center_distances_y_m)
        self._load_axis_field(self.center_distance_z_m, self.grid.center_distances_z_m)

    def _load_multigrid_axis_fields(self) -> None:
        for level, axis_widths in enumerate(self._mg_axis_widths_m[1:], start=1):
            widths_x, widths_y, widths_z = axis_widths
            centers_x = CartesianGrid._axis_centers(self.bounds_min[0], widths_x)
            centers_y = CartesianGrid._axis_centers(self.bounds_min[1], widths_y)
            centers_z = CartesianGrid._axis_centers(self.bounds_min[2], widths_z)
            self._load_axis_field(self._mg_cell_width_x_m[level], widths_x)
            self._load_axis_field(self._mg_cell_width_y_m[level], widths_y)
            self._load_axis_field(self._mg_cell_width_z_m[level], widths_z)
            self._load_axis_field(
                self._mg_center_distance_x_m[level],
                CartesianGrid._axis_center_distances(centers_x, widths_x),
            )
            self._load_axis_field(
                self._mg_center_distance_y_m[level],
                CartesianGrid._axis_center_distances(centers_y, widths_y),
            )
            self._load_axis_field(
                self._mg_center_distance_z_m[level],
                CartesianGrid._axis_center_distances(centers_z, widths_z),
            )

    @ti.kernel
    def _clear_kernel(self):
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.saved_velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.pressure[i, j, k] = 0.0
            self.saved_pressure[i, j, k] = 0.0
            self.fsi_pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0
            self.pressure_accum[i, j, k] = 0.0
            self.divergence[i, j, k] = 0.0
            self.volume_source_s[i, j, k] = 0.0
            self.pressure_interface_matrix_diagonal[i, j, k] = 0.0
            self.pressure_interface_matrix_rhs[i, j, k] = 0.0
            self.pressure_interface_coupling_active[i, j, k] = 0
            self.pressure_interface_coupling_neighbor[i, j, k] = ti.Vector([0, 0, 0])
            self.pressure_interface_coupling_coefficient[i, j, k] = 0.0
            self.fv_diag[i, j, k] = 0.0
            self.cg_r[i, j, k] = 0.0
            self.cg_z[i, j, k] = 0.0
            self.cg_d[i, j, k] = 0.0
            self.cg_Ad[i, j, k] = 0.0
            self.cg_r_old[i, j, k] = 0.0
            self.cg_mg_rhs[i, j, k] = 0.0
            self.cg_mg_residual[i, j, k] = 0.0
            self.force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.obstacle[i, j, k] = 0
            self.hibm_base_obstacle[i, j, k] = 0
            self.hibm_pressure_outlet_reachable[i, j, k] = 0
            self.hibm_pressure_outlet_reachable_next[i, j, k] = 0
            self.hibm_fresh_fluid_cell[i, j, k] = 0
            self.hibm_pressure_filled[i, j, k] = 0
            self.hibm_pressure_filled_next[i, j, k] = 0
            self.hibm_pressure_fill_next[i, j, k] = 0.0
            self.sampling_obstacle[i, j, k] = 0
            self.velocity_constraint_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_weight[i, j, k] = 0.0
            self.velocity_constraint_primary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_primary_weight[i, j, k] = 0.0
            self.velocity_constraint_secondary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_secondary_weight[i, j, k] = 0.0
            self.velocity_dirichlet_boundary_active[i, j, k] = 0
            self.velocity_dirichlet_boundary_value_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0
        self.report_surface_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_force_spread_relative_error[None] = 0.0
        self.report_active_force_cells[None] = 0
        self.report_grid_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_impulse_relative_error[None] = 0.0
        self.report_active_velocity_cells[None] = 0
        self.report_velocity_constraint_cells[None] = 0
        self.report_velocity_constraint_delta_sum[None] = 0.0
        self.report_velocity_constraint_delta_max[None] = 0.0
        self.report_velocity_constraint_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_primary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_secondary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.velocity_constraint_primary_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.velocity_constraint_secondary_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_dirichlet_boundary_cells[None] = 0
        self.report_velocity_dirichlet_boundary_delta_sum[None] = 0.0
        self.report_velocity_dirichlet_boundary_delta_max[None] = 0.0
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_source_volume_flux_m3s[None] = 0.0
        self.report_zmin_pressure_outlet_flux_m3s[None] = 0.0
        self.report_zmin_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_outlet_flux_ratio[None] = 0.0
        self.report_zmin_velocity_outlet_flux_ratio[None] = 0.0
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None] = 0.0
        self.pressure_outlet_report_snapshot[None] = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.cg_rz[None] = 0.0
        self.cg_rz_new[None] = 0.0
        self.cg_dAd[None] = 0.0
        self.cg_rr[None] = 0.0
        self.cg_beta_numerator[None] = 0.0
        self.cg_alpha[None] = 0.0
        self.cg_beta[None] = 0.0
        self.cg_weighted_sum[None] = 0.0
        self.cg_free_volume[None] = 0.0
        self.cg_weighted_mean[None] = 0.0
        self.cg_breakdown_code[None] = 0
        self.cg_breakdown_dAd[None] = 0.0
        self.report_hibm_internal_obstacle_cells[None] = 0
        self.report_hibm_fresh_fluid_cells[None] = 0
        self.report_hibm_solid_band_nonprojectable_cells[None] = 0
        self.report_hibm_solid_band_interior_cells[None] = 0
        self.report_hibm_solid_band_enclosed_water_cells[None] = 0
        self.report_hibm_pressure_disconnected_nonprojectable_cells[None] = 0
        self.report_hibm_pressure_filled_cells[None] = 0

    def clear(self) -> None:
        self._clear_kernel()
        self._hibm_base_obstacle_initialized = False

    @ti.kernel
    def _snapshot_hibm_base_obstacle_kernel(self):
        for i, j, k in self.obstacle:
            self.hibm_base_obstacle[i, j, k] = self.obstacle[i, j, k]

    def snapshot_hibm_base_obstacle(self) -> None:
        self._snapshot_hibm_base_obstacle_kernel()
        self._hibm_base_obstacle_initialized = True

    def _ensure_hibm_base_obstacle(self) -> None:
        if not self._hibm_base_obstacle_initialized:
            self.snapshot_hibm_base_obstacle()

    @ti.kernel
    def _reset_obstacle_to_hibm_base_kernel(self):
        for i, j, k in self.obstacle:
            self.obstacle[i, j, k] = self.hibm_base_obstacle[i, j, k]

    def reset_obstacle_to_hibm_base(self) -> None:
        self._ensure_hibm_base_obstacle()
        self._reset_obstacle_to_hibm_base_kernel()

    @ti.kernel
    def _apply_hibm_internal_obstacles_kernel(
        self,
        node_kind_code: ti.template(),
        internal_node_code: ti.i32,
    ):
        self.report_hibm_internal_obstacle_cells[None] = 0
        self.report_hibm_fresh_fluid_cells[None] = 0
        for i, j, k in self.obstacle:
            old_obstacle = self.obstacle[i, j, k]
            base_obstacle = self.hibm_base_obstacle[i, j, k]
            self.hibm_fresh_fluid_cell[i, j, k] = 0
            self.obstacle[i, j, k] = base_obstacle
            if node_kind_code[i, j, k] == internal_node_code:
                self.obstacle[i, j, k] = 1
                self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                ti.atomic_add(self.report_hibm_internal_obstacle_cells[None], 1)
            elif old_obstacle != 0 and base_obstacle == 0:
                self.hibm_fresh_fluid_cell[i, j, k] = 1
                ti.atomic_add(self.report_hibm_fresh_fluid_cells[None], 1)

    @ti.kernel
    def _reconstruct_hibm_fresh_fluid_cells_kernel(self):
        for i, j, k in self.hibm_fresh_fluid_cell:
            if self.hibm_fresh_fluid_cell[i, j, k] != 0 and self.obstacle[i, j, k] == 0:
                velocity_sum = ti.Vector([0.0, 0.0, 0.0])
                weight = 0.0
                if self.velocity_dirichlet_boundary_active[i, j, k] != 0:
                    velocity_sum += self.velocity_dirichlet_boundary_value_mps[i, j, k]
                    weight += 1.0
                for offset in ti.static(
                    (
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    )
                ):
                    ni = i + offset[0]
                    nj = j + offset[1]
                    nk = k + offset[2]
                    if (
                        0 <= ni
                        and ni < self.nx
                        and 0 <= nj
                        and nj < self.ny
                        and 0 <= nk
                        and nk < self.nz
                        and self.obstacle[ni, nj, nk] == 0
                        and self.hibm_fresh_fluid_cell[ni, nj, nk] == 0
                    ):
                        velocity_sum += self.velocity[ni, nj, nk]
                        weight += 1.0
                if weight > 0.0:
                    reconstructed = velocity_sum / weight
                    self.velocity[i, j, k] = reconstructed
                    self.velocity_prev[i, j, k] = reconstructed

    def apply_hibm_internal_obstacles(
        self,
        node_kind_code,
        *,
        internal_node_code: int,
    ) -> int:
        self._ensure_hibm_base_obstacle()
        self._apply_hibm_internal_obstacles_kernel(
            node_kind_code,
            int(internal_node_code),
        )
        self._reconstruct_hibm_fresh_fluid_cells_kernel()
        return int(self.report_hibm_internal_obstacle_cells[None])

    @ti.kernel
    def _mark_hibm_solid_band_nonprojectable_cells_kernel(
        self,
        pressure_outlet_zmin: ti.i32,
        convert_to_obstacle: ti.i32,
    ):
        self.report_hibm_solid_band_nonprojectable_cells[None] = 0
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and not self._divergence_stencil_has_pressure_correctable_face(
                    i,
                    j,
                    k,
                    pressure_outlet_zmin,
                )
            ):
                if convert_to_obstacle == 1:
                    self.obstacle[i, j, k] = 1
                    self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    self.velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    self.volume_source_s[i, j, k] = 0.0
                    self.divergence[i, j, k] = 0.0
                ti.atomic_add(self.report_hibm_solid_band_nonprojectable_cells[None], 1)

    @ti.kernel
    def _mark_hibm_solid_band_population_split_kernel(
        self,
        node_kind_code: ti.template(),
        unclassified_node_code: ti.i32,
        pressure_outlet_zmin: ti.i32,
        convert_to_obstacle: ti.i32,
        interior_only: ti.i32,
    ):
        self.report_hibm_solid_band_nonprojectable_cells[None] = 0
        self.report_hibm_solid_band_interior_cells[None] = 0
        self.report_hibm_solid_band_enclosed_water_cells[None] = 0
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and not self._divergence_stencil_has_pressure_correctable_face(
                    i,
                    j,
                    k,
                    pressure_outlet_zmin,
                )
            ):
                # Population split (S2-A8'): a candidate the IB node
                # search classified (any code other than the unclassified
                # sentinel) lies inside the near-surface band - a
                # membrane-interior sliver. An unclassified candidate has
                # no marker within the search radius: real enclosed water
                # sealed off only by the Dirichlet row cloud.
                is_interior_sliver = 0
                if node_kind_code[i, j, k] != unclassified_node_code:
                    is_interior_sliver = 1
                if is_interior_sliver == 1:
                    ti.atomic_add(
                        self.report_hibm_solid_band_interior_cells[None],
                        1,
                    )
                else:
                    ti.atomic_add(
                        self.report_hibm_solid_band_enclosed_water_cells[None],
                        1,
                    )
                if convert_to_obstacle == 1 and (
                    interior_only == 0 or is_interior_sliver == 1
                ):
                    self.obstacle[i, j, k] = 1
                    self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    self.velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    self.volume_source_s[i, j, k] = 0.0
                    self.divergence[i, j, k] = 0.0
                if interior_only == 0 or is_interior_sliver == 1:
                    ti.atomic_add(
                        self.report_hibm_solid_band_nonprojectable_cells[None],
                        1,
                    )

    def mark_hibm_solid_band_nonprojectable_cells(
        self,
        *,
        pressure_outlet_zmin: bool = False,
        node_kind_code=None,
        unclassified_node_code: int = 0,
    ) -> int:
        """Mark one band sweep and return the newly marked cell count.

        The kernel resets the 0-D counter each call, so the return value is
        already the per-round increment the caller's fixed-round cap loops on.
        ``last_hibm_solid_band_marked_increment`` keeps the most recent
        increment host-side (R2-M3): a non-zero value at projection-report
        time means the caller's band budget was exhausted before saturation.

        Population split (S2-A8'): when ``node_kind_code`` (the IB node
        search classification field, cell-shaped) is provided, band
        candidates split into two populations:

        - interior slivers: candidates the search classified near the
          marker surface (``node_kind_code != unclassified_node_code``) -
          membrane-interior quasi-solid cells inside the row cloud;
        - enclosed water: unclassified candidates (no marker within the
          search radius) - real water sealed off by the Dirichlet row
          cloud, solvable by the per-component zero-mean anchoring chain.

        Mode table (environment gates read per call; default OFF =
        bitwise-unchanged legacy band):

        - both gates unset: convert every candidate through the legacy
          kernel; the population mirrors stay -1 (not measured) even when
          a classification field is supplied.
        - ``HIBM_BAND_INTERIOR_ONLY=1`` (diagnostic): convert interior
          slivers only; enclosed water stays ACTIVE fluid for the
          anchoring chain. Requires ``node_kind_code`` (raises
          ``ValueError`` otherwise). The return value and the legacy
          counter then cover conversions only, so the caller's fixed-round
          loop still saturates monotonically: conversions never revert
          during the loop and the candidate test only consumes the frozen
          classification plus the growing obstacle set.
        - ``HIBM_BAND_COUNT_ONLY=1`` (A8 diagnostic, wins over the
          interior-only gate): convert nothing; the legacy counter covers
          every candidate. With ``node_kind_code`` the two populations are
          still counted.

        ``last_hibm_solid_band_interior_cells`` /
        ``last_hibm_solid_band_enclosed_water_cells`` mirror the per-sweep
        populations host-side (-1 when the sweep ran without a split).
        """
        count_only = os.environ.get("HIBM_BAND_COUNT_ONLY") == "1"
        interior_only = (
            not count_only and os.environ.get("HIBM_BAND_INTERIOR_ONLY") == "1"
        )
        if interior_only and node_kind_code is None:
            raise ValueError(
                "HIBM_BAND_INTERIOR_ONLY=1 requires the IB node classification "
                "field (node_kind_code) so the band can split interior slivers "
                "from enclosed water"
            )
        self.last_hibm_solid_band_interior_cells = -1
        self.last_hibm_solid_band_enclosed_water_cells = -1
        if node_kind_code is not None and (interior_only or count_only):
            if tuple(node_kind_code.shape) != tuple(self.obstacle.shape):
                raise ValueError(
                    "node_kind_code shape "
                    f"{tuple(node_kind_code.shape)} does not match the fluid "
                    f"cell grid {tuple(self.obstacle.shape)}"
                )
            self._mark_hibm_solid_band_population_split_kernel(
                node_kind_code,
                int(unclassified_node_code),
                1 if pressure_outlet_zmin else 0,
                0 if count_only else 1,
                1 if interior_only else 0,
            )
            self.last_hibm_solid_band_interior_cells = int(
                self.report_hibm_solid_band_interior_cells[None]
            )
            self.last_hibm_solid_band_enclosed_water_cells = int(
                self.report_hibm_solid_band_enclosed_water_cells[None]
            )
        else:
            self._mark_hibm_solid_band_nonprojectable_cells_kernel(
                1 if pressure_outlet_zmin else 0,
                0 if count_only else 1,
            )
        marked = int(self.report_hibm_solid_band_nonprojectable_cells[None])
        self.last_hibm_solid_band_marked_increment = marked
        return marked

    @ti.func
    def _pressure_outlet_reachable_neighbor_exists(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        reachable = False
        if (
            i > 0
            and self.hibm_pressure_outlet_reachable[i - 1, j, k] != 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            reachable = True
        if (
            i < self.nx - 1
            and self.hibm_pressure_outlet_reachable[i + 1, j, k] != 0
            and self.velocity_dirichlet_boundary_active[i + 1, j, k] == 0
        ):
            reachable = True
        if (
            j > 0
            and self.hibm_pressure_outlet_reachable[i, j - 1, k] != 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            reachable = True
        if (
            j < self.ny - 1
            and self.hibm_pressure_outlet_reachable[i, j + 1, k] != 0
            and self.velocity_dirichlet_boundary_active[i, j + 1, k] == 0
        ):
            reachable = True
        if (
            k > 0
            and self.hibm_pressure_outlet_reachable[i, j, k - 1] != 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            reachable = True
        if (
            k < self.nz - 1
            and self.hibm_pressure_outlet_reachable[i, j, k + 1] != 0
            and self.velocity_dirichlet_boundary_active[i, j, k + 1] == 0
        ):
            reachable = True
        return reachable

    @ti.kernel
    def _hibm_reachability_pattern_checksum_kernel(self) -> ti.f64:
        total = ti.cast(0.0, ti.f64)
        for i, j, k in self.obstacle:
            linear = ti.cast((i * self.ny + j) * self.nz + k, ti.f64)
            if self.velocity_dirichlet_boundary_active[i, j, k] != 0:
                total += linear * 3.0 + 1.0
            if self.obstacle[i, j, k] != 0:
                total += linear * 7.0 + 5.0
        return total

    @ti.kernel
    def _init_hibm_pressure_outlet_reachable_kernel(self):
        for i, j, k in self.obstacle:
            self.hibm_pressure_outlet_reachable[i, j, k] = 0
            self.hibm_pressure_outlet_reachable_next[i, j, k] = 0
            if (
                self.obstacle[i, j, k] == 0
                and k == 0
                and self.velocity_dirichlet_boundary_active[i, j, k] == 0
            ):
                self.hibm_pressure_outlet_reachable[i, j, k] = 1
                self.hibm_pressure_outlet_reachable_next[i, j, k] = 1

    @ti.kernel
    def _expand_hibm_pressure_outlet_reachable_kernel(self):
        for i, j, k in self.obstacle:
            reachable = self.hibm_pressure_outlet_reachable[i, j, k]
            if (
                self.obstacle[i, j, k] == 0
                and reachable == 0
                and self._pressure_outlet_reachable_neighbor_exists(i, j, k)
            ):
                reachable = 1
            self.hibm_pressure_outlet_reachable_next[i, j, k] = reachable

    @ti.kernel
    def _commit_hibm_pressure_outlet_reachable_kernel(self):
        for i, j, k in self.obstacle:
            self.hibm_pressure_outlet_reachable[i, j, k] = (
                self.hibm_pressure_outlet_reachable_next[i, j, k]
            )

    @ti.kernel
    def _count_hibm_pressure_outlet_reachable_kernel(self) -> ti.i32:
        count = 0
        for i, j, k in self.obstacle:
            if self.hibm_pressure_outlet_reachable[i, j, k] != 0:
                count += 1
        return count

    @ti.kernel
    def _count_hibm_pressure_outlet_unreached_cells_kernel(self):
        self.report_hibm_pressure_disconnected_nonprojectable_cells[None] = 0
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                ti.atomic_add(
                    self.report_hibm_pressure_disconnected_nonprojectable_cells[None],
                    1,
                )

    def mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
        self,
        *,
        pressure_outlet_zmin: bool = False,
    ) -> int:
        """Compute z-min outlet pressure reachability and report unreached cells.

        Unreached active fluid cells are NOT converted into obstacle cells and
        their velocity is preserved. The pressure solve anchors the unreached
        set's nullspace through a set-restricted zero-mean projection instead
        of freezing the cells out of the flow solution.
        """
        if not pressure_outlet_zmin:
            self.last_hibm_pressure_unreached_cell_count = 0
            self.last_hibm_pressure_reachability_converged = True
            self.last_hibm_pressure_reachability_sweeps = 0
            self.last_hibm_pressure_reachability_reused = False
            self._hibm_pressure_unreached_count = 0
            self.last_hibm_pressure_unreached_component_count = 0
            self._hibm_pressure_unreached_component_count = 0
            self.last_hibm_pressure_unreached_component_overflow = False
            self.last_hibm_pressure_component_labels_converged = True
            self.last_hibm_unreached_cells_with_interface_diagonal = 0
            self.last_hibm_unreached_cells_with_interface_coupling = 0
            self.last_hibm_unreached_components_with_interface_hits = 0
            self._hibm_reachability_checksum = None
            return 0
        # Reuse is opt-in only: an A/B resume on 2026-06-11 proved the checksum
        # skip corrupted the step-163 guard state in the 2-second squid run
        # (clean once disabled), so a full recompute is the safe default. The
        # full-grid checksum kernel only runs when reuse is opted in (R2-M1);
        # with reuse disabled the stored checksum stays None, which preserves
        # the always-recompute semantics below.
        reuse_enabled = os.environ.get("HIBM_ENABLE_REACHABILITY_REUSE") == "1"
        pattern_checksum: float | None = None
        if reuse_enabled:
            pattern_checksum = float(self._hibm_reachability_pattern_checksum_kernel())
            if (
                self._hibm_reachability_checksum is not None
                and pattern_checksum == self._hibm_reachability_checksum
            ):
                self.last_hibm_pressure_reachability_reused = True
                return int(self._hibm_pressure_unreached_count)
        self._hibm_reachability_checksum = None
        self.last_hibm_pressure_unreached_cell_count = 0
        self.last_hibm_pressure_reachability_converged = True
        self.last_hibm_pressure_reachability_sweeps = 0
        self.last_hibm_pressure_reachability_reused = False
        self.last_hibm_pressure_component_labels_converged = True
        self.last_hibm_unreached_cells_with_interface_diagonal = 0
        self.last_hibm_unreached_cells_with_interface_coupling = 0
        self.last_hibm_unreached_components_with_interface_hits = 0
        self._hibm_pressure_unreached_count = 0
        self._init_hibm_pressure_outlet_reachable_kernel()
        sweep_block = max(1, int(self.nx + self.ny + self.nz))
        max_blocks = 32
        previous_reachable = -1
        converged = False
        for _ in range(max_blocks):
            for _ in range(sweep_block):
                self._expand_hibm_pressure_outlet_reachable_kernel()
                self._commit_hibm_pressure_outlet_reachable_kernel()
            self.last_hibm_pressure_reachability_sweeps += sweep_block
            reachable = int(self._count_hibm_pressure_outlet_reachable_kernel())
            if reachable == previous_reachable:
                converged = True
                break
            previous_reachable = reachable
        self.last_hibm_pressure_reachability_converged = converged
        self._count_hibm_pressure_outlet_unreached_cells_kernel()
        unreached = int(
            self.report_hibm_pressure_disconnected_nonprojectable_cells[None]
        )
        self.last_hibm_pressure_unreached_cell_count = unreached
        self._hibm_pressure_unreached_count = unreached
        self.last_hibm_pressure_unreached_component_count = 0
        self._hibm_pressure_unreached_component_count = 0
        self.last_hibm_pressure_unreached_component_overflow = False
        if unreached > 0:
            self._init_hibm_unreached_component_labels_kernel()
            labels_converged = False
            for _ in range(max_blocks):
                for _ in range(sweep_block):
                    self._propagate_hibm_unreached_component_labels_kernel()
                if int(self._propagate_hibm_unreached_component_labels_kernel()) == 0:
                    labels_converged = True
                    break
            # R2-M5: False means the block budget ran out before the label
            # propagation reached a fixed point, so one physical component may
            # still carry several labels (partial merge).
            self.last_hibm_pressure_component_labels_converged = labels_converged
            component_count = 0
            for component_index in range(32):
                self._scan_min_unreached_raw_label_kernel()
                raw_label = int(self.cg_unreached_component_scan[None])
                if raw_label >= (1 << 30):
                    break
                self._assign_unreached_component_id_kernel(
                    raw_label,
                    -(component_index + 1),
                )
                component_count += 1
            if component_count == 32:
                self._scan_min_unreached_raw_label_kernel()
                if int(self.cg_unreached_component_scan[None]) < (1 << 30):
                    self.last_hibm_pressure_unreached_component_overflow = True
            self.last_hibm_pressure_unreached_component_count = component_count
            self._hibm_pressure_unreached_component_count = component_count
        self._hibm_reachability_checksum = pattern_checksum
        return unreached

    @ti.kernel
    def _init_hibm_converted_cell_pressure_fill_kernel(self):
        for i, j, k in self.hibm_pressure_filled:
            self.hibm_pressure_filled[i, j, k] = 0
            self.hibm_pressure_filled_next[i, j, k] = 0
            self.hibm_pressure_fill_next[i, j, k] = self.pressure[i, j, k]

    @ti.kernel
    def _expand_hibm_converted_cell_pressure_fill_kernel(self):
        # One Jacobi sweep (read pressure / filled, write the *_next
        # buffers): every hibm-converted non-base cell with at least one
        # available neighbor recomputes pressure = mean(available
        # neighbor pressures) and is marked filled. Available = a
        # 6-neighbor that is solved water (obstacle == 0) or a converted
        # cell already marked in a PREVIOUS sweep, so the fill front
        # advances exactly one cell layer per sweep from the solved
        # water into the sealed interior. Cells with no available
        # neighbor keep their value and stay unmarked. Non-converted
        # cells are exact copy-through.
        for i, j, k in self.obstacle:
            next_pressure = self.pressure[i, j, k]
            next_filled = self.hibm_pressure_filled[i, j, k]
            if (
                self.obstacle[i, j, k] != 0
                and self.hibm_base_obstacle[i, j, k] == 0
            ):
                neighbor_sum = 0.0
                neighbor_count = 0
                for offset in ti.static(
                    (
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    )
                ):
                    ni = i + offset[0]
                    nj = j + offset[1]
                    nk = k + offset[2]
                    if (
                        0 <= ni
                        and ni < self.nx
                        and 0 <= nj
                        and nj < self.ny
                        and 0 <= nk
                        and nk < self.nz
                    ):
                        if self.obstacle[ni, nj, nk] == 0:
                            neighbor_sum += self.pressure[ni, nj, nk]
                            neighbor_count += 1
                        elif (
                            self.hibm_base_obstacle[ni, nj, nk] == 0
                            and self.hibm_pressure_filled[ni, nj, nk] != 0
                        ):
                            neighbor_sum += self.pressure[ni, nj, nk]
                            neighbor_count += 1
                if neighbor_count > 0:
                    next_pressure = neighbor_sum / ti.cast(
                        neighbor_count,
                        ti.f32,
                    )
                    next_filled = 1
            self.hibm_pressure_fill_next[i, j, k] = next_pressure
            self.hibm_pressure_filled_next[i, j, k] = next_filled

    @ti.kernel
    def _commit_hibm_converted_cell_pressure_fill_kernel(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = self.hibm_pressure_fill_next[i, j, k]
            self.hibm_pressure_filled[i, j, k] = self.hibm_pressure_filled_next[
                i,
                j,
                k,
            ]

    @ti.kernel
    def _count_hibm_pressure_filled_cells_kernel(self):
        self.report_hibm_pressure_filled_cells[None] = 0
        for i, j, k in self.hibm_pressure_filled:
            if self.hibm_pressure_filled[i, j, k] != 0:
                ti.atomic_add(self.report_hibm_pressure_filled_cells[None], 1)

    def fill_hibm_converted_cell_pressures(self, sweeps: int = 8) -> int:
        """Back-fill stale pressures of hibm-converted cells (S2-A8'').

        The band's full conversion is the correct projection behavior
        (zero-correctable cells are zero matrix rows), but converted
        cells drop out of the pressure solve, so their ``pressure``
        values go stale exactly where the dedicated sampling view
        (:meth:`build_hibm_sampling_obstacle`) re-exposes sealed water to
        the closure stress sampler. This iterative 6-neighbor average
        (Jacobi expand/commit double buffer, one cell layer per sweep
        from the solved water inward) replaces the stale values with a
        diffused estimate of the adjacent solved-water pressure.

        Targets exactly the hibm-converted non-base population
        (``obstacle != 0 and hibm_base_obstacle == 0``): solved water and
        base geometry obstacles are exact copy-through. Cells never
        reached within ``sweeps`` (no available neighbor chain to solved
        water) keep their value and stay unmarked.

        Returns the number of filled cells and mirrors it host-side in
        ``last_hibm_pressure_filled_cell_count`` (-1 until the first
        call; reset by :meth:`restore_state`). NOT mounted inside
        :meth:`project` - the HIBM assemble calls it explicitly after
        the projection returns and before stress sampling, only when the
        far-pressure closure is enabled, so the default projection path
        stays bitwise-unchanged.
        """
        sweep_count = int(sweeps)
        if sweep_count <= 0:
            raise ValueError("sweeps must be positive")
        self._ensure_hibm_base_obstacle()
        self._init_hibm_converted_cell_pressure_fill_kernel()
        for _ in range(sweep_count):
            self._expand_hibm_converted_cell_pressure_fill_kernel()
            self._commit_hibm_converted_cell_pressure_fill_kernel()
        self._count_hibm_pressure_filled_cells_kernel()
        filled = int(self.report_hibm_pressure_filled_cells[None])
        self.last_hibm_pressure_filled_cell_count = filled
        return filled

    @ti.kernel
    def _build_hibm_sampling_obstacle_kernel(
        self,
        node_kind_code: ti.template(),
        unclassified_node_code: ti.i32,
    ):
        for i, j, k in self.sampling_obstacle:
            sampling_dry = 0
            if (
                self.hibm_base_obstacle[i, j, k] != 0
                or node_kind_code[i, j, k] != unclassified_node_code
            ):
                sampling_dry = 1
            self.sampling_obstacle[i, j, k] = sampling_dry

    def build_hibm_sampling_obstacle(
        self,
        node_kind_code,
        *,
        unclassified_node_code: int = 0,
    ) -> None:
        """Build the dedicated closure stress-sampling view (S2-A8'').

        Truth table (``sampling_obstacle = 1`` means dry for sampling):

        - base geometry obstacle (``hibm_base_obstacle != 0``): dry;
        - classified row-cloud envelope (``node_kind_code !=
          unclassified_node_code``, regardless of conversion state): dry
          - the A8 experiment proved opening the envelope makes every
          marker sample zero-pressure dead water on both sides;
        - everything else (free solved water AND NONE-classified
          hibm-converted sealed water): samplable water - the converted
          population carries the back-filled pressure from
          :meth:`fill_hibm_converted_cell_pressures`.

        The view is consumed only by the closure stress sampling
        (``sampling_obstacle_field`` keyword); the projection, no-slip
        residual, Neumann gradient sampling and every other consumer
        keep reading ``obstacle``.
        """
        if tuple(node_kind_code.shape) != tuple(self.obstacle.shape):
            raise ValueError(
                "node_kind_code shape "
                f"{tuple(node_kind_code.shape)} does not match the fluid "
                f"cell grid {tuple(self.obstacle.shape)}"
            )
        self._ensure_hibm_base_obstacle()
        self._build_hibm_sampling_obstacle_kernel(
            node_kind_code,
            int(unclassified_node_code),
        )

    @ti.kernel
    def _save_state_kernel(self):
        for i, j, k in self.velocity:
            self.saved_velocity[i, j, k] = self.velocity[i, j, k]
            self.saved_pressure[i, j, k] = self.pressure[i, j, k]

    @ti.kernel
    def _restore_state_kernel(self):
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = self.saved_velocity[i, j, k]
            self.velocity_prev[i, j, k] = self.saved_velocity[i, j, k]
            self.pressure[i, j, k] = self.saved_pressure[i, j, k]
            self.pressure_tmp[i, j, k] = self.saved_pressure[i, j, k]
            self.pressure_accum[i, j, k] = self.saved_pressure[i, j, k]
            self.volume_source_s[i, j, k] = 0.0
            self.pressure_interface_matrix_diagonal[i, j, k] = 0.0
            self.pressure_interface_matrix_rhs[i, j, k] = 0.0
            self.pressure_interface_coupling_active[i, j, k] = 0
            self.pressure_interface_coupling_neighbor[i, j, k] = ti.Vector([0, 0, 0])
            self.pressure_interface_coupling_coefficient[i, j, k] = 0.0
            self.force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_weight[i, j, k] = 0.0
            self.velocity_constraint_primary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_primary_weight[i, j, k] = 0.0
            self.velocity_constraint_secondary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_secondary_weight[i, j, k] = 0.0
            self.velocity_dirichlet_boundary_active[i, j, k] = 0
            self.velocity_dirichlet_boundary_value_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0

    def save_state(self) -> None:
        self._save_state_kernel()

    def restore_state(self) -> None:
        self._restore_state_kernel()
        self._hibm_pressure_unreached_count = 0
        self.last_hibm_pressure_unreached_cell_count = 0
        self._hibm_pressure_unreached_component_count = 0
        self.last_hibm_pressure_unreached_component_count = 0
        self.last_hibm_pressure_unreached_component_overflow = False
        self.last_hibm_pressure_reachability_converged = True
        self.last_hibm_pressure_reachability_sweeps = 0
        self.last_hibm_pressure_reachability_reused = False
        self.last_hibm_pressure_component_labels_converged = True
        self.last_hibm_unreached_cells_with_interface_diagonal = 0
        self.last_hibm_unreached_cells_with_interface_coupling = 0
        self.last_hibm_unreached_components_with_interface_hits = 0
        self.last_hibm_solid_band_marked_increment = 0
        self.last_hibm_solid_band_interior_cells = -1
        self.last_hibm_solid_band_enclosed_water_cells = -1
        self.last_hibm_pressure_filled_cell_count = -1
        self._hibm_reachability_checksum = None

    @ti.kernel
    def _clear_pressure_interface_matrix_terms_kernel(self):
        for i, j, k in self.pressure_interface_matrix_diagonal:
            self.pressure_interface_matrix_diagonal[i, j, k] = 0.0
            self.pressure_interface_matrix_rhs[i, j, k] = 0.0
            self.pressure_interface_coupling_active[i, j, k] = 0
            self.pressure_interface_coupling_neighbor[i, j, k] = ti.Vector([0, 0, 0])
            self.pressure_interface_coupling_coefficient[i, j, k] = 0.0

    def clear_pressure_interface_matrix_terms(self) -> None:
        self._clear_pressure_interface_matrix_terms_kernel()

    @ti.kernel
    def _clear_pressure_interface_matrix_rhs_kernel(self):
        for i, j, k in self.pressure_interface_matrix_rhs:
            self.pressure_interface_matrix_rhs[i, j, k] = 0.0

    @ti.kernel
    def _clear_velocity_dirichlet_boundary_rows_kernel(self):
        for i, j, k in self.velocity_dirichlet_boundary_active:
            self.velocity_dirichlet_boundary_active[i, j, k] = 0
            self.velocity_dirichlet_boundary_value_mps[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
            self.velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0
        self.report_velocity_dirichlet_boundary_cells[None] = 0
        self.report_velocity_dirichlet_boundary_delta_sum[None] = 0.0
        self.report_velocity_dirichlet_boundary_delta_max[None] = 0.0
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s[None] = ti.Vector(
            [0.0, 0.0, 0.0]
        )

    def clear_velocity_dirichlet_boundary_rows(self) -> None:
        self._clear_velocity_dirichlet_boundary_rows_kernel()

    @ti.kernel
    def _clear_force_kernel(self):
        for i, j, k in self.force:
            self.force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        self.report_surface_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_force_spread_relative_error[None] = 0.0
        self.report_active_force_cells[None] = 0
        self.report_grid_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_impulse_relative_error[None] = 0.0
        self.report_active_velocity_cells[None] = 0

    def clear_force(self) -> None:
        self._clear_force_kernel()

    @ti.kernel
    def _clear_volume_source_kernel(self):
        for i, j, k in self.volume_source_s:
            self.volume_source_s[i, j, k] = 0.0

    def clear_volume_source(self) -> None:
        self._clear_volume_source_kernel()

    @ti.kernel
    def _copy_velocity_to_prev_kernel(self):
        for i, j, k in self.velocity:
            self.velocity_prev[i, j, k] = self.velocity[i, j, k]

    @ti.func
    def _sample_velocity_prev_trilinear(self, gx, gy, gz, fallback):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), self.nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), self.ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), self.nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.Vector([0.0, 0.0, 0.0])
        fluid_weight = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if self.obstacle[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * self.velocity_prev[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        else:
            value = fallback
        return value

    @ti.func
    def _axis_grid_coordinate_device(
        self,
        value,
        faces: ti.template(),
        centers: ti.template(),
        count: ti.i32,
    ):
        coordinate = 0.0
        if value <= centers[0]:
            half_width = ti.max(centers[0] - faces[0], 1.0e-18)
            coordinate = -0.5 * (centers[0] - value) / half_width
        elif value >= centers[count - 1]:
            half_width = ti.max(faces[count] - centers[count - 1], 1.0e-18)
            coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (value - centers[count - 1]) / half_width
        else:
            lower = 0
            upper = count - 1
            while upper - lower > 1:
                middle = (lower + upper) // 2
                if value >= centers[middle]:
                    lower = middle
                else:
                    upper = middle
            upper = ti.min(lower + 1, count - 1)
            distance = ti.max(centers[upper] - centers[lower], 1.0e-18)
            coordinate = ti.cast(lower, ti.f32) + (value - centers[lower]) / distance
        return coordinate

    @ti.func
    def _grid_coordinate_x(self, x):
        return self._axis_grid_coordinate_device(x, self.cell_face_x_m, self.cell_center_x_m, self.nx)

    @ti.func
    def _grid_coordinate_y(self, y):
        return self._axis_grid_coordinate_device(y, self.cell_face_y_m, self.cell_center_y_m, self.ny)

    @ti.func
    def _grid_coordinate_z(self, z):
        return self._axis_grid_coordinate_device(z, self.cell_face_z_m, self.cell_center_z_m, self.nz)

    @ti.kernel
    def _predict_kernel(
        self,
        dt_s: ti.f32,
        nu_m2_s: ti.f32,
        advection_scheme_code: ti.i32,
    ):
        for i, j, k in self.velocity:
            if self.obstacle[i, j, k] == 1:
                self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            else:
                center = self.velocity_prev[i, j, k]
                x = self.cell_center_x_m[i]
                y = self.cell_center_y_m[j]
                z = self.cell_center_z_m[k]
                trace_velocity = center
                if advection_scheme_code == 1:
                    mid_x = x - 0.5 * dt_s * center.x
                    mid_y = y - 0.5 * dt_s * center.y
                    mid_z = z - 0.5 * dt_s * center.z
                    mid_gx = self._grid_coordinate_x(mid_x)
                    mid_gy = self._grid_coordinate_y(mid_y)
                    mid_gz = self._grid_coordinate_z(mid_z)
                    trace_velocity = self._sample_velocity_prev_trilinear(
                        mid_gx,
                        mid_gy,
                        mid_gz,
                        center,
                    )
                back_x = x - dt_s * trace_velocity.x
                back_y = y - dt_s * trace_velocity.y
                back_z = z - dt_s * trace_velocity.z
                gx = self._grid_coordinate_x(back_x)
                gy = self._grid_coordinate_y(back_y)
                gz = self._grid_coordinate_z(back_z)
                advected = self._sample_velocity_prev_trilinear(gx, gy, gz, center)

                im = ti.max(i - 1, 0)
                ip = ti.min(i + 1, self.nx - 1)
                jm = ti.max(j - 1, 0)
                jp = ti.min(j + 1, self.ny - 1)
                km = ti.max(k - 1, 0)
                kp = ti.min(k + 1, self.nz - 1)
                flux_x_backward = ti.Vector([0.0, 0.0, 0.0])
                flux_x_forward = ti.Vector([0.0, 0.0, 0.0])
                flux_y_backward = ti.Vector([0.0, 0.0, 0.0])
                flux_y_forward = ti.Vector([0.0, 0.0, 0.0])
                flux_z_backward = ti.Vector([0.0, 0.0, 0.0])
                flux_z_forward = ti.Vector([0.0, 0.0, 0.0])
                if i > 0 and self.obstacle[im, j, k] == 0:
                    flux_x_backward = (center - self.velocity_prev[im, j, k]) / self.center_distance_x_m[i]
                if i < self.nx - 1 and self.obstacle[ip, j, k] == 0:
                    flux_x_forward = (self.velocity_prev[ip, j, k] - center) / self.center_distance_x_m[i + 1]
                if j > 0 and self.obstacle[i, jm, k] == 0:
                    flux_y_backward = (center - self.velocity_prev[i, jm, k]) / self.center_distance_y_m[j]
                if j < self.ny - 1 and self.obstacle[i, jp, k] == 0:
                    flux_y_forward = (self.velocity_prev[i, jp, k] - center) / self.center_distance_y_m[j + 1]
                if k > 0 and self.obstacle[i, j, km] == 0:
                    flux_z_backward = (center - self.velocity_prev[i, j, km]) / self.center_distance_z_m[k]
                if k < self.nz - 1 and self.obstacle[i, j, kp] == 0:
                    flux_z_forward = (self.velocity_prev[i, j, kp] - center) / self.center_distance_z_m[k + 1]
                laplacian = (
                    (flux_x_forward - flux_x_backward) / self.cell_width_x_m[i]
                    + (flux_y_forward - flux_y_backward) / self.cell_width_y_m[j]
                    + (flux_z_forward - flux_z_backward) / self.cell_width_z_m[k]
                )
                self.velocity[i, j, k] = advected + dt_s * nu_m2_s * laplacian

    def predict(self, dt_s: float | None = None, *, advection_scheme: str = "euler") -> None:
        step_dt_s = self.dt if dt_s is None else float(dt_s)
        if step_dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        scheme = str(advection_scheme).lower()
        if scheme not in {"euler", "rk2"}:
            raise ValueError(f"unsupported advection_scheme: {advection_scheme!r}")
        self._copy_velocity_to_prev_kernel()
        self._predict_kernel(
            float(step_dt_s),
            float(self.mu / self.rho),
            1 if scheme == "rk2" else 0,
        )

    @ti.kernel
    def _clear_velocity_constraints_kernel(self):
        for i, j, k in self.velocity_constraint_weight:
            self.velocity_constraint_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_weight[i, j, k] = 0.0
            self.velocity_constraint_primary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_primary_weight[i, j, k] = 0.0
            self.velocity_constraint_secondary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_secondary_weight[i, j, k] = 0.0
        self.report_velocity_constraint_cells[None] = 0
        self.report_velocity_constraint_delta_sum[None] = 0.0
        self.report_velocity_constraint_delta_max[None] = 0.0
        self.report_velocity_constraint_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_primary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_secondary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])

    def clear_velocity_constraints(self) -> None:
        self._clear_velocity_constraints_kernel()

    @ti.kernel
    def _apply_velocity_dirichlet_boundary_rows_kernel(self, read_report: ti.i32):
        self.report_velocity_dirichlet_boundary_cells[None] = 0
        self.report_velocity_dirichlet_boundary_delta_sum[None] = 0.0
        self.report_velocity_dirichlet_boundary_delta_max[None] = 0.0
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s[None] = ti.Vector(
            [0.0, 0.0, 0.0]
        )
        for i, j, k in self.velocity:
            if (
                self.velocity_dirichlet_boundary_active[i, j, k] != 0
                and self.obstacle[i, j, k] == 0
            ):
                old_velocity = self.velocity[i, j, k]
                new_velocity = self.velocity_dirichlet_boundary_value_mps[i, j, k]
                velocity_delta = new_velocity - old_velocity
                self.velocity[i, j, k] = new_velocity
                if read_report != 0:
                    delta = velocity_delta.norm()
                    momentum_delta = (
                        self.rho
                        * velocity_delta
                        * self._cell_volume_m3(i, j, k)
                    )
                    ti.atomic_add(
                        self.report_velocity_dirichlet_boundary_cells[None],
                        1,
                    )
                    ti.atomic_add(
                        self.report_velocity_dirichlet_boundary_delta_sum[None],
                        delta,
                    )
                    ti.atomic_max(
                        self.report_velocity_dirichlet_boundary_delta_max[None],
                        delta,
                    )
                    self._atomic_add_report_vector(
                        self.report_velocity_dirichlet_boundary_momentum_delta_n_s,
                        momentum_delta,
                    )

    def apply_velocity_dirichlet_boundary_rows(
        self,
        *,
        read_report: bool = True,
    ) -> VelocityDirichletBoundaryReport | None:
        self._apply_velocity_dirichlet_boundary_rows_kernel(1 if read_report else 0)
        if not read_report:
            return None
        return self.velocity_dirichlet_boundary_report()

    def velocity_dirichlet_boundary_report(self) -> VelocityDirichletBoundaryReport:
        active_cells = int(self.report_velocity_dirichlet_boundary_cells[None])
        mean_delta = (
            float(self.report_velocity_dirichlet_boundary_delta_sum[None]) / active_cells
            if active_cells > 0
            else 0.0
        )
        return VelocityDirichletBoundaryReport(
            active_cells=active_cells,
            max_delta_mps=float(self.report_velocity_dirichlet_boundary_delta_max[None]),
            mean_delta_mps=mean_delta,
            momentum_delta_n_s=self._read_vector(
                self.report_velocity_dirichlet_boundary_momentum_delta_n_s
            ),
        )

    @ti.kernel
    def reset_velocity_constraint_impulse_accumulator(self):
        self.velocity_constraint_primary_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.velocity_constraint_secondary_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _apply_velocity_constraints_kernel(
        self,
        blend: ti.f32,
        solid_mobility_ratio: ti.f32,
        read_report: ti.i32,
    ):
        self.report_velocity_constraint_cells[None] = 0
        self.report_velocity_constraint_delta_sum[None] = 0.0
        self.report_velocity_constraint_delta_max[None] = 0.0
        self.report_velocity_constraint_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_primary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_secondary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        for i, j, k in self.velocity:
            weight = self.velocity_constraint_weight[i, j, k]
            if weight > 0.0 and self.obstacle[i, j, k] == 0:
                target = self.velocity_constraint_sum[i, j, k] / weight
                old_velocity = self.velocity[i, j, k]
                correction_scale = blend / (1.0 + solid_mobility_ratio)
                new_velocity = old_velocity + correction_scale * (target - old_velocity)
                velocity_delta = new_velocity - old_velocity
                delta = (new_velocity - old_velocity).norm()
                self.velocity[i, j, k] = new_velocity
                momentum_delta = self.rho * velocity_delta * self._cell_volume_m3(i, j, k)
                primary_weight = self.velocity_constraint_primary_weight[i, j, k]
                secondary_weight = self.velocity_constraint_secondary_weight[i, j, k]
                primary_velocity_delta = (
                    correction_scale
                    * (
                        self.velocity_constraint_primary_sum[i, j, k]
                        - old_velocity * primary_weight
                    )
                    / weight
                )
                secondary_velocity_delta = (
                    correction_scale
                    * (
                        self.velocity_constraint_secondary_sum[i, j, k]
                        - old_velocity * secondary_weight
                    )
                    / weight
                )
                primary_momentum_delta = (
                    self.rho * primary_velocity_delta * self._cell_volume_m3(i, j, k)
                )
                secondary_momentum_delta = (
                    self.rho * secondary_velocity_delta * self._cell_volume_m3(i, j, k)
                )
                self._atomic_add_report_vector(
                    self.velocity_constraint_primary_impulse_n_s,
                    primary_momentum_delta,
                )
                self._atomic_add_report_vector(
                    self.velocity_constraint_secondary_impulse_n_s,
                    secondary_momentum_delta,
                )
                if read_report != 0:
                    ti.atomic_add(self.report_velocity_constraint_cells[None], 1)
                    ti.atomic_add(self.report_velocity_constraint_delta_sum[None], delta)
                    ti.atomic_max(self.report_velocity_constraint_delta_max[None], delta)
                    self._atomic_add_report_vector(
                        self.report_velocity_constraint_momentum_delta_n_s,
                        momentum_delta,
                    )
                    self._atomic_add_report_vector(
                        self.report_velocity_constraint_primary_momentum_delta_n_s,
                        primary_momentum_delta,
                    )
                    self._atomic_add_report_vector(
                        self.report_velocity_constraint_secondary_momentum_delta_n_s,
                        secondary_momentum_delta,
                    )

    def apply_velocity_constraints(
        self,
        blend: float = 1.0,
        *,
        solid_mobility_ratio: float = 0.0,
        read_report: bool = True,
    ) -> VelocityConstraintReport | None:
        blend_value = float(blend)
        if not 0.0 <= blend_value <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        solid_mobility_ratio_value = float(solid_mobility_ratio)
        if (
            not math.isfinite(solid_mobility_ratio_value)
            or solid_mobility_ratio_value < 0.0
        ):
            raise ValueError("solid_mobility_ratio must be a finite non-negative number")
        self._apply_velocity_constraints_kernel(
            blend_value,
            solid_mobility_ratio_value,
            1 if read_report else 0,
        )
        if not read_report:
            return None
        return self.velocity_constraint_report()

    def velocity_constraint_report(self) -> VelocityConstraintReport:
        active_cells = int(self.report_velocity_constraint_cells[None])
        mean_delta = (
            float(self.report_velocity_constraint_delta_sum[None]) / active_cells
            if active_cells > 0
            else 0.0
        )
        return VelocityConstraintReport(
            active_cells=active_cells,
            max_delta_mps=float(self.report_velocity_constraint_delta_max[None]),
            mean_delta_mps=mean_delta,
            momentum_delta_n_s=self._read_vector(
                self.report_velocity_constraint_momentum_delta_n_s
            ),
            primary_momentum_delta_n_s=self._read_vector(
                self.report_velocity_constraint_primary_momentum_delta_n_s
            ),
            secondary_momentum_delta_n_s=self._read_vector(
                self.report_velocity_constraint_secondary_momentum_delta_n_s
            ),
        )

    def velocity_constraint_impulse_report(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        self.last_velocity_constraint_impulse_host_reads += 1
        return (
            self._read_vector(self.velocity_constraint_primary_impulse_n_s),
            self._read_vector(self.velocity_constraint_secondary_impulse_n_s),
        )

    @ti.kernel
    def _set_vertical_pressure_gradient_kernel(
        self,
        reference_height_m: ti.f32,
        gradient_z_pa_per_m: ti.f32,
    ):
        for i, j, k in self.pressure:
            z = self.cell_center_z_m[k]
            self.pressure[i, j, k] = gradient_z_pa_per_m * (z - reference_height_m)

    def set_vertical_pressure_gradient(
        self,
        reference_height_m: float,
        gradient_z_pa_per_m: float,
    ) -> None:
        self._set_vertical_pressure_gradient_kernel(
            float(reference_height_m),
            float(gradient_z_pa_per_m),
        )

    @ti.kernel
    def _set_uniform_velocity_kernel(self, vx: ti.f32, vy: ti.f32, vz: ti.f32):
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = ti.Vector([vx, vy, vz])

    def set_uniform_velocity(self, velocity_mps: tuple[float, float, float]) -> None:
        self._set_uniform_velocity_kernel(
            float(velocity_mps[0]),
            float(velocity_mps[1]),
            float(velocity_mps[2]),
        )

    @ti.kernel
    def _set_sinusoidal_divergent_velocity_kernel(
        self,
        amplitude_mps: ti.f32,
    ):
        for i, j, k in self.velocity:
            x = (ti.cast(i, ti.f32) + 0.5) / ti.cast(self.nx, ti.f32)
            y = (ti.cast(j, ti.f32) + 0.5) / ti.cast(self.ny, ti.f32)
            z = (ti.cast(k, ti.f32) + 0.5) / ti.cast(self.nz, ti.f32)
            self.velocity[i, j, k] = ti.Vector(
                [
                    amplitude_mps * ti.sin(2.0 * ti.math.pi * x),
                    0.5 * amplitude_mps * ti.sin(2.0 * ti.math.pi * y),
                    -0.25 * amplitude_mps * ti.sin(2.0 * ti.math.pi * z),
                ]
            )

    def set_sinusoidal_divergent_velocity(self, amplitude_mps: float) -> None:
        if amplitude_mps <= 0.0:
            raise ValueError("amplitude_mps must be positive")
        self._set_sinusoidal_divergent_velocity_kernel(float(amplitude_mps))

    @ti.kernel
    def _set_simple_shear_velocity_kernel(
        self,
        shear_rate_s: ti.f32,
        center_y_m: ti.f32,
    ):
        for i, j, k in self.velocity:
            y = self.cell_center_y_m[j]
            self.velocity[i, j, k] = ti.Vector([shear_rate_s * (y - center_y_m), 0.0, 0.0])

    def set_simple_shear_velocity(self, shear_rate_s: float, center_y_m: float = 0.0) -> None:
        self._set_simple_shear_velocity_kernel(
            float(shear_rate_s),
            float(center_y_m),
        )

    @ti.kernel
    def _mark_sphere_kernel(
        self,
        cx: ti.f32,
        cy: ti.f32,
        cz: ti.f32,
        radius_m: ti.f32,
    ):
        radius2 = radius_m * radius_m
        for i, j, k in self.obstacle:
            x = self.cell_center_x_m[i]
            y = self.cell_center_y_m[j]
            z = self.cell_center_z_m[k]
            rx = x - cx
            ry = y - cy
            rz = z - cz
            self.obstacle[i, j, k] = 1 if rx * rx + ry * ry + rz * rz <= radius2 else 0

    def mark_sphere_obstacle(
        self,
        center_m: tuple[float, float, float],
        radius_m: float,
    ) -> None:
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        self._mark_sphere_kernel(
            float(center_m[0]),
            float(center_m[1]),
            float(center_m[2]),
            float(radius_m),
        )

    @ti.func
    def _cell_volume_m3(self, i, j, k):
        return self.cell_width_x_m[i] * self.cell_width_y_m[j] * self.cell_width_z_m[k]

    @ti.kernel
    def _pressure_interface_matrix_terms_report_kernel(self):
        self.report_pressure_interface_matrix_diagonal_integral[None] = 0.0
        self.report_pressure_interface_matrix_rhs_integral[None] = 0.0
        self.report_pressure_interface_matrix_max_abs_diagonal[None] = 0.0
        self.report_pressure_interface_matrix_active_cells[None] = 0
        for i, j, k in self.pressure_interface_matrix_diagonal:
            diagonal = self.pressure_interface_matrix_diagonal[i, j, k]
            rhs = self.pressure_interface_matrix_rhs[i, j, k]
            if ti.abs(diagonal) > 0.0 or ti.abs(rhs) > 0.0:
                ti.atomic_add(self.report_pressure_interface_matrix_active_cells[None], 1)
            cell_volume_m3 = self._cell_volume_m3(i, j, k)
            ti.atomic_add(
                self.report_pressure_interface_matrix_diagonal_integral[None],
                ti.cast(diagonal * cell_volume_m3, ti.f64),
            )
            ti.atomic_add(
                self.report_pressure_interface_matrix_rhs_integral[None],
                ti.cast(rhs * cell_volume_m3, ti.f64),
            )
            ti.atomic_max(
                self.report_pressure_interface_matrix_max_abs_diagonal[None],
                ti.abs(diagonal),
            )

    def pressure_interface_matrix_terms_report(self) -> dict[str, float | int]:
        self._pressure_interface_matrix_terms_report_kernel()
        return {
            "diagonal_integral": float(
                self.report_pressure_interface_matrix_diagonal_integral[None]
            ),
            "rhs_integral": float(self.report_pressure_interface_matrix_rhs_integral[None]),
            "max_abs_diagonal": float(
                self.report_pressure_interface_matrix_max_abs_diagonal[None]
            ),
            "active_cells": int(self.report_pressure_interface_matrix_active_cells[None]),
        }

    @ti.kernel
    def _sum_obstacle_volume_kernel(self):
        self.reduction_sum[None] = 0.0
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 1:
                ti.atomic_add(self.reduction_sum[None], self._cell_volume_m3(i, j, k))

    def obstacle_volume_m3(self) -> float:
        self._sum_obstacle_volume_kernel()
        return float(self.reduction_sum[None])

    @ti.kernel
    def _count_obstacle_cells_kernel(self):
        self.reduction_count[None] = 0
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 1:
                ti.atomic_add(self.reduction_count[None], 1)

    def obstacle_cell_count(self) -> int:
        self._count_obstacle_cells_kernel()
        return int(self.reduction_count[None])

    @ti.kernel
    def _apply_obstacle_velocity_kernel(self, vx: ti.f32, vy: ti.f32, vz: ti.f32):
        obstacle_velocity = ti.Vector([vx, vy, vz])
        for i, j, k in self.velocity:
            if self.obstacle[i, j, k] == 1:
                self.velocity[i, j, k] = obstacle_velocity

    def apply_obstacle_velocity(self, velocity_mps: tuple[float, float, float]) -> None:
        self._apply_obstacle_velocity_kernel(
            float(velocity_mps[0]),
            float(velocity_mps[1]),
            float(velocity_mps[2]),
        )

    @ti.kernel
    def _obstacle_velocity_error_kernel(self, vx: ti.f32, vy: ti.f32, vz: ti.f32):
        self.reduction_sum[None] = 0.0
        self.reduction_max[None] = 0.0
        self.reduction_count[None] = 0
        self.cleanup_target_l2_sq[None] = 0.0
        self.cleanup_required[None] = 0
        target = ti.Vector([vx, vy, vz])
        for i, j, k in self.velocity:
            if self.obstacle[i, j, k] == 1:
                diff = self.velocity[i, j, k] - target
                ti.atomic_max(self.reduction_max[None], ti.abs(diff.x))
                ti.atomic_max(self.reduction_max[None], ti.abs(diff.y))
                ti.atomic_max(self.reduction_max[None], ti.abs(diff.z))
                ti.atomic_add(self.reduction_sum[None], diff.norm())
                ti.atomic_add(self.reduction_count[None], 1)

    def obstacle_velocity_error(self, velocity_mps: tuple[float, float, float]) -> dict[str, float]:
        self._obstacle_velocity_error_kernel(
            float(velocity_mps[0]),
            float(velocity_mps[1]),
            float(velocity_mps[2]),
        )
        count = int(self.reduction_count[None])
        if count <= 0:
            return {"max_abs": 0.0, "mean_l2": 0.0}
        return {
            "max_abs": float(self.reduction_max[None]),
            "mean_l2": float(self.reduction_sum[None]) / count,
        }

    @ti.func
    def _atomic_add_vector_force(self, i, j, k, value):
        ti.atomic_add(self.force[i, j, k].x, value.x)
        ti.atomic_add(self.force[i, j, k].y, value.y)
        ti.atomic_add(self.force[i, j, k].z, value.z)

    @ti.func
    def _atomic_add_report_vector(self, field, value):
        ti.atomic_add(field[None].x, value.x)
        ti.atomic_add(field[None].y, value.y)
        ti.atomic_add(field[None].z, value.z)

    @ti.kernel
    def _spread_surface_forces_kernel(
        self,
        surface_position_m: ti.template(),
        surface_force_n: ti.template(),
        vertex_count: ti.i32,
        center_x: ti.f32,
        center_y: ti.f32,
        center_z: ti.f32,
        force_sign: ti.f32,
    ):
        for i, j, k in self.force:
            self.force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        self.report_surface_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_force_spread_relative_error[None] = 0.0
        self.report_active_force_cells[None] = 0

        center = ti.Vector([center_x, center_y, center_z])
        for vertex in range(vertex_count):
            position = surface_position_m[vertex] + center
            solid_force = surface_force_n[vertex]
            fluid_force = force_sign * solid_force
            self._atomic_add_report_vector(self.report_surface_force_n, solid_force)

            gx = self._grid_coordinate_x(position.x)
            gy = self._grid_coordinate_y(position.y)
            gz = self._grid_coordinate_z(position.z)
            base_i = ti.floor(gx, ti.i32)
            base_j = ti.floor(gy, ti.i32)
            base_k = ti.floor(gz, ti.i32)
            fx = gx - ti.cast(base_i, ti.f32)
            fy = gy - ti.cast(base_j, ti.f32)
            fz = gz - ti.cast(base_k, ti.f32)

            active_weight_sum = 0.0
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                    wx = 1.0 - fx if oi == 0 else fx
                    wy = 1.0 - fy if oj == 0 else fy
                    wz = 1.0 - fz if ok == 0 else fz
                    weight = wx * wy * wz
                    if self.obstacle[ii, jj, kk] == 0:
                        active_weight_sum += weight

            if active_weight_sum > 0.0:
                for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                    ii = base_i + oi
                    jj = base_j + oj
                    kk = base_k + ok
                    if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                        wx = 1.0 - fx if oi == 0 else fx
                        wy = 1.0 - fy if oj == 0 else fy
                        wz = 1.0 - fz if ok == 0 else fz
                        weight = wx * wy * wz
                        if self.obstacle[ii, jj, kk] == 0:
                            weight = weight / active_weight_sum
                            cell_force = fluid_force * weight
                            self._atomic_add_vector_force(
                                ii,
                                jj,
                                kk,
                                cell_force / self._cell_volume_m3(ii, jj, kk),
                            )
                            self._atomic_add_report_vector(self.report_grid_force_n, cell_force)

        expected_grid_force = force_sign * self.report_surface_force_n[None]
        denominator = ti.max(expected_grid_force.norm(), 1.0e-12)
        self.report_force_spread_relative_error[None] = (
            self.report_grid_force_n[None] - expected_grid_force
        ).norm() / denominator

        for i, j, k in self.force:
            if self.force[i, j, k].norm() > 0.0:
                ti.atomic_add(self.report_active_force_cells[None], 1)

    def spread_surface_forces(
        self,
        surface_position_m,
        surface_force_n,
        vertex_count: int,
        center_m: tuple[float, float, float],
        force_sign: float = -1.0,
    ) -> ForceSpreadingReport:
        if vertex_count <= 0:
            raise ValueError("vertex_count must be positive")
        self._spread_surface_forces_kernel(
            surface_position_m,
            surface_force_n,
            int(vertex_count),
            float(center_m[0]),
            float(center_m[1]),
            float(center_m[2]),
            float(force_sign),
        )
        return ForceSpreadingReport(
            surface_force_n=self._read_vector(self.report_surface_force_n),
            grid_force_n=self._read_vector(self.report_grid_force_n),
            action_reaction_relative_error=float(self.report_force_spread_relative_error[None]),
            active_grid_cells=int(self.report_active_force_cells[None]),
        )

    @ti.kernel
    def _apply_body_force_kernel(
        self,
        dt_s: ti.f32,
        density_kgm3: ti.f32,
    ):
        self.report_grid_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_impulse_relative_error[None] = 0.0
        self.report_active_velocity_cells[None] = 0

        for i, j, k in self.velocity:
            force_density_n_m3 = self.force[i, j, k]
            cell_volume_m3 = self._cell_volume_m3(i, j, k)
            cell_impulse = force_density_n_m3 * cell_volume_m3 * dt_s
            if self.obstacle[i, j, k] == 0:
                delta_v = force_density_n_m3 * (dt_s / density_kgm3)
                self.velocity[i, j, k] += delta_v
                momentum_delta = density_kgm3 * delta_v * cell_volume_m3
                self._atomic_add_report_vector(self.report_momentum_delta_n_s, momentum_delta)
                self._atomic_add_report_vector(self.report_grid_impulse_n_s, cell_impulse)
                if delta_v.norm() > 0.0:
                    ti.atomic_add(self.report_active_velocity_cells[None], 1)

        denominator = ti.max(self.report_grid_impulse_n_s[None].norm(), 1.0e-12)
        self.report_impulse_relative_error[None] = (
            self.report_momentum_delta_n_s[None] - self.report_grid_impulse_n_s[None]
        ).norm() / denominator

    def apply_body_force(
        self,
        dt_s: float | None = None,
        *,
        read_report: bool = True,
    ) -> FluidImpulseReport | None:
        step_dt_s = self.dt if dt_s is None else float(dt_s)
        if step_dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self._apply_body_force_kernel(
            float(step_dt_s),
            float(self.rho),
        )
        if not read_report:
            return None
        return self.body_force_impulse_report()

    def body_force_impulse_report(self) -> FluidImpulseReport:
        return FluidImpulseReport(
            grid_impulse_n_s=self._read_vector(self.report_grid_impulse_n_s),
            momentum_delta_n_s=self._read_vector(self.report_momentum_delta_n_s),
            impulse_relative_error=float(self.report_impulse_relative_error[None]),
            active_velocity_cells=int(self.report_active_velocity_cells[None]),
        )

    @ti.func
    def _sample_pressure_trilinear(self, gx, gy, gz):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), self.nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), self.ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), self.nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            value += wx * wy * wz * self.pressure[i0 + oi, j0 + oj, k0 + ok]
        return value

    @ti.func
    def _sample_velocity_trilinear(self, gx, gy, gz):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), self.nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), self.ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), self.nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.Vector([0.0, 0.0, 0.0])
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            value += wx * wy * wz * self.velocity[i0 + oi, j0 + oj, k0 + ok]
        return value

    @ti.kernel
    def _write_xz_plane_kernel(
        self,
        pressure_plane: ti.template(),
        velocity_u_plane: ti.template(),
        velocity_w_plane: ti.template(),
        plane_nx: ti.i32,
        plane_nz: ti.i32,
        x_min: ti.f32,
        x_max: ti.f32,
        y_m: ti.f32,
        z_min: ti.f32,
        z_max: ti.f32,
        xmin: ti.f32,
        ymin: ti.f32,
        zmin: ti.f32,
        dx: ti.f32,
        dy: ti.f32,
        dz: ti.f32,
    ):
        for k, i in pressure_plane:
            x = x_min + (x_max - x_min) * ti.cast(i, ti.f32) / ti.cast(plane_nx - 1, ti.f32)
            z = z_min + (z_max - z_min) * ti.cast(k, ti.f32) / ti.cast(plane_nz - 1, ti.f32)
            gx = self._grid_coordinate_x(x)
            gy = self._grid_coordinate_y(y_m)
            gz = self._grid_coordinate_z(z)
            pressure_plane[k, i] = self._sample_pressure_trilinear(gx, gy, gz)
            velocity = self._sample_velocity_trilinear(gx, gy, gz)
            velocity_u_plane[k, i] = velocity.x
            velocity_w_plane[k, i] = velocity.z

    def write_xz_plane(
        self,
        pressure_plane,
        velocity_u_plane,
        velocity_w_plane,
        plane_shape: tuple[int, int],
        x_bounds_m: tuple[float, float],
        y_m: float,
        z_bounds_m: tuple[float, float],
    ) -> None:
        plane_nz, plane_nx = plane_shape
        if plane_nx < 2 or plane_nz < 2:
            raise ValueError("plane_shape must be at least (2, 2)")
        self._write_xz_plane_kernel(
            pressure_plane,
            velocity_u_plane,
            velocity_w_plane,
            int(plane_nx),
            int(plane_nz),
            float(x_bounds_m[0]),
            float(x_bounds_m[1]),
            float(y_m),
            float(z_bounds_m[0]),
            float(z_bounds_m[1]),
            float(self.bounds_min[0]),
            float(self.bounds_min[1]),
            float(self.bounds_min[2]),
            float(self.dx),
            float(self.dy),
            float(self.dz),
        )

    @ti.kernel
    def _compute_divergence_kernel(
        self,
        pressure_outlet_zmin: ti.i32,
    ):
        for i, j, k in self.divergence:
            if self.obstacle[i, j, k] == 1:
                self.divergence[i, j, k] = 0.0
            else:
                left_velocity_x = 0.0
                right_velocity_x = 0.0
                if i > 0 and self.obstacle[i - 1, j, k] == 0:
                    left_velocity_x = self.velocity[i, j, k].x
                if i < self.nx - 1 and self.obstacle[i + 1, j, k] == 0:
                    right_velocity_x = self.velocity[i + 1, j, k].x

                back_velocity_y = 0.0
                front_velocity_y = 0.0
                if j > 0 and self.obstacle[i, j - 1, k] == 0:
                    back_velocity_y = self.velocity[i, j, k].y
                if j < self.ny - 1 and self.obstacle[i, j + 1, k] == 0:
                    front_velocity_y = self.velocity[i, j + 1, k].y

                bottom_velocity_z = 0.0
                top_velocity_z = 0.0
                if pressure_outlet_zmin == 1 and k == 0:
                    bottom_velocity_z = self.velocity[i, j, k].z
                elif k > 0 and self.obstacle[i, j, k - 1] == 0:
                    bottom_velocity_z = self.velocity[i, j, k].z
                if k < self.nz - 1 and self.obstacle[i, j, k + 1] == 0:
                    top_velocity_z = self.velocity[i, j, k + 1].z

                self.divergence[i, j, k] = (
                    (right_velocity_x - left_velocity_x) / self.cell_width_x_m[i]
                    + (front_velocity_y - back_velocity_y) / self.cell_width_y_m[j]
                    + (top_velocity_z - bottom_velocity_z) / self.cell_width_z_m[k]
                )

    def compute_divergence(self, *, pressure_outlet_zmin: bool = False) -> None:
        self._compute_divergence_kernel(
            1 if pressure_outlet_zmin else 0,
        )

    @ti.kernel
    def _divergence_stats_kernel(self, interior_only: ti.i32):
        self.reduction_sum[None] = 0.0
        self.reduction_max[None] = 0.0
        self.reduction_count[None] = 0
        for i, j, k in self.divergence:
            is_interior = i > 0 and i < self.nx - 1
            is_interior = (
                is_interior
                and j > 0
                and j < self.ny - 1
                and k > 0
                and k < self.nz - 1
            )
            if self.obstacle[i, j, k] == 0 and (interior_only == 0 or is_interior):
                value = ti.abs(self.divergence[i, j, k])
                ti.atomic_max(self.reduction_max[None], value)
                ti.atomic_add(self.reduction_sum[None], value * value)
                ti.atomic_add(self.reduction_count[None], 1)
        self.divergence_report_snapshot[None] = ti.Vector(
            [
                ti.cast(self.reduction_max[None], ti.f64),
                ti.cast(self.reduction_sum[None], ti.f64),
                ti.cast(self.reduction_count[None], ti.f64),
            ]
        )

    @ti.kernel
    def _divergence_residual_stats_kernel(
        self,
        interior_only: ti.i32,
        exclude_unreached: ti.i32,
    ):
        self.reduction_sum[None] = 0.0
        self.reduction_max[None] = 0.0
        self.reduction_count[None] = 0
        for i, j, k in self.divergence:
            is_interior = i > 0 and i < self.nx - 1
            is_interior = (
                is_interior
                and j > 0
                and j < self.ny - 1
                and k > 0
                and k < self.nz - 1
            )
            is_unreached = (
                exclude_unreached == 1
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            )
            if (
                self.obstacle[i, j, k] == 0
                and not is_unreached
                and (interior_only == 0 or is_interior)
            ):
                value = ti.abs(self.divergence[i, j, k] - self.volume_source_s[i, j, k])
                ti.atomic_max(self.reduction_max[None], value)
                ti.atomic_add(self.reduction_sum[None], value * value)
                ti.atomic_add(self.reduction_count[None], 1)
        self.divergence_report_snapshot[None] = ti.Vector(
            [
                ti.cast(self.reduction_max[None], ti.f64),
                ti.cast(self.reduction_sum[None], ti.f64),
                ti.cast(self.reduction_count[None], ti.f64),
            ]
        )

    @ti.func
    def _accumulate_divergence_report_slot(self, slot: ti.i32, value: ti.f32):
        ti.atomic_max(self.divergence_combined_max[slot], value)
        ti.atomic_add(self.divergence_combined_sum[slot], value * value)
        ti.atomic_add(self.divergence_combined_count[slot], 1)

    @ti.kernel
    def _divergence_final_report_kernel(
        self,
        pressure_outlet_zmin: ti.i32,
        exclude_unreached: ti.i32,
    ):
        for slot in ti.static(range(18)):
            self.divergence_combined_sum[slot] = 0.0
            self.divergence_combined_max[slot] = 0.0
            self.divergence_combined_count[slot] = 0
        for i, j, k in self.divergence:
            is_interior = i > 0 and i < self.nx - 1
            is_interior = (
                is_interior
                and j > 0
                and j < self.ny - 1
                and k > 0
                and k < self.nz - 1
            )
            is_unreached = (
                exclude_unreached == 1
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            )
            if self.obstacle[i, j, k] == 0 and is_unreached:
                raw_value = ti.abs(self.divergence[i, j, k])
                residual_value = ti.abs(
                    self.divergence[i, j, k] - self.volume_source_s[i, j, k]
                )
                self._accumulate_divergence_report_slot(16, raw_value)
                self._accumulate_divergence_report_slot(17, residual_value)
            elif self.obstacle[i, j, k] == 0:
                raw_value = ti.abs(self.divergence[i, j, k])
                residual_value = ti.abs(
                    self.divergence[i, j, k] - self.volume_source_s[i, j, k]
                )
                self._accumulate_divergence_report_slot(0, raw_value)
                self._accumulate_divergence_report_slot(1, residual_value)
                if is_interior:
                    self._accumulate_divergence_report_slot(2, raw_value)
                    self._accumulate_divergence_report_slot(3, residual_value)
                if self._divergence_stencil_uses_velocity_dirichlet(i, j, k):
                    self._accumulate_divergence_report_slot(4, raw_value)
                    self._accumulate_divergence_report_slot(5, residual_value)
                else:
                    self._accumulate_divergence_report_slot(6, raw_value)
                    self._accumulate_divergence_report_slot(7, residual_value)
                if self._divergence_stencil_has_pressure_correctable_face(
                    i,
                    j,
                    k,
                    pressure_outlet_zmin,
                ):
                    self._accumulate_divergence_report_slot(8, raw_value)
                    self._accumulate_divergence_report_slot(9, residual_value)
                    if is_interior:
                        self._accumulate_divergence_report_slot(12, raw_value)
                        self._accumulate_divergence_report_slot(13, residual_value)
                else:
                    self._accumulate_divergence_report_slot(10, raw_value)
                    self._accumulate_divergence_report_slot(11, residual_value)
                    if is_interior:
                        self._accumulate_divergence_report_slot(14, raw_value)
                        self._accumulate_divergence_report_slot(15, residual_value)
        for slot in ti.static(range(18)):
            offset = slot * 3
            snapshot_index = offset // 24
            snapshot_offset = offset - snapshot_index * 24
            self.divergence_final_report_snapshot[snapshot_index][snapshot_offset] = ti.cast(
                self.divergence_combined_max[slot],
                ti.f64,
            )
            self.divergence_final_report_snapshot[snapshot_index][
                snapshot_offset + 1
            ] = ti.cast(
                self.divergence_combined_sum[slot],
                ti.f64,
            )
            self.divergence_final_report_snapshot[snapshot_index][
                snapshot_offset + 2
            ] = ti.cast(
                self.divergence_combined_count[slot],
                ti.f64,
            )

    @ti.func
    def _divergence_stencil_uses_velocity_dirichlet(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        uses_dirichlet = self.velocity_dirichlet_boundary_active[i, j, k] != 0
        if i < self.nx - 1:
            uses_dirichlet = (
                uses_dirichlet
                or self.velocity_dirichlet_boundary_active[i + 1, j, k] != 0
            )
        if j < self.ny - 1:
            uses_dirichlet = (
                uses_dirichlet
                or self.velocity_dirichlet_boundary_active[i, j + 1, k] != 0
            )
        if k < self.nz - 1:
            uses_dirichlet = (
                uses_dirichlet
                or self.velocity_dirichlet_boundary_active[i, j, k + 1] != 0
            )
        return uses_dirichlet

    @ti.func
    def _divergence_stencil_has_pressure_correctable_face(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        pressure_outlet_zmin: ti.i32,
    ):
        has_correctable = False
        if (
            i > 0
            and self.obstacle[i - 1, j, k] == 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            has_correctable = True
        if (
            i < self.nx - 1
            and self.obstacle[i + 1, j, k] == 0
            and self.velocity_dirichlet_boundary_active[i + 1, j, k] == 0
        ):
            has_correctable = True
        if (
            j > 0
            and self.obstacle[i, j - 1, k] == 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            has_correctable = True
        if (
            j < self.ny - 1
            and self.obstacle[i, j + 1, k] == 0
            and self.velocity_dirichlet_boundary_active[i, j + 1, k] == 0
        ):
            has_correctable = True
        if (
            k > 0
            and self.obstacle[i, j, k - 1] == 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            has_correctable = True
        if (
            k < self.nz - 1
            and self.obstacle[i, j, k + 1] == 0
            and self.velocity_dirichlet_boundary_active[i, j, k + 1] == 0
        ):
            has_correctable = True
        if (
            pressure_outlet_zmin == 1
            and k == 0
            and self.velocity_dirichlet_boundary_active[i, j, k] == 0
        ):
            has_correctable = True
        return has_correctable

    @ti.kernel
    def _divergence_dirichlet_partition_report_kernel(self):
        for slot in ti.static(range(4)):
            self.divergence_combined_sum[slot] = 0.0
            self.divergence_combined_max[slot] = 0.0
            self.divergence_combined_count[slot] = 0
        for i, j, k in self.divergence:
            if self.obstacle[i, j, k] == 0:
                raw_value = ti.abs(self.divergence[i, j, k])
                residual_value = ti.abs(
                    self.divergence[i, j, k] - self.volume_source_s[i, j, k]
                )
                if self._divergence_stencil_uses_velocity_dirichlet(i, j, k):
                    self._accumulate_divergence_report_slot(0, raw_value)
                    self._accumulate_divergence_report_slot(1, residual_value)
                else:
                    self._accumulate_divergence_report_slot(2, raw_value)
                    self._accumulate_divergence_report_slot(3, residual_value)
        self.divergence_dirichlet_partition_snapshot[None] = ti.Vector(
            [
                ti.cast(self.divergence_combined_max[0], ti.f64),
                ti.cast(self.divergence_combined_sum[0], ti.f64),
                ti.cast(self.divergence_combined_count[0], ti.f64),
                ti.cast(self.divergence_combined_max[1], ti.f64),
                ti.cast(self.divergence_combined_sum[1], ti.f64),
                ti.cast(self.divergence_combined_count[1], ti.f64),
                ti.cast(self.divergence_combined_max[2], ti.f64),
                ti.cast(self.divergence_combined_sum[2], ti.f64),
                ti.cast(self.divergence_combined_count[2], ti.f64),
                ti.cast(self.divergence_combined_max[3], ti.f64),
                ti.cast(self.divergence_combined_sum[3], ti.f64),
                ti.cast(self.divergence_combined_count[3], ti.f64),
            ]
        )

    @ti.kernel
    def _store_cleanup_target_l2_sq_from_reduction_kernel(
        self,
        tolerance_scale_sq: ti.f64,
        min_l2_sq: ti.f64,
    ):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        self.cleanup_target_l2_sq[None] = ti.max(l2_sq * tolerance_scale_sq, min_l2_sq)

    @ti.kernel
    def _update_cleanup_required_from_reduction_kernel(self):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        self.cleanup_required[None] = 0
        if l2_sq > self.cleanup_target_l2_sq[None]:
            self.cleanup_required[None] = 1

    def divergence_stats(self, *, interior_only: bool = False) -> dict[str, float]:
        self._divergence_stats_kernel(1 if interior_only else 0)
        self.last_divergence_report_host_reads += 1
        snapshot = self.divergence_report_snapshot[None]
        count = max(1, int(snapshot[2]))
        return {
            "max_abs": float(snapshot[0]),
            "l2": sqrt(float(snapshot[1]) / count),
        }

    def divergence_residual_stats(self, *, interior_only: bool = False) -> dict[str, float]:
        self._divergence_residual_stats_kernel(
            1 if interior_only else 0,
            1 if int(self._hibm_pressure_unreached_count) > 0 else 0,
        )
        self.last_divergence_report_host_reads += 1
        snapshot = self.divergence_report_snapshot[None]
        count = max(1, int(snapshot[2]))
        return {
            "max_abs": float(snapshot[0]),
            "l2": sqrt(float(snapshot[1]) / count),
        }

    @staticmethod
    def _decode_divergence_report_snapshot(snapshot, offset: int) -> dict[str, float]:
        actual_count = int(snapshot[offset + 2])
        count = max(1, actual_count)
        return {
            "max_abs": float(snapshot[offset]),
            "l2": sqrt(float(snapshot[offset + 1]) / count),
            "count": actual_count,
        }

    def _read_divergence_final_report_snapshot(self) -> tuple[float, ...]:
        first = self.divergence_final_report_snapshot[0]
        second = self.divergence_final_report_snapshot[1]
        third = self.divergence_final_report_snapshot[2]
        return (
            tuple(float(first[index]) for index in range(24))
            + tuple(float(second[index]) for index in range(24))
            + tuple(float(third[index]) for index in range(24))
        )

    def final_divergence_report_stats(
        self,
        *,
        pressure_outlet_zmin: bool = False,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        self._divergence_final_report_kernel(
            1 if pressure_outlet_zmin else 0,
            1 if int(self._hibm_pressure_unreached_count) > 0 else 0,
        )
        self.last_divergence_report_host_reads += 1
        snapshot = self._read_divergence_final_report_snapshot()
        final_raw_stats = self._decode_divergence_report_snapshot(snapshot, 0)
        final_stats = self._decode_divergence_report_snapshot(snapshot, 3)
        final_interior_raw_stats = self._decode_divergence_report_snapshot(snapshot, 6)
        final_interior_stats = self._decode_divergence_report_snapshot(snapshot, 9)
        self.last_unreached_divergence_raw_stats = (
            self._decode_divergence_report_snapshot(snapshot, 48)
        )
        self.last_unreached_divergence_stats = (
            self._decode_divergence_report_snapshot(snapshot, 51)
        )
        return (
            final_raw_stats,
            final_stats,
            final_interior_raw_stats,
            final_interior_stats,
        )

    def final_and_dirichlet_partition_report_stats(
        self,
        *,
        pressure_outlet_zmin: bool = False,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        self._divergence_final_report_kernel(
            1 if pressure_outlet_zmin else 0,
            1 if int(self._hibm_pressure_unreached_count) > 0 else 0,
        )
        self.last_divergence_report_host_reads += 1
        snapshot = self._read_divergence_final_report_snapshot()
        final_raw_stats = self._decode_divergence_report_snapshot(snapshot, 0)
        final_stats = self._decode_divergence_report_snapshot(snapshot, 3)
        final_interior_raw_stats = self._decode_divergence_report_snapshot(snapshot, 6)
        final_interior_stats = self._decode_divergence_report_snapshot(snapshot, 9)
        self.last_unreached_divergence_raw_stats = (
            self._decode_divergence_report_snapshot(snapshot, 48)
        )
        self.last_unreached_divergence_stats = (
            self._decode_divergence_report_snapshot(snapshot, 51)
        )
        near_raw_stats = self._decode_divergence_report_snapshot(snapshot, 12)
        near_stats = self._decode_divergence_report_snapshot(snapshot, 15)
        far_raw_stats = self._decode_divergence_report_snapshot(snapshot, 18)
        far_stats = self._decode_divergence_report_snapshot(snapshot, 21)
        pressure_correctable_raw_stats = self._decode_divergence_report_snapshot(
            snapshot,
            24,
        )
        pressure_correctable_stats = self._decode_divergence_report_snapshot(
            snapshot,
            27,
        )
        pressure_fixed_raw_stats = self._decode_divergence_report_snapshot(snapshot, 30)
        pressure_fixed_stats = self._decode_divergence_report_snapshot(snapshot, 33)
        interior_pressure_correctable_raw_stats = (
            self._decode_divergence_report_snapshot(snapshot, 36)
        )
        interior_pressure_correctable_stats = self._decode_divergence_report_snapshot(
            snapshot,
            39,
        )
        interior_pressure_fixed_raw_stats = self._decode_divergence_report_snapshot(
            snapshot,
            42,
        )
        interior_pressure_fixed_stats = self._decode_divergence_report_snapshot(
            snapshot,
            45,
        )
        return (
            final_raw_stats,
            final_stats,
            final_interior_raw_stats,
            final_interior_stats,
            near_raw_stats,
            near_stats,
            far_raw_stats,
            far_stats,
            pressure_correctable_raw_stats,
            pressure_correctable_stats,
            pressure_fixed_raw_stats,
            pressure_fixed_stats,
            interior_pressure_correctable_raw_stats,
            interior_pressure_correctable_stats,
            interior_pressure_fixed_raw_stats,
            interior_pressure_fixed_stats,
        )

    def divergence_dirichlet_partition_report_stats(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        self._divergence_dirichlet_partition_report_kernel()
        self.last_divergence_report_host_reads += 1
        snapshot = self.divergence_dirichlet_partition_snapshot[None]
        near_raw_stats = self._decode_divergence_report_snapshot(snapshot, 0)
        near_stats = self._decode_divergence_report_snapshot(snapshot, 3)
        far_raw_stats = self._decode_divergence_report_snapshot(snapshot, 6)
        far_stats = self._decode_divergence_report_snapshot(snapshot, 9)
        return near_raw_stats, near_stats, far_raw_stats, far_stats

    @ti.kernel
    def _pressure_jacobi_kernel(
        self,
        rhs_scale: ti.f32,
        inv_dx2: ti.f32,
        inv_dy2: ti.f32,
        inv_dz2: ti.f32,
        pressure_outlet_zmin: ti.i32,
    ):
        for i, j, k in self.pressure:
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, self.nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, self.ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, self.nz - 1)
            neighbor_sum = (
                inv_dx2 * (self.pressure[im, j, k] + self.pressure[ip, j, k])
                + inv_dy2 * (self.pressure[i, jm, k] + self.pressure[i, jp, k])
                + inv_dz2 * (self.pressure[i, j, km] + self.pressure[i, j, kp])
            )
            denominator = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2)
            obstacle_adjacent = (
                self.obstacle[im, j, k] == 1
                or self.obstacle[ip, j, k] == 1
                or self.obstacle[i, jm, k] == 1
                or self.obstacle[i, jp, k] == 1
                or self.obstacle[i, j, km] == 1
                or self.obstacle[i, j, kp] == 1
            )
            if obstacle_adjacent or (pressure_outlet_zmin == 1 and k == 0):
                neighbor_sum = 0.0
                denominator = 0.0
                if i > 0 and self.obstacle[i - 1, j, k] == 0:
                    neighbor_sum += inv_dx2 * self.pressure[i - 1, j, k]
                    denominator += inv_dx2
                if i < self.nx - 1 and self.obstacle[i + 1, j, k] == 0:
                    neighbor_sum += inv_dx2 * self.pressure[i + 1, j, k]
                    denominator += inv_dx2
                if j > 0 and self.obstacle[i, j - 1, k] == 0:
                    neighbor_sum += inv_dy2 * self.pressure[i, j - 1, k]
                    denominator += inv_dy2
                if j < self.ny - 1 and self.obstacle[i, j + 1, k] == 0:
                    neighbor_sum += inv_dy2 * self.pressure[i, j + 1, k]
                    denominator += inv_dy2
                if k > 0 and self.obstacle[i, j, k - 1] == 0:
                    neighbor_sum += inv_dz2 * self.pressure[i, j, k - 1]
                    denominator += inv_dz2
                if k < self.nz - 1 and self.obstacle[i, j, k + 1] == 0:
                    neighbor_sum += inv_dz2 * self.pressure[i, j, k + 1]
                    denominator += inv_dz2
                if pressure_outlet_zmin == 1 and k == 0:
                    denominator += 2.0 * inv_dz2
            rhs = rhs_scale * (self.divergence[i, j, k] - self.volume_source_s[i, j, k])
            if self.obstacle[i, j, k] == 1 or denominator <= 0.0:
                self.pressure_tmp[i, j, k] = 0.0
            else:
                self.pressure_tmp[i, j, k] = (neighbor_sum - rhs) / denominator

    @ti.kernel
    def _copy_pressure_kernel(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = self.pressure_tmp[i, j, k]

    @ti.kernel
    def _clear_pressure_kernel(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0
            self.pressure_accum[i, j, k] = 0.0

    @ti.kernel
    def _clear_pressure_correction_kernel(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0

    def clear_pressure(self) -> None:
        self._clear_pressure_kernel()

    def snapshot_pressure(self) -> None:
        self._copy_scalar_field_kernel(self.fsi_pressure, self.pressure)

    @ti.func
    def _fv_pressure_neighbor_sum_denominator(
        self,
        pressure: ti.template(),
        obstacle: ti.template(),
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_distance_x_m: ti.template(),
        center_distance_y_m: ti.template(),
        center_distance_z_m: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        pressure_outlet_zmin: ti.i32,
    ):
        neighbor_sum = 0.0
        denominator = 0.0
        if i > 0 and obstacle[i - 1, j, k] == 0:
            face_weight = 1.0
            if velocity_dirichlet_boundary_active[i, j, k] != 0:
                face_weight = 0.0
            coeff = 1.0 / (cell_width_x_m[i] * center_distance_x_m[i])
            neighbor_sum += coeff * face_weight * pressure[i - 1, j, k]
            denominator += coeff * face_weight
        if i < nx - 1 and obstacle[i + 1, j, k] == 0:
            face_weight = 1.0
            if velocity_dirichlet_boundary_active[i + 1, j, k] != 0:
                face_weight = 0.0
            coeff = 1.0 / (cell_width_x_m[i] * center_distance_x_m[i + 1])
            neighbor_sum += coeff * face_weight * pressure[i + 1, j, k]
            denominator += coeff * face_weight
        if j > 0 and obstacle[i, j - 1, k] == 0:
            face_weight = 1.0
            if velocity_dirichlet_boundary_active[i, j, k] != 0:
                face_weight = 0.0
            coeff = 1.0 / (cell_width_y_m[j] * center_distance_y_m[j])
            neighbor_sum += coeff * face_weight * pressure[i, j - 1, k]
            denominator += coeff * face_weight
        if j < ny - 1 and obstacle[i, j + 1, k] == 0:
            face_weight = 1.0
            if velocity_dirichlet_boundary_active[i, j + 1, k] != 0:
                face_weight = 0.0
            coeff = 1.0 / (cell_width_y_m[j] * center_distance_y_m[j + 1])
            neighbor_sum += coeff * face_weight * pressure[i, j + 1, k]
            denominator += coeff * face_weight
        if k > 0 and obstacle[i, j, k - 1] == 0:
            face_weight = 1.0
            if velocity_dirichlet_boundary_active[i, j, k] != 0:
                face_weight = 0.0
            coeff = 1.0 / (cell_width_z_m[k] * center_distance_z_m[k])
            neighbor_sum += coeff * face_weight * pressure[i, j, k - 1]
            denominator += coeff * face_weight
        if k < nz - 1 and obstacle[i, j, k + 1] == 0:
            face_weight = 1.0
            if velocity_dirichlet_boundary_active[i, j, k + 1] != 0:
                face_weight = 0.0
            coeff = 1.0 / (cell_width_z_m[k] * center_distance_z_m[k + 1])
            neighbor_sum += coeff * face_weight * pressure[i, j, k + 1]
            denominator += coeff * face_weight
        if pressure_outlet_zmin == 1 and k == 0:
            denominator += 2.0 / (cell_width_z_m[k] * cell_width_z_m[k])
        return neighbor_sum, denominator

    @ti.kernel
    def _pressure_fv_jacobi_kernel(
        self,
        pressure: ti.template(),
        rhs: ti.template(),
        pressure_interface_matrix_diagonal: ti.template(),
        obstacle: ti.template(),
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        tmp: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_distance_x_m: ti.template(),
        center_distance_y_m: ti.template(),
        center_distance_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        pressure_outlet_zmin: ti.i32,
        omega: ti.f32,
    ):
        for i, j, k in pressure:
            if obstacle[i, j, k] == 1:
                tmp[i, j, k] = 0.0
            else:
                neighbor_sum, denominator = self._fv_pressure_neighbor_sum_denominator(
                    pressure,
                    obstacle,
                    velocity_dirichlet_boundary_active,
                    velocity_dirichlet_boundary_projection_weight,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    center_distance_x_m,
                    center_distance_y_m,
                    center_distance_z_m,
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    pressure_outlet_zmin,
                )
                denominator += pressure_interface_matrix_diagonal[i, j, k]
                if denominator <= 0.0:
                    tmp[i, j, k] = 0.0
                else:
                    jacobi_value = (neighbor_sum - rhs[i, j, k]) / denominator
                    tmp[i, j, k] = pressure[i, j, k] + omega * (jacobi_value - pressure[i, j, k])

    @ti.kernel
    def _copy_scalar_field_kernel(self, destination: ti.template(), source: ti.template()):
        for i, j, k in destination:
            destination[i, j, k] = source[i, j, k]

    @ti.kernel
    def _fv_diagonal_kernel(
        self,
        diagonal: ti.template(),
        pressure_outlet_zmin: ti.i32,
    ):
        for i, j, k in diagonal:
            if self.obstacle[i, j, k] == 1:
                diagonal[i, j, k] = 0.0
            else:
                _, denominator = self._fv_pressure_neighbor_sum_denominator(
                    self.pressure,
                    self.obstacle,
                    self.velocity_dirichlet_boundary_active,
                    self.velocity_dirichlet_boundary_projection_weight,
                    self.cell_width_x_m,
                    self.cell_width_y_m,
                    self.cell_width_z_m,
                    self.center_distance_x_m,
                    self.center_distance_y_m,
                    self.center_distance_z_m,
                    i,
                    j,
                    k,
                    self.nx,
                    self.ny,
                    self.nz,
                    pressure_outlet_zmin,
                )
                diagonal[i, j, k] = denominator + self.pressure_interface_matrix_diagonal[i, j, k]

    @ti.kernel
    def _fv_laplacian_apply_kernel(
        self,
        pressure: ti.template(),
        output: ti.template(),
        pressure_outlet_zmin: ti.i32,
    ):
        for i, j, k in output:
            if self.obstacle[i, j, k] == 1:
                output[i, j, k] = 0.0
            else:
                neighbor_sum, denominator = self._fv_pressure_neighbor_sum_denominator(
                    pressure,
                    self.obstacle,
                    self.velocity_dirichlet_boundary_active,
                    self.velocity_dirichlet_boundary_projection_weight,
                    self.cell_width_x_m,
                    self.cell_width_y_m,
                    self.cell_width_z_m,
                    self.center_distance_x_m,
                    self.center_distance_y_m,
                    self.center_distance_z_m,
                    i,
                    j,
                    k,
                    self.nx,
                    self.ny,
                    self.nz,
                    pressure_outlet_zmin,
                )
                output[i, j, k] = (
                    (denominator + self.pressure_interface_matrix_diagonal[i, j, k])
                    * pressure[i, j, k]
                    - neighbor_sum
                )
        for i, j, k in output:
            if (
                self.pressure_interface_coupling_active[i, j, k] == 1
                and self.obstacle[i, j, k] == 0
            ):
                neighbor = self.pressure_interface_coupling_neighbor[i, j, k]
                ni = neighbor.x
                nj = neighbor.y
                nk = neighbor.z
                if (
                    0 <= ni < self.nx
                    and 0 <= nj < self.ny
                    and 0 <= nk < self.nz
                    and self.obstacle[ni, nj, nk] == 0
                ):
                    transmissibility = self.pressure_interface_coupling_coefficient[
                        i,
                        j,
                        k,
                    ]
                    active_volume_m3 = ti.max(self._cell_volume_m3(i, j, k), 1.0e-30)
                    neighbor_volume_m3 = ti.max(
                        self._cell_volume_m3(ni, nj, nk),
                        1.0e-30,
                    )
                    if self.velocity_dirichlet_boundary_active[i, j, k] != 0:
                        ti.atomic_add(
                            output[i, j, k],
                            -(transmissibility / active_volume_m3)
                            * pressure[i, j, k],
                        )
                        ti.atomic_add(
                            output[ni, nj, nk],
                            -(transmissibility / neighbor_volume_m3)
                            * pressure[ni, nj, nk],
                        )
                    else:
                        ti.atomic_add(
                            output[i, j, k],
                            -(transmissibility / active_volume_m3)
                            * pressure[ni, nj, nk],
                        )
                        ti.atomic_add(
                            output[ni, nj, nk],
                            -(transmissibility / neighbor_volume_m3)
                            * pressure[i, j, k],
                        )

    @ti.kernel
    def _cg_build_positive_rhs_kernel(
        self,
        legacy_rhs: ti.template(),
        rhs_out: ti.template(),
        rhs_weighted_mean: ti.f32,
    ):
        for i, j, k in rhs_out:
            if self.obstacle[i, j, k] == 1:
                rhs_out[i, j, k] = 0.0
            else:
                rhs_out[i, j, k] = -legacy_rhs[i, j, k] - rhs_weighted_mean

    @ti.kernel
    def _cg_build_positive_rhs_from_cg_mean_kernel(
        self,
        legacy_rhs: ti.template(),
        rhs_out: ti.template(),
    ):
        rhs_weighted_mean = ti.cast(self.cg_weighted_mean[None], ti.f32)
        for i, j, k in rhs_out:
            if self.obstacle[i, j, k] == 1:
                rhs_out[i, j, k] = 0.0
            else:
                rhs_out[i, j, k] = -legacy_rhs[i, j, k] - rhs_weighted_mean

    @ti.kernel
    def _pcg_prepare_mg_level0_kernel(
        self,
        positive_rhs: ti.template(),
        legacy_rhs: ti.template(),
        pressure: ti.template(),
        tmp: ti.template(),
        residual: ti.template(),
    ):
        for i, j, k in positive_rhs:
            pressure[i, j, k] = 0.0
            tmp[i, j, k] = 0.0
            residual[i, j, k] = 0.0
            if self.obstacle[i, j, k] == 1:
                legacy_rhs[i, j, k] = 0.0
            else:
                legacy_rhs[i, j, k] = -positive_rhs[i, j, k]

    @ti.kernel
    def _apply_jacobi_preconditioner_kernel(
        self,
        residual: ti.template(),
        preconditioned: ti.template(),
    ):
        for i, j, k in preconditioned:
            if self.obstacle[i, j, k] == 1:
                preconditioned[i, j, k] = 0.0
            else:
                denominator = self.fv_diag[i, j, k]
                if denominator > 0.0:
                    preconditioned[i, j, k] = residual[i, j, k] / denominator
                else:
                    preconditioned[i, j, k] = 0.0

    @ti.kernel
    def _axpby_scalar_field_kernel(
        self,
        out: ti.template(),
        a: ti.f32,
        x: ti.template(),
        b: ti.f32,
        y: ti.template(),
    ):
        for i, j, k in out:
            if self.obstacle[i, j, k] == 1:
                out[i, j, k] = 0.0
            else:
                out[i, j, k] = a * x[i, j, k] + b * y[i, j, k]

    @ti.kernel
    def _weighted_dot_kernel(self, a: ti.template(), b: ti.template()) -> ti.f64:
        total = ti.cast(0.0, ti.f64)
        for i, j, k in a:
            if self.obstacle[i, j, k] == 0:
                volume_m3 = (
                    ti.cast(self.cell_width_x_m[i], ti.f64)
                    * ti.cast(self.cell_width_y_m[j], ti.f64)
                    * ti.cast(self.cell_width_z_m[k], ti.f64)
                )
                total += ti.cast(a[i, j, k], ti.f64) * ti.cast(b[i, j, k], ti.f64) * volume_m3
        return total

    @ti.kernel
    def _weighted_dot_to_field_kernel(
        self,
        a: ti.template(),
        b: ti.template(),
        out: ti.template(),
    ):
        out[None] = ti.cast(0.0, ti.f64)
        for i, j, k in a:
            if self.obstacle[i, j, k] == 0:
                volume_m3 = (
                    ti.cast(self.cell_width_x_m[i], ti.f64)
                    * ti.cast(self.cell_width_y_m[j], ti.f64)
                    * ti.cast(self.cell_width_z_m[k], ti.f64)
                )
                ti.atomic_add(
                    out[None],
                    ti.cast(a[i, j, k], ti.f64) * ti.cast(b[i, j, k], ti.f64) * volume_m3,
                )

    @ti.kernel
    def _cg_compute_alpha_kernel(self):
        self.cg_alpha[None] = 0.0
        if self.cg_dAd[None] <= 0.0:
            if self.cg_breakdown_code[None] == 0:
                self.cg_breakdown_dAd[None] = self.cg_dAd[None]
            self.cg_breakdown_code[None] = 1
        else:
            self.cg_alpha[None] = self.cg_rz[None] / self.cg_dAd[None]

    @ti.kernel
    def _cg_apply_alpha_kernel(self):
        alpha = ti.cast(self.cg_alpha[None], ti.f32)
        for i, j, k in self.pressure:
            if self.obstacle[i, j, k] == 1:
                self.cg_r_old[i, j, k] = 0.0
                self.pressure[i, j, k] = 0.0
                self.cg_r[i, j, k] = 0.0
            else:
                self.cg_r_old[i, j, k] = self.cg_r[i, j, k]
                self.pressure[i, j, k] += alpha * self.cg_d[i, j, k]
                self.cg_r[i, j, k] -= alpha * self.cg_Ad[i, j, k]

    @ti.kernel
    def _cg_compute_beta_kernel(self, use_mg_preconditioner: ti.i32):
        self.cg_beta[None] = 0.0
        if self.cg_rz[None] <= 1.0e-300 or self.cg_rz_new[None] <= 0.0:
            self.cg_breakdown_code[None] = 2
        else:
            if use_mg_preconditioner != 0:
                beta = self.cg_beta_numerator[None] / self.cg_rz[None]
                if beta < 0.0:
                    beta = 0.0
                self.cg_beta[None] = beta
            else:
                self.cg_beta[None] = self.cg_rz_new[None] / self.cg_rz[None]

    @ti.kernel
    def _cg_update_direction_and_rz_kernel(self):
        beta = ti.cast(self.cg_beta[None], ti.f32)
        for i, j, k in self.cg_d:
            if self.obstacle[i, j, k] == 1:
                self.cg_d[i, j, k] = 0.0
            else:
                self.cg_d[i, j, k] = self.cg_z[i, j, k] + beta * self.cg_d[i, j, k]
        self.cg_rz[None] = self.cg_rz_new[None]

    @ti.kernel
    def _weighted_sum_kernel(self, field: ti.template()) -> ti.f64:
        total = ti.cast(0.0, ti.f64)
        for i, j, k in field:
            if self.obstacle[i, j, k] == 0:
                volume_m3 = (
                    ti.cast(self.cell_width_x_m[i], ti.f64)
                    * ti.cast(self.cell_width_y_m[j], ti.f64)
                    * ti.cast(self.cell_width_z_m[k], ti.f64)
                )
                total += ti.cast(field[i, j, k], ti.f64) * volume_m3
        return total

    @ti.kernel
    def _weighted_mean_to_cg_field_kernel(self, field: ti.template(), multiplier: ti.f64):
        total = ti.cast(0.0, ti.f64)
        volume_total = ti.cast(0.0, ti.f64)
        for i, j, k in field:
            if self.obstacle[i, j, k] == 0:
                volume_m3 = (
                    ti.cast(self.cell_width_x_m[i], ti.f64)
                    * ti.cast(self.cell_width_y_m[j], ti.f64)
                    * ti.cast(self.cell_width_z_m[k], ti.f64)
                )
                total += ti.cast(field[i, j, k], ti.f64) * volume_m3
                volume_total += volume_m3
        self.cg_weighted_sum[None] = total
        self.cg_free_volume[None] = volume_total
        self.cg_weighted_mean[None] = 0.0
        if volume_total > 0.0:
            self.cg_weighted_mean[None] = multiplier * total / volume_total

    @ti.kernel
    def _free_cell_volume_sum_kernel(self) -> ti.f64:
        total = ti.cast(0.0, ti.f64)
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 0:
                total += (
                    ti.cast(self.cell_width_x_m[i], ti.f64)
                    * ti.cast(self.cell_width_y_m[j], ti.f64)
                    * ti.cast(self.cell_width_z_m[k], ti.f64)
                )
        return total

    @ti.kernel
    def _subtract_weighted_mean_kernel(self, field: ti.template(), weighted_mean: ti.f32):
        for i, j, k in field:
            if self.obstacle[i, j, k] == 0:
                field[i, j, k] -= weighted_mean
            else:
                field[i, j, k] = 0.0

    @ti.kernel
    def _subtract_cg_weighted_mean_kernel(self, field: ti.template()):
        weighted_mean = ti.cast(self.cg_weighted_mean[None], ti.f32)
        for i, j, k in field:
            if self.obstacle[i, j, k] == 0:
                field[i, j, k] -= weighted_mean
            else:
                field[i, j, k] = 0.0

    def _subtract_weighted_mean_device(self, field: ti.template()) -> None:
        self._weighted_mean_to_cg_field_kernel(field, 1.0)
        self._subtract_cg_weighted_mean_kernel(field)

    @ti.kernel
    def _weighted_unreached_set_mean_to_cg_field_kernel(self, field: ti.template()):
        total = ti.cast(0.0, ti.f64)
        volume_total = ti.cast(0.0, ti.f64)
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                volume_m3 = (
                    ti.cast(self.cell_width_x_m[i], ti.f64)
                    * ti.cast(self.cell_width_y_m[j], ti.f64)
                    * ti.cast(self.cell_width_z_m[k], ti.f64)
                )
                total += ti.cast(field[i, j, k], ti.f64) * volume_m3
                volume_total += volume_m3
        self.cg_weighted_mean[None] = 0.0
        if volume_total > 0.0:
            self.cg_weighted_mean[None] = total / volume_total

    @ti.kernel
    def _subtract_cg_weighted_mean_from_unreached_set_kernel(
        self,
        field: ti.template(),
    ):
        mean = ti.cast(self.cg_weighted_mean[None], ti.f32)
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                field[i, j, k] -= mean

    @ti.kernel
    def _init_hibm_unreached_component_labels_kernel(self):
        for i, j, k in self.obstacle:
            label = 1 << 30
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                label = (i * self.ny + j) * self.nz + k
            self.hibm_pressure_unreached_component_label[i, j, k] = label

    @ti.kernel
    def _propagate_hibm_unreached_component_labels_kernel(self) -> ti.i32:
        changed = 0
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                label = self.hibm_pressure_unreached_component_label[i, j, k]
                best = label
                if (
                    i > 0
                    and self.obstacle[i - 1, j, k] == 0
                    and self.hibm_pressure_outlet_reachable[i - 1, j, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] == 0
                ):
                    best = ti.min(
                        best,
                        self.hibm_pressure_unreached_component_label[i - 1, j, k],
                    )
                if (
                    i < self.nx - 1
                    and self.obstacle[i + 1, j, k] == 0
                    and self.hibm_pressure_outlet_reachable[i + 1, j, k] == 0
                    and self.velocity_dirichlet_boundary_active[i + 1, j, k] == 0
                ):
                    best = ti.min(
                        best,
                        self.hibm_pressure_unreached_component_label[i + 1, j, k],
                    )
                if (
                    j > 0
                    and self.obstacle[i, j - 1, k] == 0
                    and self.hibm_pressure_outlet_reachable[i, j - 1, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] == 0
                ):
                    best = ti.min(
                        best,
                        self.hibm_pressure_unreached_component_label[i, j - 1, k],
                    )
                if (
                    j < self.ny - 1
                    and self.obstacle[i, j + 1, k] == 0
                    and self.hibm_pressure_outlet_reachable[i, j + 1, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, j + 1, k] == 0
                ):
                    best = ti.min(
                        best,
                        self.hibm_pressure_unreached_component_label[i, j + 1, k],
                    )
                if (
                    k > 0
                    and self.obstacle[i, j, k - 1] == 0
                    and self.hibm_pressure_outlet_reachable[i, j, k - 1] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] == 0
                ):
                    best = ti.min(
                        best,
                        self.hibm_pressure_unreached_component_label[i, j, k - 1],
                    )
                if (
                    k < self.nz - 1
                    and self.obstacle[i, j, k + 1] == 0
                    and self.hibm_pressure_outlet_reachable[i, j, k + 1] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k + 1] == 0
                ):
                    best = ti.min(
                        best,
                        self.hibm_pressure_unreached_component_label[i, j, k + 1],
                    )
                if best < label:
                    self.hibm_pressure_unreached_component_label[i, j, k] = best
                    changed += 1
        return changed

    @ti.kernel
    def _scan_min_unreached_raw_label_kernel(self):
        self.cg_unreached_component_scan[None] = 1 << 30
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                label = self.hibm_pressure_unreached_component_label[i, j, k]
                if label >= 0:
                    ti.atomic_min(self.cg_unreached_component_scan[None], label)

    @ti.kernel
    def _assign_unreached_component_id_kernel(
        self,
        raw_label: ti.i32,
        compact_label: ti.i32,
    ):
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
                and self.hibm_pressure_unreached_component_label[i, j, k] == raw_label
            ):
                self.hibm_pressure_unreached_component_label[i, j, k] = compact_label

    @ti.kernel
    def _accumulate_unreached_component_means_kernel(
        self,
        field: ti.template(),
        component_count: ti.i32,
    ):
        for c in range(32):
            self.cg_unreached_component_sum[c] = 0.0
            self.cg_unreached_component_volume[c] = 0.0
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                component = (
                    -self.hibm_pressure_unreached_component_label[i, j, k] - 1
                )
                if 0 <= component < component_count:
                    volume_m3 = (
                        ti.cast(self.cell_width_x_m[i], ti.f64)
                        * ti.cast(self.cell_width_y_m[j], ti.f64)
                        * ti.cast(self.cell_width_z_m[k], ti.f64)
                    )
                    ti.atomic_add(
                        self.cg_unreached_component_sum[component],
                        ti.cast(field[i, j, k], ti.f64) * volume_m3,
                    )
                    ti.atomic_add(
                        self.cg_unreached_component_volume[component],
                        volume_m3,
                    )

    @ti.kernel
    def _subtract_unreached_component_means_kernel(
        self,
        field: ti.template(),
        component_count: ti.i32,
    ):
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
            ):
                component = (
                    -self.hibm_pressure_unreached_component_label[i, j, k] - 1
                )
                if 0 <= component < component_count:
                    volume_total = self.cg_unreached_component_volume[component]
                    if volume_total > 0.0:
                        field[i, j, k] -= ti.cast(
                            self.cg_unreached_component_sum[component]
                            / volume_total,
                            ti.f32,
                        )

    def _subtract_unreached_set_mean_device(self, field: ti.template()) -> None:
        component_count = int(
            getattr(self, "_hibm_pressure_unreached_component_count", 0)
        )
        if component_count > 0:
            self._accumulate_unreached_component_means_kernel(
                field,
                component_count,
            )
            self._subtract_unreached_component_means_kernel(
                field,
                component_count,
            )
        else:
            self._weighted_unreached_set_mean_to_cg_field_kernel(field)
            self._subtract_cg_weighted_mean_from_unreached_set_kernel(field)
        self.last_cg_unreached_set_mean_projection_count += 1

    @ti.func
    def _flag_unreached_interface_component_hit(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_count: ti.i32,
    ):
        label = self.hibm_pressure_unreached_component_label[i, j, k]
        if label < 0:
            component = -label - 1
            if component < component_count:
                ti.atomic_or(
                    self.hibm_unreached_interface_component_hit[component],
                    1,
                )

    # R2-H1 diagnostics: count flood-unreachable cells whose pressure row is
    # nevertheless touched by interface matrix terms. A diagonal hit is an
    # unreached cell with a positive pressure_interface_matrix_diagonal entry
    # (the row is anchored). A coupling hit is an unreached cell whose row
    # receives an active interface coupling contribution, either as the owning
    # cell or as the target neighbor of another cell's edge, using exactly the
    # validity checks the FV operator applies in _fv_laplacian_apply_kernel.
    # Component hits deduplicate per compacted unreached-component label.
    @ti.kernel
    def _scan_hibm_unreached_interface_hits_kernel(self, component_count: ti.i32):
        self.report_hibm_unreached_interface_diagonal_cells[None] = 0
        self.report_hibm_unreached_interface_coupling_cells[None] = 0
        self.report_hibm_unreached_interface_component_hits[None] = 0
        for c in range(32):
            self.hibm_unreached_interface_component_hit[c] = 0
        for i, j, k in self.obstacle:
            self.hibm_unreached_interface_cell_hit[i, j, k] = 0
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.hibm_pressure_outlet_reachable[i, j, k] == 0
                and self.pressure_interface_matrix_diagonal[i, j, k] > 0.0
            ):
                ti.atomic_add(
                    self.report_hibm_unreached_interface_diagonal_cells[None],
                    1,
                )
                self._flag_unreached_interface_component_hit(
                    i,
                    j,
                    k,
                    component_count,
                )
        for i, j, k in self.obstacle:
            if (
                self.pressure_interface_coupling_active[i, j, k] == 1
                and self.obstacle[i, j, k] == 0
            ):
                neighbor = self.pressure_interface_coupling_neighbor[i, j, k]
                ni = neighbor.x
                nj = neighbor.y
                nk = neighbor.z
                if (
                    0 <= ni < self.nx
                    and 0 <= nj < self.ny
                    and 0 <= nk < self.nz
                    and self.obstacle[ni, nj, nk] == 0
                ):
                    if self.hibm_pressure_outlet_reachable[i, j, k] == 0:
                        ti.atomic_or(
                            self.hibm_unreached_interface_cell_hit[i, j, k],
                            1,
                        )
                        self._flag_unreached_interface_component_hit(
                            i,
                            j,
                            k,
                            component_count,
                        )
                    if self.hibm_pressure_outlet_reachable[ni, nj, nk] == 0:
                        ti.atomic_or(
                            self.hibm_unreached_interface_cell_hit[ni, nj, nk],
                            1,
                        )
                        self._flag_unreached_interface_component_hit(
                            ni,
                            nj,
                            nk,
                            component_count,
                        )
        for i, j, k in self.obstacle:
            if self.hibm_unreached_interface_cell_hit[i, j, k] != 0:
                ti.atomic_add(
                    self.report_hibm_unreached_interface_coupling_cells[None],
                    1,
                )
        for c in range(32):
            if self.hibm_unreached_interface_component_hit[c] != 0:
                ti.atomic_add(
                    self.report_hibm_unreached_interface_component_hits[None],
                    1,
                )

    def _record_unreached_interface_hit_diagnostics(self) -> None:
        """Run the device scan and read back the three scalar counters."""
        self._scan_hibm_unreached_interface_hits_kernel(
            int(self._hibm_pressure_unreached_component_count)
        )
        self.last_hibm_unreached_cells_with_interface_diagonal = int(
            self.report_hibm_unreached_interface_diagonal_cells[None]
        )
        self.last_hibm_unreached_cells_with_interface_coupling = int(
            self.report_hibm_unreached_interface_coupling_cells[None]
        )
        self.last_hibm_unreached_components_with_interface_hits = int(
            self.report_hibm_unreached_interface_component_hits[None]
        )

    @ti.kernel
    def _mg_prepare_level0_kernel(
        self,
        rhs: ti.template(),
        residual: ti.template(),
        rhs_scale: ti.f32,
    ):
        for i, j, k in rhs:
            if self.obstacle[i, j, k] == 1:
                rhs[i, j, k] = 0.0
                residual[i, j, k] = 0.0
            else:
                rhs[i, j, k] = (
                    rhs_scale * (self.divergence[i, j, k] - self.volume_source_s[i, j, k])
                    - self.pressure_interface_matrix_rhs[i, j, k]
                )
                residual[i, j, k] = 0.0

    @ti.kernel
    def _mg_clear_level_kernel(
        self,
        pressure: ti.template(),
        tmp: ti.template(),
        rhs: ti.template(),
        residual: ti.template(),
        pressure_interface_matrix_diagonal: ti.template(),
    ):
        for i, j, k in pressure:
            pressure[i, j, k] = 0.0
            tmp[i, j, k] = 0.0
            rhs[i, j, k] = 0.0
            residual[i, j, k] = 0.0
            pressure_interface_matrix_diagonal[i, j, k] = 0.0

    @ti.kernel
    def _mg_compute_residual_kernel(
        self,
        pressure: ti.template(),
        rhs: ti.template(),
        pressure_interface_matrix_diagonal: ti.template(),
        obstacle: ti.template(),
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        residual: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_distance_x_m: ti.template(),
        center_distance_y_m: ti.template(),
        center_distance_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        pressure_outlet_zmin: ti.i32,
    ):
        for i, j, k in pressure:
            if obstacle[i, j, k] == 1:
                residual[i, j, k] = 0.0
            else:
                neighbor_sum, denominator = self._fv_pressure_neighbor_sum_denominator(
                    pressure,
                    obstacle,
                    velocity_dirichlet_boundary_active,
                    velocity_dirichlet_boundary_projection_weight,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    center_distance_x_m,
                    center_distance_y_m,
                    center_distance_z_m,
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    pressure_outlet_zmin,
                )
                denominator += pressure_interface_matrix_diagonal[i, j, k]
                applied = neighbor_sum - denominator * pressure[i, j, k]
                residual[i, j, k] = rhs[i, j, k] - applied

    @ti.kernel
    def _mg_restrict_residual_kernel(
        self,
        fine_residual: ti.template(),
        fine_pressure_interface_matrix_diagonal: ti.template(),
        fine_obstacle: ti.template(),
        fine_velocity_dirichlet_boundary_active: ti.template(),
        fine_velocity_dirichlet_boundary_projection_weight: ti.template(),
        fine_cell_width_x_m: ti.template(),
        fine_cell_width_y_m: ti.template(),
        fine_cell_width_z_m: ti.template(),
        coarse_rhs: ti.template(),
        coarse_pressure_interface_matrix_diagonal: ti.template(),
        coarse_pressure: ti.template(),
        coarse_tmp: ti.template(),
        coarse_residual: ti.template(),
        coarse_obstacle: ti.template(),
        coarse_velocity_dirichlet_boundary_active: ti.template(),
        coarse_velocity_dirichlet_boundary_projection_weight: ti.template(),
        fine_nx: ti.i32,
        fine_ny: ti.i32,
        fine_nz: ti.i32,
    ):
        for i, j, k in coarse_rhs:
            weighted_residual_sum = 0.0
            weighted_interface_diagonal_sum = 0.0
            free_volume_m3 = 0.0
            obstacle_count = 0
            velocity_dirichlet_count = 0
            velocity_dirichlet_weight = 0.0
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                fi = 2 * i + oi
                fj = 2 * j + oj
                fk = 2 * k + ok
                if fi < fine_nx and fj < fine_ny and fk < fine_nz:
                    if fine_velocity_dirichlet_boundary_active[fi, fj, fk] != 0:
                        velocity_dirichlet_count += 1
                        velocity_dirichlet_weight = ti.max(
                            velocity_dirichlet_weight,
                            fine_velocity_dirichlet_boundary_projection_weight[
                                fi,
                                fj,
                                fk,
                            ],
                        )
                    if fine_obstacle[fi, fj, fk] == 1:
                        obstacle_count += 1
                    else:
                        volume_m3 = (
                            fine_cell_width_x_m[fi]
                            * fine_cell_width_y_m[fj]
                            * fine_cell_width_z_m[fk]
                        )
                        weighted_residual_sum += fine_residual[fi, fj, fk] * volume_m3
                        weighted_interface_diagonal_sum += (
                            fine_pressure_interface_matrix_diagonal[fi, fj, fk]
                            * volume_m3
                        )
                        free_volume_m3 += volume_m3
            if obstacle_count > 0:
                coarse_obstacle[i, j, k] = 1
                coarse_rhs[i, j, k] = 0.0
                coarse_pressure_interface_matrix_diagonal[i, j, k] = 0.0
            else:
                coarse_obstacle[i, j, k] = 0
                coarse_rhs[i, j, k] = weighted_residual_sum / ti.max(free_volume_m3, 1.0e-30)
                coarse_pressure_interface_matrix_diagonal[i, j, k] = (
                    weighted_interface_diagonal_sum / ti.max(free_volume_m3, 1.0e-30)
                )
            if velocity_dirichlet_count > 0:
                coarse_velocity_dirichlet_boundary_active[i, j, k] = 1
                coarse_velocity_dirichlet_boundary_projection_weight[i, j, k] = (
                    velocity_dirichlet_weight
                )
            else:
                coarse_velocity_dirichlet_boundary_active[i, j, k] = 0
                coarse_velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0
            coarse_pressure[i, j, k] = 0.0
            coarse_tmp[i, j, k] = 0.0
            coarse_residual[i, j, k] = 0.0

    @ti.kernel
    def _mg_prolongate_add_kernel(
        self,
        fine_pressure: ti.template(),
        fine_obstacle: ti.template(),
        coarse_pressure: ti.template(),
        coarse_nx: ti.i32,
        coarse_ny: ti.i32,
        coarse_nz: ti.i32,
        correction_scale: ti.f32,
    ):
        for i, j, k in fine_pressure:
            if fine_obstacle[i, j, k] == 0:
                ci = ti.min(i // 2, coarse_nx - 1)
                cj = ti.min(j // 2, coarse_ny - 1)
                ck = ti.min(k // 2, coarse_nz - 1)
                fine_pressure[i, j, k] += correction_scale * coarse_pressure[ci, cj, ck]

    @ti.kernel
    def _reset_zmin_projection_flux_report_kernel(self):
        self.report_zmin_pressure_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None] = 0.0

    @ti.kernel
    def _record_zmin_projection_pre_velocity_flux_kernel(self):
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                ti.atomic_add(
                    self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None],
                    -self.velocity[i, j, 0].z * outlet_cell_area_m2,
                )

    @ti.kernel
    def _record_zmin_pressure_step_pre_velocity_flux_kernel(self):
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                ti.atomic_add(
                    self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None],
                    -self.velocity[i, j, 0].z * outlet_cell_area_m2,
                )

    @ti.kernel
    def _accumulate_zmin_pressure_correction_flux_kernel(self):
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                ti.atomic_add(
                    self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None],
                    -self.velocity[i, j, 0].z * outlet_cell_area_m2,
                )
        self.report_zmin_pressure_outlet_flux_m3s[None] += (
            self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None]
            - self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None]
        )

    @ti.kernel
    def _record_zmin_projection_post_boundary_velocity_flux_kernel(self):
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                ti.atomic_add(
                    self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None],
                    -self.velocity[i, j, 0].z * outlet_cell_area_m2,
                )

    @ti.kernel
    def _pressure_outlet_fv_flux_report_kernel(
        self,
        dt_over_rho: ti.f32,
    ):
        self.report_source_volume_flux_m3s[None] = 0.0
        self.report_zmin_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_outlet_flux_ratio[None] = 0.0
        self.report_zmin_velocity_outlet_flux_ratio[None] = 0.0
        for i, j, k in self.volume_source_s:
            if self.obstacle[i, j, k] == 0:
                cell_volume_m3 = self.cell_width_x_m[i] * self.cell_width_y_m[j] * self.cell_width_z_m[k]
                ti.atomic_add(
                    self.report_source_volume_flux_m3s[None],
                    self.volume_source_s[i, j, k] * cell_volume_m3,
                )
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                velocity_outflow_m3s = -self.velocity[i, j, 0].z * outlet_cell_area_m2
                ti.atomic_add(
                    self.report_zmin_velocity_outlet_flux_m3s[None],
                    velocity_outflow_m3s,
                )
        source_abs = ti.abs(self.report_source_volume_flux_m3s[None])
        if source_abs > 1.0e-18:
            self.report_zmin_pressure_outlet_flux_ratio[None] = (
                self.report_zmin_pressure_outlet_flux_m3s[None] / source_abs
            )
            self.report_zmin_velocity_outlet_flux_ratio[None] = (
                self.report_zmin_velocity_outlet_flux_m3s[None] / source_abs
            )
        self.pressure_outlet_report_snapshot[None] = ti.Vector(
            [
                self.report_source_volume_flux_m3s[None],
                self.report_zmin_pressure_outlet_flux_m3s[None],
                self.report_zmin_velocity_outlet_flux_m3s[None],
                self.report_zmin_pressure_outlet_flux_ratio[None],
                self.report_zmin_velocity_outlet_flux_ratio[None],
                self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None],
                self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None],
                self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None],
            ]
        )

    @ti.kernel
    def _copy_pressure_to_accum_kernel(self):
        for i, j, k in self.pressure:
            self.pressure_accum[i, j, k] = self.pressure[i, j, k]

    @ti.kernel
    def _accumulate_pressure_correction_kernel(self):
        for i, j, k in self.pressure:
            corrected = self.pressure_accum[i, j, k] + self.pressure[i, j, k]
            self.pressure_accum[i, j, k] = corrected
            self.pressure[i, j, k] = corrected
            self.pressure_tmp[i, j, k] = corrected

    @ti.kernel
    def _apply_pressure_outlet_zmin_kernel(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.pressure[i, j, 0] = 0.0
            self.pressure_tmp[i, j, 0] = 0.0

    @ti.kernel
    def _apply_closed_boundary_no_normal_flow_kernel(self, pressure_outlet_zmin: ti.i32):
        for j, k in ti.ndrange(self.ny, self.nz):
            self.velocity[0, j, k].x = 0.0
        for i, k in ti.ndrange(self.nx, self.nz):
            self.velocity[i, 0, k].y = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if pressure_outlet_zmin == 0:
                self.velocity[i, j, 0].z = 0.0

    @ti.kernel
    def _subtract_pressure_gradient_kernel(
        self,
        dt_over_rho: ti.f32,
        pressure_outlet_zmin: ti.i32,
    ):
        for i, j, k in self.velocity:
            im = ti.max(i - 1, 0)
            jm = ti.max(j - 1, 0)
            km = ti.max(k - 1, 0)
            center = self.pressure[i, j, k]
            if self.obstacle[i, j, k] == 1:
                self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            elif self.velocity_dirichlet_boundary_active[i, j, k] != 0:
                self.velocity[i, j, k] = self.velocity_dirichlet_boundary_value_mps[
                    i, j, k
                ]
            else:
                grad = ti.Vector([0.0, 0.0, 0.0])
                if i > 0 and self.obstacle[im, j, k] == 0:
                    grad.x = (center - self.pressure[im, j, k]) / self.center_distance_x_m[i]
                if j > 0 and self.obstacle[i, jm, k] == 0:
                    grad.y = (center - self.pressure[i, jm, k]) / self.center_distance_y_m[j]
                if k > 0 and self.obstacle[i, j, km] == 0:
                    grad.z = (center - self.pressure[i, j, km]) / self.center_distance_z_m[k]
                elif pressure_outlet_zmin == 1 and k == 0:
                    grad.z = 2.0 * center / self.cell_width_z_m[k]
                self.velocity[i, j, k] -= dt_over_rho * grad

    @ti.kernel
    def _apply_zmin_no_backflow_kernel(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0 and self.velocity[i, j, 0].z > 0.0:
                self.velocity[i, j, 0].z = 0.0

    @ti.kernel
    def _local_divergence_cleanup_kernel(
        self,
        dx: ti.f32,
        dy: ti.f32,
        dz: ti.f32,
        relaxation: ti.f32,
    ):
        for i, j, k in self.divergence:
            if self.obstacle[i, j, k] == 0:
                div = self.divergence[i, j, k] - self.volume_source_s[i, j, k]
                im = ti.max(i - 1, 0)
                ip = ti.min(i + 1, self.nx - 1)
                jm = ti.max(j - 1, 0)
                jp = ti.min(j + 1, self.ny - 1)
                km = ti.max(k - 1, 0)
                kp = ti.min(k + 1, self.nz - 1)
                use_x_left = (
                    i > 0
                    and self.obstacle[im, j, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] == 0
                )
                use_x_right = (
                    i < self.nx - 1
                    and self.obstacle[ip, j, k] == 0
                    and self.velocity_dirichlet_boundary_active[ip, j, k] == 0
                )
                use_y_back = (
                    j > 0
                    and self.obstacle[i, jm, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] == 0
                )
                use_y_front = (
                    j < self.ny - 1
                    and self.obstacle[i, jp, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, jp, k] == 0
                )
                use_z_bottom = (
                    k > 0
                    and self.obstacle[i, j, km] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] == 0
                )
                use_z_top = (
                    k < self.nz - 1
                    and self.obstacle[i, j, kp] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, kp] == 0
                )
                denom = 0.0
                if use_x_left:
                    denom += 1.0 / (dx * dx)
                if use_x_right:
                    denom += 1.0 / (dx * dx)
                if use_y_back:
                    denom += 1.0 / (dy * dy)
                if use_y_front:
                    denom += 1.0 / (dy * dy)
                if use_z_bottom:
                    denom += 1.0 / (dz * dz)
                if use_z_top:
                    denom += 1.0 / (dz * dz)
                if denom > 0.0:
                    scale = -relaxation * div / denom
                    if use_x_left:
                        ti.atomic_add(self.velocity[i, j, k].x, scale * (-1.0 / dx))
                    if use_x_right:
                        ti.atomic_add(self.velocity[ip, j, k].x, scale * (1.0 / dx))
                    if use_y_back:
                        ti.atomic_add(self.velocity[i, j, k].y, scale * (-1.0 / dy))
                    if use_y_front:
                        ti.atomic_add(self.velocity[i, jp, k].y, scale * (1.0 / dy))
                    if use_z_bottom:
                        ti.atomic_add(self.velocity[i, j, k].z, scale * (-1.0 / dz))
                    if use_z_top:
                        ti.atomic_add(self.velocity[i, j, kp].z, scale * (1.0 / dz))

    def _solve_pressure_poisson(
        self,
        *,
        iterations: int,
        rhs_scale: float,
        inv_dx2: float,
        inv_dy2: float,
        inv_dz2: float,
        pressure_outlet_zmin: bool,
    ) -> None:
        for _ in range(iterations):
            self._pressure_jacobi_kernel(
                float(rhs_scale),
                float(inv_dx2),
                float(inv_dy2),
                float(inv_dz2),
                1 if pressure_outlet_zmin else 0,
            )
            self._copy_pressure_kernel()

    def _smooth_fv_pressure_fields(
        self,
        *,
        pressure: object,
        rhs: object,
        pressure_interface_matrix_diagonal: object,
        obstacle: object,
        velocity_dirichlet_boundary_active: object,
        velocity_dirichlet_boundary_projection_weight: object,
        tmp: object,
        cell_width_x_m: object,
        cell_width_y_m: object,
        cell_width_z_m: object,
        center_distance_x_m: object,
        center_distance_y_m: object,
        center_distance_z_m: object,
        shape: tuple[int, int, int],
        iterations: int,
        pressure_outlet_zmin: bool,
        omega: float,
    ) -> None:
        for _ in range(max(0, int(iterations))):
            self._pressure_fv_jacobi_kernel(
                pressure,
                rhs,
                pressure_interface_matrix_diagonal,
                obstacle,
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                tmp,
                cell_width_x_m,
                cell_width_y_m,
                cell_width_z_m,
                center_distance_x_m,
                center_distance_y_m,
                center_distance_z_m,
                int(shape[0]),
                int(shape[1]),
                int(shape[2]),
                1 if pressure_outlet_zmin else 0,
                float(omega),
            )
            self._copy_scalar_field_kernel(pressure, tmp)

    def _smooth_fv_pressure_level(
        self,
        level: int,
        *,
        iterations: int,
        pressure_outlet_zmin: bool,
        omega: float,
    ) -> None:
        self._smooth_fv_pressure_fields(
            pressure=self._mg_pressure[level],
            rhs=self._mg_rhs[level],
            pressure_interface_matrix_diagonal=self._mg_pressure_interface_matrix_diagonal[
                level
            ],
            obstacle=self._mg_obstacle[level],
            velocity_dirichlet_boundary_active=self._mg_velocity_dirichlet_boundary_active[
                level
            ],
            velocity_dirichlet_boundary_projection_weight=self._mg_velocity_dirichlet_boundary_projection_weight[
                level
            ],
            tmp=self._mg_tmp[level],
            cell_width_x_m=self._mg_cell_width_x_m[level],
            cell_width_y_m=self._mg_cell_width_y_m[level],
            cell_width_z_m=self._mg_cell_width_z_m[level],
            center_distance_x_m=self._mg_center_distance_x_m[level],
            center_distance_y_m=self._mg_center_distance_y_m[level],
            center_distance_z_m=self._mg_center_distance_z_m[level],
            shape=self._mg_shapes[level],
            iterations=int(iterations),
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            omega=float(omega),
        )

    def _compute_fv_residual_level(self, level: int, *, pressure_outlet_zmin: bool) -> None:
        shape = self._mg_shapes[level]
        self._mg_compute_residual_kernel(
            self._mg_pressure[level],
            self._mg_rhs[level],
            self._mg_pressure_interface_matrix_diagonal[level],
            self._mg_obstacle[level],
            self._mg_velocity_dirichlet_boundary_active[level],
            self._mg_velocity_dirichlet_boundary_projection_weight[level],
            self._mg_residual[level],
            self._mg_cell_width_x_m[level],
            self._mg_cell_width_y_m[level],
            self._mg_cell_width_z_m[level],
            self._mg_center_distance_x_m[level],
            self._mg_center_distance_y_m[level],
            self._mg_center_distance_z_m[level],
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            1 if pressure_outlet_zmin else 0,
        )

    def _prepare_fv_multigrid_rhs(self, rhs_scale: float) -> None:
        self._mg_prepare_level0_kernel(
            self._mg_rhs[0],
            self._mg_residual[0],
            float(rhs_scale),
        )
        for level in range(1, len(self._mg_shapes)):
            self._mg_clear_level_kernel(
                self._mg_pressure[level],
                self._mg_tmp[level],
                self._mg_rhs[level],
                self._mg_residual[level],
                self._mg_pressure_interface_matrix_diagonal[level],
            )

    def _solve_pressure_poisson_fv_jacobi(
        self,
        *,
        iterations: int,
        rhs_scale: float,
        pressure_outlet_zmin: bool,
    ) -> None:
        self._prepare_fv_multigrid_rhs(rhs_scale)
        self._smooth_fv_pressure_level(
            0,
            iterations=int(iterations),
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            omega=0.8,
        )

    def _solve_pressure_poisson_fv_multigrid(
        self,
        *,
        iterations: int,
        rhs_scale: float,
        pressure_outlet_zmin: bool,
        multigrid_cycles: int | None,
    ) -> None:
        self._prepare_fv_multigrid_rhs(rhs_scale)
        level_count = len(self._mg_shapes)
        if level_count <= 1:
            self._smooth_fv_pressure_level(
                0,
                iterations=max(1, int(iterations)),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                omega=0.8,
            )
            return

        cycles = (
            int(multigrid_cycles)
            if multigrid_cycles is not None
            else self.default_multigrid_cycles()
        )
        if cycles <= 0:
            raise ValueError("multigrid_cycles must be positive")
        coarse_correction_scale = 1.0 if self.grid.is_uniform else 0.7
        for _ in range(cycles):
            for level in range(level_count - 1):
                self._smooth_fv_pressure_level(
                    level,
                    iterations=3,
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                    omega=0.8,
                )
                self._compute_fv_residual_level(
                    level,
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
                fine_shape = self._mg_shapes[level]
                self._mg_restrict_residual_kernel(
                    self._mg_residual[level],
                    self._mg_pressure_interface_matrix_diagonal[level],
                    self._mg_obstacle[level],
                    self._mg_velocity_dirichlet_boundary_active[level],
                    self._mg_velocity_dirichlet_boundary_projection_weight[level],
                    self._mg_cell_width_x_m[level],
                    self._mg_cell_width_y_m[level],
                    self._mg_cell_width_z_m[level],
                    self._mg_rhs[level + 1],
                    self._mg_pressure_interface_matrix_diagonal[level + 1],
                    self._mg_pressure[level + 1],
                    self._mg_tmp[level + 1],
                    self._mg_residual[level + 1],
                    self._mg_obstacle[level + 1],
                    self._mg_velocity_dirichlet_boundary_active[level + 1],
                    self._mg_velocity_dirichlet_boundary_projection_weight[level + 1],
                    int(fine_shape[0]),
                    int(fine_shape[1]),
                    int(fine_shape[2]),
                )

            self._smooth_fv_pressure_level(
                level_count - 1,
                iterations=36,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                omega=0.8,
            )
            for level in range(level_count - 2, -1, -1):
                coarse_shape = self._mg_shapes[level + 1]
                self._mg_prolongate_add_kernel(
                    self._mg_pressure[level],
                    self._mg_obstacle[level],
                    self._mg_pressure[level + 1],
                    int(coarse_shape[0]),
                    int(coarse_shape[1]),
                    int(coarse_shape[2]),
                    float(coarse_correction_scale),
                )
                self._smooth_fv_pressure_level(
                    level,
                    iterations=3,
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                    omega=0.8,
                )
        if not self.grid.is_uniform:
            self._smooth_fv_pressure_level(
                0,
                iterations=max(1, int(iterations)),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                omega=0.8,
            )

    def _apply_fv_multigrid_preconditioner(
        self,
        residual: object,
        *,
        pressure_outlet_zmin: bool,
        pre_smooth_iterations: int = 2,
        coarse_smooth_iterations: int = 24,
        post_smooth_iterations: int = 2,
    ) -> None:
        pre_iterations = max(1, int(pre_smooth_iterations))
        coarse_iterations = max(1, int(coarse_smooth_iterations))
        post_iterations = max(1, int(post_smooth_iterations))
        self._pcg_prepare_mg_level0_kernel(
            residual,
            self._pcg_mg_rhs[0],
            self._pcg_mg_pressure[0],
            self._pcg_mg_tmp[0],
            self._pcg_mg_residual[0],
        )
        level_count = len(self._mg_shapes)
        if level_count <= 1:
            self._smooth_fv_pressure_fields(
                pressure=self._pcg_mg_pressure[0],
                rhs=self._pcg_mg_rhs[0],
                pressure_interface_matrix_diagonal=self._mg_pressure_interface_matrix_diagonal[
                    0
                ],
                obstacle=self._mg_obstacle[0],
                velocity_dirichlet_boundary_active=self._mg_velocity_dirichlet_boundary_active[
                    0
                ],
                velocity_dirichlet_boundary_projection_weight=self._mg_velocity_dirichlet_boundary_projection_weight[
                    0
                ],
                tmp=self._pcg_mg_tmp[0],
                cell_width_x_m=self._mg_cell_width_x_m[0],
                cell_width_y_m=self._mg_cell_width_y_m[0],
                cell_width_z_m=self._mg_cell_width_z_m[0],
                center_distance_x_m=self._mg_center_distance_x_m[0],
                center_distance_y_m=self._mg_center_distance_y_m[0],
                center_distance_z_m=self._mg_center_distance_z_m[0],
                shape=self._mg_shapes[0],
                iterations=max(pre_iterations + post_iterations, coarse_iterations),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                omega=0.8,
            )
            return

        coarse_correction_scale = 1.0 if self.grid.is_uniform else 0.7
        for level in range(level_count - 1):
            self._smooth_fv_pressure_fields(
                pressure=self._pcg_mg_pressure[level],
                rhs=self._pcg_mg_rhs[level],
                pressure_interface_matrix_diagonal=self._mg_pressure_interface_matrix_diagonal[
                    level
                ],
                obstacle=self._mg_obstacle[level],
                velocity_dirichlet_boundary_active=self._mg_velocity_dirichlet_boundary_active[
                    level
                ],
                velocity_dirichlet_boundary_projection_weight=self._mg_velocity_dirichlet_boundary_projection_weight[
                    level
                ],
                tmp=self._pcg_mg_tmp[level],
                cell_width_x_m=self._mg_cell_width_x_m[level],
                cell_width_y_m=self._mg_cell_width_y_m[level],
                cell_width_z_m=self._mg_cell_width_z_m[level],
                center_distance_x_m=self._mg_center_distance_x_m[level],
                center_distance_y_m=self._mg_center_distance_y_m[level],
                center_distance_z_m=self._mg_center_distance_z_m[level],
                shape=self._mg_shapes[level],
                iterations=pre_iterations,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                omega=0.8,
            )
            shape = self._mg_shapes[level]
            self._mg_compute_residual_kernel(
                self._pcg_mg_pressure[level],
                self._pcg_mg_rhs[level],
                self._mg_pressure_interface_matrix_diagonal[level],
                self._mg_obstacle[level],
                self._mg_velocity_dirichlet_boundary_active[level],
                self._mg_velocity_dirichlet_boundary_projection_weight[level],
                self._pcg_mg_residual[level],
                self._mg_cell_width_x_m[level],
                self._mg_cell_width_y_m[level],
                self._mg_cell_width_z_m[level],
                self._mg_center_distance_x_m[level],
                self._mg_center_distance_y_m[level],
                self._mg_center_distance_z_m[level],
                int(shape[0]),
                int(shape[1]),
                int(shape[2]),
                1 if pressure_outlet_zmin else 0,
            )
            self._mg_restrict_residual_kernel(
                self._pcg_mg_residual[level],
                self._mg_pressure_interface_matrix_diagonal[level],
                self._mg_obstacle[level],
                self._mg_velocity_dirichlet_boundary_active[level],
                self._mg_velocity_dirichlet_boundary_projection_weight[level],
                self._mg_cell_width_x_m[level],
                self._mg_cell_width_y_m[level],
                self._mg_cell_width_z_m[level],
                self._pcg_mg_rhs[level + 1],
                self._mg_pressure_interface_matrix_diagonal[level + 1],
                self._pcg_mg_pressure[level + 1],
                self._pcg_mg_tmp[level + 1],
                self._pcg_mg_residual[level + 1],
                self._mg_obstacle[level + 1],
                self._mg_velocity_dirichlet_boundary_active[level + 1],
                self._mg_velocity_dirichlet_boundary_projection_weight[level + 1],
                int(shape[0]),
                int(shape[1]),
                int(shape[2]),
            )

        self._smooth_fv_pressure_fields(
            pressure=self._pcg_mg_pressure[level_count - 1],
            rhs=self._pcg_mg_rhs[level_count - 1],
            pressure_interface_matrix_diagonal=self._mg_pressure_interface_matrix_diagonal[
                level_count - 1
            ],
            obstacle=self._mg_obstacle[level_count - 1],
            velocity_dirichlet_boundary_active=self._mg_velocity_dirichlet_boundary_active[
                level_count - 1
            ],
            velocity_dirichlet_boundary_projection_weight=self._mg_velocity_dirichlet_boundary_projection_weight[
                level_count - 1
            ],
            tmp=self._pcg_mg_tmp[level_count - 1],
            cell_width_x_m=self._mg_cell_width_x_m[level_count - 1],
            cell_width_y_m=self._mg_cell_width_y_m[level_count - 1],
            cell_width_z_m=self._mg_cell_width_z_m[level_count - 1],
            center_distance_x_m=self._mg_center_distance_x_m[level_count - 1],
            center_distance_y_m=self._mg_center_distance_y_m[level_count - 1],
            center_distance_z_m=self._mg_center_distance_z_m[level_count - 1],
            shape=self._mg_shapes[level_count - 1],
            iterations=coarse_iterations,
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            omega=0.8,
        )
        for level in range(level_count - 2, -1, -1):
            coarse_shape = self._mg_shapes[level + 1]
            self._mg_prolongate_add_kernel(
                self._pcg_mg_pressure[level],
                self._mg_obstacle[level],
                self._pcg_mg_pressure[level + 1],
                int(coarse_shape[0]),
                int(coarse_shape[1]),
                int(coarse_shape[2]),
                float(coarse_correction_scale),
            )
            self._smooth_fv_pressure_fields(
                pressure=self._pcg_mg_pressure[level],
                rhs=self._pcg_mg_rhs[level],
                pressure_interface_matrix_diagonal=self._mg_pressure_interface_matrix_diagonal[
                    level
                ],
                obstacle=self._mg_obstacle[level],
                velocity_dirichlet_boundary_active=self._mg_velocity_dirichlet_boundary_active[
                    level
                ],
                velocity_dirichlet_boundary_projection_weight=self._mg_velocity_dirichlet_boundary_projection_weight[
                    level
                ],
                tmp=self._pcg_mg_tmp[level],
                cell_width_x_m=self._mg_cell_width_x_m[level],
                cell_width_y_m=self._mg_cell_width_y_m[level],
                cell_width_z_m=self._mg_cell_width_z_m[level],
                center_distance_x_m=self._mg_center_distance_x_m[level],
                center_distance_y_m=self._mg_center_distance_y_m[level],
                center_distance_z_m=self._mg_center_distance_z_m[level],
                shape=self._mg_shapes[level],
                iterations=post_iterations,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                omega=0.8,
            )

    def _solve_pressure_poisson_fv_cg(
        self,
        *,
        iterations: int,
        rhs_scale: float,
        pressure_outlet_zmin: bool,
        tolerance: float,
        preconditioner: str = "auto",
        remove_nullspace_mean: bool = True,
    ) -> None:
        max_iters = max(1, int(iterations))
        relative_tolerance = max(0.0, float(tolerance))
        preconditioner_name = str(preconditioner)
        if preconditioner_name not in CG_PRECONDITIONER_CHOICES:
            raise ValueError(f"unsupported cg_preconditioner: {preconditioner!r}")
        outlet = 1 if pressure_outlet_zmin else 0
        self.last_cg_iterations = 0
        self.last_cg_initial_relative_residual = math.inf
        self.last_cg_relative_residual = math.inf
        self.last_cg_converged = False
        self.last_cg_breakdown = ""
        self.last_cg_host_residual_checks = 0
        self.last_cg_mean_host_reads = 0
        self.last_cg_mean_projection_count = 0
        self.last_cg_unreached_set_mean_projection_count = 0
        self.last_cg_restart_count = 0
        self.last_cg_restart_count_measured = False
        self.last_cg_restart_policy = "not_implemented"
        self.last_cg_breakdown_dAd = 0.0
        self.cg_breakdown_code[None] = 0
        self.cg_breakdown_dAd[None] = 0.0
        anchor_unreached = bool(pressure_outlet_zmin) and (
            int(self._hibm_pressure_unreached_count) > 0
        )
        self.last_hibm_unreached_cells_with_interface_diagonal = 0
        self.last_hibm_unreached_cells_with_interface_coupling = 0
        self.last_hibm_unreached_components_with_interface_hits = 0
        if anchor_unreached:
            # R2-H1: before the unreached-set mean subtraction anchors the
            # flood-disconnected components, measure how much of that set the
            # interface matrix terms already anchor/connect. Overlap here means
            # the zero-mean projection perturbs a non-singular subsystem.
            self._record_unreached_interface_hit_diagnostics()

        self._prepare_fv_multigrid_rhs(rhs_scale)
        self._fv_diagonal_kernel(self.fv_diag, outlet)
        if anchor_unreached:
            self._subtract_unreached_set_mean_device(self._mg_rhs[0])
        if remove_nullspace_mean:
            self._weighted_mean_to_cg_field_kernel(self._mg_rhs[0], -1.0)
            self.last_cg_mean_projection_count += 1
            self._cg_build_positive_rhs_from_cg_mean_kernel(
                self._mg_rhs[0],
                self.cg_z,
            )
        else:
            self._cg_build_positive_rhs_kernel(
                self._mg_rhs[0],
                self.cg_z,
                0.0,
            )
        b_norm = sqrt(max(float(self._weighted_dot_kernel(self.cg_z, self.cg_z)), 0.0))
        if b_norm <= 1.0e-30:
            self._copy_scalar_field_kernel(self.cg_r, self.cg_z)
            self._clear_pressure_kernel()
            self.last_cg_iterations = 0
            self.last_cg_initial_relative_residual = 0.0
            self.last_cg_relative_residual = 0.0
            self.last_cg_converged = True
            return

        self._fv_laplacian_apply_kernel(self.pressure, self.cg_Ad, outlet)
        self._axpby_scalar_field_kernel(self.cg_r, 1.0, self.cg_z, -1.0, self.cg_Ad)
        if anchor_unreached:
            self._subtract_unreached_set_mean_device(self.cg_r)
        if remove_nullspace_mean:
            self._subtract_weighted_mean_device(self.cg_r)
            self.last_cg_mean_projection_count += 1
        self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
        self.last_cg_host_residual_checks += 1
        r_norm = sqrt(max(float(self.cg_rr[None]), 0.0))
        initial_relative_residual = r_norm / b_norm
        self.last_cg_initial_relative_residual = initial_relative_residual
        self.last_cg_relative_residual = initial_relative_residual
        if r_norm <= relative_tolerance * b_norm:
            self.last_cg_converged = True
            return

        use_mg_preconditioner = (
            len(self._mg_shapes) > 1
            and (
                preconditioner_name in {"fv_multigrid", "fv_multigrid_light"}
                or (preconditioner_name == "auto" and not self.grid.is_uniform)
            )
        )
        mg_pre_smooth_iterations = 1 if preconditioner_name == "fv_multigrid_light" else 2
        mg_coarse_smooth_iterations = 12 if preconditioner_name == "fv_multigrid_light" else 24
        mg_post_smooth_iterations = 1 if preconditioner_name == "fv_multigrid_light" else 2
        if use_mg_preconditioner:
            self._apply_fv_multigrid_preconditioner(
                self.cg_r,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                pre_smooth_iterations=mg_pre_smooth_iterations,
                coarse_smooth_iterations=mg_coarse_smooth_iterations,
                post_smooth_iterations=mg_post_smooth_iterations,
            )
        else:
            self._apply_jacobi_preconditioner_kernel(self.cg_r, self.cg_z)
        if anchor_unreached:
            self._subtract_unreached_set_mean_device(self.cg_z)
        if remove_nullspace_mean:
            self._subtract_weighted_mean_device(self.cg_z)
            self.last_cg_mean_projection_count += 1
        self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
        rz = float(self.cg_rz[None])
        if not math.isfinite(rz) or abs(rz) <= 1.0e-300:
            use_mg_preconditioner = False
            self._apply_jacobi_preconditioner_kernel(self.cg_r, self.cg_z)
            self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
            rz = float(self.cg_rz[None])
        if not math.isfinite(rz) or rz <= 0.0:
            self.last_cg_breakdown = "initial residual/preconditioner dot is not positive finite"
            return
        self._copy_scalar_field_kernel(self.cg_d, self.cg_z)

        residual_check_interval = 16
        for iteration in range(1, max_iters + 1):
            self._fv_laplacian_apply_kernel(self.cg_d, self.cg_Ad, outlet)
            self._weighted_dot_to_field_kernel(self.cg_d, self.cg_Ad, self.cg_dAd)
            self._cg_compute_alpha_kernel()

            self._cg_apply_alpha_kernel()
            if anchor_unreached and iteration % 16 == 0:
                self._subtract_unreached_set_mean_device(self.pressure)
                self._subtract_unreached_set_mean_device(self.cg_r)
            if remove_nullspace_mean and iteration % 16 == 0:
                self._subtract_weighted_mean_device(self.pressure)
                self._subtract_weighted_mean_device(self.cg_r)
                self.last_cg_mean_projection_count += 2

            self.last_cg_iterations = iteration
            should_check_residual = (
                iteration % residual_check_interval == 0 or iteration == max_iters
            )
            if should_check_residual:
                self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
                self.last_cg_host_residual_checks += 1
                r_norm = sqrt(max(float(self.cg_rr[None]), 0.0))
                relative_residual = r_norm / b_norm
                self.last_cg_relative_residual = relative_residual
                if relative_residual <= relative_tolerance:
                    self.last_cg_converged = True
                    return
                breakdown_code = int(self.cg_breakdown_code[None])
                if breakdown_code != 0:
                    self.last_cg_breakdown_dAd = float(self.cg_breakdown_dAd[None])
                    self.last_cg_breakdown = "device-side CG scalar update failed"
                    return

            if use_mg_preconditioner:
                self._apply_fv_multigrid_preconditioner(
                    self.cg_r,
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                    pre_smooth_iterations=mg_pre_smooth_iterations,
                    coarse_smooth_iterations=mg_coarse_smooth_iterations,
                    post_smooth_iterations=mg_post_smooth_iterations,
                )
            else:
                self._apply_jacobi_preconditioner_kernel(self.cg_r, self.cg_z)
            if anchor_unreached:
                self._subtract_unreached_set_mean_device(self.cg_z)
            if remove_nullspace_mean:
                self._subtract_weighted_mean_device(self.cg_z)
                self.last_cg_mean_projection_count += 1
            self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz_new)
            if use_mg_preconditioner:
                self._axpby_scalar_field_kernel(self.cg_Ad, 1.0, self.cg_r, -1.0, self.cg_r_old)
                self._weighted_dot_to_field_kernel(self.cg_z, self.cg_Ad, self.cg_beta_numerator)
            self._cg_compute_beta_kernel(1 if use_mg_preconditioner else 0)
            self._cg_update_direction_and_rz_kernel()

    def _solve_pressure_poisson_with_solver(
        self,
        *,
        iterations: int,
        rhs_scale: float,
        inv_dx2: float,
        inv_dy2: float,
        inv_dz2: float,
        pressure_outlet_zmin: bool,
        pressure_solver: str,
        multigrid_cycles: int | None,
        cg_tolerance: float,
        cg_preconditioner: str = "auto",
        remove_nullspace_mean: bool = True,
    ) -> None:
        if pressure_solver in {"jacobi", "compact_jacobi"}:
            self._solve_pressure_poisson(
                iterations=int(iterations),
                rhs_scale=float(rhs_scale),
                inv_dx2=float(inv_dx2),
                inv_dy2=float(inv_dy2),
                inv_dz2=float(inv_dz2),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
            return
        if pressure_solver == "fv_jacobi":
            self._solve_pressure_poisson_fv_jacobi(
                iterations=int(iterations),
                rhs_scale=float(rhs_scale),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
            return
        if pressure_solver == "fv_multigrid":
            self._solve_pressure_poisson_fv_multigrid(
                iterations=int(iterations),
                rhs_scale=float(rhs_scale),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                multigrid_cycles=multigrid_cycles,
            )
            return
        if pressure_solver == "fv_cg":
            self._solve_pressure_poisson_fv_cg(
                iterations=int(iterations),
                rhs_scale=float(rhs_scale),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                tolerance=float(cg_tolerance),
                preconditioner=str(cg_preconditioner),
                remove_nullspace_mean=bool(remove_nullspace_mean),
            )
            return
        raise ValueError(f"unsupported pressure_solver: {pressure_solver!r}")

    def pressure_outlet_fv_flux_report(self, dt_s: float | None = None) -> dict[str, float]:
        step_dt_s = self.dt if dt_s is None else float(dt_s)
        if step_dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self._pressure_outlet_fv_flux_report_kernel(
            float(step_dt_s / self.rho),
        )
        snapshot = self.pressure_outlet_report_snapshot[None]
        self.last_pressure_outlet_report_host_reads = 1
        return {
            "source_volume_flux_m3s": float(snapshot[0]),
            "zmin_pressure_outlet_flux_m3s": float(snapshot[1]),
            "zmin_velocity_outlet_flux_m3s": float(snapshot[2]),
            "zmin_pressure_outlet_to_source_ratio": float(snapshot[3]),
            "zmin_velocity_outlet_to_source_ratio": float(snapshot[4]),
            "zmin_projection_pre_velocity_outlet_flux_m3s": float(snapshot[5]),
            "zmin_projection_post_pressure_velocity_outlet_flux_m3s": float(snapshot[6]),
            "zmin_projection_post_boundary_velocity_outlet_flux_m3s": float(snapshot[7]),
        }

    def project(
        self,
        iterations: int = 40,
        pressure_outlet_zmin: bool = False,
        dt_s: float | None = None,
        preserve_velocity_constraints: bool = False,
        velocity_constraint_blend: float = 1.0,
        velocity_constraint_solid_mobility_ratio: float = 0.0,
        divergence_cleanup_iterations: int = 0,
        divergence_cleanup_relaxation: float = 0.7,
        reset_pressure: bool = False,
        pressure_solver: str = "jacobi",
        multigrid_cycles: int | None = None,
        cg_tolerance: float = 1.0e-6,
        cg_preconditioner: str = "auto",
        pressure_solve_failure_policy: str = "report",
        read_report: bool = True,
    ) -> dict[str, float]:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        step_dt_s = self.dt if dt_s is None else float(dt_s)
        if step_dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        constraint_blend = float(velocity_constraint_blend)
        if not 0.0 <= constraint_blend <= 1.0:
            raise ValueError("velocity_constraint_blend must be in [0, 1]")
        constraint_solid_mobility_ratio = float(velocity_constraint_solid_mobility_ratio)
        if (
            not math.isfinite(constraint_solid_mobility_ratio)
            or constraint_solid_mobility_ratio < 0.0
        ):
            raise ValueError(
                "velocity_constraint_solid_mobility_ratio must be a finite non-negative number"
            )
        cleanup_relaxation = float(divergence_cleanup_relaxation)
        if not 0.0 <= cleanup_relaxation <= 1.0:
            raise ValueError("divergence_cleanup_relaxation must be in [0, 1]")
        cg_relative_tolerance = float(cg_tolerance)
        if not math.isfinite(cg_relative_tolerance) or cg_relative_tolerance < 0.0:
            raise ValueError("cg_tolerance must be a finite non-negative number")
        cg_preconditioner_name = str(cg_preconditioner)
        if cg_preconditioner_name not in CG_PRECONDITIONER_CHOICES:
            raise ValueError(f"unsupported cg_preconditioner: {cg_preconditioner!r}")
        pressure_solve_failure_policy_name = str(pressure_solve_failure_policy)
        if pressure_solve_failure_policy_name not in {"report", "raise"}:
            raise ValueError(
                "pressure_solve_failure_policy must be 'report' or 'raise'"
            )
        pressure_solver_name = str(pressure_solver)
        if pressure_solver_name not in {"jacobi", "compact_jacobi", "fv_jacobi", "fv_multigrid", "fv_cg"}:
            raise ValueError(f"unsupported pressure_solver: {pressure_solver!r}")
        if not self.grid.is_uniform:
            if pressure_solver_name not in {"fv_jacobi", "fv_multigrid", "fv_cg"}:
                raise ValueError(
                    "non-uniform CartesianGrid projection requires an FV pressure solver"
                )
            if divergence_cleanup_iterations > 0:
                raise ValueError(
                    "non-uniform CartesianGrid divergence cleanup requires non-uniform cleanup operators"
                )
        if multigrid_cycles is not None and int(multigrid_cycles) <= 0:
            raise ValueError("multigrid_cycles must be positive")
        report_requested = bool(read_report)
        self.last_divergence_report_host_reads = 0
        empty_stats = {"max_abs": math.nan, "l2": math.nan}
        pressure_interface_policy_report = self.pressure_interface_matrix_terms_report()
        pressure_system_anchored_by_interface_matrix = (
            not bool(pressure_outlet_zmin)
            and float(pressure_interface_policy_report["max_abs_diagonal"]) > 0.0
        )
        pressure_nullspace_zero_mean_projection_applied = (
            not bool(pressure_outlet_zmin)
            and pressure_solver_name == "fv_cg"
            and not pressure_system_anchored_by_interface_matrix
        )
        if pressure_outlet_zmin:
            pressure_nullspace_policy = "pressure_outlet_dirichlet"
            pressure_nullspace_compatibility_measured = True
        elif pressure_system_anchored_by_interface_matrix:
            pressure_nullspace_policy = "interface_matrix_anchored"
            pressure_nullspace_compatibility_measured = pressure_solver_name == "fv_cg"
        elif pressure_solver_name == "fv_cg":
            pressure_nullspace_policy = "closed_neumann_fv_cg_zero_mean"
            pressure_nullspace_compatibility_measured = True
        else:
            pressure_nullspace_policy = "closed_neumann_non_cg_unmeasured"
            pressure_nullspace_compatibility_measured = False
        self.last_project_cg_project_calls = 0
        self.last_project_cg_iterations_total = 0
        self.last_project_cg_iterations_max = 0
        self.last_project_cg_host_residual_checks = 0
        self.last_project_cg_mean_host_reads = 0
        self.last_project_cg_mean_projection_count = 0
        self.last_project_cg_unreached_set_mean_projection_count = 0
        self.last_project_cg_restart_count = 0
        self.last_project_cg_restart_count_measured = False
        self.last_project_cg_restart_policy = (
            "not_implemented"
            if pressure_solver_name == "fv_cg"
            else "not_applicable_non_cg"
        )
        self.last_project_cg_initial_relative_residual_max = 0.0
        self.last_project_cg_relative_residual_max = 0.0
        self.last_project_cg_converged_all = True
        self.last_project_cg_breakdown_count = 0
        self.last_project_cg_breakdown_code = 0
        self.last_project_cg_breakdown_dAd = 0.0
        self.last_project_cg_breakdown = ""
        pressure_solve_failed = False

        def record_cg_stats() -> None:
            nonlocal pressure_solve_failed
            if pressure_solver_name != "fv_cg":
                return
            self.last_project_cg_project_calls += 1
            self.last_project_cg_iterations_total += int(self.last_cg_iterations)
            self.last_project_cg_iterations_max = max(
                int(self.last_project_cg_iterations_max),
                int(self.last_cg_iterations),
            )
            self.last_project_cg_host_residual_checks += int(self.last_cg_host_residual_checks)
            self.last_project_cg_mean_host_reads += int(self.last_cg_mean_host_reads)
            self.last_project_cg_mean_projection_count += int(
                self.last_cg_mean_projection_count
            )
            self.last_project_cg_unreached_set_mean_projection_count += int(
                self.last_cg_unreached_set_mean_projection_count
            )
            self.last_project_cg_restart_count += int(self.last_cg_restart_count)
            if math.isfinite(float(self.last_cg_initial_relative_residual)):
                self.last_project_cg_initial_relative_residual_max = max(
                    float(self.last_project_cg_initial_relative_residual_max),
                    float(self.last_cg_initial_relative_residual),
                )
            if math.isfinite(float(self.last_cg_relative_residual)):
                self.last_project_cg_relative_residual_max = max(
                    float(self.last_project_cg_relative_residual_max),
                    float(self.last_cg_relative_residual),
                )
            if not bool(self.last_cg_converged):
                pressure_solve_failed = True
                self.last_project_cg_converged_all = False
                self.last_project_cg_breakdown_count += 1
                self.last_project_cg_breakdown_code = max(
                    int(self.last_project_cg_breakdown_code),
                    int(self.cg_breakdown_code[None]),
                )
                if math.isfinite(float(self.last_cg_breakdown_dAd)):
                    self.last_project_cg_breakdown_dAd = float(
                        self.last_cg_breakdown_dAd
                    )
                if self.last_cg_breakdown:
                    self.last_project_cg_breakdown = str(self.last_cg_breakdown)

        def handle_pressure_solve_failure() -> None:
            if pressure_solver_name != "fv_cg" or bool(self.last_cg_converged):
                return
            if pressure_solve_failure_policy_name != "raise":
                return
            raise RuntimeError(
                "FV-CG pressure solve did not converge before pressure-gradient "
                "velocity correction "
                f"(iterations={int(self.last_cg_iterations)}, "
                f"relative_residual={float(self.last_cg_relative_residual):.6g}, "
                f"breakdown={self.last_cg_breakdown!r})"
            )

        def apply_velocity_boundary_conditions() -> None:
            self._apply_velocity_dirichlet_boundary_rows_kernel(0)
            if pressure_outlet_zmin:
                self._apply_zmin_no_backflow_kernel()
                self._apply_closed_boundary_no_normal_flow_kernel(1)

        def cleanup_required_from_current_residual() -> bool:
            self._divergence_residual_stats_kernel(
                0,
                1 if int(self._hibm_pressure_unreached_count) > 0 else 0,
            )
            self._update_cleanup_required_from_reduction_kernel()
            self.last_divergence_report_host_reads += 1
            return bool(int(self.cleanup_required[None]))

        final_closed_boundary_cleanup = not pressure_outlet_zmin

        self._reset_zmin_projection_flux_report_kernel()
        apply_velocity_boundary_conditions()
        if pressure_outlet_zmin:
            self._record_zmin_projection_pre_velocity_flux_kernel()
        self.compute_divergence(pressure_outlet_zmin=pressure_outlet_zmin)
        if report_requested:
            pre_projection_raw_stats = self.divergence_stats()
            pre_projection_stats = self.divergence_residual_stats()
        else:
            pre_projection_raw_stats = empty_stats
            pre_projection_stats = empty_stats
        if reset_pressure:
            self._clear_pressure_kernel()
        rhs_scale = self.rho / step_dt_s
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        inv_dz2 = 1.0 / (self.dz * self.dz)
        self._solve_pressure_poisson_with_solver(
            iterations=int(iterations),
            rhs_scale=float(rhs_scale),
            inv_dx2=float(inv_dx2),
            inv_dy2=float(inv_dy2),
            inv_dz2=float(inv_dz2),
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            pressure_solver=pressure_solver_name,
            multigrid_cycles=multigrid_cycles,
            cg_tolerance=cg_relative_tolerance,
            cg_preconditioner=cg_preconditioner_name,
            remove_nullspace_mean=pressure_nullspace_zero_mean_projection_applied,
        )
        record_cg_stats()
        handle_pressure_solve_failure()
        if pressure_outlet_zmin:
            self._record_zmin_pressure_step_pre_velocity_flux_kernel()
        self._subtract_pressure_gradient_kernel(
            float(step_dt_s / self.rho),
            1 if pressure_outlet_zmin else 0,
        )
        if pressure_outlet_zmin:
            self._accumulate_zmin_pressure_correction_flux_kernel()
        apply_velocity_boundary_conditions()
        self.compute_divergence(pressure_outlet_zmin=pressure_outlet_zmin)
        if report_requested:
            projection_raw_stats = self.divergence_stats()
            projection_stats = self.divergence_residual_stats()
        elif pressure_outlet_zmin:
            projection_raw_stats = empty_stats
            projection_stats = empty_stats
            self._divergence_residual_stats_kernel(
                0,
                1 if int(self._hibm_pressure_unreached_count) > 0 else 0,
            )
            self._store_cleanup_target_l2_sq_from_reduction_kernel(
                1.05 * 1.05,
                1.0e-16,
            )
        else:
            projection_raw_stats = empty_stats
            projection_stats = empty_stats
        post_boundary_stats = projection_stats
        post_boundary_raw_stats = projection_raw_stats
        if pressure_outlet_zmin:
            self._copy_pressure_to_accum_kernel()
            apply_velocity_boundary_conditions()
            self.compute_divergence(pressure_outlet_zmin=pressure_outlet_zmin)
            if report_requested:
                post_boundary_raw_stats = self.divergence_stats()
                post_boundary_stats = self.divergence_residual_stats()
                cleanup_target_l2 = max(float(projection_stats["l2"]) * 1.05, 1.0e-8)
            else:
                post_boundary_raw_stats = empty_stats
                post_boundary_stats = empty_stats
                cleanup_needed = cleanup_required_from_current_residual()
            cleanup_iterations = max(1, min(int(iterations), 256))
            for _ in range(3):
                if report_requested:
                    if post_boundary_stats["l2"] <= cleanup_target_l2:
                        break
                elif not cleanup_needed:
                    break
                self._clear_pressure_correction_kernel()
                self._clear_pressure_interface_matrix_rhs_kernel()
                self._solve_pressure_poisson_with_solver(
                    iterations=int(cleanup_iterations),
                    rhs_scale=float(rhs_scale),
                    inv_dx2=float(inv_dx2),
                    inv_dy2=float(inv_dy2),
                    inv_dz2=float(inv_dz2),
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                    pressure_solver=pressure_solver_name,
                    multigrid_cycles=multigrid_cycles,
                    cg_tolerance=cg_relative_tolerance,
                    cg_preconditioner=cg_preconditioner_name,
                    remove_nullspace_mean=pressure_nullspace_zero_mean_projection_applied,
                )
                record_cg_stats()
                handle_pressure_solve_failure()
                if pressure_outlet_zmin:
                    self._record_zmin_pressure_step_pre_velocity_flux_kernel()
                self._subtract_pressure_gradient_kernel(
                    float(step_dt_s / self.rho),
                    1 if pressure_outlet_zmin else 0,
                )
                if pressure_outlet_zmin:
                    self._accumulate_zmin_pressure_correction_flux_kernel()
                self._accumulate_pressure_correction_kernel()
                apply_velocity_boundary_conditions()
                self.compute_divergence(pressure_outlet_zmin=pressure_outlet_zmin)
                if report_requested:
                    post_boundary_raw_stats = self.divergence_stats()
                    post_boundary_stats = self.divergence_residual_stats()
                else:
                    post_boundary_raw_stats = empty_stats
                    post_boundary_stats = empty_stats
                    cleanup_needed = cleanup_required_from_current_residual()
        if preserve_velocity_constraints:
            self._apply_velocity_constraints_kernel(
                constraint_blend,
                constraint_solid_mobility_ratio,
                1 if report_requested else 0,
            )
            apply_velocity_boundary_conditions()
        if final_closed_boundary_cleanup and int(divergence_cleanup_iterations) > 0:
            self._apply_closed_boundary_no_normal_flow_kernel(0)
        for _ in range(max(0, int(divergence_cleanup_iterations))):
            self.compute_divergence(pressure_outlet_zmin=pressure_outlet_zmin)
            self._local_divergence_cleanup_kernel(
                float(self.dx),
                float(self.dy),
                float(self.dz),
                float(cleanup_relaxation),
            )
            apply_velocity_boundary_conditions()
            if final_closed_boundary_cleanup:
                self._apply_closed_boundary_no_normal_flow_kernel(0)
        if final_closed_boundary_cleanup:
            self._apply_closed_boundary_no_normal_flow_kernel(0)
        if pressure_outlet_zmin:
            self._record_zmin_projection_post_boundary_velocity_flux_kernel()
        if not report_requested:
            return {}
        self.compute_divergence(pressure_outlet_zmin=pressure_outlet_zmin)
        (
            final_raw_stats,
            final_stats,
            final_interior_raw_stats,
            final_interior_stats,
            velocity_dirichlet_near_raw_stats,
            velocity_dirichlet_near_stats,
            velocity_dirichlet_far_raw_stats,
            velocity_dirichlet_far_stats,
            pressure_correctable_raw_stats,
            pressure_correctable_stats,
            pressure_fixed_raw_stats,
            pressure_fixed_stats,
            interior_pressure_correctable_raw_stats,
            interior_pressure_correctable_stats,
            interior_pressure_fixed_raw_stats,
            interior_pressure_fixed_stats,
        ) = self.final_and_dirichlet_partition_report_stats(
            pressure_outlet_zmin=pressure_outlet_zmin,
        )
        return {
            "l2": final_stats["l2"],
            "max_abs": final_stats["max_abs"],
            "raw_l2": final_raw_stats["l2"],
            "raw_max_abs": final_raw_stats["max_abs"],
            "interior_l2": final_interior_stats["l2"],
            "interior_max_abs": final_interior_stats["max_abs"],
            "interior_raw_l2": final_interior_raw_stats["l2"],
            "interior_raw_max_abs": final_interior_raw_stats["max_abs"],
            "pre_projection_l2": pre_projection_stats["l2"],
            "pre_projection_max_abs": pre_projection_stats["max_abs"],
            "pre_projection_raw_l2": pre_projection_raw_stats["l2"],
            "pre_projection_raw_max_abs": pre_projection_raw_stats["max_abs"],
            "projection_l2": projection_stats["l2"],
            "projection_max_abs": projection_stats["max_abs"],
            "projection_raw_l2": projection_raw_stats["l2"],
            "projection_raw_max_abs": projection_raw_stats["max_abs"],
            "post_boundary_l2": post_boundary_stats["l2"],
            "post_boundary_max_abs": post_boundary_stats["max_abs"],
            "post_boundary_raw_l2": post_boundary_raw_stats["l2"],
            "post_boundary_raw_max_abs": post_boundary_raw_stats["max_abs"],
            "post_constraint_l2": final_stats["l2"],
            "post_constraint_max_abs": final_stats["max_abs"],
            "post_constraint_raw_l2": final_raw_stats["l2"],
            "post_constraint_raw_max_abs": final_raw_stats["max_abs"],
            "velocity_dirichlet_near_l2": velocity_dirichlet_near_stats["l2"],
            "velocity_dirichlet_near_max_abs": velocity_dirichlet_near_stats["max_abs"],
            "velocity_dirichlet_near_raw_l2": velocity_dirichlet_near_raw_stats["l2"],
            "velocity_dirichlet_near_raw_max_abs": (
                velocity_dirichlet_near_raw_stats["max_abs"]
            ),
            "velocity_dirichlet_far_l2": velocity_dirichlet_far_stats["l2"],
            "velocity_dirichlet_far_max_abs": velocity_dirichlet_far_stats["max_abs"],
            "velocity_dirichlet_far_raw_l2": velocity_dirichlet_far_raw_stats["l2"],
            "velocity_dirichlet_far_raw_max_abs": (
                velocity_dirichlet_far_raw_stats["max_abs"]
            ),
            "pressure_correctable_l2": pressure_correctable_stats["l2"],
            "pressure_correctable_max_abs": pressure_correctable_stats["max_abs"],
            "pressure_correctable_cell_count": pressure_correctable_stats["count"],
            "pressure_correctable_raw_l2": pressure_correctable_raw_stats["l2"],
            "pressure_correctable_raw_max_abs": (
                pressure_correctable_raw_stats["max_abs"]
            ),
            "pressure_fixed_l2": pressure_fixed_stats["l2"],
            "pressure_fixed_max_abs": pressure_fixed_stats["max_abs"],
            "pressure_fixed_cell_count": pressure_fixed_stats["count"],
            "pressure_fixed_raw_l2": pressure_fixed_raw_stats["l2"],
            "pressure_fixed_raw_max_abs": pressure_fixed_raw_stats["max_abs"],
            "interior_pressure_correctable_l2": (
                interior_pressure_correctable_stats["l2"]
            ),
            "interior_pressure_correctable_max_abs": (
                interior_pressure_correctable_stats["max_abs"]
            ),
            "interior_pressure_correctable_cell_count": (
                interior_pressure_correctable_stats["count"]
            ),
            "interior_pressure_correctable_raw_l2": (
                interior_pressure_correctable_raw_stats["l2"]
            ),
            "interior_pressure_correctable_raw_max_abs": (
                interior_pressure_correctable_raw_stats["max_abs"]
            ),
            "interior_pressure_fixed_l2": interior_pressure_fixed_stats["l2"],
            "interior_pressure_fixed_max_abs": (
                interior_pressure_fixed_stats["max_abs"]
            ),
            "interior_pressure_fixed_cell_count": (
                interior_pressure_fixed_stats["count"]
            ),
            "interior_pressure_fixed_raw_l2": (
                interior_pressure_fixed_raw_stats["l2"]
            ),
            "interior_pressure_fixed_raw_max_abs": (
                interior_pressure_fixed_raw_stats["max_abs"]
            ),
            "pressure_nullspace_policy": pressure_nullspace_policy,
            "pressure_nullspace_compatibility_measured": bool(
                pressure_nullspace_compatibility_measured
            ),
            "pressure_nullspace_zero_mean_projection_applied": bool(
                pressure_nullspace_zero_mean_projection_applied
            ),
            "pressure_system_anchored_by_interface_matrix": bool(
                pressure_system_anchored_by_interface_matrix
            ),
            "pressure_solve_failure_policy": pressure_solve_failure_policy_name,
            "pressure_solve_failed": bool(pressure_solve_failed),
            "pressure_solve_failure_action": (
                "reported" if pressure_solve_failed else "none"
            ),
            "cg_project_calls": int(self.last_project_cg_project_calls),
            "cg_iterations_total": int(self.last_project_cg_iterations_total),
            "cg_iterations_max": int(self.last_project_cg_iterations_max),
            "cg_host_residual_checks": int(self.last_project_cg_host_residual_checks),
            "cg_mean_host_reads": int(self.last_project_cg_mean_host_reads),
            "cg_mean_projection_count": int(
                self.last_project_cg_mean_projection_count
            ),
            "cg_unreached_set_mean_projection_count": int(
                self.last_project_cg_unreached_set_mean_projection_count
            ),
            "hibm_pressure_unreached_cell_count": int(
                self.last_hibm_pressure_unreached_cell_count
            ),
            "hibm_pressure_reachability_converged": bool(
                self.last_hibm_pressure_reachability_converged
            ),
            "cg_unreached_component_count": int(
                self.last_hibm_pressure_unreached_component_count
            ),
            "cg_unreached_component_overflow": bool(
                self.last_hibm_pressure_unreached_component_overflow
            ),
            "unreached_cells_with_interface_diagonal": int(
                self.last_hibm_unreached_cells_with_interface_diagonal
            ),
            "unreached_cells_with_interface_coupling": int(
                self.last_hibm_unreached_cells_with_interface_coupling
            ),
            "unreached_components_with_interface_hits": int(
                self.last_hibm_unreached_components_with_interface_hits
            ),
            "hibm_pressure_component_labels_converged": bool(
                self.last_hibm_pressure_component_labels_converged
            ),
            "hibm_solid_band_last_marked_increment": int(
                self.last_hibm_solid_band_marked_increment
            ),
            "hibm_solid_band_interior_cells": int(
                self.last_hibm_solid_band_interior_cells
            ),
            "hibm_solid_band_enclosed_water_cells": int(
                self.last_hibm_solid_band_enclosed_water_cells
            ),
            # S2-A8'' host-mirror passthrough: -1 = the converted-cell
            # pressure fill never ran; otherwise the count from the most
            # recent fill_hibm_converted_cell_pressures() call. The fill
            # runs post-projection (HIBM assemble), so within one
            # assemble step this key shows the PREVIOUS step's fill.
            "hibm_pressure_filled_cell_count": int(
                self.last_hibm_pressure_filled_cell_count
            ),
            "cg_restart_count": int(self.last_project_cg_restart_count),
            "cg_restart_count_measured": bool(
                self.last_project_cg_restart_count_measured
            ),
            "cg_restart_policy": str(self.last_project_cg_restart_policy),
            "cg_initial_relative_residual_max": float(
                self.last_project_cg_initial_relative_residual_max
            ),
            "cg_relative_residual_max": float(self.last_project_cg_relative_residual_max),
            "cg_converged_all": bool(self.last_project_cg_converged_all),
            "cg_breakdown_count": int(self.last_project_cg_breakdown_count),
            "cg_breakdown_code": int(self.last_project_cg_breakdown_code),
            "cg_breakdown_dAd": float(self.last_project_cg_breakdown_dAd),
            "cg_breakdown": str(self.last_project_cg_breakdown),
        }

    @staticmethod
    def _read_vector(field: ti.template()) -> tuple[float, float, float]:
        value = field[None]
        return (float(value[0]), float(value[1]), float(value[2]))
