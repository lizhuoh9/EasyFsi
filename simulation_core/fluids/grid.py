from dataclasses import dataclass


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
