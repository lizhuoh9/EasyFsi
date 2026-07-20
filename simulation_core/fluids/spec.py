from dataclasses import dataclass

from simulation_core.fluids.grid import CartesianGrid, GradedGridSpec, build_graded_grid


class _GradedCartesianGrid(CartesianGrid):
    """Materialized grid tagged so dataclasses.replace preserves provenance."""


def _materialize_graded_grid(spec: GradedGridSpec) -> _GradedCartesianGrid:
    grid = build_graded_grid(spec)
    return _GradedCartesianGrid(
        bounds_min_m=grid.bounds_min_m,
        cell_widths_x_m=grid.cell_widths_x_m,
        cell_widths_y_m=grid.cell_widths_y_m,
        cell_widths_z_m=grid.cell_widths_z_m,
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
        if self.graded_grid is not None:
            grid = _materialize_graded_grid(self.graded_grid)
            # dataclasses.replace() copies the materialized public
            # cartesian_grid from an existing graded spec. Accept exactly
            # that derived grid, but reject a genuinely distinct pair.
            if self.cartesian_grid is not None and (
                not isinstance(self.cartesian_grid, _GradedCartesianGrid)
                or self.cartesian_grid != grid
            ):
                raise ValueError("cartesian_grid and graded_grid are mutually exclusive")
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
