from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


class CartesianGridDeviceGeometryContracts(unittest.TestCase):
    """Host geometry must remain finite and strictly positive on the device."""

    def test_cartesian_grid_rejects_nonfinite_axis_widths(self) -> None:
        valid_axis = (0.25,) * 4
        for axis_name in ("cell_widths_x_m", "cell_widths_y_m", "cell_widths_z_m"):
            for invalid_width in (float("nan"), float("inf"), float("-inf")):
                axes = {
                    "cell_widths_x_m": valid_axis,
                    "cell_widths_y_m": valid_axis,
                    "cell_widths_z_m": valid_axis,
                }
                axes[axis_name] = (0.25, invalid_width, 0.25, 0.25)
                with self.subTest(axis=axis_name, invalid_width=invalid_width):
                    with self.assertRaises(ValueError):
                        CartesianGrid(
                            bounds_min_m=(0.0, 0.0, 0.0),
                            **axes,
                        )

    def test_solver_rejects_positive_width_that_underflows_in_f32_geometry(
        self,
    ) -> None:
        underflowing_width_m = 1.0e-50
        self.assertGreater(underflowing_width_m, 0.0)
        self.assertEqual(float(np.float32(underflowing_width_m)), 0.0)
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(underflowing_width_m,) * 4,
            cell_widths_y_m=(0.25,) * 4,
            cell_widths_z_m=(0.25,) * 4,
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )

        with self.assertRaises(ValueError):
            CartesianFluidSolver(
                spec,
                runtime=TaichiRuntimeConfig(arch="cuda"),
            )


if __name__ == "__main__":
    unittest.main()
