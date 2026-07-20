from __future__ import annotations

import inspect
import re
import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


_F64_FLUX_LEDGER_FIELDS = (
    "report_source_volume_flux_m3s",
    "report_positive_source_volume_flux_m3s",
    "report_abs_source_volume_flux_m3s",
    "report_zmin_reachable_source_volume_flux_m3s",
    "report_zmin_unreached_source_volume_flux_m3s",
    "report_zmin_unreached_source_abs_flux_m3s",
    "report_zmin_pressure_outlet_flux_m3s",
    "report_zmin_velocity_outlet_flux_m3s",
    "report_zmin_pressure_outlet_flux_ratio",
    "report_zmin_velocity_outlet_flux_ratio",
    "report_zmin_projection_pre_velocity_outlet_flux_m3s",
    "report_zmin_pressure_step_pre_velocity_outlet_flux_m3s",
    "report_zmin_projection_post_pressure_velocity_outlet_flux_m3s",
    "report_zmin_projection_post_boundary_velocity_outlet_flux_m3s",
)


def _graded_precision_solver() -> tuple[CartesianFluidSolver, CartesianGrid]:
    grid = CartesianGrid(
        bounds_min_m=(0.0, 0.0, 0.0),
        cell_widths_x_m=(0.0012, 0.0013, 0.0014, 0.0015),
        cell_widths_y_m=(0.000390625, 0.00042, 0.00047, 0.00051),
        cell_widths_z_m=(0.00046875, 0.00053, 0.00059, 0.00067),
    )
    solver = CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=5.0e-4,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )
    return solver, grid


class PressureOutletLedgerPrecisionContracts(unittest.TestCase):
    """Keep health/report geometry out of pre-cast f32 products."""

    def test_z_face_area_helper_casts_each_width_before_multiplication(self) -> None:
        helper = getattr(CartesianFluidSolver, "_z_face_area_f64_m2", None)
        self.assertIsNotNone(
            helper,
            "zmin flux reports need one reusable f64 face-area authority",
        )
        source = " ".join(inspect.getsource(helper).split())
        self.assertIn("ti.cast(self.cell_width_x_m[i], ti.f64)", source)
        self.assertIn("ti.cast(self.cell_width_y_m[j], ti.f64)", source)
        self.assertNotIn(
            "ti.cast(self.cell_width_x_m[i] * self.cell_width_y_m[j], ti.f64)",
            source,
        )

    def test_flux_report_kernels_use_f64_geometry_helpers(self) -> None:
        face_area_consumers = (
            "_record_zmin_projection_pre_velocity_flux_kernel",
            "_record_zmin_pressure_step_pre_velocity_flux_kernel",
            "_accumulate_zmin_pressure_correction_flux_kernel",
            "_record_zmin_projection_post_boundary_velocity_flux_kernel",
            "_pressure_outlet_fv_flux_report_kernel",
        )
        direct_area_product = (
            "self.cell_width_x_m[i] * self.cell_width_y_m[j]"
        )
        for kernel_name in face_area_consumers:
            with self.subTest(kernel=kernel_name):
                source = inspect.getsource(getattr(CartesianFluidSolver, kernel_name))
                self.assertIn("self._z_face_area_f64_m2(i, j)", source)
                self.assertNotIn(direct_area_product, source)

        report_source = inspect.getsource(
            CartesianFluidSolver._pressure_outlet_fv_flux_report_kernel
        )
        self.assertIn("self._cell_volume_f64_m3(i, j, k)", report_source)
        self.assertNotIn(
            "self.cell_width_x_m[i] * self.cell_width_y_m[j] * self.cell_width_z_m[k]",
            report_source,
        )

    def test_air_backed_volume_uses_f64_cell_volume_authority(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._mark_hibm_air_backed_cells_kernel
        )
        self.assertIn("self._cell_volume_f64_m3(i, j, k)", source)
        self.assertNotIn("self._cell_volume_m3(i, j, k)", source)

    def test_pressure_outlet_health_flux_ledgers_are_f64(self) -> None:
        source = inspect.getsource(CartesianFluidSolver.__init__)
        for field_name in _F64_FLUX_LEDGER_FIELDS:
            with self.subTest(field=field_name):
                declaration = re.compile(
                    rf"self\.{field_name}\s*=\s*ti\.field\(\s*"
                    rf"dtype\s*=\s*ti\.f64\s*,\s*shape\s*=\s*\(\)\s*\)",
                    re.DOTALL,
                )
                self.assertRegex(source, declaration)
        self.assertRegex(
            source,
            re.compile(
                r"self\.pressure_outlet_report_snapshot\s*=\s*"
                r"ti\.Vector\.field\(\s*10\s*,\s*dtype\s*=\s*ti\.f64",
                re.DOTALL,
            ),
        )

    def test_graded_source_and_outlet_flux_match_promoted_f64_geometry(self) -> None:
        solver, grid = _graded_precision_solver()
        source = np.zeros(grid.grid_nodes, dtype=np.float32)
        velocity = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
        source[0, 0, 1] = np.float32(1.0e12)
        velocity[0, 0, 0, 2] = np.float32(-1.0e8)
        solver.volume_source_s.from_numpy(source)
        solver.velocity.from_numpy(velocity)

        report = solver.pressure_outlet_fv_flux_report()
        width_x_f64 = np.float64(np.float32(grid.cell_widths_x_m[0]))
        width_y_f64 = np.float64(np.float32(grid.cell_widths_y_m[0]))
        width_z_f64 = np.float64(np.float32(grid.cell_widths_z_m[1]))
        expected_source_flux_m3s = (
            np.float64(source[0, 0, 1])
            * width_x_f64
            * width_y_f64
            * width_z_f64
        )
        expected_outlet_flux_m3s = (
            -np.float64(velocity[0, 0, 0, 2])
            * width_x_f64
            * width_y_f64
        )

        self.assertAlmostEqual(
            report["source_volume_flux_m3s"],
            expected_source_flux_m3s,
            delta=1.0e-11,
        )
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_flux_m3s"],
            expected_outlet_flux_m3s,
            delta=1.0e-11,
        )

    def test_air_backed_volume_matches_promoted_f64_geometry(self) -> None:
        solver, grid = _graded_precision_solver()
        solver.hibm_air_component_selected[0] = 1
        solver.hibm_pressure_unreached_component_label[0, 0, 0] = -1

        solver._mark_hibm_air_backed_cells_kernel()

        expected_volume_m3 = (
            np.float64(np.float32(grid.cell_widths_x_m[0]))
            * np.float64(np.float32(grid.cell_widths_y_m[0]))
            * np.float64(np.float32(grid.cell_widths_z_m[0]))
        )
        self.assertEqual(int(solver.report_hibm_air_backed_cells[None]), 1)
        self.assertAlmostEqual(
            float(solver.report_hibm_air_backed_cell_volume_m3[None]),
            expected_volume_m3,
            delta=1.0e-24,
        )


if __name__ == "__main__":
    unittest.main()
