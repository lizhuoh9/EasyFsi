from __future__ import annotations

import math
import os
import unittest

import numpy as np

from benchmarks.official import solid_mpm_fsi_runner
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig
from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)


RUNTIME = TaichiRuntimeConfig(arch="cuda")


@unittest.skipIf(
    os.environ.get("GITHUB_ACTIONS") == "true"
    and os.environ.get("HIBM_RUN_CUDA_TRACTION_PROBE_TESTS") != "1",
    "simulation_core is GPU-only; set HIBM_RUN_CUDA_TRACTION_PROBE_TESTS=1 on a CUDA runner",
)
class HibmMpmTractionProbeLadderControlTests(unittest.TestCase):
    def test_default_ladder_controls_preserve_current_pressure_only_path(self):
        default_markers, default_fluid = _single_marker_fixture()
        explicit_default_markers, explicit_default_fluid = _single_marker_fixture()
        _set_step_pressure(default_fluid)
        _set_step_pressure(explicit_default_fluid)

        _sample(default_markers, default_fluid)
        _sample(
            explicit_default_markers,
            explicit_default_fluid,
            pressure_probe_ladder_start_offset_cells=None,
            pressure_probe_ladder_spacing_cells=0.5,
            pressure_probe_ladder_rung_count=5,
            pressure_probe_ladder_mode="current_normal_cell_ladder",
        )

        default = default_markers.stress_marker_diagnostics()[0]
        explicit_default = explicit_default_markers.stress_marker_diagnostics()[0]

        self.assertEqual(default["position_m"], explicit_default["position_m"])
        self.assertEqual(
            default["pressure_probe_origin_m"],
            explicit_default["pressure_probe_origin_m"],
        )
        self.assertEqual(
            default["inside_probe_nearest_cell"],
            explicit_default["inside_probe_nearest_cell"],
        )
        self.assertEqual(
            default["outside_probe_nearest_cell"],
            explicit_default["outside_probe_nearest_cell"],
        )
        self.assertEqual(default["inside_probe_rung"], explicit_default["inside_probe_rung"])
        self.assertEqual(default["outside_probe_rung"], explicit_default["outside_probe_rung"])
        self.assertAlmostEqual(
            default["pressure_jump_pa"],
            explicit_default["pressure_jump_pa"],
        )
        self.assertAlmostEqual(
            default["total_traction_pa"][2],
            explicit_default["total_traction_pa"][2],
        )

    def test_explicit_ladder_start_changes_sampling_without_moving_marker(self):
        default_markers, default_fluid = _single_marker_fixture()
        shifted_markers, shifted_fluid = _single_marker_fixture()
        _set_step_pressure(default_fluid)
        _set_step_pressure(shifted_fluid)

        _sample(default_markers, default_fluid)
        _sample(
            shifted_markers,
            shifted_fluid,
            pressure_probe_ladder_start_offset_cells=0.25,
            pressure_probe_ladder_spacing_cells=0.25,
            pressure_probe_ladder_rung_count=5,
            pressure_probe_ladder_mode="current_normal_cell_ladder",
        )

        default = default_markers.stress_marker_diagnostics()[0]
        shifted = shifted_markers.stress_marker_diagnostics()[0]

        self.assertEqual(shifted["position_m"], default["position_m"])
        self.assertEqual(
            shifted["pressure_probe_origin_m"],
            default["pressure_probe_origin_m"],
        )
        self.assertEqual(shifted["pressure_probe_origin_source"], "marker_position")
        self.assertFalse(shifted["pressure_probe_origin_explicit"])
        self.assertNotEqual(
            shifted["inside_probe_grid_coordinate"],
            default["inside_probe_grid_coordinate"],
        )
        self.assertNotEqual(
            shifted["outside_probe_grid_coordinate"],
            default["outside_probe_grid_coordinate"],
        )
        self.assertAlmostEqual(shifted["inside_probe_multiplier"], 0.25)
        self.assertAlmostEqual(shifted["outside_probe_multiplier"], 0.25)

    def test_invalid_ladder_controls_fail_fast(self):
        markers, fluid = _single_marker_fixture()

        with self.assertRaisesRegex(ValueError, "pressure_probe_ladder_start_offset_cells"):
            _sample(
                markers,
                fluid,
                pressure_probe_ladder_start_offset_cells=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "pressure_probe_ladder_spacing_cells"):
            _sample(markers, fluid, pressure_probe_ladder_spacing_cells=math.nan)
        with self.assertRaisesRegex(ValueError, "pressure_probe_ladder_rung_count"):
            _sample(markers, fluid, pressure_probe_ladder_rung_count=0)
        with self.assertRaisesRegex(ValueError, "pressure_probe_ladder_mode"):
            _sample(markers, fluid, pressure_probe_ladder_mode="unsupported")
        with self.assertRaisesRegex(ValueError, "pressure-only diagnostics"):
            _sample(
                markers,
                fluid,
                viscosity_pa_s=1.0,
                pressure_probe_ladder_start_offset_cells=0.25,
            )

    def test_config_ladder_controls_are_diagnostic_only(self):
        default_config = VerticalFlapFsiConfig()
        self.assertIsNone(default_config.traction_pressure_probe_start_offset_cells)
        self.assertAlmostEqual(
            default_config.traction_pressure_probe_ladder_spacing_cells,
            0.5,
        )
        self.assertEqual(default_config.traction_pressure_probe_ladder_rung_count, 5)
        self.assertEqual(
            default_config.traction_pressure_probe_ladder_mode,
            "current_normal_cell_ladder",
        )
        self.assertTrue(
            solid_mpm_fsi_runner._is_default_traction_formulation(default_config)
        )

        non_default = VerticalFlapFsiConfig(
            traction_pressure_probe_start_offset_cells=0.75,
            traction_pressure_probe_ladder_spacing_cells=0.5,
            traction_pressure_probe_ladder_rung_count=5,
        )
        self.assertFalse(solid_mpm_fsi_runner._is_default_traction_formulation(non_default))
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(non_default)

        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=1,
                traction_pressure_probe_start_offset_cells=0.75,
                traction_pressure_probe_ladder_spacing_cells=0.5,
                traction_pressure_probe_ladder_rung_count=5,
            )
        )


def _single_marker_fixture():
    markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
    markers.load_markers(
        positions_m=((0.625, 0.625, 0.5),),
        velocities_mps=((0.0, 0.0, 0.0),),
        normals=((0.0, 0.0, 1.0),),
        areas_m2=(1.0,),
        region_ids=(101,),
    )
    fluid = CartesianFluidSolver(
        FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
        runtime=RUNTIME,
    )
    return markers, fluid


def _set_step_pressure(fluid: CartesianFluidSolver) -> None:
    pressure = np.zeros((8, 8, 8), dtype=np.float32)
    pressure[:, :, :4] = 5.0
    pressure[:, :, 4:] = 1.0
    fluid.pressure.from_numpy(pressure)


def _sample(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    *,
    viscosity_pa_s: float = 0.0,
    **ladder_controls,
):
    return markers.sample_fluid_stress_to_marker_tractions(
        fluid.velocity,
        fluid.pressure,
        fluid.obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        fluid.grid.grid_nodes,
        viscosity_pa_s=viscosity_pa_s,
        two_sided_pressure=True,
        **ladder_controls,
    )


if __name__ == "__main__":
    unittest.main()
