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
class HibmMpmTractionPressurePairAnchorMapTests(unittest.TestCase):
    def test_default_independent_ladder_is_unchanged_by_anchor_storage(self):
        baseline_markers, baseline_fluid = _single_marker_fixture()
        anchored_markers, anchored_fluid = _single_marker_fixture()
        _set_step_pressure(baseline_fluid)
        _set_step_pressure(anchored_fluid)
        anchored_markers.set_pressure_pair_anchor_cells(
            inside_cells=((2, 2, 2),),
            outside_cells=((2, 2, 5),),
        )

        _sample(baseline_markers, baseline_fluid)
        _sample(anchored_markers, anchored_fluid)

        baseline = baseline_markers.stress_marker_diagnostics()[0]
        anchored = anchored_markers.stress_marker_diagnostics()[0]
        self.assertEqual(anchored["pressure_pair_policy"], "independent_ladder")
        self.assertFalse(anchored["pressure_pair_selected"])
        self.assertTrue(anchored["pressure_pair_anchor_active"])
        self.assertEqual(anchored["pressure_pair_anchor_source"], "api")
        self.assertFalse(anchored["pressure_pair_anchor_fallback_used"])
        self.assertAlmostEqual(anchored["pressure_jump_pa"], baseline["pressure_jump_pa"])
        self.assertEqual(
            anchored["inside_probe_nearest_cell"],
            baseline["inside_probe_nearest_cell"],
        )
        self.assertEqual(
            anchored["outside_probe_nearest_cell"],
            baseline["outside_probe_nearest_cell"],
        )

    def test_baseline_anchored_cell_pair_reads_configured_pressure_cells(self):
        markers, fluid = _single_marker_fixture()
        _set_custom_pressure(fluid, inside_value=7.5, outside_value=1.25)
        markers.set_pressure_pair_anchor_cells(
            inside_cells=((2, 2, 2),),
            outside_cells=((2, 2, 5),),
        )

        _sample(markers, fluid, pressure_pair_policy="baseline_anchored_cell_pair")

        diagnostic = markers.stress_marker_diagnostics()[0]
        self.assertTrue(diagnostic["valid"])
        self.assertEqual(
            diagnostic["pressure_pair_policy"],
            "baseline_anchored_cell_pair",
        )
        self.assertTrue(diagnostic["pressure_pair_selected"])
        self.assertTrue(diagnostic["pressure_pair_anchor_active"])
        self.assertEqual(diagnostic["pressure_pair_anchor_source"], "api")
        self.assertFalse(diagnostic["pressure_pair_fallback_used"])
        self.assertFalse(diagnostic["pressure_pair_anchor_fallback_used"])
        self.assertEqual(diagnostic["pressure_pair_inside_cell"], [2, 2, 2])
        self.assertEqual(diagnostic["pressure_pair_outside_cell"], [2, 2, 5])
        self.assertEqual(diagnostic["pressure_pair_anchor_inside_cell"], [2, 2, 2])
        self.assertEqual(diagnostic["pressure_pair_anchor_outside_cell"], [2, 2, 5])
        self.assertEqual(diagnostic["inside_probe_nearest_cell"], [2, 2, 2])
        self.assertEqual(diagnostic["outside_probe_nearest_cell"], [2, 2, 5])
        self.assertAlmostEqual(diagnostic["inside_pressure_pa"], 7.5)
        self.assertAlmostEqual(diagnostic["outside_pressure_pa"], 1.25)
        self.assertAlmostEqual(diagnostic["pressure_jump_pa"], 6.25)
        self.assertAlmostEqual(diagnostic["total_traction_pa"][2], 6.25)

    def test_probe_origin_changes_do_not_change_anchor_cells(self):
        first_markers, first_fluid = _single_marker_fixture()
        shifted_markers, shifted_fluid = _single_marker_fixture(
            pressure_probe_origins_m=((0.625, 0.625, 0.75),)
        )
        _set_custom_pressure(first_fluid, inside_value=9.0, outside_value=2.0)
        _set_custom_pressure(shifted_fluid, inside_value=9.0, outside_value=2.0)
        for markers in (first_markers, shifted_markers):
            markers.set_pressure_pair_anchor_cells(
                inside_cells=((2, 2, 2),),
                outside_cells=((2, 2, 5),),
            )

        _sample(first_markers, first_fluid, pressure_pair_policy="baseline_anchored_cell_pair")
        _sample(shifted_markers, shifted_fluid, pressure_pair_policy="baseline_anchored_cell_pair")

        first = first_markers.stress_marker_diagnostics()[0]
        shifted = shifted_markers.stress_marker_diagnostics()[0]
        self.assertEqual(shifted["pressure_probe_origin_source"], "explicit")
        self.assertEqual(first["pressure_pair_inside_cell"], shifted["pressure_pair_inside_cell"])
        self.assertEqual(first["pressure_pair_outside_cell"], shifted["pressure_pair_outside_cell"])
        self.assertEqual(
            first["pressure_pair_anchor_inside_cell"],
            shifted["pressure_pair_anchor_inside_cell"],
        )
        self.assertEqual(
            first["pressure_pair_anchor_outside_cell"],
            shifted["pressure_pair_anchor_outside_cell"],
        )
        self.assertAlmostEqual(first["pressure_jump_pa"], shifted["pressure_jump_pa"])
        self.assertAlmostEqual(first["total_traction_pa"][2], shifted["total_traction_pa"][2])

    def test_missing_anchor_fails_closed_without_fallback(self):
        markers, fluid = _single_marker_fixture()
        _set_step_pressure(fluid)

        _sample(markers, fluid, pressure_pair_policy="baseline_anchored_cell_pair")

        diagnostic = markers.stress_marker_diagnostics()[0]
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(
            diagnostic["invalid_reason"],
            "two_sided_pressure_missing",
        )
        self.assertEqual(
            diagnostic["pressure_pair_policy"],
            "baseline_anchored_cell_pair",
        )
        self.assertFalse(diagnostic["pressure_pair_selected"])
        self.assertFalse(diagnostic["pressure_pair_fallback_used"])
        self.assertFalse(diagnostic["pressure_pair_anchor_active"])
        self.assertEqual(diagnostic["pressure_pair_anchor_source"], "unset")
        self.assertFalse(diagnostic["pressure_pair_anchor_fallback_used"])

    def test_invalid_anchor_cells_fail_fast(self):
        markers, _fluid = _single_marker_fixture()

        with self.assertRaisesRegex(ValueError, "marker count"):
            markers.set_pressure_pair_anchor_cells(
                inside_cells=((2, 2, 2), (2, 2, 3)),
                outside_cells=((2, 2, 5),),
            )
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            markers.set_pressure_pair_anchor_cells(
                inside_cells=((2, 2),),
                outside_cells=((2, 2, 5),),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            markers.set_pressure_pair_anchor_cells(
                inside_cells=((-1, 2, 2),),
                outside_cells=((2, 2, 5),),
            )

    def test_runner_anchor_policy_is_fixed_solid_diagnostic_only(self):
        default_config = VerticalFlapFsiConfig()
        self.assertTrue(solid_mpm_fsi_runner._is_default_traction_formulation(default_config))

        non_default = VerticalFlapFsiConfig(
            traction_pressure_pair_policy="baseline_anchored_cell_pair",
        )
        self.assertFalse(solid_mpm_fsi_runner._is_default_traction_formulation(non_default))
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(non_default)

        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=1,
                traction_pressure_pair_policy="baseline_anchored_cell_pair",
            )
        )


def _single_marker_fixture(
    *,
    pressure_probe_origins_m: tuple[tuple[float, float, float], ...] | None = None,
):
    markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
    markers.load_markers(
        positions_m=((0.625, 0.625, 0.5),),
        velocities_mps=((0.0, 0.0, 0.0),),
        normals=((0.0, 0.0, 1.0),),
        areas_m2=(1.0,),
        region_ids=(101,),
        pressure_probe_origins_m=pressure_probe_origins_m,
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


def _set_custom_pressure(
    fluid: CartesianFluidSolver,
    *,
    inside_value: float,
    outside_value: float,
) -> None:
    pressure = np.zeros((8, 8, 8), dtype=np.float32)
    pressure[2, 2, 2] = float(inside_value)
    pressure[2, 2, 5] = float(outside_value)
    fluid.pressure.from_numpy(pressure)


def _sample(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    *,
    viscosity_pa_s: float = 0.0,
    two_sided_pressure: bool = True,
    **controls,
):
    if not math.isfinite(viscosity_pa_s):
        raise ValueError("viscosity_pa_s must be finite")
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
        two_sided_pressure=two_sided_pressure,
        **controls,
    )


if __name__ == "__main__":
    unittest.main()
