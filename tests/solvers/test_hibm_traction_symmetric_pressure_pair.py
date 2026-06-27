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
class HibmMpmTractionSymmetricPressurePairTests(unittest.TestCase):
    def test_independent_ladder_is_the_default_pair_policy(self):
        implicit_markers, implicit_fluid = _single_marker_fixture()
        explicit_markers, explicit_fluid = _single_marker_fixture()
        _set_step_pressure(implicit_fluid)
        _set_step_pressure(explicit_fluid)

        _sample(implicit_markers, implicit_fluid)
        _sample(
            explicit_markers,
            explicit_fluid,
            pressure_pair_policy="independent_ladder",
        )

        implicit = implicit_markers.stress_marker_diagnostics()[0]
        explicit = explicit_markers.stress_marker_diagnostics()[0]

        self.assertEqual(implicit["pressure_pair_policy"], "independent_ladder")
        self.assertEqual(explicit["pressure_pair_policy"], "independent_ladder")
        self.assertFalse(implicit["pressure_pair_selected"])
        self.assertFalse(explicit["pressure_pair_selected"])
        self.assertFalse(explicit["pressure_pair_fallback_used"])
        self.assertEqual(implicit["inside_probe_nearest_cell"], explicit["inside_probe_nearest_cell"])
        self.assertEqual(implicit["outside_probe_nearest_cell"], explicit["outside_probe_nearest_cell"])
        self.assertAlmostEqual(implicit["pressure_jump_pa"], explicit["pressure_jump_pa"])
        self.assertAlmostEqual(
            implicit["total_traction_pa"][2],
            explicit["total_traction_pa"][2],
        )

    def test_symmetric_cell_pair_selects_same_rung_two_sided_pressure(self):
        markers, fluid = _single_marker_fixture()
        _set_step_pressure(fluid)

        _sample(markers, fluid, pressure_pair_policy="symmetric_cell_pair")

        diagnostic = markers.stress_marker_diagnostics()[0]
        self.assertTrue(diagnostic["valid"])
        self.assertEqual(diagnostic["pressure_pair_policy"], "symmetric_cell_pair")
        self.assertTrue(diagnostic["pressure_pair_selected"])
        self.assertFalse(diagnostic["pressure_pair_fallback_used"])
        self.assertTrue(diagnostic["inside_pressure_found"])
        self.assertTrue(diagnostic["outside_pressure_found"])
        self.assertEqual(diagnostic["inside_probe_rung"], diagnostic["outside_probe_rung"])
        self.assertAlmostEqual(
            diagnostic["inside_probe_multiplier"],
            diagnostic["outside_probe_multiplier"],
        )
        self.assertLessEqual(diagnostic["pressure_pair_symmetry_residual_cells"], 1.0e-8)
        self.assertGreaterEqual(diagnostic["pressure_pair_cell_delta"], 0)
        self.assertEqual(len(diagnostic["pressure_pair_inside_cell"]), 3)
        self.assertEqual(len(diagnostic["pressure_pair_outside_cell"]), 3)
        self.assertAlmostEqual(diagnostic["pressure_jump_pa"], 4.0)
        self.assertAlmostEqual(diagnostic["total_traction_pa"][2], 4.0)

    def test_symmetric_cell_pair_preserves_pairing_with_explicit_probe_origin(self):
        markers, fluid = _single_marker_fixture(
            pressure_probe_origins_m=((0.625, 0.625, 0.625),)
        )
        _set_step_pressure(fluid)

        _sample(
            markers,
            fluid,
            pressure_pair_policy="symmetric_cell_pair",
            pressure_probe_ladder_start_offset_cells=0.25,
            pressure_probe_ladder_spacing_cells=0.25,
            pressure_probe_ladder_rung_count=5,
        )

        diagnostic = markers.stress_marker_diagnostics()[0]
        self.assertEqual(diagnostic["pressure_probe_origin_source"], "explicit")
        self.assertTrue(diagnostic["pressure_probe_origin_explicit"])
        self.assertEqual(diagnostic["pressure_pair_policy"], "symmetric_cell_pair")
        self.assertTrue(diagnostic["pressure_pair_selected"])
        self.assertFalse(diagnostic["pressure_pair_fallback_used"])
        self.assertEqual(diagnostic["inside_probe_rung"], diagnostic["outside_probe_rung"])
        self.assertAlmostEqual(
            diagnostic["inside_probe_multiplier"],
            diagnostic["outside_probe_multiplier"],
        )
        self.assertLessEqual(diagnostic["pressure_pair_symmetry_residual_cells"], 1.0e-8)

    def test_invalid_symmetric_pair_controls_fail_fast(self):
        markers, fluid = _single_marker_fixture()

        with self.assertRaisesRegex(ValueError, "pressure_pair_policy"):
            _sample(markers, fluid, pressure_pair_policy="unsupported")
        with self.assertRaisesRegex(ValueError, "pressure_pair_max_cell_delta"):
            _sample(
                markers,
                fluid,
                pressure_pair_policy="symmetric_cell_pair",
                pressure_pair_max_cell_delta=-1,
            )
        with self.assertRaisesRegex(ValueError, "pressure-only two-sided diagnostics"):
            _sample(
                markers,
                fluid,
                viscosity_pa_s=1.0,
                pressure_pair_policy="symmetric_cell_pair",
            )
        with self.assertRaisesRegex(ValueError, "pressure-only two-sided diagnostics"):
            _sample(
                markers,
                fluid,
                two_sided_pressure=False,
                pressure_pair_policy="symmetric_cell_pair",
            )

    def test_runner_pair_policy_is_fixed_solid_diagnostic_only(self):
        default_config = VerticalFlapFsiConfig()
        self.assertEqual(default_config.traction_pressure_pair_policy, "independent_ladder")
        self.assertEqual(default_config.traction_pressure_pair_max_cell_delta, 1)
        self.assertTrue(default_config.traction_pressure_pair_require_opposite_sides)
        self.assertTrue(solid_mpm_fsi_runner._is_default_traction_formulation(default_config))

        non_default = VerticalFlapFsiConfig(
            traction_pressure_pair_policy="symmetric_cell_pair",
        )
        self.assertFalse(solid_mpm_fsi_runner._is_default_traction_formulation(non_default))
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(non_default)

        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=1,
                traction_pressure_pair_policy="symmetric_cell_pair",
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported traction_pressure_pair_policy"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=0,
                    preflow_steps=1,
                    traction_pressure_pair_policy="unsupported",
                )
            )
        with self.assertRaisesRegex(ValueError, "traction_pressure_pair_max_cell_delta"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=0,
                    preflow_steps=1,
                    traction_pressure_pair_policy="symmetric_cell_pair",
                    traction_pressure_pair_max_cell_delta=-1,
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
