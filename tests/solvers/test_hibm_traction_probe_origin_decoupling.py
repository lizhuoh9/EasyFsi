from __future__ import annotations

import os
import unittest

import numpy as np

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
class HibmMpmTractionProbeOriginDecouplingTests(unittest.TestCase):
    def test_default_probe_origin_tracks_marker_position(self):
        markers, fluid = _single_marker_fixture()
        _set_step_pressure(fluid)

        _sample(markers, fluid)
        diagnostic = markers.stress_marker_diagnostics()[0]

        self.assertEqual(diagnostic["pressure_probe_origin_source"], "marker_position")
        self.assertFalse(diagnostic["pressure_probe_origin_explicit"])
        self.assertEqual(diagnostic["pressure_probe_origin_m"], diagnostic["position_m"])
        self.assertAlmostEqual(diagnostic["pressure_jump_pa"], 4.0)
        self.assertAlmostEqual(diagnostic["total_traction_pa"][2], 4.0)

    def test_explicit_probe_origin_changes_sampling_without_moving_marker(self):
        default_markers, default_fluid = _single_marker_fixture()
        explicit_markers, explicit_fluid = _single_marker_fixture()
        _set_step_pressure(default_fluid)
        _set_step_pressure(explicit_fluid)

        _sample(default_markers, default_fluid)
        explicit_markers.set_pressure_probe_origins_m(((0.625, 0.625, 0.75),))
        _sample(explicit_markers, explicit_fluid)

        default = default_markers.stress_marker_diagnostics()[0]
        explicit = explicit_markers.stress_marker_diagnostics()[0]

        self.assertEqual(explicit["position_m"], default["position_m"])
        self.assertEqual(explicit["normal"], default["normal"])
        self.assertEqual(explicit["pressure_probe_origin_source"], "explicit")
        self.assertTrue(explicit["pressure_probe_origin_explicit"])
        self.assertNotEqual(
            explicit["pressure_probe_origin_m"],
            explicit["position_m"],
        )
        self.assertNotEqual(
            explicit["inside_probe_grid_coordinate"],
            default["inside_probe_grid_coordinate"],
        )
        self.assertNotEqual(
            explicit["outside_probe_grid_coordinate"],
            default["outside_probe_grid_coordinate"],
        )
        self.assertLess(abs(explicit["pressure_jump_pa"]), abs(default["pressure_jump_pa"]))
        self.assertLess(abs(explicit["total_traction_pa"][2]), abs(default["total_traction_pa"][2]))

    def test_invalid_probe_origins_fail_fast(self):
        markers, _ = _single_marker_fixture()

        with self.assertRaisesRegex(ValueError, "marker count"):
            markers.set_pressure_probe_origins_m(
                (
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.75),
                )
            )

        with self.assertRaisesRegex(ValueError, "pressure_probe_origins_m"):
            markers.set_pressure_probe_origins_m(((0.5, 0.5),))

        with self.assertRaisesRegex(ValueError, "finite"):
            markers.set_pressure_probe_origins_m(((0.5, 0.5, float("nan")),))


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


def _sample(markers: HibmMpmSurfaceMarkers, fluid: CartesianFluidSolver):
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
        viscosity_pa_s=0.0,
        two_sided_pressure=True,
    )


if __name__ == "__main__":
    unittest.main()
