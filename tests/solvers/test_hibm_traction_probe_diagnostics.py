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
class HibmMpmTractionProbeDiagnosticsTests(unittest.TestCase):
    def test_uniform_two_sided_pressure_reports_zero_jump_and_probe_evidence(self):
        markers, fluid = _single_marker_fixture()
        fluid.pressure.fill(7.0)

        report = _sample(markers, fluid, two_sided_pressure=True, viscosity_pa_s=0.0)
        diagnostic = markers.stress_marker_diagnostics()[0]
        face = markers.stress_face_diagnostics(primary_region_id=101)

        self.assertEqual(report.valid_marker_count, 1)
        self.assertTrue(diagnostic["inside_pressure_found"])
        self.assertTrue(diagnostic["outside_pressure_found"])
        self.assertAlmostEqual(diagnostic["inside_pressure_pa"], 7.0)
        self.assertAlmostEqual(diagnostic["outside_pressure_pa"], 7.0)
        self.assertAlmostEqual(diagnostic["pressure_jump_pa"], 0.0)
        self.assertAlmostEqual(diagnostic["traction_decomposition_residual_pa"], 0.0)
        self.assertFalse(diagnostic["fluid_side_pressure_defined"])
        self.assertGreaterEqual(diagnostic["inside_probe_rung"], 0)
        self.assertGreaterEqual(diagnostic["outside_probe_rung"], 0)
        self.assertNotEqual(diagnostic["inside_probe_nearest_cell"], [-1, -1, -1])
        self.assertNotEqual(diagnostic["outside_probe_nearest_cell"], [-1, -1, -1])
        self.assertGreater(diagnostic["inside_probe_fluid_weight"], 0.0)
        self.assertGreater(diagnostic["outside_probe_fluid_weight"], 0.0)
        self.assertEqual(face["primary_face_pressure_complete_marker_count"], 1)
        self.assertEqual(face["primary_face_pressure_missing_marker_count"], 0)
        self.assertEqual(face["primary_face_inside_pressure_found_marker_count"], 1)
        self.assertEqual(face["primary_face_outside_pressure_found_marker_count"], 1)
        self.assertEqual(face["primary_face_inside_probe_rung_histogram"], {"0": 1})
        self.assertGreater(face["primary_face_inside_unique_nearest_cell_count"], 0)
        self.assertAlmostEqual(
            face["primary_face_traction_decomposition_max_abs_residual_pa"],
            0.0,
        )

    def test_piecewise_two_sided_pressure_reports_jump_sign_and_nearest_cells(self):
        markers, fluid = _single_marker_fixture()
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[:, :, :4] = 5.0
        pressure[:, :, 4:] = 1.0
        fluid.pressure.from_numpy(pressure)

        report = _sample(markers, fluid, two_sided_pressure=True, viscosity_pa_s=0.0)
        diagnostic = markers.stress_marker_diagnostics()[0]

        self.assertEqual(report.valid_marker_count, 1)
        self.assertAlmostEqual(diagnostic["inside_pressure_pa"], 5.0)
        self.assertAlmostEqual(diagnostic["outside_pressure_pa"], 1.0)
        self.assertAlmostEqual(diagnostic["pressure_jump_pa"], 4.0)
        self.assertAlmostEqual(diagnostic["total_traction_pa"][2], 4.0)
        self.assertEqual(
            diagnostic["inside_probe_ladder_mode"],
            "pressure_only_integer_ladder",
        )
        self.assertGreater(diagnostic["inside_probe_multiplier"], 0.0)
        self.assertNotEqual(diagnostic["inside_probe_grid_coordinate"], [-1.0] * 3)

    def test_missing_two_sided_probe_resets_sentinel_fields_and_reason_code(self):
        markers, fluid = _single_marker_fixture()
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[:, :, :4] = 5.0
        pressure[:, :, 4:] = 1.0
        fluid.pressure.from_numpy(pressure)
        _sample(markers, fluid, two_sided_pressure=True, viscosity_pa_s=0.0)

        fluid.obstacle.from_numpy(np.ones((8, 8, 8), dtype=np.int32))
        report = _sample(markers, fluid, two_sided_pressure=True, viscosity_pa_s=0.0)
        diagnostic = markers.stress_marker_diagnostics()[0]

        self.assertEqual(report.valid_marker_count, 0)
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(diagnostic["invalid_reason"], "two_sided_pressure_missing")
        self.assertIsInstance(diagnostic["invalid_reason_code"], int)
        self.assertEqual(diagnostic["inside_probe_rung"], -1)
        self.assertEqual(diagnostic["outside_probe_rung"], -1)
        self.assertEqual(diagnostic["inside_probe_nearest_cell"], [-1, -1, -1])
        self.assertEqual(diagnostic["outside_probe_nearest_cell"], [-1, -1, -1])
        self.assertEqual(diagnostic["inside_probe_grid_coordinate"], [-1.0] * 3)
        self.assertEqual(diagnostic["outside_probe_grid_coordinate"], [-1.0] * 3)
        self.assertAlmostEqual(diagnostic["total_traction_pa"][2], 0.0)

    def test_pressure_only_and_general_paths_report_equivalent_pressure_traction(self):
        fast_markers, fast_fluid = _single_marker_fixture()
        general_markers, general_fluid = _single_marker_fixture()
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[:, :, :4] = 5.0
        pressure[:, :, 4:] = 1.0
        fast_fluid.pressure.from_numpy(pressure)
        general_fluid.pressure.from_numpy(pressure)

        _sample(fast_markers, fast_fluid, two_sided_pressure=True, viscosity_pa_s=0.0)
        _sample(
            general_markers,
            general_fluid,
            two_sided_pressure=True,
            viscosity_pa_s=0.0,
            sampling_obstacle_field=general_fluid.obstacle,
        )

        fast = fast_markers.stress_marker_diagnostics()[0]
        general = general_markers.stress_marker_diagnostics()[0]
        self.assertAlmostEqual(
            fast["pressure_traction_pa"][2],
            general["pressure_traction_pa"][2],
            delta=1.0e-6,
        )
        self.assertEqual(fast["inside_pressure_found"], general["inside_pressure_found"])
        self.assertEqual(
            fast["outside_pressure_found"],
            general["outside_pressure_found"],
        )

    def test_base_viscous_split_preserves_pressure_viscous_total_decomposition(self):
        markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 1.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(101,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                viscosity_pa_s=2.0,
                dt_s=1.0e-3,
            ),
            runtime=RUNTIME,
        )
        fluid.set_simple_shear_velocity(shear_rate_s=3.0, center_y_m=0.0)

        report = _sample(markers, fluid, viscosity_pa_s=fluid.mu)
        diagnostic = markers.stress_marker_diagnostics()[0]

        self.assertEqual(report.valid_marker_count, 1)
        self.assertAlmostEqual(diagnostic["pressure_traction_pa"][0], 0.0)
        self.assertAlmostEqual(diagnostic["viscous_traction_pa"][0], 6.0)
        self.assertAlmostEqual(diagnostic["total_traction_pa"][0], 6.0)
        self.assertAlmostEqual(diagnostic["traction_decomposition_residual_pa"], 0.0)

    def test_external_total_traction_keeps_public_decomposition_consistent(self):
        markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
        markers.load_markers(
            positions_m=((0.0, 0.0, 0.0),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(101,),
        )

        markers.set_marker_tractions_pa(((1.0, 2.0, 3.0),))
        diagnostic = markers.stress_marker_diagnostics()[0]

        self.assertTrue(diagnostic["valid"])
        self.assertEqual(diagnostic["pressure_traction_pa"], [1.0, 2.0, 3.0])
        self.assertEqual(diagnostic["viscous_traction_pa"], [0.0, 0.0, 0.0])
        self.assertEqual(diagnostic["total_traction_pa"], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(diagnostic["traction_decomposition_residual_pa"], 0.0)
        self.assertFalse(diagnostic["inside_pressure_found"])
        self.assertEqual(diagnostic["inside_probe_nearest_cell"], [-1, -1, -1])


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


def _sample(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    *,
    two_sided_pressure: bool = False,
    viscosity_pa_s: float,
    sampling_obstacle_field=None,
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
        two_sided_pressure=two_sided_pressure,
        sampling_obstacle_field=sampling_obstacle_field,
    )


if __name__ == "__main__":
    unittest.main()
