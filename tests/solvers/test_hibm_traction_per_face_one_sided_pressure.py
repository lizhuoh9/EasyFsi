from __future__ import annotations

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
class HibmMpmTractionPerFaceOneSidedPressureTests(unittest.TestCase):
    def test_default_disabled_path_keeps_two_sided_pressure_jump(self):
        markers, fluid = _two_marker_fixture()
        _set_anchor_pressure(fluid, inside_value=7.0, outside_value=2.0)
        markers.set_pressure_pair_anchor_cells(
            inside_cells=((2, 2, 2), (2, 2, 2)),
            outside_cells=((2, 2, 5), (2, 2, 5)),
        )

        report = _sample(markers, fluid)

        self.assertEqual(report.two_sided_pressure_marker_count, 2)
        self.assertEqual(report.one_sided_pressure_marker_count, 0)
        for marker in markers.stress_marker_diagnostics():
            self.assertTrue(marker["valid"])
            self.assertEqual(marker["probe_mode"], "two_sided_pressure_jump")
            self.assertEqual(marker["one_sided_policy"], "disabled")
            self.assertAlmostEqual(marker["pressure_jump_pa"], 5.0)

    def test_per_face_one_sided_selects_declared_sides_and_references(self):
        markers, fluid = _two_marker_fixture()
        _set_anchor_pressure(fluid, inside_value=7.0, outside_value=2.0)
        markers.set_pressure_pair_anchor_cells(
            inside_cells=((2, 2, 2), (2, 2, 2)),
            outside_cells=((2, 2, 5), (2, 2, 5)),
        )

        report = _sample(
            markers,
            fluid,
            one_sided_pressure_primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
            one_sided_pressure_secondary_region_id=solid_mpm_fsi_runner.SECONDARY_REGION_ID,
            one_sided_primary_reference_pressure_pa=1.0,
            one_sided_secondary_reference_pressure_pa=10.0,
            one_sided_primary_fluid_side_normal_sign=-1.0,
            one_sided_secondary_fluid_side_normal_sign=1.0,
        )

        diagnostics = markers.stress_marker_diagnostics()
        primary = diagnostics[0]
        secondary = diagnostics[1]
        self.assertEqual(report.one_sided_pressure_marker_count, 2)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(primary["one_sided_policy"], "per_face_region")
        self.assertEqual(primary["one_sided_side_selected"], "inside")
        self.assertEqual(primary["one_sided_region_id"], solid_mpm_fsi_runner.PRIMARY_REGION_ID)
        self.assertAlmostEqual(primary["one_sided_fluid_side_pressure_pa"], 7.0)
        self.assertAlmostEqual(primary["one_sided_reference_pressure_pa"], 1.0)
        self.assertAlmostEqual(primary["pressure_jump_pa"], 6.0)
        self.assertAlmostEqual(primary["total_traction_pa"][2], 6.0)
        self.assertTrue(primary["one_sided_anchor_selected"])
        self.assertFalse(primary["one_sided_anchor_fallback_used"])

        self.assertEqual(secondary["one_sided_policy"], "per_face_region")
        self.assertEqual(secondary["one_sided_side_selected"], "outside")
        self.assertEqual(
            secondary["one_sided_region_id"],
            solid_mpm_fsi_runner.SECONDARY_REGION_ID,
        )
        self.assertAlmostEqual(secondary["one_sided_fluid_side_pressure_pa"], 2.0)
        self.assertAlmostEqual(secondary["one_sided_reference_pressure_pa"], 10.0)
        self.assertAlmostEqual(secondary["pressure_jump_pa"], 8.0)
        self.assertAlmostEqual(secondary["total_traction_pa"][2], -8.0)
        self.assertTrue(secondary["one_sided_anchor_selected"])
        self.assertFalse(secondary["one_sided_anchor_fallback_used"])

    def test_per_face_one_sided_missing_anchor_fails_closed(self):
        markers, fluid = _two_marker_fixture()
        _set_anchor_pressure(fluid, inside_value=7.0, outside_value=2.0)

        report = _sample(
            markers,
            fluid,
            one_sided_pressure_primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
            one_sided_pressure_secondary_region_id=solid_mpm_fsi_runner.SECONDARY_REGION_ID,
            one_sided_primary_fluid_side_normal_sign=-1.0,
            one_sided_secondary_fluid_side_normal_sign=1.0,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 2)
        for marker in markers.stress_marker_diagnostics():
            self.assertFalse(marker["valid"])
            self.assertEqual(marker["invalid_reason"], "two_sided_pressure_missing")
            self.assertFalse(marker["pressure_pair_selected"])
            self.assertFalse(marker["one_sided_anchor_selected"])

    def test_invalid_per_face_signs_fail_fast(self):
        markers, fluid = _two_marker_fixture()
        _set_anchor_pressure(fluid, inside_value=7.0, outside_value=2.0)

        with self.assertRaisesRegex(ValueError, "primary_fluid_side_normal_sign"):
            _sample(
                markers,
                fluid,
                one_sided_pressure_primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
                one_sided_primary_fluid_side_normal_sign=0.0,
            )

    def test_runner_config_allows_per_face_only_for_diagnostics(self):
        diagnostic = VerticalFlapFsiConfig(
            step_count=0,
            preflow_steps=1,
            traction_pressure_sampling_mode="one_sided_surface_pressure",
            traction_pressure_pair_policy="baseline_anchored_cell_pair",
            traction_one_sided_pressure_policy="per_face_mirrored",
            traction_one_sided_primary_fluid_side_normal_sign=1.0,
            traction_one_sided_secondary_fluid_side_normal_sign=1.0,
        )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(diagnostic)

        positive_step = VerticalFlapFsiConfig(
            traction_pressure_sampling_mode="one_sided_surface_pressure",
            traction_pressure_pair_policy="baseline_anchored_cell_pair",
            traction_one_sided_pressure_policy="per_face_mirrored",
            traction_one_sided_primary_fluid_side_normal_sign=1.0,
            traction_one_sided_secondary_fluid_side_normal_sign=1.0,
        )
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(positive_step)


def _two_marker_fixture():
    markers = HibmMpmSurfaceMarkers(marker_capacity=2, runtime=RUNTIME)
    markers.load_markers(
        positions_m=((0.625, 0.625, 0.5), (0.625, 0.625, 0.5)),
        velocities_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        normals=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        areas_m2=(1.0, 1.0),
        region_ids=(
            solid_mpm_fsi_runner.PRIMARY_REGION_ID,
            solid_mpm_fsi_runner.SECONDARY_REGION_ID,
        ),
    )
    fluid = CartesianFluidSolver(
        FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
        runtime=RUNTIME,
    )
    return markers, fluid


def _set_anchor_pressure(
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
    **controls,
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
        viscosity_pa_s=0.0,
        two_sided_pressure=True,
        pressure_pair_policy="baseline_anchored_cell_pair",
        **controls,
    )


if __name__ == "__main__":
    unittest.main()
