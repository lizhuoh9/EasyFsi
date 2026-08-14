import unittest

import numpy as np
import taichi as ti

from benchmarks.official.solid_mpm_fsi_runner import _build_markers
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig
from simulation_core import CartesianFluidSolver, FluidDomainSpec, HibmMpmSurfaceMarkers
from simulation_core import TaichiRuntimeConfig
from simulation_core.coupling.hibm_mpm.core import HibmMpmSharpCouplingState
from src.refactored.validation.ansys_vertical_flap_fsi.native_fine_final_contracts import (
    FINAL_FINE_CONFIG_IDENTITY,
)


class TipCapTractionContracts(unittest.TestCase):
    def test_coupling_state_preserves_fractional_count_for_strict_loader(self) -> None:
        calls: dict[str, object] = {}

        class Markers:
            def load_markers_from_surface_fields(self, *_args, **kwargs):
                calls["marker_count"] = kwargs["marker_count"]
                return 0

        state = object.__new__(HibmMpmSharpCouplingState)
        state.markers = Markers()
        state.load_markers_from_surface_fields(
            object(), object(), object(), object(), marker_count=1.5
        )

        self.assertEqual(calls["marker_count"], 1.5)

    def test_coupling_state_preflights_triangles_before_replacing_markers(self) -> None:
        events: list[str] = []

        class Markers:
            def _preflight_projection_triangles_from_field(self, *_args, **kwargs):
                events.append("triangle_preflight")
                self.assert_fractional(kwargs["triangle_count"])
                raise ValueError("triangle_count must be an integer")

            @staticmethod
            def assert_fractional(value):
                if value != 1.5:
                    raise AssertionError(value)

            def load_markers_from_surface_fields(self, *_args, **_kwargs):
                events.append("marker_commit")
                return 1

        state = object.__new__(HibmMpmSharpCouplingState)
        state.markers = Markers()

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            state.load_markers_from_surface_fields(
                object(),
                object(),
                object(),
                object(),
                marker_count=1,
                projection_triangle_indices=object(),
                projection_triangle_count=1.5,
            )

        self.assertEqual(events, ["triangle_preflight"])

    @staticmethod
    def _markers() -> HibmMpmSurfaceMarkers:
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=8,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        markers.load_markers(
            positions_m=(
                (0.5, 0.25, 0.45),
                (0.5, 0.75, 0.45),
                (0.5, 0.25, 0.55),
                (0.5, 0.75, 0.55),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 4,
            normals=((0.0, 0.0, 1.0),) * 2 + ((0.0, 0.0, -1.0),) * 2,
            areas_m2=(0.1,) * 4,
            region_ids=(101, 101, 202, 202),
        )
        markers.configure_open_ribbon_tip_cap(
            primary_previous_marker_index=0,
            primary_tip_marker_index=1,
            secondary_previous_marker_index=2,
            secondary_tip_marker_index=3,
            cap_region_id=303,
            cap_area_m2=0.2,
            inactive_axis=0,
        )
        return markers

    @staticmethod
    def _uniform_fluid(pressure_pa: float, *, all_obstacle: bool = False):
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.pressure.from_numpy(
            np.full((4, 8, 8), pressure_pa, dtype=np.float32)
        )
        fluid.obstacle.from_numpy(
            np.full((4, 8, 8), int(all_obstacle), dtype=np.int32)
        )
        return fluid

    def test_cap_vertices_have_cap_area_outward_normal_and_region(self) -> None:
        markers = self._markers()

        self.assertEqual(markers.marker_count, 4)
        self.assertEqual(markers.marker_region_id(6), 303)
        self.assertEqual(markers.marker_region_id(7), 303)
        self.assertAlmostEqual(float(markers.A_gamma_m2[4]), 0.0)
        self.assertAlmostEqual(float(markers.A_gamma_m2[5]), 0.0)
        self.assertAlmostEqual(float(markers.A_gamma_m2[6]), 0.1, delta=1.0e-6)
        self.assertAlmostEqual(float(markers.A_gamma_m2[7]), 0.1, delta=1.0e-6)
        self.assertGreater(float(markers.marker_normal(6)[1]), 0.99)
        self.assertGreater(float(markers.marker_normal(7)[1]), 0.99)

    def test_one_sided_cap_pressure_has_known_constant_pressure_force(self) -> None:
        markers = self._markers()
        fluid = self._uniform_fluid(pressure_pa=2.5)

        report = markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity, fluid.pressure, fluid.obstacle,
            fluid.cell_face_x_m, fluid.cell_face_y_m, fluid.cell_face_z_m,
            fluid.cell_center_x_m, fluid.cell_center_y_m, fluid.cell_center_z_m,
            fluid.cell_width_x_m, fluid.cell_width_y_m, fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=0.0,
            two_sided_pressure=True,
            tip_cap_pressure_enabled=True,
            tip_cap_region_id=303,
            tip_cap_reference_pressure_pa=0.0,
        )
        markers.compute_marker_forces()
        forces = markers.aggregate_region_forces(
            primary_region_id=101, secondary_region_id=202
        )

        self.assertEqual(report.tip_cap_marker_count, 2)
        self.assertEqual(report.tip_cap_invalid_marker_count, 0)
        self.assertAlmostEqual(forces.tip_cap_marker_force_n[1], -0.5, delta=1.0e-5)
        self.assertAlmostEqual(forces.total_marker_force_n[1], -0.5, delta=1.0e-5)

    def test_cap_sampling_and_scatter_fail_closed_without_fluid_support(self) -> None:
        markers = self._markers()
        fluid = self._uniform_fluid(pressure_pa=2.5, all_obstacle=True)

        report = markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity, fluid.pressure, fluid.obstacle,
            fluid.cell_face_x_m, fluid.cell_face_y_m, fluid.cell_face_z_m,
            fluid.cell_center_x_m, fluid.cell_center_y_m, fluid.cell_center_z_m,
            fluid.cell_width_x_m, fluid.cell_width_y_m, fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=0.0,
            two_sided_pressure=True,
            tip_cap_pressure_enabled=True,
            tip_cap_region_id=303,
        )

        self.assertEqual(report.tip_cap_marker_count, 2)
        self.assertEqual(report.tip_cap_invalid_marker_count, 2)

    def test_cap_scatter_conserves_total_marker_force(self) -> None:
        markers = self._markers()
        markers.set_tip_cap_gauge_pressure_tractions_pa((2.5, 2.5))
        markers.compute_marker_forces()
        external_force = ti.Vector.field(3, dtype=ti.f32, shape=1)
        position = ti.Vector.field(3, dtype=ti.f32, shape=1)
        position[0] = (0.5, 1.0, 0.5)
        external_force[0] = (0.0, 0.0, 0.0)

        scatter = markers.scatter_marker_forces_to_mpm_particles(
            external_force, position, particle_count=1, support_radius_m=1.0
        )

        self.assertEqual(scatter.active_marker_count, 6)
        self.assertAlmostEqual(scatter.total_marker_force_n[1], -0.5, delta=1.0e-5)
        self.assertLess(scatter.action_reaction_residual_n, 1.0e-5)

    def test_geometry_refresh_retires_active_cap_load_until_resampled(self) -> None:
        markers = self._markers()
        markers.set_tip_cap_gauge_pressure_tractions_pa((2.5, 2.5))
        markers.refresh_open_ribbon_tip_cap_projection_vertices()

        self.assertEqual(markers._tip_cap_force_layout()[1], 0)
        self.assertEqual(int(markers._stress_pressure_valid[6]), 0)
        self.assertEqual(int(markers._stress_pressure_valid[7]), 0)

    def test_cached_inactive_scatter_has_cap_slots_after_activation(self) -> None:
        markers = self._markers()
        external_force = ti.Vector.field(3, dtype=ti.f32, shape=1)
        position = ti.Vector.field(3, dtype=ti.f32, shape=1)
        position[0] = (0.5, 1.0, 0.5)
        markers.scatter_marker_forces_to_mpm_particles(
            external_force, position, particle_count=1, support_radius_m=1.0
        )
        markers.set_tip_cap_gauge_pressure_tractions_pa((2.5, 2.5))
        markers.compute_marker_forces()
        external_force[0] = (0.0, 0.0, 0.0)

        scatter = markers.scatter_marker_forces_to_mpm_particles(
            external_force, position, particle_count=1, support_radius_m=1.0
        )

        self.assertEqual(scatter.active_marker_count, 6)
        self.assertAlmostEqual(scatter.total_marker_force_n[1], -0.5, delta=1.0e-5)
        self.assertLess(scatter.action_reaction_residual_n, 1.0e-5)

    def test_side_force_is_unchanged_when_cap_is_not_loaded(self) -> None:
        markers = self._markers()
        markers.set_marker_tractions_pa(
            ((1.0, 0.0, 0.0),) * 4
        )
        markers.compute_marker_forces()
        report = markers.aggregate_region_forces(
            primary_region_id=101, secondary_region_id=202
        )

        self.assertAlmostEqual(report.primary_marker_force_n[0], 0.2, delta=1.0e-6)
        self.assertAlmostEqual(report.secondary_marker_force_n[0], 0.2, delta=1.0e-6)
        self.assertEqual(report.tip_cap_marker_force_n, (0.0, 0.0, 0.0))

    def test_vertical_flap_and_final_identity_lock_tip_cap_pressure_enabled(self) -> None:
        config = VerticalFlapFsiConfig()

        self.assertTrue(config.traction_tip_cap_pressure_enabled)
        self.assertTrue(FINAL_FINE_CONFIG_IDENTITY["traction_tip_cap_pressure_enabled"])
        markers = _build_markers(config, TaichiRuntimeConfig(arch="cuda"))
        self.assertEqual(markers.projection_vertex_count, markers.marker_count + 4)


if __name__ == "__main__":
    unittest.main()
