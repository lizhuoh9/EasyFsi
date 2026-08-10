import ast
import os
import unittest
from pathlib import Path

import numpy as np
import taichi as ti

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from simulation_core.coupling.hibm_mpm.core import (
    assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
)


CORE_PATH = Path("simulation_core/coupling/hibm_mpm/core.py")
RUNTIME = TaichiRuntimeConfig(arch="cuda")


@ti.kernel
def _probe_path_crossed_for_test(
    markers: ti.template(),
    obstacle: ti.template(),
    sampling_obstacle: ti.template(),
) -> ti.i32:
    return markers._probe_path_crossed_obstacle_sampling_view(
        obstacle,
        sampling_obstacle,
        1,
        ti.Vector([7.5, 7.5, 7.5]),
        ti.Vector([7.5, 7.5, 13.5]),
        16,
        16,
        16,
    )


def _core_tree() -> ast.Module:
    return ast.parse(CORE_PATH.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _contains_call(node: ast.AST, attribute: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == attribute
        for candidate in ast.walk(node)
    )


class HibmProbeAndFinalPressureHostContractTests(unittest.TestCase):
    def test_air_backed_pressure_requires_the_generic_outlet_reference(self):
        with self.assertRaisesRegex(ValueError, "pressure_outlet_zmin"):
            assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
                fluid=None,
                markers=None,
                ib_search=None,
                ib_boundary=None,
                mpm_external_force_n=None,
                mpm_particle_position_m=None,
                mpm_particle_count=1,
                marker_pressure_neumann_gradient_pa_per_m_field=None,
                search_radius_m=1.0,
                interior_probe_distance_m=1.0,
                mpm_support_radius_m=1.0,
                far_pressure_air_backed=True,
                pressure_outlet_zmin=False,
            )

    def test_both_extended_kernels_share_the_effective_view_path_guard(self):
        tree = _core_tree()
        for function_name in (
            "_sample_fluid_stress_to_marker_tractions_kernel",
            "_add_split_viscous_mode_marker_tractions_kernel",
        ):
            with self.subTest(function_name=function_name):
                function = _function(tree, function_name)
                self.assertTrue(
                    _contains_call(
                        function,
                        "_probe_path_crossed_obstacle_sampling_view",
                    )
                )
                self.assertNotIn(
                    "marker_near_is_obstacle",
                    ast.get_source_segment(
                        CORE_PATH.read_text(encoding="utf-8"),
                        function,
                    ),
                )

    def test_post_solid_air_pressure_stamp_is_after_conversion_and_optional_project(self):
        advance = _function(_core_tree(), "advance_hibm_mpm_sharp_mpm_step")
        top_level_stamp_index = next(
            index
            for index, statement in enumerate(advance.body)
            if _contains_call(statement, "write_hibm_air_backed_cell_pressures")
        )
        stamp_statement = advance.body[top_level_stamp_index]
        conversion_lines = [
            call.lineno
            for call in ast.walk(advance)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "convert_hibm_air_backed_cells"
        ]
        project_lines = [
            call.lineno
            for call in ast.walk(advance)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "project"
        ]
        stamp_lines = [
            call.lineno
            for call in ast.walk(stamp_statement)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "write_hibm_air_backed_cell_pressures"
        ]

        self.assertEqual(len(stamp_lines), 1)
        self.assertTrue(project_lines)
        self.assertGreater(stamp_lines[0], max(project_lines))
        self.assertGreater(stamp_lines[0], max(conversion_lines))

    def test_air_seed_kernel_resets_the_fallback_counter_each_execution(self):
        kernel = _function(
            _core_tree(),
            "_mark_far_pressure_air_backed_seed_components_kernel",
        )
        first_assignments = [
            statement
            for statement in kernel.body[:6]
            if isinstance(statement, ast.Assign)
        ]
        assigned_targets = {
            ast.unparse(assignment.targets[0]) for assignment in first_assignments
        }
        self.assertIn(
            "self.report_air_backed_seed_fallback_cell_count[None]",
            assigned_targets,
        )


@unittest.skipIf(
    os.environ.get("GITHUB_ACTIONS") == "true"
    and os.environ.get("HIBM_RUN_CUDA_TRACTION_PROBE_TESTS") != "1",
    "simulation_core is GPU-only; set HIBM_RUN_CUDA_TRACTION_PROBE_TESTS=1 on a CUDA runner",
)
class HibmProbeEffectiveSamplingViewRuntimeTests(unittest.TestCase):
    def test_extension_leaves_sampling_view_self_obstacle_without_projection_false_cross(self):
        markers, fluid = _single_marker_fixture()
        projection_obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        sampling_obstacle = np.zeros_like(projection_obstacle)
        # The sampling view owns the marker-connected self envelope.  Its
        # first extension sample can read adjacent sealed water, although the
        # projection view still calls that nearest cell an obstacle.
        sampling_obstacle[:, :, 7:12] = 1
        projection_obstacle[:, :, 11] = 1
        fluid.obstacle.from_numpy(projection_obstacle)
        fluid.hibm_no_slip_sampling_obstacle.from_numpy(sampling_obstacle)

        report = _sample(
            markers,
            fluid,
            sampling_obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
            two_sided_probe_max_multiplier=5.0,
        )
        diagnostic = markers.stress_marker_diagnostics()[0]

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.two_sided_extended_marker_count, 1)
        self.assertTrue(diagnostic["outside_pressure_found"])
        self.assertGreaterEqual(diagnostic["outside_probe_rung"], 10)

    def test_one_sided_marker_does_not_inherit_two_sided_extension_multiplier(self):
        markers, fluid = _single_marker_fixture()
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 4:13] = 1
        fluid.obstacle.from_numpy(obstacle)
        fluid.hibm_no_slip_sampling_obstacle.from_numpy(obstacle)

        report = _sample(
            markers,
            fluid,
            sampling_obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
            one_sided_pressure_region_id=101,
            one_sided_reference_pressure_pa=0.0,
            two_sided_probe_max_multiplier=5.0,
            one_sided_probe_max_multiplier=3.0,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.one_sided_extended_marker_count, 0)

    def test_two_sided_marker_does_not_inherit_one_sided_extension_multiplier(self):
        markers, fluid = _single_marker_fixture()
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 4:13] = 1
        fluid.obstacle.from_numpy(obstacle)
        fluid.hibm_no_slip_sampling_obstacle.from_numpy(obstacle)

        report = _sample(
            markers,
            fluid,
            sampling_obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
            two_sided_probe_max_multiplier=3.0,
            one_sided_probe_max_multiplier=5.0,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.two_sided_extended_marker_count, 0)

    def test_extension_rejects_fluid_to_foreign_obstacle_reentry(self):
        markers, fluid = _single_marker_fixture()
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        # The marker begins in its contiguous self-obstacle band, exits into
        # fluid at z=10, and then meets a separate obstacle plane at z=11.
        # Fluid exists beyond that plane, but the do-not-tunnel guard must
        # make the remote rung unreachable.
        obstacle[:, :, 7:10] = 1
        obstacle[:, :, 11] = 1
        fluid.obstacle.from_numpy(obstacle)
        fluid.hibm_no_slip_sampling_obstacle.from_numpy(obstacle)

        crossed = _probe_path_crossed_for_test(
            markers,
            fluid.obstacle,
            fluid.hibm_no_slip_sampling_obstacle,
        )

        self.assertEqual(crossed, 1)


def _single_marker_fixture():
    markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
    markers.load_markers(
        positions_m=((0.5, 0.5, 0.5),),
        velocities_mps=((0.0, 0.0, 0.0),),
        normals=((0.0, 0.0, 1.0),),
        areas_m2=(1.0,),
        region_ids=(101,),
    )
    fluid = CartesianFluidSolver(
        FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
        runtime=RUNTIME,
    )
    fluid.pressure.fill(7.0)
    return markers, fluid


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
        **controls,
    )


if __name__ == "__main__":
    unittest.main()
