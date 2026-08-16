import inspect
import math
import unittest
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.official import solid_mpm_fsi_runner
import cases.ansys_vertical_flap_fsi as vertical_flap_case
from cases import CASE_MODULES
from cases.ansys_vertical_flap_fsi import (
    ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS,
    ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
    VerticalFlapFsiConfig,
    build_ansys_vertical_flap_generic_problem,
    run_vertical_flap_fsi_smoke,
    selected_formulation_solver_config,
    surface_force_support_radius_m,
    thin_wall_pressure_probe_max_multiplier,
)
from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


SELECTED_ANCHOR_MARKERS_JSON = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "traction_fixed_solid_selected_formulation_diagnostics"
    / "marker_diagnostics"
    / "fixed_solid_selected_per_face_one_sided_probe0p51_markers.json"
).as_posix()


class _NumpyField:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values)

    def to_numpy(self) -> np.ndarray:
        return np.array(self._values, copy=True)


def _fake_flow_report_fluid(pressure: np.ndarray) -> SimpleNamespace:
    pressure_values = np.asarray(pressure, dtype=np.float32)
    return SimpleNamespace(
        obstacle=_NumpyField(np.zeros(pressure_values.shape, dtype=np.int32)),
        velocity=_NumpyField(np.zeros(pressure_values.shape + (3,), dtype=np.float32)),
        pressure=_NumpyField(np.zeros_like(pressure_values)),
        fsi_pressure=_NumpyField(pressure_values),
    )


class AnsysVerticalFlapFsiSmokeTests(unittest.TestCase):
    def test_benchmark_results_must_be_computed_not_assigned_from_reference(self):
        config_fields = VerticalFlapFsiConfig.__dataclass_fields__
        self.assertNotIn("pressure_scale", config_fields)
        self.assertFalse(
            hasattr(vertical_flap_case, "_reference_equivalent_pressure_jump_pa")
        )

        run_source = inspect.getsource(vertical_flap_case.run_vertical_flap_fsi_smoke)
        self.assertNotIn("_load_reference_pressure_and_velocity", run_source)
        self.assertNotIn("pressure_jump_pa", run_source)

    def test_case_metadata_matches_ansys_tutorial_boundaries_and_targets(self):
        bc = ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS
        metadata = ANSYS_VERTICAL_FLAP_CASE_METADATA
        reference = ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS

        self.assertAlmostEqual(metadata["geometry"]["duct_length_m"], 0.10)
        self.assertAlmostEqual(metadata["geometry"]["duct_height_m"], 0.04)
        self.assertAlmostEqual(metadata["geometry"]["modeled_height_m"], 0.02)
        self.assertEqual(metadata["geometry"]["modeled_domain"], "lower-symmetry-half")
        self.assertAlmostEqual(metadata["geometry"]["flap_height_m"], 0.01)
        self.assertAlmostEqual(metadata["geometry"]["flap_thickness_m"], 0.003)
        self.assertAlmostEqual(metadata["geometry"]["flap_streamwise_min_m"], 0.050)
        self.assertAlmostEqual(metadata["geometry"]["flap_streamwise_max_m"], 0.053)
        self.assertEqual(metadata["fluid"]["material"], "air")
        self.assertAlmostEqual(metadata["fluid"]["density_kgm3"], 1.2)
        self.assertAlmostEqual(metadata["fluid"]["viscosity_pa_s"], 1.8e-5)
        self.assertAlmostEqual(metadata["fluid"]["inlet_velocity_mps"], 10.0)
        self.assertAlmostEqual(metadata["solid"]["density_kgm3"], 1600.0)
        self.assertAlmostEqual(metadata["solid"]["young_modulus_pa"], 1.0e6)
        self.assertAlmostEqual(metadata["solid"]["poisson_ratio"], 0.47)
        self.assertEqual(metadata["solid"]["constitutive_model"], "linear-elastic")
        self.assertEqual(metadata["solid"]["stress_state"], "plane-stress")
        self.assertEqual(metadata["time_integration"]["dt_s"], 5.0e-4)
        self.assertEqual(metadata["time_integration"]["step_count"], 50)
        self.assertAlmostEqual(metadata["time_integration"]["total_time_s"], 0.025)
        self.assertEqual(bc["inlet"]["type"], "velocity-inlet")
        self.assertAlmostEqual(bc["inlet"]["velocity_mps"], 10.0)
        self.assertEqual(bc["outlet"]["type"], "pressure-outlet")
        self.assertEqual(bc["symmetry"]["type"], "symmetry")
        self.assertEqual(bc["flap_root"]["structure"], "fixed-displacement")
        self.assertEqual(bc["flap_wall"]["coupling"], "intrinsic-two-way-fsi")
        self.assertEqual(
            metadata["fsi_interface"]["thin_wall_pressure_sampling"],
            ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
        )
        self.assertEqual(
            ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING["model"],
            "two-sided-fluid-pressure",
        )
        self.assertNotIn(
            "reference_pressure_pa",
            ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
        )
        self.assertGreaterEqual(
            ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING[
                "probe_max_multiplier"
            ],
            3.0,
        )
        self.assertEqual(
            CASE_MODULES["ansys-vertical-flap-fsi"],
            "cases.ansys_vertical_flap_fsi",
        )
        # Official monitor solid_max_total_col0_col6_m at step 50 (t=0.025 s).
        self.assertAlmostEqual(reference["max_displacement_m"], 5.8296e-5)
        # Peak over the 50-step run (step 9, t=0.0045 s) plus ringing period:
        # the official response is a lightly damped oscillation, so same-time
        # snapshots are phase-sensitive while the peak is not.
        self.assertAlmostEqual(
            reference["max_displacement_peak_over_run_m"], 4.3164e-4
        )
        self.assertAlmostEqual(reference["max_displacement_peak_time_s"], 4.5e-3)
        self.assertAlmostEqual(reference["displacement_ringing_period_s"], 8.5e-3)
        self.assertLess(
            reference["max_displacement_m"],
            reference["max_displacement_peak_over_run_m"],
        )
        self.assertEqual(reference["time_step_s"], 5.0e-4)
        self.assertEqual(reference["step_count"], 50)
        self.assertGreaterEqual(reference["local_velocity_peak_mps"], 20.0)
        self.assertLessEqual(reference["local_velocity_peak_mps"], 29.0)

    def test_core_uses_the_case_spec_authoritative_reference_subset(self):
        with patch.object(
            vertical_flap_case,
            "run_rectangular_solid_marker_mpm_fsi_smoke",
            return_value={},
        ) as run:
            vertical_flap_case._run_vertical_flap_fsi_core(
                VerticalFlapFsiConfig(step_count=0, preflow_steps=0)
            )

        self.assertEqual(
            run.call_args.kwargs["reference_results"],
            vertical_flap_case.CASE_SPEC.reference_results,
        )

    def test_config_uses_official_air_density_and_preserves_marker_area_feedback(
        self,
    ):
        config = VerticalFlapFsiConfig()

        self.assertAlmostEqual(config.air_density_kgm3, 1.2)
        self.assertEqual(config.solid_constitutive_model, "plane_stress_linear_elastic")
        self.assertFalse(config.enforce_plane_strain_x)
        self.assertTrue(config.preserve_marker_area_during_surface_feedback)
        self.assertFalse(config.update_fluid_obstacle_from_solid)
        # Root-clamp integrity: "pure_fixed_mass" leaves mixed fixed/free grid
        # nodes mobile with zero fixed-particle stress, so the cantilever root
        # creeps past static equilibrium (see NeoHookeanMpm static-cantilever
        # regression test); the flap case must lock any node touched by a
        # fixed particle to reproduce the tutorial's rigid flap_attach BC.
        self.assertEqual(config.fixed_node_lock_policy, "any_fixed_particle")
        self.assertAlmostEqual(config.solid_velocity_transfer_flip_blend, 0.0)
        self.assertAlmostEqual(
            config.flow_predictor_kinematic_viscosity_multiplier,
            1.0,
        )
        self.assertEqual(config.flow_predictor_substeps, 8)
        self.assertEqual(config.flow_predictor_no_slip_domain_walls, ("ymin",))
        self.assertEqual(config.flow_symmetry_domain_walls, ("ymax",))
        self.assertEqual(config.flow_ymin_no_slip_rows, 0)
        self.assertEqual(config.flow_obstacle_no_slip_layers, 0)
        self.assertEqual(config.flow_solid_boundary_mode, "hibm_sharp_marker_rows")
        self.assertEqual(config.flow_pressure_outlet_backflow_policy, "allow")
        self.assertEqual(config.flow_obstacle_normal_velocity_policy, "face_clamp")

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
        )
        run_source += inspect.getsource(
            solid_mpm_fsi_runner._advance_solid_substeps_batched
        )
        self.assertIn("preserve_marker_area_during_surface_feedback", run_source)
        self.assertIn("preserve_marker_area=", run_source)
        self.assertIn("solid_constitutive_model", run_source)
        self.assertIn("constitutive_model=", run_source)
        self.assertIn("velocity_transfer_flip_blend", run_source)
        self.assertIn("update_fluid_obstacle_from_solid", run_source)
        self.assertIn("fixed_node_lock_policy", run_source)

    def test_solid_seeding_guard_flags_underseeded_fine_grid_combination(self):
        """2026-07-03 fine-flap ejection audit: solid_particle_counts
        (1, 64, 12) on grid 4x256x320 leaves ~2 background cells between
        wall-normal particle layers, the MPM body numerically fractures at
        the root clamp, free-falls, and ejects particles. The seeding report
        must expose the spacing ratios and the opt-in guard must fail loud
        on that combination while accepting the repaired (1, 256, 20)
        seeding and the coarse production configuration."""
        coarse = VerticalFlapFsiConfig()
        coarse_report = solid_mpm_fsi_runner.solid_seeding_report(coarse)
        self.assertFalse(coarse_report["solid_seeding_guard_enabled"])
        self.assertTrue(coarse_report["solid_seeding_guard_satisfied"])
        self.assertLessEqual(
            coarse_report["solid_seeding_worst_guarded_spacing_cells"],
            1.5,
        )

        underseeded = replace(
            coarse,
            grid_nodes=(4, 256, 320),
            solid_particle_counts=(1, 64, 12),
            enforce_solid_seeding_limit=True,
        )
        underseeded_report = solid_mpm_fsi_runner.solid_seeding_report(
            underseeded
        )
        self.assertFalse(underseeded_report["solid_seeding_guard_satisfied"])
        self.assertGreater(
            underseeded_report["solid_particle_spacing_cells"][1],
            1.5,
        )
        with self.assertRaisesRegex(ValueError, "seeding is too sparse"):
            solid_mpm_fsi_runner._enforce_solid_seeding_limit(underseeded)

        repaired = replace(
            underseeded,
            solid_particle_counts=(1, 256, 20),
        )
        repaired_report = solid_mpm_fsi_runner._enforce_solid_seeding_limit(
            repaired
        )
        self.assertTrue(repaired_report["solid_seeding_guard_satisfied"])
        # The span axis intentionally stays single-column (2D-equivalent
        # slab); it must not participate in the guard.
        self.assertEqual(
            repaired_report["solid_seeding_guarded_axes"], (1, 2)
        )

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
        )
        self.assertIn("_enforce_solid_seeding_limit", run_source)

    def test_hibm_sharp_mode_starts_without_legacy_solid_obstacle(self):
        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 8, 40),
            flow_solid_boundary_mode="hibm_sharp_marker_rows",
        )

        initial = solid_mpm_fsi_runner._initial_fluid_obstacle(config)

        self.assertEqual(int(np.count_nonzero(initial)), 0)

    def test_legacy_cell_obstacle_mode_keeps_static_solid_obstacle(self):
        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 8, 40),
            flow_solid_boundary_mode="cell_obstacle_layers",
        )

        initial = solid_mpm_fsi_runner._initial_fluid_obstacle(config)

        self.assertGreater(int(np.count_nonzero(initial)), 0)

    def test_ymin_no_slip_rows_constrain_only_fluid_wall_cells(self):
        active = np.zeros((1, 3, 4), dtype=np.int32)
        values = np.ones((1, 3, 4, 3), dtype=np.float32)
        weights = np.zeros((1, 3, 4), dtype=np.float32)
        marker_regions = np.full((1, 3, 4), -1, dtype=np.int32)
        hard_masks = np.zeros((1, 3, 4), dtype=np.int32)
        external_exact_masks = np.zeros((1, 3, 4), dtype=np.int32)
        owned_rows = np.zeros((1, 3, 4), dtype=np.int32)
        obstacle = np.zeros((1, 3, 4), dtype=np.int32)
        obstacle[0, 0, 2] = 1
        config = SimpleNamespace(flow_ymin_no_slip_rows=1)

        solid_mpm_fsi_runner._apply_ymin_no_slip_rows(
            active,
            values,
            weights,
            marker_regions,
            hard_masks,
            external_exact_masks,
            owned_rows,
            obstacle,
            config,
        )

        self.assertEqual(active[0, 0, 0], 1)
        self.assertEqual(active[0, 0, 2], 0)
        np.testing.assert_allclose(values[0, 0, 0], [0.0, 0.0, 0.0])
        self.assertEqual(float(weights[0, 0, 0]), 1.0)
        self.assertEqual(float(weights[0, 0, 2]), 0.0)
        self.assertEqual(int(marker_regions[0, 0, 0]), -1)
        self.assertEqual(int(hard_masks[0, 0, 0]), 0b111)
        self.assertEqual(int(external_exact_masks[0, 0, 0]), 0b010)
        self.assertEqual(int(external_exact_masks[0, 0, 2]), 0)
        self.assertEqual(int(owned_rows[0, 0, 0]), 0)
        self.assertEqual(active[0, 1, 0], 0)

    def test_ymin_no_slip_rows_preserve_active_marker_feedback_targets(self):
        active = np.zeros((1, 3, 4), dtype=np.int32)
        values = np.ones((1, 3, 4, 3), dtype=np.float32)
        weights = np.zeros((1, 3, 4), dtype=np.float32)
        marker_regions = np.full((1, 3, 4), -1, dtype=np.int32)
        hard_masks = np.zeros((1, 3, 4), dtype=np.int32)
        external_exact_masks = np.zeros((1, 3, 4), dtype=np.int32)
        owned_rows = np.zeros((1, 3, 4), dtype=np.int32)
        obstacle = np.zeros((1, 3, 4), dtype=np.int32)
        active[0, 0, 1] = 1
        values[0, 0, 1] = (2.5, -0.5, 0.25)
        weights[0, 0, 1] = 1.0
        marker_regions[0, 0, 1] = 7
        hard_masks[0, 0, 1] = 0b111
        owned_rows[0, 0, 1] = 1
        config = SimpleNamespace(flow_ymin_no_slip_rows=1)

        solid_mpm_fsi_runner._apply_ymin_no_slip_rows(
            active,
            values,
            weights,
            marker_regions,
            hard_masks,
            external_exact_masks,
            owned_rows,
            obstacle,
            config,
        )

        self.assertEqual(active[0, 0, 1], 1)
        np.testing.assert_allclose(values[0, 0, 1], (2.5, -0.5, 0.25))
        self.assertEqual(float(weights[0, 0, 1]), 1.0)
        self.assertEqual(int(marker_regions[0, 0, 1]), 7)
        self.assertEqual(int(hard_masks[0, 0, 1]), 0b111)
        self.assertEqual(int(external_exact_masks[0, 0, 1]), 0)
        self.assertEqual(int(owned_rows[0, 0, 1]), 1)
        self.assertEqual(active[0, 0, 0], 1)
        np.testing.assert_allclose(values[0, 0, 0], (0.0, 0.0, 0.0))
        self.assertEqual(int(external_exact_masks[0, 0, 0]), 0b010)

    def test_obstacle_no_slip_layers_constrain_only_adjacent_fluid_cells(self):
        active = np.zeros((1, 4, 5), dtype=np.int32)
        values = np.ones((1, 4, 5, 3), dtype=np.float32)
        weights = np.zeros((1, 4, 5), dtype=np.float32)
        obstacle = np.zeros((1, 4, 5), dtype=np.int32)
        obstacle[0, 1, 2] = 1
        config = SimpleNamespace(flow_obstacle_no_slip_layers=1)

        count = solid_mpm_fsi_runner._apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )

        self.assertEqual(count, 4)
        for cell in ((0, 0, 2), (0, 2, 2), (0, 1, 1), (0, 1, 3)):
            self.assertEqual(int(active[cell]), 1)
            self.assertEqual(float(weights[cell]), 1.0)
            np.testing.assert_allclose(values[cell], [0.0, 0.0, 0.0])
        self.assertEqual(int(active[0, 1, 2]), 0)
        self.assertEqual(float(weights[0, 1, 2]), 0.0)
        self.assertEqual(int(active[0, 3, 4]), 0)

    def test_plane_stress_constitutive_model_uses_official_two_dimensional_lame(
        self,
    ):
        config = VerticalFlapFsiConfig()
        mu, lam = solid_mpm_fsi_runner._lame_parameters(config)

        expected_mu = config.young_modulus_pa / (2.0 * (1.0 + config.poisson_ratio))
        expected_lam = (
            config.young_modulus_pa
            * config.poisson_ratio
            / (1.0 - config.poisson_ratio * config.poisson_ratio)
        )
        legacy_lam = (
            config.young_modulus_pa
            * config.poisson_ratio
            / ((1.0 + config.poisson_ratio) * (1.0 - 2.0 * config.poisson_ratio))
        )

        self.assertAlmostEqual(mu, expected_mu)
        self.assertAlmostEqual(lam, expected_lam)
        self.assertLess(lam, 0.2 * legacy_lam)

    def test_solid_velocity_damping_is_distributed_over_substeps(self):
        config = VerticalFlapFsiConfig(velocity_damping=0.995, solid_substeps=1600)

        substep_damping = solid_mpm_fsi_runner._solid_substep_velocity_damping(
            config,
            solid_substeps=config.solid_substeps,
        )

        self.assertGreater(substep_damping, 0.999)
        self.assertAlmostEqual(substep_damping**config.solid_substeps, 0.995)

    def test_selected_official_config_uses_local_surface_force_support(self):
        config = selected_formulation_solver_config(step_count=50)

        self.assertAlmostEqual(
            config.mpm_support_radius_m,
            surface_force_support_radius_m(config),
        )
        self.assertLess(config.mpm_support_radius_m, 0.5 * config.flap_height_m)

    def test_selected_official_config_uses_bounded_marker_mac_failure_ceiling(self):
        config = selected_formulation_solver_config(step_count=50)

        self.assertEqual(config.flow_hibm_marker_mac_constraint_iterations, 64)

    def test_selected_official_config_keeps_boundary_markers_on_physical_faces(self):
        class FakeMarkers:
            def __init__(self, marker_capacity, runtime):
                self.marker_capacity = marker_capacity
                self.marker_count = 0
                self.projection_vertex_count = 0

            def load_markers(self, **kwargs):
                self.positions_m = list(kwargs["positions_m"])
                self.pressure_probe_origins_m = list(
                    kwargs["pressure_probe_origins_m"]
                )
                self.marker_count = len(self.positions_m)
                self.projection_vertex_count = self.marker_count

            def configure_open_ribbon_tip_cap(self, **kwargs):
                self.tip_cap = dict(kwargs)
                self.projection_vertex_count = self.marker_count + 4
                return self.projection_vertex_count

            def set_projection_segments(self, segment_indices):
                self.projection_segments = tuple(segment_indices)
                return len(self.projection_segments)

        config = replace(
            selected_formulation_solver_config(step_count=5),
            marker_count=2,
        )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(config)
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)
        dz = config.duct_length_m / float(config.grid_nodes[2])

        with patch.object(solid_mpm_fsi_runner, "HibmMpmSurfaceMarkers", FakeMarkers):
            markers = solid_mpm_fsi_runner._build_markers(config, runtime=None)

        self.assertEqual(config.flow_solid_boundary_mode, "hibm_sharp_marker_rows")
        self.assertAlmostEqual(config.traction_marker_face_offset_cells, 0.0)
        self.assertEqual(
            config.traction_pressure_probe_origin_mode,
            "physical_face_offset",
        )
        self.assertAlmostEqual(
            config.traction_pressure_probe_origin_offset_cells,
            0.51,
        )
        self.assertTrue(
            all(
                math.isclose(position[2], solid_max[2])
                for position in markers.positions_m[:2]
            )
        )
        self.assertTrue(
            all(
                math.isclose(position[2], solid_min[2])
                for position in markers.positions_m[2:]
            )
        )
        self.assertTrue(
            all(
                math.isclose(origin[2], solid_max[2] + 0.51 * dz)
                for origin in markers.pressure_probe_origins_m[:2]
            )
        )
        self.assertTrue(
            all(
                math.isclose(origin[2], solid_min[2] - 0.51 * dz)
                for origin in markers.pressure_probe_origins_m[2:]
            )
        )

    def test_production_geometry_reports_all_128_marker_constraint_semantics(self):
        test_started = perf_counter()
        stage_started = test_started

        def report_stage(stage_name: str) -> None:
            nonlocal stage_started
            now = perf_counter()
            print(
                "production marker gate timing:",
                {
                    "stage": stage_name,
                    "stage_elapsed_s": now - stage_started,
                    "total_elapsed_s": now - test_started,
                },
                flush=True,
            )
            stage_started = now

        grid_nodes = (4, 256, 320)
        base_config = selected_formulation_solver_config(step_count=0)
        config = replace(
            base_config,
            grid_nodes=grid_nodes,
            solid_particle_counts=(1, 256, 20),
            marker_count=64,
            # Retain the v4 launch identity, but do not invoke the preflow
            # loop: this fixture stops after one initial sharp assembly.
            preflow_steps=40,
            preflow_convergence_mode="single_step_legacy",
            flow_post_dirichlet_consistency_projection_iterations=3,
            flow_cg_preconditioner="fv_multigrid",
            flow_predictor_substeps=64,
            flow_hibm_sharp_search_radius_m=1.7e-3,
            flow_hibm_sharp_search_radius_xyz_m=(
                0.5 * float(base_config.span_m)
                - 0.4 * (float(base_config.span_m) / float(grid_nodes[0])),
                5.0
                * (
                    0.5
                    * float(base_config.duct_height_m)
                    / float(grid_nodes[1])
                ),
                1.5
                * (float(base_config.duct_length_m) / float(grid_nodes[2])),
            ),
            flow_hibm_sharp_interior_probe_distance_m=(
                1.5
                * max(
                    float(base_config.span_m) / float(grid_nodes[0]),
                    0.5
                    * float(base_config.duct_height_m)
                    / float(grid_nodes[1]),
                    float(base_config.duct_length_m) / float(grid_nodes[2]),
                )
            ),
            flow_hibm_sharp_interpolate_velocity_rows=False,
            flow_hibm_dynamic_solid_volume_enabled=True,
            flow_hibm_tiny_unreached_cleanup_component_cells=128,
            update_fluid_obstacle_from_solid=True,
            enforce_solid_seeding_limit=True,
        )
        config = vertical_flap_case.with_local_surface_force_support(config)
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = None
        markers = None
        solid = None
        sampling_identity = None
        sharp_boundary_cache: dict[str, object] = {}

        try:
            solid_mpm_fsi_runner._validate_rectangular_solid_config(config)
            report_stage("validate_config")
            fluid = solid_mpm_fsi_runner._build_fluid(config, runtime)
            report_stage("build_fluid")
            solid_mpm_fsi_runner._initialize_computed_flow(fluid, config)
            report_stage("initialize_flow")
            markers = solid_mpm_fsi_runner._build_markers(config, runtime)
            report_stage("build_markers")
            self.assertEqual(int(markers.marker_count), 128)
            self.assertEqual(int(markers.projection_vertex_count), 132)
            self.assertEqual(int(markers.projection_segment_count), 129)
            solid = solid_mpm_fsi_runner._build_solid(config, runtime)
            report_stage("build_solid")
            dynamic_obstacle_report = (
                solid_mpm_fsi_runner._update_fluid_obstacle_from_solid(
                    fluid,
                    solid,
                    config,
                )
            )
            report_stage("update_dynamic_obstacle")
            self.assertTrue(
                dynamic_obstacle_report[
                    "fluid_dynamic_obstacle_is_hibm_solid_volume"
                ]
            )
            boundary_report = (
                solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid(
                markers,
                fluid,
                config,
                update_pressure_gradient=False,
                boundary_cache=sharp_boundary_cache,
                )
            )
            report_stage("apply_sharp_boundary")
            canonical_report = boundary_report[
                "canonical_velocity_dirichlet_report"
            ]
            self.assertEqual(canonical_report["region_conflict_count"], 0)
            self.assertGreater(
                canonical_report[
                    "projection_only_region_seam_merged_count"
                ],
                0,
            )
            closure_report = canonical_report["marker_target_closure"]
            self.assertTrue(closure_report["enabled"])
            self.assertEqual(
                closure_report["projection_only_marker_count"],
                4,
            )
            self.assertEqual(
                closure_report["projection_only_evaluated_axis_count"],
                12,
            )
            self.assertEqual(
                closure_report["projection_only_invalid_axis_count"],
                0,
            )
            self.assertTrue(
                np.isfinite(
                    closure_report["projection_only_max_residual_mps"]
                )
            )
            self.assertLessEqual(
                closure_report["projection_only_max_residual_mps"],
                closure_report["absolute_tolerance_mps"],
            )

            component_face_valid_mask = (
                fluid.prepare_hibm_no_slip_component_face_valid_mask()
            )
            report_stage("prepare_component_face_valid_mask")
            sampling_identity = markers.prepare_no_slip_sampling_identity(
                obstacle_field=fluid.hibm_no_slip_sampling_obstacle,
                component_face_valid_mask=component_face_valid_mask,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                topology_generation=int(fluid.hibm_reachability_revision),
                component_face_valid_mask_generation=int(
                    fluid.velocity_dirichlet_component_ledger_generation
                ),
            )
            report_stage("prepare_no_slip_sampling_identity")

            marker_count = int(sampling_identity.marker_count)
            sample_valid = sampling_identity.sample_valid.to_numpy()[
                :marker_count
            ].astype(bool)
            sample_source = sampling_identity.sample_source_code.to_numpy()[
                :marker_count
            ]
            sample_position_m = sampling_identity.sample_position_m.to_numpy()[
                :marker_count
            ].astype(np.float64)
            marker_position_m = (
                sampling_identity.marker_position_snapshot_m.to_numpy()[
                    :marker_count
                ].astype(np.float64)
            )
            marker_normal = (
                sampling_identity.marker_normal_snapshot.to_numpy()[
                    :marker_count
                ].astype(np.float64)
            )

            self.assertTrue(np.all(np.isfinite(sample_position_m)))
            self.assertTrue(np.all(np.isfinite(marker_position_m)))
            self.assertTrue(np.all(np.isfinite(marker_normal)))
            sample_offset_m = sample_position_m - marker_position_m
            total_offset_m = np.linalg.norm(sample_offset_m, axis=1)
            normal_norm = np.linalg.norm(marker_normal, axis=1)
            unit_normal = marker_normal / np.where(
                normal_norm > 0.0,
                normal_norm,
                1.0,
            )[:, None]
            signed_normal_offset_m = np.einsum(
                "ij,ij->i",
                sample_offset_m,
                unit_normal,
            )
            normal_offset_m = np.abs(signed_normal_offset_m)
            tangential_offset_m = np.linalg.norm(
                sample_offset_m - signed_normal_offset_m[:, None] * unit_normal,
                axis=1,
            )

            valid_count = int(np.count_nonzero(sample_valid))
            invalid_count = int(marker_count - valid_count)
            direct_count = int(
                np.count_nonzero(sample_valid & (sample_source == 1))
            )
            normal_walk_count = int(
                np.count_nonzero(sample_valid & (sample_source == 2))
            )
            nearest_count = int(
                np.count_nonzero(sample_valid & (sample_source == 3))
            )
            moved_constraint_count = int(
                np.count_nonzero(
                    sample_valid
                    & (sample_source != 1)
                    & (total_offset_m > 0.0)
                )
            )
            max_total_offset_m = (
                float(np.max(total_offset_m[sample_valid]))
                if valid_count
                else 0.0
            )
            max_normal_offset_m = (
                float(np.max(normal_offset_m[sample_valid]))
                if valid_count
                else 0.0
            )
            max_tangential_offset_m = (
                float(np.max(tangential_offset_m[sample_valid]))
                if valid_count
                else 0.0
            )
            diagnostics = {
                "marker_count": marker_count,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "direct_count": direct_count,
                "normal_walk_count": normal_walk_count,
                "nearest_count": nearest_count,
                "moved_constraint_count": moved_constraint_count,
                "max_total_offset_m": max_total_offset_m,
                "max_normal_offset_m": max_normal_offset_m,
                "max_tangential_offset_m": max_tangential_offset_m,
            }
            print("production marker constraint semantics:", diagnostics)

            self.assertEqual(marker_count, 128, diagnostics)
            self.assertEqual(valid_count + invalid_count, 128, diagnostics)
            self.assertEqual(
                direct_count
                + normal_walk_count
                + nearest_count
                + invalid_count,
                128,
                diagnostics,
            )
            self.assertEqual(invalid_count, 0, diagnostics)
            self.assertEqual(direct_count, 128, diagnostics)
            self.assertEqual(normal_walk_count, 0, diagnostics)
            self.assertEqual(nearest_count, 0, diagnostics)
            self.assertEqual(moved_constraint_count, 0, diagnostics)
        finally:
            sharp_boundary_cache.clear()
            sampling_identity = None
            solid = None
            markers = None
            fluid = None

    def test_refined_solid_mpm_bounds_pad_root_particle_stencil(self):
        config = VerticalFlapFsiConfig(
            grid_nodes=(4, 128, 256),
            solid_particle_counts=(1, 80, 24),
        )
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)
        bounds_min, bounds_max = solid_mpm_fsi_runner._solid_mpm_bounds(config)
        dy = (bounds_max[1] - bounds_min[1]) / float(config.grid_nodes[1])
        particle_dy = (solid_max[1] - solid_min[1]) / float(
            config.solid_particle_counts[1]
        )
        first_root_y = solid_min[1] + 0.5 * particle_dy
        first_grid_coordinate = (first_root_y - bounds_min[1]) / dy

        self.assertLess(bounds_min[1], solid_min[1])
        self.assertGreaterEqual(first_grid_coordinate, 0.5)

    def test_dynamic_fluid_obstacle_follows_deformed_solid_particles(self):
        class FakeField:
            def __init__(self, values):
                self._values = values

            def to_numpy(self):
                return self._values.copy()

        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 8, 40),
            solid_particle_counts=(1, 4, 2),
        )
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)
        rest_rows = []
        row_height = config.flap_height_m / config.solid_particle_counts[1]
        for row in range(config.solid_particle_counts[1]):
            y = solid_min[1] + (row + 0.5) * row_height
            for z in (solid_min[2], solid_max[2]):
                rest_rows.append([0.5 * config.span_m, y, z])
        rest = np.asarray(rest_rows, dtype=np.float32)
        moved = rest.copy()
        moved[:, 2] -= 0.004
        solid = SimpleNamespace(
            particle_count=len(rest),
            x=FakeField(moved),
            rest_x=FakeField(rest),
        )

        static = solid_mpm_fsi_runner._solid_obstacle(config)
        dynamic = solid_mpm_fsi_runner._solid_obstacle_from_mpm_particles(
            solid,
            config,
        )

        static_k = np.argwhere(static != 0)[:, 2]
        dynamic_k = np.argwhere(dynamic != 0)[:, 2]
        self.assertGreater(static_k.size, 0)
        self.assertGreater(dynamic_k.size, 0)
        self.assertLess(int(dynamic_k.min()), int(static_k.min()))
        self.assertLess(int(dynamic_k.max()), int(static_k.max()))

    def test_dynamic_fluid_obstacle_update_prefers_solver_device_api(self):
        class FailingField:
            def to_numpy(self):
                raise AssertionError("device obstacle update must not download fields")

        class FakeFluid:
            def __init__(self):
                self.calls = []

            def update_dynamic_solid_obstacle_from_particles(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {
                    "fluid_dynamic_obstacle_cell_count": 3,
                    "fluid_dynamic_obstacle_added_cell_count": 2,
                    "fluid_dynamic_obstacle_removed_cell_count": 1,
                }

        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 8, 40),
            solid_particle_counts=(1, 4, 2),
            flow_solid_boundary_mode="cell_obstacle_layers",
            update_fluid_obstacle_from_solid=True,
        )
        fake_fluid = FakeFluid()
        fake_solid = SimpleNamespace(
            particle_count=8,
            x=FailingField(),
            rest_x=FailingField(),
        )

        report = solid_mpm_fsi_runner._update_fluid_obstacle_from_solid(
            fake_fluid,
            fake_solid,
            config,
        )

        self.assertEqual(len(fake_fluid.calls), 1)
        self.assertEqual(len(fake_fluid.calls[0][0]), 1)
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)
        expected_support = tuple(
            (solid_max[axis] - solid_min[axis])
            / config.solid_particle_counts[axis]
            for axis in range(3)
        )
        np.testing.assert_allclose(
            fake_fluid.calls[0][1]["particle_support_size_m"],
            expected_support,
        )
        self.assertTrue(report["fluid_dynamic_obstacle_update_enabled"])
        self.assertEqual(report["fluid_dynamic_obstacle_cell_count"], 3)
        self.assertEqual(report["fluid_dynamic_obstacle_added_cell_count"], 2)
        self.assertEqual(report["fluid_dynamic_obstacle_removed_cell_count"], 1)

    def test_particle_obstacle_fallback_vectorizes_row_rasterization(self):
        source = inspect.getsource(
            solid_mpm_fsi_runner._solid_obstacle_from_mpm_particles
        )

        self.assertIn("np.bincount", source)
        self.assertIn("np.minimum.at", source)
        self.assertIn("np.maximum.at", source)
        self.assertNotIn("for row_index in range", source)
        self.assertNotIn("for i in range(nx)", source)
        self.assertNotIn("for j in range(ny)", source)
        self.assertNotIn("for k in range(nz)", source)

    def test_hibm_sharp_mode_can_store_particle_obstacle_as_dynamic_volume(self):
        class FailingField:
            def to_numpy(self):
                raise AssertionError("sharp dynamic volume must stay on device")

        class FakeFluid:
            def __init__(self):
                self.calls = []

            def update_dynamic_solid_obstacle_from_particles(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {
                    "fluid_dynamic_obstacle_cell_count": 7,
                    "fluid_dynamic_obstacle_added_cell_count": 7,
                    "fluid_dynamic_obstacle_removed_cell_count": 0,
                }

        fake_fluid = FakeFluid()
        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 8, 40),
            solid_particle_counts=(1, 4, 2),
            flow_solid_boundary_mode="hibm_sharp_marker_rows",
            update_fluid_obstacle_from_solid=True,
            flow_hibm_dynamic_solid_volume_enabled=True,
        )
        fake_solid = SimpleNamespace(
            particle_count=8,
            x=FailingField(),
            rest_x=FailingField(),
        )

        report = solid_mpm_fsi_runner._update_fluid_obstacle_from_solid(
            fake_fluid,
            fake_solid,
            config,
        )

        self.assertEqual(len(fake_fluid.calls), 1)
        self.assertTrue(
            fake_fluid.calls[0][1]["store_as_hibm_dynamic_solid_volume"]
        )
        self.assertTrue(report["fluid_dynamic_obstacle_update_enabled"])
        self.assertTrue(report["fluid_dynamic_obstacle_is_hibm_solid_volume"])

    def test_hibm_dynamic_solid_volume_requires_device_update(self):
        config = VerticalFlapFsiConfig(
            flow_solid_boundary_mode="hibm_sharp_marker_rows",
            flow_hibm_dynamic_solid_volume_enabled=True,
            update_fluid_obstacle_from_solid=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "dynamic solid volume requires update_fluid_obstacle_from_solid",
        ):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(config)

    def test_hibm_tiny_unreached_cleanup_threshold_must_be_non_negative(self):
        config = VerticalFlapFsiConfig(
            flow_hibm_tiny_unreached_cleanup_component_cells=-1,
        )

        with self.assertRaisesRegex(
            ValueError,
            "flow_hibm_tiny_unreached_cleanup_component_cells must be non-negative",
        ):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(config)

    def test_sharp_boundary_is_refreshed_before_predictor_and_after_predictor(self):
        source = inspect.getsource(solid_mpm_fsi_runner._flow_advance_current_step)

        pre_refresh = source.index("pre_predictor_sharp_boundary_report")
        predictor = source.index("fluid.predict(")
        post_refresh = source.index(
            "_apply_hibm_sharp_marker_boundary_to_fluid(",
            predictor,
        )
        projection = source.index("main_flow_report = _project_current_flow(")
        consistency_refresh = source.index(
            "_apply_hibm_sharp_marker_boundary_to_fluid(",
            projection,
        )
        consistency_projection = source.index(
            "consistency_flow_report = _project_current_flow(",
            consistency_refresh,
        )

        self.assertLess(pre_refresh, predictor)
        self.assertLess(predictor, post_refresh)
        self.assertLess(post_refresh, projection)
        self.assertLess(projection, consistency_refresh)
        self.assertLess(consistency_refresh, consistency_projection)

    def test_observer_boundary_rows_refresh_after_solid_volume_commit(self):
        source = inspect.getsource(
            solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
        )

        volume_commit = source.index("_update_fluid_obstacle_from_solid(")
        observer_refresh = source.index(
            "latest_observer_topology_report =", volume_commit
        )
        full_row_rebuild = source.index("topology_only=False", observer_refresh)
        observer = source.index("if step_observer is not None:", full_row_rebuild)

        self.assertLess(volume_commit, observer_refresh)
        self.assertLess(observer_refresh, full_row_rebuild)
        self.assertLess(full_row_rebuild, observer)

    def test_sharp_topology_cleanup_runs_before_pressure_rows_and_is_not_repeated(self):
        boundary_source = inspect.getsource(
            solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid
        )
        advance_source = inspect.getsource(
            solid_mpm_fsi_runner._flow_advance_current_step
        )

        first_velocity_rows = boundary_source.index(
            "velocity_report = assemble_velocity_rows()"
        )
        reachability_helper = boundary_source.index(
            "def refresh_pressure_reachability"
        )
        reachability = boundary_source.index(
            "mark_hibm_pressure_outlet_disconnected_nonprojectable_cells",
            reachability_helper,
        )
        rebuild_helper = boundary_source.index(
            "def rebuild_velocity_rows_after_topology_mutation",
            reachability,
        )
        cleanup_loop = boundary_source.index(
            "for _topology_cleanup_pass in range(8):",
            rebuild_helper,
        )
        reachability_call = boundary_source.index(
            "refresh_pressure_reachability()",
            cleanup_loop,
        )
        overflow_conversion = boundary_source.index(
            "fluid.convert_hibm_row_cloud_orphan_components(",
            reachability_call,
        )
        transactional_rebuild_callback = boundary_source.index(
            "after_topology_mutation=(",
            overflow_conversion,
        )
        overflow_rejection_guard = boundary_source.index(
            "if overflow_rejected_cell_count > 0:",
            transactional_rebuild_callback,
        )
        cleanup = boundary_source.index(
            "cleanup_hibm_pressure_outlet_tiny_unreached_components",
            overflow_rejection_guard,
        )
        pressure_rows = boundary_source.index(
            "assemble_pressure_neumann_matrix_rows"
        )

        self.assertLess(first_velocity_rows, reachability_helper)
        self.assertLess(reachability, rebuild_helper)
        self.assertLess(rebuild_helper, cleanup_loop)
        self.assertLess(cleanup_loop, reachability_call)
        self.assertLess(reachability_call, overflow_conversion)
        self.assertLess(overflow_conversion, transactional_rebuild_callback)
        self.assertLess(transactional_rebuild_callback, overflow_rejection_guard)
        self.assertLess(overflow_rejection_guard, cleanup)
        self.assertIn(
            '"hibm_preassembly_tiny_unreached_cleanup_cell_count": 0',
            boundary_source[
                overflow_rejection_guard:cleanup
            ],
        )
        self.assertLess(cleanup, pressure_rows)
        self.assertIn("reuse_topology_from_previous_assembly=True", advance_source)

    def test_formal_runner_maps_official_streamwise_flap_box_to_solver_z(self):
        config = VerticalFlapFsiConfig()
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)

        self.assertAlmostEqual(solid_min[0], 0.0)
        self.assertAlmostEqual(solid_max[0], config.span_m)
        self.assertAlmostEqual(
            solid_min[2],
            config.duct_length_m - config.flap_streamwise_max_m,
        )
        self.assertAlmostEqual(
            solid_max[2],
            config.duct_length_m - config.flap_streamwise_min_m,
        )
        self.assertAlmostEqual(
            config.duct_length_m - solid_max[2],
            config.flap_streamwise_min_m,
        )
        self.assertAlmostEqual(
            config.duct_length_m - solid_min[2],
            config.flap_streamwise_max_m,
        )

    def test_formal_runner_places_both_streamwise_marker_faces(self):
        class FakeMarkers:
            def __init__(self, marker_capacity, runtime):
                self.marker_capacity = marker_capacity
                self.runtime = runtime
                self.marker_count = 0
                self.positions_m = []
                self.normals = []
                self.areas_m2 = []
                self.region_ids = []
                self.projection_segments = ()
                self.projection_vertex_count = 0
                self.tip_cap = None

            def load_markers(
                self,
                *,
                positions_m,
                velocities_mps,
                normals,
                areas_m2,
                region_ids,
                pressure_probe_origins_m=None,
            ):
                self.positions_m = list(positions_m)
                self.normals = list(normals)
                self.areas_m2 = list(areas_m2)
                self.region_ids = list(region_ids)
                self.pressure_probe_origins_m = pressure_probe_origins_m
                self.marker_count = len(self.positions_m)
                self.projection_vertex_count = self.marker_count

            def configure_open_ribbon_tip_cap(self, **kwargs):
                self.tip_cap = dict(kwargs)
                self.projection_vertex_count = self.marker_count + 4
                return self.projection_vertex_count

            def set_projection_segments(self, segment_indices):
                self.projection_segments = tuple(tuple(row) for row in segment_indices)
                return len(self.projection_segments)

        config = VerticalFlapFsiConfig(marker_count=3)
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)
        dz = config.duct_length_m / float(config.grid_nodes[2])
        with patch.object(solid_mpm_fsi_runner, "HibmMpmSurfaceMarkers", FakeMarkers):
            markers = solid_mpm_fsi_runner._build_markers(config, runtime=None)

        self.assertEqual(markers.marker_capacity, 10)
        self.assertEqual(markers.marker_count, 6)
        self.assertEqual(markers.projection_vertex_count, 10)
        self.assertEqual(
            markers.projection_segments,
            ((0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (5, 7), (8, 9)),
        )
        self.assertEqual(
            markers.tip_cap["cap_region_id"],
            solid_mpm_fsi_runner.TIP_CAP_BOUNDARY_REGION_ID,
        )
        self.assertEqual(markers.tip_cap["inactive_axis"], 0)
        self.assertAlmostEqual(
            markers.tip_cap["cap_area_m2"],
            (solid_max[0] - solid_min[0]) * (solid_max[2] - solid_min[2]),
        )
        boundary_report = (
            solid_mpm_fsi_runner._marker_projection_boundary_report_fields(markers)
        )
        self.assertEqual(boundary_report["marker_physical_traction_count"], 6)
        self.assertEqual(boundary_report["marker_projection_vertex_count"], 10)
        self.assertEqual(boundary_report["marker_boundary_only_vertex_count"], 4)
        self.assertTrue(boundary_report["tip_cap_boundary_enabled"])
        self.assertTrue(boundary_report["tip_cap_force_included"])
        self.assertFalse(boundary_report["tip_cap_no_slip_closure_included"])
        self.assertEqual(
            boundary_report["tip_cap_no_slip_health_policy"],
            "missing_kernel_evidence",
        )
        measured_boundary_report = (
            solid_mpm_fsi_runner._marker_projection_boundary_report_fields(
                markers,
                canonical_velocity_dirichlet_report={
                    "marker_target_closure": {
                        "enabled": True,
                        "constraint_count": 0,
                        "adjustable_constraint_count": 0,
                        "immutable_constraint_count": 0,
                        "solver": "weighted_minimum_norm_lstsq",
                        "solve_count": 0,
                        "matrix_rank": 0,
                        "adjustable_dof_count": 0,
                        "least_squares_max_residual_mps": 0.0,
                        "materialized_max_residual_mps": 0.0,
                        "max_abs_correction_mps": 0.0,
                        "initial_max_residual_mps": 0.0,
                        "final_max_residual_mps": 0.0,
                        "final_max_adjustable_residual_mps": 0.0,
                        "final_max_immutable_residual_mps": 0.0,
                        "absolute_tolerance_mps": 1.0e-4,
                        "closure_tolerance_mps": 1.0e-6,
                        "density_kgm3": 1.2,
                        "projection_only_marker_count": 4,
                        "projection_only_evaluated_axis_count": 12,
                        "projection_only_invalid_axis_count": 0,
                        "projection_only_constraint_count": 0,
                        "projection_only_max_residual_mps": 0.0,
                    }
                },
            )
        )
        self.assertTrue(
            measured_boundary_report["tip_cap_no_slip_closure_included"]
        )
        self.assertEqual(
            measured_boundary_report["tip_cap_no_slip_health_policy"],
            "canonical_marker_target_closure_kernel_evidence",
        )
        self.assertEqual(
            measured_boundary_report[
                "tip_cap_marker_target_closure_evaluated_axis_count"
            ],
            12,
        )
        self.assertEqual(
            boundary_report["tip_cap_traction_policy"],
            "one_sided_gauge_pressure_outward_normal",
        )
        self.assertEqual(markers.normals[:3], [(0.0, 0.0, 1.0)] * 3)
        self.assertEqual(markers.normals[3:], [(0.0, 0.0, -1.0)] * 3)
        self.assertEqual(
            markers.region_ids[:3],
            [solid_mpm_fsi_runner.PRIMARY_REGION_ID] * 3,
        )
        self.assertEqual(
            markers.region_ids[3:],
            [solid_mpm_fsi_runner.SECONDARY_REGION_ID] * 3,
        )
        self.assertTrue(
            all(
                math.isclose(position[2], solid_max[2] + 0.51 * dz)
                for position in markers.positions_m[:3]
            )
        )
        self.assertTrue(
            all(
                math.isclose(position[2], solid_min[2] - 0.51 * dz)
                for position in markers.positions_m[3:]
            )
        )

        offset_config = VerticalFlapFsiConfig(
            marker_count=2,
            traction_marker_face_offset_cells=1.0,
        )
        offset_solid_min, offset_solid_max = solid_mpm_fsi_runner._solid_box(
            offset_config
        )
        with patch.object(solid_mpm_fsi_runner, "HibmMpmSurfaceMarkers", FakeMarkers):
            offset_markers = solid_mpm_fsi_runner._build_markers(
                offset_config,
                runtime=None,
            )

        self.assertEqual(offset_markers.marker_capacity, 8)
        self.assertEqual(offset_markers.marker_count, 4)
        self.assertEqual(offset_markers.projection_vertex_count, 8)
        self.assertEqual(
            offset_markers.projection_segments,
            ((0, 1), (2, 3), (1, 4), (3, 5), (6, 7)),
        )
        self.assertTrue(
            all(
                math.isclose(
                    position[2],
                    offset_solid_max[2] + dz,
                )
                for position in offset_markers.positions_m[:2]
            )
        )
        self.assertTrue(
            all(
                math.isclose(
                    position[2],
                    offset_solid_min[2] - dz,
                )
                for position in offset_markers.positions_m[2:]
            )
        )

        single_config = VerticalFlapFsiConfig(
            marker_count=3,
            traction_marker_layout="single_mid_surface",
            traction_marker_face_offset_cells=0.0,
        )
        with patch.object(solid_mpm_fsi_runner, "HibmMpmSurfaceMarkers", FakeMarkers):
            single_markers = solid_mpm_fsi_runner._build_markers(
                single_config,
                runtime=None,
            )

        single_solid_min, single_solid_max = solid_mpm_fsi_runner._solid_box(
            single_config
        )
        midpoint_z = 0.5 * (
            single_solid_min[2]
            + single_solid_max[2]
        )
        self.assertEqual(single_markers.marker_capacity, 3)
        self.assertEqual(single_markers.marker_count, 3)
        self.assertEqual(single_markers.projection_vertex_count, 3)
        self.assertIsNone(single_markers.tip_cap)
        self.assertEqual(single_markers.projection_segments, ((0, 1), (1, 2)))
        single_boundary_report = (
            solid_mpm_fsi_runner._marker_projection_boundary_report_fields(
                single_markers
            )
        )
        self.assertEqual(single_boundary_report["marker_boundary_only_vertex_count"], 0)
        self.assertFalse(single_boundary_report["tip_cap_boundary_enabled"])
        self.assertIsNone(single_boundary_report["tip_cap_boundary_region_id"])
        self.assertFalse(
            single_boundary_report["tip_cap_no_slip_closure_included"]
        )
        self.assertEqual(
            single_boundary_report["tip_cap_traction_policy"],
            "not_applicable",
        )
        self.assertEqual(single_markers.normals, [(0.0, 0.0, 1.0)] * 3)
        self.assertEqual(
            single_markers.region_ids,
            [solid_mpm_fsi_runner.PRIMARY_REGION_ID] * 3,
        )
        self.assertTrue(
            all(
                math.isclose(position[2], midpoint_z)
                for position in single_markers.positions_m
            )
        )
        self.assertEqual(solid_mpm_fsi_runner._traction_marker_face_count(config), 2)
        self.assertEqual(
            solid_mpm_fsi_runner._traction_marker_face_count(single_config),
            1,
        )

    def test_dual_face_tip_cap_requires_two_markers_per_face(self):
        with self.assertRaisesRegex(ValueError, "at least two markers per face"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(marker_count=1)
            )

    def test_generic_problem_defaults_to_runtime_pressure_pairs_without_json(self):
        problem = build_ansys_vertical_flap_generic_problem(step_count=50)
        provider = problem.traction_config.pressure_sampling.pair_provider

        self.assertEqual(provider.mode, "runtime_anchored_cell_pair")
        self.assertEqual(provider.pair_source_status, "runtime_generated")
        self.assertEqual(provider.source, "")
        self.assertFalse(provider.transition_backed)
        self.assertNotIn("selected_anchor_markers_json", provider.as_diagnostics()["source"])

        config = selected_formulation_solver_config(step_count=50)
        self.assertIsNone(config.traction_pressure_pair_anchor_markers_json)
        self.assertEqual(
            config.traction_pressure_pair_runtime_provider_mode,
            "runtime_anchored_cell_pair",
        )

    def test_generic_replay_pressure_pairs_require_anchor_json(self):
        with self.assertRaisesRegex(
            ValueError,
            "replay_from_diagnostics requires selected_anchor_markers_json",
        ):
            build_ansys_vertical_flap_generic_problem(
                pressure_pair_provider_mode="replay_from_diagnostics",
                step_count=50,
            )

        problem = build_ansys_vertical_flap_generic_problem(
            pressure_pair_provider_mode="replay_from_diagnostics",
            selected_anchor_markers_json=SELECTED_ANCHOR_MARKERS_JSON,
            step_count=50,
        )
        provider = problem.traction_config.pressure_sampling.pair_provider
        self.assertEqual(provider.mode, "replay_from_diagnostics")
        self.assertEqual(provider.source, SELECTED_ANCHOR_MARKERS_JSON)
        self.assertTrue(provider.transition_backed)

    def test_generic_pressure_pair_provider_mode_is_fail_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "unsupported pressure_pair_provider_mode",
        ):
            build_ansys_vertical_flap_generic_problem(
                pressure_pair_provider_mode="transition_seeded_from_anchor_artifact",
                step_count=50,
            )

    def test_runtime_pressure_pair_mode_ignores_supplied_anchor_json_source(self):
        problem = build_ansys_vertical_flap_generic_problem(
            selected_anchor_markers_json=SELECTED_ANCHOR_MARKERS_JSON,
            step_count=50,
        )
        provider = problem.traction_config.pressure_sampling.pair_provider

        self.assertEqual(provider.mode, "runtime_anchored_cell_pair")
        self.assertEqual(provider.pair_source_status, "runtime_generated")
        self.assertEqual(provider.source, "")
        self.assertFalse(provider.transition_backed)

        config = selected_formulation_solver_config(
            step_count=50,
            selected_anchor_markers_json=SELECTED_ANCHOR_MARKERS_JSON,
        )
        self.assertIsNone(config.traction_pressure_pair_anchor_markers_json)
        self.assertEqual(
            config.traction_pressure_pair_runtime_provider_mode,
            "runtime_anchored_cell_pair",
        )

    def test_generic_problem_reports_runtime_half_domain_metadata(self):
        problem = build_ansys_vertical_flap_generic_problem(step_count=50)
        config = VerticalFlapFsiConfig(step_count=50)

        self.assertEqual(
            problem.fluid_domain.coordinate_model,
            "cartesian-3d-half-domain",
        )
        self.assertEqual(problem.fluid_domain.grid_nodes, (4, 32, 64))
        self.assertEqual(problem.fluid_domain.bounds_m[0], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(problem.fluid_domain.bounds_m[1][0], config.span_m)
        self.assertAlmostEqual(
            problem.fluid_domain.bounds_m[1][1],
            0.5 * config.duct_height_m,
        )
        self.assertAlmostEqual(
            problem.fluid_domain.bounds_m[1][1],
            ANSYS_VERTICAL_FLAP_CASE_METADATA["geometry"]["modeled_height_m"],
        )
        self.assertAlmostEqual(
            problem.fluid_domain.bounds_m[1][2],
            config.duct_length_m,
        )
        self.assertEqual(
            problem.metadata["conceptual_coordinate_model"],
            "cartesian-2d",
        )
        self.assertEqual(
            problem.metadata["runtime_discretization_model"],
            "cartesian-3d-half-domain",
        )
        self.assertAlmostEqual(problem.metadata["extrusion_depth_m"], config.span_m)
        self.assertEqual(
            problem.metadata["extrusion_depth_source"],
            "VerticalFlapFsiConfig.span_m",
        )
        self.assertEqual(
            problem.metadata["out_of_plane_boundary_policy"],
            "strict_periodic_or_slip",
        )

    def test_slab_equivalence_diagnostics_expose_3d_and_per_depth_quantities(self):
        config = VerticalFlapFsiConfig(
            span_m=0.003,
            marker_count=4,
            solid_particle_counts=(1, 10, 3),
        )

        report = solid_mpm_fsi_runner.slab_equivalence_diagnostics(
            config,
            interface_force_total_n=(0.0, 0.0, -3.0e-4),
            pressure_force_total_n=(0.0, 0.0, -2.4e-4),
            max_displacement_m=2.0e-6,
            conceptual_coordinate_model="cartesian-2d",
            runtime_discretization_model="cartesian-3d-half-domain",
        )

        self.assertEqual(report["conceptual_coordinate_model"], "cartesian-2d")
        self.assertEqual(report["runtime_discretization_model"], "cartesian-3d-half-domain")
        self.assertEqual(report["out_of_plane_axis"], "x")
        self.assertEqual(report["streamwise_axis"], "z")
        self.assertAlmostEqual(report["extrusion_depth_m"], config.span_m)
        self.assertEqual(report["extrusion_depth_source"], "VerticalFlapFsiConfig.span_m")
        self.assertAlmostEqual(report["flap_streamwise_thickness_m"], config.flap_thickness_m)
        self.assertEqual(report["flap_streamwise_thickness_source"], "VerticalFlapFsiConfig.flap_thickness_m")
        self.assertEqual(report["marker_face_count"], 2)
        self.assertAlmostEqual(
            report["marker_total_area_m2"],
            2.0 * config.flap_height_m * config.span_m,
        )
        self.assertAlmostEqual(
            report["marker_total_area_per_depth_m"],
            2.0 * config.flap_height_m,
        )
        expected_mass = (
            config.solid_density_kgm3
            * config.span_m
            * config.flap_height_m
            * config.flap_thickness_m
        )
        self.assertAlmostEqual(report["solid_mass_total_kg"], expected_mass)
        self.assertAlmostEqual(
            report["solid_mass_per_depth_kgpm"],
            expected_mass / config.span_m,
        )
        self.assertEqual(report["interface_force_total_n"], (0.0, 0.0, -3.0e-4))
        self.assertEqual(report["pressure_force_total_n"], (0.0, 0.0, -2.4e-4))
        self.assertAlmostEqual(report["interface_force_z_per_depth_N_per_m"], -0.1)
        self.assertAlmostEqual(report["pressure_force_z_per_depth_N_per_m"], -0.08)
        self.assertEqual(report["max_displacement_m"], 2.0e-6)
        self.assertEqual(
            report["displacement_depth_scaling_expectation"],
            "depth_invariant_when_force_and_mass_scale_together",
        )
        self.assertTrue(report["out_of_plane_boundary_residual_modeling_error"])

    def test_slab_depth_scaling_keeps_two_dimensional_quantities_invariant(self):
        base = VerticalFlapFsiConfig(span_m=0.003)
        doubled = replace(base, span_m=0.006)
        force_per_depth_z = -0.1

        base_report = solid_mpm_fsi_runner.slab_equivalence_diagnostics(
            base,
            interface_force_total_n=(0.0, 0.0, force_per_depth_z * base.span_m),
            pressure_force_total_n=(0.0, 0.0, 0.8 * force_per_depth_z * base.span_m),
            max_displacement_m=2.0e-6,
        )
        doubled_report = solid_mpm_fsi_runner.slab_equivalence_diagnostics(
            doubled,
            interface_force_total_n=(0.0, 0.0, force_per_depth_z * doubled.span_m),
            pressure_force_total_n=(0.0, 0.0, 0.8 * force_per_depth_z * doubled.span_m),
            max_displacement_m=2.0e-6,
        )

        self.assertAlmostEqual(
            doubled_report["marker_total_area_m2"] / base_report["marker_total_area_m2"],
            2.0,
        )
        self.assertAlmostEqual(
            doubled_report["solid_mass_total_kg"] / base_report["solid_mass_total_kg"],
            2.0,
        )
        self.assertAlmostEqual(
            doubled_report["interface_force_z_N"] / base_report["interface_force_z_N"],
            2.0,
        )
        self.assertAlmostEqual(
            doubled_report["marker_total_area_per_depth_m"],
            base_report["marker_total_area_per_depth_m"],
        )
        self.assertAlmostEqual(
            doubled_report["solid_mass_per_depth_kgpm"],
            base_report["solid_mass_per_depth_kgpm"],
        )
        self.assertAlmostEqual(
            doubled_report["interface_force_z_per_depth_N_per_m"],
            base_report["interface_force_z_per_depth_N_per_m"],
        )
        self.assertEqual(
            doubled_report["max_displacement_m"],
            base_report["max_displacement_m"],
        )

    def test_formal_runner_uses_public_stress_face_diagnostics(self):
        source = inspect.getsource(solid_mpm_fsi_runner._marker_traction_report_fields)

        self.assertIn("stress_face_diagnostics(", source)
        self.assertNotIn("._stress_pressure_valid", source)
        self.assertNotIn("._stress_inside_pressure_found", source)
        self.assertNotIn("._stress_outside_pressure_found", source)

    def test_solid_substep_cfl_report_preserves_explicit_higher_count(self):
        unstable = VerticalFlapFsiConfig(
            grid_nodes=(4, 320, 640),
            solid_substeps=200,
        )
        unstable_report = solid_mpm_fsi_runner.solid_substep_cfl_report(unstable)
        solid_spacing = solid_mpm_fsi_runner._solid_mpm_grid_spacing_m(unstable)
        fluid_spacing = solid_mpm_fsi_runner._grid_spacing_m(unstable)

        self.assertAlmostEqual(
            unstable_report["solid_min_grid_spacing_m"],
            min(solid_spacing),
        )
        self.assertNotAlmostEqual(min(solid_spacing), min(fluid_spacing))

        mu, lam = solid_mpm_fsi_runner._lame_parameters(unstable)
        expected_minimum = math.ceil(
            math.sqrt((lam + 2.0 * mu) / unstable.solid_density_kgm3)
            * unstable.dt_s
            / (
                unstable.solid_cfl_target
                * unstable_report["solid_min_grid_spacing_m"]
            )
        )
        self.assertEqual(
            unstable_report["solid_substeps_cfl_minimum"],
            expected_minimum,
        )
        self.assertGreater(
            unstable_report["solid_substeps_cfl_minimum"],
            unstable.solid_substeps,
        )
        self.assertEqual(
            unstable_report["solid_substeps_selected"],
            unstable_report["solid_substeps_cfl_minimum"],
        )
        self.assertTrue(unstable_report["solid_substeps_auto_applied"])
        self.assertLessEqual(
            unstable_report["solid_estimated_cfl"],
            unstable_report["solid_cfl_target"],
        )

        explicit = VerticalFlapFsiConfig(
            grid_nodes=(4, 320, 640),
            solid_substeps=1200,
        )
        explicit_report = solid_mpm_fsi_runner.solid_substep_cfl_report(explicit)

        self.assertEqual(explicit_report["solid_substeps_selected"], 1200)
        self.assertFalse(explicit_report["solid_substeps_auto_applied"])

    def test_preflow_controls_are_exposed_without_changing_default_smoke(self):
        config = VerticalFlapFsiConfig()

        self.assertEqual(config.preflow_steps, 0)
        self.assertEqual(config.preflow_convergence_tolerance, 0.0)
        self.assertTrue(config.apply_marker_feedback_to_fluid)
        self.assertFalse(config.flow_reset_pressure_each_step)
        self.assertFalse(config.flow_reinitialize_inlet_each_step)
        self.assertEqual(config.flow_driver_mode, "projection_only")
        self.assertEqual(config.flow_predictor_substeps, 8)
        self.assertEqual(config.flow_inlet_source_strength, 1.0)
        self.assertEqual(config.flow_inlet_source_ramp_steps, 0)
        self.assertEqual(config.flow_inlet_source_profile, "constant")
        self.assertTrue(config.flow_pressure_outlet_enabled)
        self.assertEqual(config.flow_pressure_outlet_backflow_policy, "allow")
        self.assertEqual(config.flow_outlet_balance_policy, "report_only")

        parser = vertical_flap_case._build_parser()
        args = parser.parse_args(
            [
                "--steps",
                "3",
                "--preflow-steps",
                "2",
                "--preflow-convergence-tolerance",
                "0.01",
                "--disable-marker-feedback",
                "--flow-reset-pressure-each-step",
                "--flow-reinitialize-inlet-each-step",
                "--flow-driver-mode",
                "sustained_volume_source_inlet",
                "--flow-inlet-source-strength",
                "0.5",
                "--flow-inlet-source-ramp-steps",
                "5",
                "--flow-inlet-source-profile",
                "linear_ramp",
                "--disable-pressure-outlet",
                "--flow-pressure-outlet-backflow-policy",
                "allow",
                "--flow-obstacle-normal-velocity-policy",
                "cell_zero_only",
                "--flow-outlet-balance-policy",
                "report_only",
                "--json",
            ]
        )

        self.assertEqual(args.steps, 3)
        self.assertEqual(args.preflow_steps, 2)
        self.assertAlmostEqual(args.preflow_convergence_tolerance, 0.01)
        self.assertTrue(args.disable_marker_feedback)
        self.assertTrue(args.flow_reset_pressure_each_step)
        self.assertTrue(args.flow_reinitialize_inlet_each_step)
        self.assertEqual(args.flow_driver_mode, "sustained_volume_source_inlet")
        self.assertAlmostEqual(args.flow_inlet_source_strength, 0.5)
        self.assertEqual(args.flow_inlet_source_ramp_steps, 5)
        self.assertEqual(args.flow_inlet_source_profile, "linear_ramp")
        self.assertTrue(args.disable_pressure_outlet)
        self.assertEqual(args.flow_pressure_outlet_backflow_policy, "allow")
        self.assertEqual(args.flow_obstacle_normal_velocity_policy, "cell_zero_only")
        self.assertEqual(args.flow_outlet_balance_policy, "report_only")

    def test_hibm_sharp_controls_are_declared_config_fields(self):
        config = VerticalFlapFsiConfig(
            flow_hibm_sharp_search_radius_m=0.0123,
            flow_hibm_sharp_interior_probe_distance_m=0.0045,
            flow_hibm_sharp_interior_probe_distance_xyz_m=(0.001, 0.002, 0.003),
            flow_hibm_sharp_interpolate_velocity_rows=False,
        )

        for field_name in (
            "flow_hibm_sharp_search_radius_m",
            "flow_hibm_sharp_interior_probe_distance_m",
            "flow_hibm_sharp_interior_probe_distance_xyz_m",
            "flow_hibm_sharp_interpolate_velocity_rows",
        ):
            self.assertIn(field_name, VerticalFlapFsiConfig.__dataclass_fields__)
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._hibm_sharp_search_radius_m(config),
            0.0123,
        )
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._hibm_sharp_interior_probe_distance_m(config),
            0.0045,
        )
        self.assertEqual(
            solid_mpm_fsi_runner._hibm_sharp_interior_probe_distance_xyz_m(config),
            (0.001, 0.002, 0.003),
        )
        self.assertFalse(config.flow_hibm_sharp_interpolate_velocity_rows)

    def test_official_snapshot_runner_uses_case_defaults_for_core_flow_contracts(self):
        source = Path(
            "validation_runs/ansys_vertical_flap_fsi/scripts/"
            "run_official_fluent_2way_fsi50_snapshot.py"
        ).read_text(encoding="utf-8")

        # The snapshot runner builds the case config from case defaults and must
        # not override these core flow contracts; it leaves them unset so the
        # VerticalFlapFsiConfig defaults apply.
        self.assertIn("VerticalFlapFsiConfig(", source)
        self.assertIn("export_final_flow_snapshot=True", source)
        self.assertNotIn("flow_inlet_source_strength=", source)
        self.assertNotIn("flow_predictor_substeps=", source)
        self.assertNotIn("flow_inlet_source_strength: float = 0.6", source)
        self.assertNotIn("flow_predictor_substeps: int = 1", source)

    def test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance(self):
        source = inspect.getsource(solid_mpm_fsi_runner._run_fixed_solid_preflow)

        self.assertIn('"preflow_steps_requested"', source)
        self.assertIn('"preflow_steps_completed"', source)
        self.assertIn('"preflow_status"', source)
        self.assertIn('"preflow_history"', source)
        self.assertIn('"solid_fixed": True', source)
        self.assertIn('"solid_advanced": False', source)
        self.assertIn("_flow_advance_current_step(", source)
        self.assertIn("_sample_stress_to_marker_forces(markers, fluid, config)", source)
        self.assertNotIn("solid.step(", source)
        self.assertIn("scatter_marker_forces_to_mpm_particles", source)
        self.assertIn("_scatter_report_fields(scatter_report)", source)
        self.assertIn("_marker_traction_report_fields(", source)
        self.assertIn("include_face_diagnostics=False", source)

    def test_preflow_only_report_propagates_feedback_diagnostics(self):
        source = inspect.getsource(solid_mpm_fsi_runner._preflow_only_report)

        self.assertIn(
            '"no_slip_projected_residual_after_projection_mps": latest_preflow[',
            source,
        )
        self.assertIn(
            '"fluid_feedback_constraint_active_cell_count": latest_preflow[',
            source,
        )
        self.assertNotIn(
            '"no_slip_projected_residual_after_projection_mps": 0.0',
            source,
        )
        self.assertNotIn('"fluid_marker_velocity_constraints_enabled": False', source)

    def test_marker_force_report_fields_are_face_resolved(self):
        report = SimpleNamespace(
            primary_marker_force_n=(1.0, 2.0, -3.0),
            secondary_marker_force_n=(4.0, 5.0, -6.0),
            total_marker_force_n=(5.0, 7.0, -9.0),
            fluid_reaction_force_n=(-5.0, -7.0, 9.0),
            action_reaction_residual_n=0.0,
            primary_marker_count=3,
            secondary_marker_count=3,
            total_marker_count=6,
            primary_stress_valid_marker_count=3,
            secondary_stress_valid_marker_count=2,
            primary_stress_invalid_marker_count=0,
            secondary_stress_invalid_marker_count=1,
            primary_marker_force_norm_sum_n=3.0,
            secondary_marker_force_norm_sum_n=6.0,
            total_marker_force_norm_sum_n=9.0,
            primary_marker_force_norm_max_n=2.0,
            secondary_marker_force_norm_max_n=4.0,
            total_marker_force_norm_max_n=4.0,
        )

        fields = solid_mpm_fsi_runner._marker_force_report_fields(report)

        self.assertEqual(fields["primary_face_marker_count"], 3)
        self.assertEqual(fields["secondary_face_marker_count"], 3)
        self.assertEqual(fields["primary_face_force_z_N"], -3.0)
        self.assertEqual(fields["secondary_face_force_z_N"], -6.0)
        self.assertEqual(fields["primary_plus_secondary_force_z_N"], -9.0)
        self.assertEqual(fields["force_decomposition_residual_N"], 0.0)
        self.assertEqual(fields["marker_force_z_N"], -9.0)
        self.assertEqual(fields["fluid_reaction_force_z_N"], 9.0)
        self.assertEqual(fields["marker_action_reaction_residual_N"], 0.0)
        self.assertEqual(fields["total_marker_count"], 6)
        self.assertEqual(fields["primary_face_valid_marker_count"], 3)
        self.assertEqual(fields["secondary_face_invalid_marker_count"], 1)

    def test_region_split_preserves_loaded_total_marker_force(self):
        primary = solid_mpm_fsi_runner.PRIMARY_REGION_ID
        secondary = solid_mpm_fsi_runner.SECONDARY_REGION_ID
        marker_forces = [
            (0.0, 0.0, -1.0e-5),
            (0.0, 0.0, -2.0e-5),
            (0.0, 0.0, -3.0e-5),
            (0.0, 0.0, -4.0e-5),
        ]

        def aggregate(region_ids):
            primary_force_z = sum(
                force[2]
                for force, region_id in zip(marker_forces, region_ids, strict=True)
                if region_id == primary
            )
            secondary_force_z = sum(
                force[2]
                for force, region_id in zip(marker_forces, region_ids, strict=True)
                if region_id == secondary
            )
            total_force_z = sum(force[2] for force in marker_forces)
            return {
                "primary_count": sum(region_id == primary for region_id in region_ids),
                "secondary_count": sum(
                    region_id == secondary for region_id in region_ids
                ),
                "total_count": len(region_ids),
                "primary_force_z": primary_force_z,
                "secondary_force_z": secondary_force_z,
                "total_force_z": total_force_z,
            }

        all_primary = aggregate([primary, primary, primary, primary])
        split = aggregate([primary, primary, secondary, secondary])

        self.assertEqual(all_primary["total_count"], split["total_count"])
        self.assertAlmostEqual(
            all_primary["total_force_z"],
            split["total_force_z"],
            places=12,
        )
        self.assertEqual(split["primary_count"], 2)
        self.assertEqual(split["secondary_count"], 2)
        self.assertAlmostEqual(
            split["primary_force_z"] + split["secondary_force_z"],
            split["total_force_z"],
            places=12,
        )

    def test_preflow_only_step_count_zero_is_diagnostic_only(self):
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(step_count=0, preflow_steps=1)
        )
        with self.assertRaisesRegex(ValueError, "preflow-only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(step_count=0, preflow_steps=0)
            )

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
        )
        self.assertIn("config.step_count == 0 and preflow_history", run_source)
        self.assertIn("_preflow_only_report", run_source)
        self.assertLess(
            run_source.index("if config.step_count == 0 and preflow_history"),
            run_source.index("_require_preflow_ready_for_fsi(preflow_report)"),
        )

    def test_diagnostic_flow_controls_are_explicit_and_default_safe(self):
        config = VerticalFlapFsiConfig()

        self.assertTrue(config.apply_marker_feedback_to_fluid)
        self.assertFalse(config.flow_reset_pressure_each_step)
        self.assertFalse(config.flow_reinitialize_inlet_each_step)
        self.assertEqual(config.flow_driver_mode, "projection_only")
        self.assertEqual(config.flow_inlet_source_strength, 1.0)
        self.assertEqual(config.flow_inlet_source_profile, "constant")
        self.assertEqual(config.flow_inlet_source_schedule_scope, "global")
        self.assertEqual(config.traction_marker_layout, "dual_physical_faces")
        self.assertEqual(
            config.traction_pressure_sampling_mode,
            "two_sided_pressure_jump",
        )
        self.assertFalse(config.traction_include_viscous)
        self.assertAlmostEqual(config.traction_marker_face_offset_cells, 0.51)
        self.assertAlmostEqual(config.traction_viscosity_pa_s, 0.0)

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
        )
        self.assertIn("apply_marker_feedback_to_fluid", run_source)
        self.assertIn("flow_reset_pressure_each_step", run_source)
        self.assertIn("flow_reinitialize_inlet_each_step", run_source)
        self.assertIn("_flow_advance_current_step", run_source)

    def test_traction_formulation_controls_are_explicit_and_report_unsupported(self):
        supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(
            VerticalFlapFsiConfig()
        )
        self.assertTrue(supported)
        self.assertEqual(reason, "supported")

        unsupported, unsupported_reason = (
            solid_mpm_fsi_runner.traction_formulation_supported(
                VerticalFlapFsiConfig(
                    traction_marker_layout="dual_physical_faces",
                    traction_pressure_sampling_mode="one_sided_surface_pressure",
                )
            )
        )
        self.assertFalse(unsupported)
        self.assertIn("per_face_mirrored", unsupported_reason)
        with self.assertRaisesRegex(ValueError, "unsupported traction formulation"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=0,
                    preflow_steps=20,
                    traction_marker_layout="dual_physical_faces",
                    traction_pressure_sampling_mode="one_sided_surface_pressure",
                )
            )
        per_face_one_sided = VerticalFlapFsiConfig(
            step_count=0,
            preflow_steps=20,
            traction_marker_layout="dual_physical_faces",
            traction_pressure_sampling_mode="one_sided_surface_pressure",
            traction_pressure_pair_policy="baseline_anchored_cell_pair",
            traction_one_sided_pressure_policy="per_face_mirrored",
            traction_one_sided_primary_fluid_side_normal_sign=1.0,
            traction_one_sided_secondary_fluid_side_normal_sign=1.0,
        )
        supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(
            per_face_one_sided
        )
        self.assertTrue(supported)
        self.assertEqual(reason, "supported")
        solid_mpm_fsi_runner._validate_rectangular_solid_config(per_face_one_sided)
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    traction_marker_layout="dual_physical_faces",
                    traction_pressure_sampling_mode="one_sided_surface_pressure",
                    traction_pressure_pair_policy="baseline_anchored_cell_pair",
                    traction_one_sided_pressure_policy="per_face_mirrored",
                    traction_one_sided_primary_fluid_side_normal_sign=1.0,
                    traction_one_sided_secondary_fluid_side_normal_sign=1.0,
                )
            )
        selected_coupled_smoke = VerticalFlapFsiConfig(
            step_count=5,
            traction_marker_layout="dual_physical_faces",
            traction_pressure_sampling_mode="one_sided_surface_pressure",
            traction_marker_face_offset_cells=0.0,
            traction_pressure_probe_origin_mode="physical_face_offset",
            traction_pressure_probe_origin_offset_cells=0.51,
            traction_pressure_pair_policy="baseline_anchored_cell_pair",
            traction_one_sided_pressure_policy="per_face_mirrored",
            traction_one_sided_primary_fluid_side_normal_sign=1.0,
            traction_one_sided_secondary_fluid_side_normal_sign=1.0,
            traction_pressure_pair_anchor_markers_json=SELECTED_ANCHOR_MARKERS_JSON,
            allow_selected_traction_formulation_coupled_smoke=True,
        )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            selected_coupled_smoke
        )
        with self.assertRaisesRegex(
            ValueError,
            "traction_pressure_pair_anchor_markers_json",
        ):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                replace(
                    selected_coupled_smoke,
                    traction_pressure_pair_anchor_markers_json=None,
                )
            )
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                replace(selected_coupled_smoke, step_count=50)
            )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            replace(
                selected_coupled_smoke,
                step_count=50,
                allow_selected_traction_formulation_coupled_long_validation=True,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "fixed-solid diagnostics only|traction_one_sided_pressure_pair_policy",
        ):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                replace(
                    selected_coupled_smoke,
                    traction_pressure_pair_policy="independent_ladder",
                )
            )
        with self.assertRaisesRegex(ValueError, "primary_fluid_side"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=0,
                    preflow_steps=20,
                    traction_marker_layout="dual_physical_faces",
                    traction_pressure_sampling_mode="one_sided_surface_pressure",
                    traction_pressure_pair_policy="baseline_anchored_cell_pair",
                    traction_one_sided_pressure_policy="per_face_mirrored",
                    traction_one_sided_secondary_fluid_side_normal_sign=1.0,
                )
            )

        single_supported, single_reason = (
            solid_mpm_fsi_runner.traction_formulation_supported(
                VerticalFlapFsiConfig(
                    traction_marker_layout="single_mid_surface",
                    traction_pressure_sampling_mode="two_sided_pressure_jump",
                )
            )
        )
        self.assertTrue(single_supported)
        self.assertEqual(single_reason, "supported")

        single_one_sided_supported, single_one_sided_reason = (
            solid_mpm_fsi_runner.traction_formulation_supported(
                VerticalFlapFsiConfig(
                    traction_marker_layout="single_mid_surface",
                    traction_pressure_sampling_mode="one_sided_surface_pressure",
                )
            )
        )
        self.assertFalse(single_one_sided_supported)
        self.assertIn("ambiguous fluid side", single_one_sided_reason)
        with self.assertRaisesRegex(ValueError, "unsupported traction formulation"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=0,
                    preflow_steps=20,
                    traction_marker_layout="single_mid_surface",
                    traction_pressure_sampling_mode="one_sided_surface_pressure",
                )
            )

        with self.assertRaisesRegex(ValueError, "traction_marker_layout"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    traction_marker_layout="not_a_layout",
                )
            )
        with self.assertRaisesRegex(ValueError, "traction_pressure_sampling_mode"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    traction_pressure_sampling_mode="not_a_sampling_mode",
                )
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    traction_marker_face_offset_cells=-0.01,
                )
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    traction_marker_face_offset_cells=math.nan,
                )
            )
        with self.assertRaisesRegex(ValueError, "diagnostic range"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    traction_marker_face_offset_cells=5.0,
                )
            )
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=1,
                    preflow_steps=0,
                    traction_marker_face_offset_cells=1.0,
                )
            )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=1,
                traction_marker_face_offset_cells=1.0,
            )
        )
        self.assertTrue(
            solid_mpm_fsi_runner._is_default_traction_formulation(
                VerticalFlapFsiConfig()
            )
        )
        explicit_probe_origin = VerticalFlapFsiConfig(
            traction_pressure_probe_origin_mode="physical_face_offset",
            traction_pressure_probe_origin_offset_cells=0.51,
        )
        self.assertFalse(
            solid_mpm_fsi_runner._is_default_traction_formulation(
                explicit_probe_origin
            )
        )
        with self.assertRaisesRegex(ValueError, "fixed-solid diagnostics only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(
                    step_count=1,
                    preflow_steps=0,
                    traction_pressure_probe_origin_mode="physical_face_offset",
                    traction_pressure_probe_origin_offset_cells=0.51,
                )
            )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=1,
                traction_pressure_probe_origin_mode="physical_face_offset",
                traction_pressure_probe_origin_offset_cells=0.51,
            )
        )

        source = inspect.getsource(solid_mpm_fsi_runner._sample_stress_to_marker_forces)
        self.assertIn("one_sided_pressure_region_id", source)
        self.assertIn("one_sided_pressure_primary_region_id", source)
        self.assertIn("_traction_viscosity_pa_s(config)", source)

    def test_pressure_pair_anchor_payload_cells_are_fail_closed(self):
        payload = {
            "marker_count": 1,
            "markers": [
                {
                    "marker_index": 0,
                    "pressure_pair_anchor_active": True,
                    "pressure_pair_anchor_inside_cell": [1, 2, 3],
                    "pressure_pair_anchor_outside_cell": [1, 2, 4],
                }
            ],
        }

        inside, outside = (
            solid_mpm_fsi_runner._pressure_pair_anchor_cells_from_marker_payload(
                payload
            )
        )

        self.assertEqual(inside, [(1, 2, 3)])
        self.assertEqual(outside, [(1, 2, 4)])

        inactive = {
            "markers": [
                {
                    "pressure_pair_anchor_active": False,
                    "pressure_pair_anchor_inside_cell": [1, 2, 3],
                    "pressure_pair_anchor_outside_cell": [1, 2, 4],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "inactive marker"):
            solid_mpm_fsi_runner._pressure_pair_anchor_cells_from_marker_payload(
                inactive
            )

        missing_cell = {
            "markers": [
                {
                    "pressure_pair_anchor_active": True,
                    "pressure_pair_anchor_inside_cell": [1, 2],
                    "pressure_pair_anchor_outside_cell": [1, 2, 4],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "must have 3 cells"):
            solid_mpm_fsi_runner._pressure_pair_anchor_cells_from_marker_payload(
                missing_cell
            )

        mismatched_count = {"marker_count": 2, "markers": payload["markers"]}
        with self.assertRaisesRegex(ValueError, "marker_count"):
            solid_mpm_fsi_runner._pressure_pair_anchor_cells_from_marker_payload(
                mismatched_count
            )

    def test_sustained_flow_driver_modes_are_explicit_and_default_safe(self):
        self.assertIn(
            "sustained_volume_source_inlet",
            solid_mpm_fsi_runner.SUPPORTED_FORMAL_FLOW_DRIVER_MODES,
        )
        self.assertIn(
            "sustained_boundary_predictor",
            solid_mpm_fsi_runner.SUPPORTED_FORMAL_FLOW_DRIVER_MODES,
        )
        self.assertIn(
            "reinitialize_inlet_each_step_diagnostic",
            solid_mpm_fsi_runner.SUPPORTED_FORMAL_FLOW_DRIVER_MODES,
        )
        self.assertEqual(
            solid_mpm_fsi_runner._effective_flow_driver_mode(VerticalFlapFsiConfig()),
            "projection_only",
        )
        self.assertTrue(VerticalFlapFsiConfig().preserve_marker_velocity_constraints)
        self.assertEqual(VerticalFlapFsiConfig().marker_velocity_constraint_blend, 1.0)
        self.assertEqual(
            VerticalFlapFsiConfig().marker_velocity_constraint_solid_mobility_ratio,
            0.0,
        )
        self.assertEqual(
            solid_mpm_fsi_runner._effective_flow_driver_mode(
                VerticalFlapFsiConfig(flow_reinitialize_inlet_each_step=True)
            ),
            "reinitialize_inlet_each_step_diagnostic",
        )
        split_phase = VerticalFlapFsiConfig(
            flow_driver_mode="sustained_boundary_predictor",
            preflow_flow_driver_mode="sustained_inlet_predictor",
        )
        self.assertEqual(
            solid_mpm_fsi_runner._effective_flow_driver_mode(
                split_phase,
                flow_phase="preflow",
            ),
            "sustained_inlet_predictor",
        )
        self.assertEqual(
            solid_mpm_fsi_runner._effective_flow_driver_mode(
                split_phase,
                flow_phase="fsi",
            ),
            "sustained_boundary_predictor",
        )

        advance_source = inspect.getsource(
            solid_mpm_fsi_runner._flow_advance_current_step
        )
        self.assertIn("add_zmax_velocity_inlet_volume_source", advance_source)
        self.assertIn("FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR", advance_source)
        self.assertIn("FLOW_DRIVER_SUSTAINED_SOURCE", advance_source)
        self.assertIn("FLOW_DRIVER_SUSTAINED_PREDICTOR", advance_source)
        self.assertIn("_flow_inlet_source_factor", advance_source)
        self.assertIn(
            "preserve_marker_velocity_constraints",
            inspect.getsource(solid_mpm_fsi_runner._project_current_flow),
        )
        self.assertIn(
            "flow_predictor_applied",
            inspect.getsource(solid_mpm_fsi_runner._flow_driver_report),
        )

    def test_fixed_solid_preflow_uses_marker_feedback_before_projection(self):
        preflow_source = inspect.getsource(solid_mpm_fsi_runner._run_fixed_solid_preflow)

        feedback_call = preflow_source.index("_apply_marker_feedback_to_fluid")
        flow_advance_call = preflow_source.index("_flow_advance_current_step")
        projected_residual_call = preflow_source.index(
            "_measure_projected_no_slip_residual"
        )
        row_call = preflow_source.index("row = {")
        self.assertLess(feedback_call, flow_advance_call)
        self.assertLess(flow_advance_call, projected_residual_call)
        self.assertLess(projected_residual_call, row_call)
        self.assertIn("apply_marker_feedback_to_fluid", preflow_source)
        self.assertIn("fluid_marker_velocity_constraints_enabled", preflow_source)
        self.assertIn("fluid_feedback_constraint_active_cell_count", preflow_source)
        self.assertIn(
            '"no_slip_projected_residual_after_projection_mps"',
            preflow_source,
        )
        self.assertIn("hibm_sharp_marker_boundary_search_reused", preflow_source)
        self.assertIn("hibm_sharp_marker_boundary_near_node_count", preflow_source)
        self.assertIn("hibm_sharp_marker_boundary_external_node_count", preflow_source)
        self.assertIn("hibm_sharp_marker_boundary_internal_node_count", preflow_source)
        self.assertIn(
            "hibm_sharp_marker_boundary_internal_obstacle_cell_count",
            preflow_source,
        )
        self.assertIn(
            "hibm_sharp_marker_boundary_pressure_gradient_updated",
            preflow_source,
        )
        self.assertIn(
            "hibm_pressure_neumann_skipped_velocity_dirichlet_count",
            preflow_source,
        )
        self.assertIn(
            "hibm_pressure_neumann_invalid_reconstruction_count",
            preflow_source,
        )

    def test_project_current_flow_applies_configured_symmetry_domain_wall(self):
        class Field:
            def __init__(self, value):
                self.value = value

            def to_numpy(self):
                return np.array(self.value, copy=True)

        class FakeFluid:
            def __init__(self):
                self.obstacle = Field(np.zeros((2, 2, 2), dtype=np.int32))
                self.velocity = Field(np.zeros((2, 2, 2, 3), dtype=np.float32))
                self.pressure = Field(np.zeros((2, 2, 2), dtype=np.float32))
                self.fsi_pressure = Field(np.zeros((2, 2, 2), dtype=np.float32))
                self.symmetry_calls: list[tuple[bool, ...]] = []

            def project(self, **kwargs):
                return {"l2": 0.0, "project_kwargs": kwargs}

            def apply_symmetry_domain_walls(self, symmetry_domain_walls):
                self.symmetry_calls.append(tuple(symmetry_domain_walls))

            def pressure_outlet_fv_flux_report(self, *, dt_s):
                return {"pressure_outlet_dt_s": float(dt_s)}

            def snapshot_pressure(self, *, preserve_if_current_is_zero):
                return True

        config = VerticalFlapFsiConfig(
            grid_nodes=(2, 2, 2),
            flow_symmetry_domain_walls=("ymax",),
            flow_projection_velocity_inlet_zmax=None,
        )
        fake_fluid = FakeFluid()

        report = solid_mpm_fsi_runner._project_current_flow(
            fake_fluid,
            config,
            reset_pressure=True,
            velocity_dirichlet_soft_rows_already_applied=True,
        )

        expected_flags = (False, False, False, True, False, False)
        self.assertEqual(fake_fluid.symmetry_calls, [expected_flags])
        self.assertEqual(
            report["projection_report"]["flow_symmetry_domain_walls"],
            [False, False, False, True, False, False],
        )
        self.assertIsNone(
            report["projection_report"]["project_kwargs"]["velocity_inlet_zmax"]
        )
        self.assertTrue(
            report["projection_report"]["project_kwargs"][
                "velocity_dirichlet_soft_rows_already_applied"
            ]
        )
        self.assertTrue(
            report["projection_report"][
                "velocity_dirichlet_soft_rows_already_applied"
            ]
        )

    def test_hibm_sharp_marker_boundary_mode_is_generic_and_not_reserved(self):
        config = VerticalFlapFsiConfig(
            flow_solid_boundary_mode="hibm_sharp_marker_rows",
        )

        solid_mpm_fsi_runner._validate_rectangular_solid_config(config)

        self.assertTrue(solid_mpm_fsi_runner._use_hibm_sharp_marker_boundary(config))
        self.assertNotIn(
            "cantilever",
            inspect.getsource(solid_mpm_fsi_runner).lower(),
        )
        self.assertNotIn(
            "modal",
            inspect.getsource(solid_mpm_fsi_runner).lower(),
        )

    def test_hibm_sharp_marker_boundary_cache_reclassifies_current_markers(self):
        source = inspect.getsource(
            solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid
        )
        advance_source = inspect.getsource(
            solid_mpm_fsi_runner._flow_advance_current_step
        )

        self.assertIn("reuse_topology_from_previous_assembly", source)
        self.assertIn('search_report = cache_entry["search_report"]', source)
        self.assertIn('cache_entry["search_report"] = search_report', source)
        self.assertIn("search_report = ib_search.search_and_classify_grid_fields", source)
        self.assertIn(
            "reuse_topology_from_previous_assembly=True",
            advance_source,
        )
        self.assertEqual(
            advance_source.count("reuse_topology_from_previous_assembly=True"),
            3,
        )

    def test_flow_reporting_uses_fsi_feedback_pressure_not_projection_work_field(self):
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        work_pressure = np.zeros((4, 4, 4), dtype=np.float32)
        feedback_pressure = np.zeros((4, 4, 4), dtype=np.float32)
        feedback_pressure[1, 1, 1] = -25.0
        feedback_pressure[2, 1, 1] = 75.0
        fluid.pressure.from_numpy(work_pressure)
        fluid.fsi_pressure.from_numpy(feedback_pressure)

        report = solid_mpm_fsi_runner._flow_state_report(fluid, {})
        snapshot = solid_mpm_fsi_runner._flow_field_snapshot(fluid)

        self.assertEqual(report["pressure_min_pa"], -25.0)
        self.assertEqual(report["pressure_max_pa"], 75.0)
        self.assertIn("fsi_pressure", report["pressure_sign_convention"])
        np.testing.assert_allclose(snapshot["pressure"], feedback_pressure)
        self.assertIn("velocity_dirichlet_boundary_active", snapshot)
        self.assertIn("velocity_dirichlet_boundary_projection_weight", snapshot)
        self.assertIn(
            "velocity_dirichlet_boundary_hard_fixed_component_mask",
            snapshot,
        )
        self.assertIn(
            "velocity_dirichlet_boundary_external_exact_component_mask",
            snapshot,
        )
        self.assertIn("velocity_dirichlet_boundary_owned_row", snapshot)
        self.assertIn("velocity_dirichlet_boundary_marker_region_id", snapshot)

    def test_flow_reporting_default_path_uses_solver_scalar_report(self):
        class _NoDownloadField:
            def to_numpy(self):
                raise AssertionError("default flow report should stay scalar/device-side")

        class _DeviceReportFluid:
            def __init__(self):
                self.fsi_pressure = _NoDownloadField()
                self.pressure = _NoDownloadField()
                self.obstacle = _NoDownloadField()
                self.velocity = _NoDownloadField()

            def flow_state_report(self, *, pressure_field, include_percentiles):
                self._pressure_field = pressure_field
                self._include_percentiles = include_percentiles
                return {
                    "obstacle_cell_count": 3,
                    "fluid_cell_count": 5,
                    "local_velocity_peak_mps": 2.5,
                    "fluid_speed_p99_mps": "",
                    "fluid_speed_p999_mps": "",
                    "pressure_min_pa": -4.0,
                    "pressure_max_pa": 9.0,
                }

        fluid = _DeviceReportFluid()

        report = solid_mpm_fsi_runner._flow_state_report(fluid, {})

        self.assertIs(fluid._pressure_field, fluid.fsi_pressure)
        self.assertIs(fluid._include_percentiles, False)
        self.assertEqual(report["obstacle_cell_count"], 3)
        self.assertEqual(report["fluid_cell_count"], 5)
        self.assertEqual(report["local_velocity_peak_mps"], 2.5)
        self.assertEqual(report["fluid_speed_p99_mps"], "")
        self.assertEqual(report["pressure_min_pa"], -4.0)
        self.assertEqual(report["pressure_max_pa"], 9.0)

    def test_zmax_inlet_boundary_report_default_path_uses_solver_scalar_report(self):
        class _NoDownloadField:
            def to_numpy(self):
                raise AssertionError("zmax inlet report should stay scalar/device-side")

        class _DeviceReportFluid:
            def __init__(self):
                self.velocity_dirichlet_boundary_active = _NoDownloadField()
                self.obstacle = _NoDownloadField()
                self.called = False

            def zmax_inlet_boundary_report(self):
                self.called = True
                return {
                    "flow_inlet_boundary_active_cell_count": 12,
                    "flow_inlet_boundary_obstacle_cell_count": 2,
                }

        fluid = _DeviceReportFluid()

        report = solid_mpm_fsi_runner._zmax_inlet_boundary_report(fluid)

        self.assertTrue(fluid.called)
        self.assertEqual(report["flow_inlet_boundary_active_cell_count"], 12)
        self.assertEqual(report["flow_inlet_boundary_obstacle_cell_count"], 2)

    def test_zmax_inlet_boundary_report_counts_only_top_face(self):
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity_dirichlet_boundary_active.fill(0)
        fluid.external_velocity_boundary_z_face_active_component_mask.fill(0)
        fluid.obstacle.fill(0)
        fluid.velocity_dirichlet_boundary_active[0, 0, 3] = 1
        fluid.velocity_dirichlet_boundary_active[1, 1, 3] = 1
        fluid.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        fluid.external_velocity_boundary_z_face_active_component_mask[
            1, 0, 0
        ] = 0b111
        fluid.external_velocity_boundary_z_face_active_component_mask[
            1, 1, 1
        ] = 0b111
        fluid.obstacle[1, 1, 3] = 1

        report = solid_mpm_fsi_runner._zmax_inlet_boundary_report(fluid)

        self.assertEqual(report["flow_inlet_boundary_active_cell_count"], 2)
        self.assertEqual(report["flow_inlet_boundary_obstacle_cell_count"], 1)

    def test_refresh_zmax_inlet_boundary_default_path_uses_solver_device_api(self):
        class _NoDownloadField:
            def to_numpy(self):
                raise AssertionError("zmax inlet refresh should stay device-side")

        class _DeviceRefreshFluid:
            def __init__(self):
                self.velocity_dirichlet_boundary_authority = "legacy"
                self.velocity_dirichlet_boundary_active = _NoDownloadField()
                self.velocity_dirichlet_boundary_value_mps = _NoDownloadField()
                self.velocity_dirichlet_boundary_projection_weight = _NoDownloadField()
                self.obstacle = _NoDownloadField()
                self.calls = []

            def refresh_zmax_inlet_boundary(
                self,
                *,
                inlet_velocity_mps,
                streamwise_axis_index,
            ):
                self.calls.append(
                    (float(inlet_velocity_mps), int(streamwise_axis_index))
                )
                return {
                    "flow_inlet_boundary_active_cell_count": 16,
                    "flow_inlet_boundary_obstacle_cell_count": 0,
                }

        config = VerticalFlapFsiConfig()
        fluid = _DeviceRefreshFluid()

        report = solid_mpm_fsi_runner._refresh_zmax_inlet_boundary(fluid, config)

        self.assertEqual(
            fluid.calls,
            [
                (
                    float(config.inlet_velocity_mps),
                    solid_mpm_fsi_runner.STREAMWISE_AXIS_INDEX,
                )
            ],
        )
        self.assertEqual(report["flow_inlet_boundary_active_cell_count"], 16)
        self.assertEqual(report["flow_inlet_boundary_obstacle_cell_count"], 0)

    def test_zmax_inlet_boundary_device_refresh_keeps_legacy_fallback_conditions(
        self,
    ):
        self.assertFalse(
            solid_mpm_fsi_runner._zmax_inlet_boundary_device_refresh_compatible(
                replace(VerticalFlapFsiConfig(), flow_ymin_no_slip_rows=1)
            )
        )
        self.assertFalse(
            solid_mpm_fsi_runner._zmax_inlet_boundary_device_refresh_compatible(
                replace(
                    VerticalFlapFsiConfig(),
                    flow_solid_boundary_mode=(
                        solid_mpm_fsi_runner.FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS
                    ),
                    flow_obstacle_no_slip_layers=1,
                )
            )
        )
        self.assertTrue(
            solid_mpm_fsi_runner._zmax_inlet_boundary_device_refresh_compatible(
                VerticalFlapFsiConfig()
            )
        )

    def test_solver_refresh_zmax_inlet_boundary_writes_only_top_face(self):
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, 3] = 1
        active = np.zeros((4, 4, 4), dtype=np.int32)
        active[0, 0, 2] = 1
        values = np.zeros((4, 4, 4, 3), dtype=np.float32)
        values[0, 0, 2] = (0.25, 0.5, 0.75)
        weights = np.zeros((4, 4, 4), dtype=np.float32)
        weights[0, 0, 2] = 0.5
        enforcement_weights = np.full((4, 4, 4), 0.375, dtype=np.float32)
        external_exact_masks = np.zeros((4, 4, 4), dtype=np.int32)
        external_exact_masks[0, 0, 3] = 0b010
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        fluid.velocity_dirichlet_boundary_enforcement_weight.from_numpy(
            enforcement_weights
        )
        fluid.velocity_dirichlet_boundary_external_exact_component_mask.from_numpy(
            external_exact_masks
        )

        report = fluid.refresh_zmax_inlet_boundary(
            inlet_velocity_mps=4.5,
            streamwise_axis_index=2,
        )

        refreshed_active = fluid.velocity_dirichlet_boundary_active.to_numpy()
        refreshed_values = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
        refreshed_weights = (
            fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        )
        refreshed_enforcement_weights = (
            fluid.velocity_dirichlet_boundary_enforcement_weight.to_numpy()
        )
        refreshed_external_exact_masks = (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        refreshed_external_face_masks = (
            fluid.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        )
        refreshed_external_face_values = (
            fluid.external_velocity_boundary_z_face_value_mps.to_numpy()
        )
        self.assertEqual(report["flow_inlet_boundary_active_cell_count"], 15)
        self.assertEqual(report["flow_inlet_boundary_obstacle_cell_count"], 1)
        self.assertEqual(int(refreshed_active[1, 1, 3]), 0)
        self.assertEqual(int(refreshed_active[0, 0, 3]), 0)
        self.assertEqual(int(refreshed_active[0, 0, 2]), 1)
        np.testing.assert_allclose(refreshed_values[0, 0, 3], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(refreshed_values[1, 1, 3], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(refreshed_values[0, 0, 2], (0.25, 0.5, 0.75))
        self.assertEqual(float(refreshed_weights[0, 0, 3]), 0.0)
        self.assertEqual(float(refreshed_weights[1, 1, 3]), 0.0)
        self.assertEqual(float(refreshed_weights[0, 0, 2]), 0.5)
        self.assertEqual(float(refreshed_enforcement_weights[0, 0, 3]), 0.375)
        self.assertEqual(float(refreshed_enforcement_weights[1, 1, 3]), 0.375)
        self.assertEqual(float(refreshed_enforcement_weights[0, 0, 2]), 0.375)
        self.assertEqual(int(refreshed_external_exact_masks[0, 0, 3]), 0b010)
        self.assertEqual(int(refreshed_external_exact_masks[1, 1, 3]), 0)
        self.assertEqual(int(refreshed_external_exact_masks[0, 0, 2]), 0)
        self.assertTrue(np.all(refreshed_external_face_masks[1] == 0b111))
        np.testing.assert_allclose(
            refreshed_external_face_values[1],
            np.broadcast_to((0.0, 0.0, -4.5), (4, 4, 3)),
        )

    def test_initial_inlet_flow_registers_external_pressure_face_provenance(self):
        config = replace(VerticalFlapFsiConfig(), grid_nodes=(4, 4, 8))
        fluid = solid_mpm_fsi_runner._build_fluid(
            config,
            TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity_dirichlet_boundary_enforcement_weight.fill(0.375)

        solid_mpm_fsi_runner._initialize_inlet_flow(fluid, config)

        hard_masks = (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        )
        external_exact_masks = (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        enforcement_weights = (
            fluid.velocity_dirichlet_boundary_enforcement_weight.to_numpy()
        )
        projection_weights = (
            fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        )
        directed_masks = (
            fluid.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        )
        directed_values = (
            fluid.external_velocity_boundary_z_face_value_mps.to_numpy()
        )
        np.testing.assert_array_equal(
            hard_masks[:, :, -1],
            np.zeros(config.grid_nodes[:2], dtype=np.int32),
            err_msg=(
                "the physical zmax face must not alias the final cell's "
                "backward internal MAC row"
            ),
        )
        np.testing.assert_array_equal(
            external_exact_masks[:, :, -1],
            np.zeros(config.grid_nodes[:2], dtype=np.int32),
        )
        self.assertEqual(
            int(np.count_nonzero(external_exact_masks)),
            0,
        )
        np.testing.assert_array_equal(
            directed_masks[1],
            np.full(config.grid_nodes[:2], 0b111, dtype=np.int32),
        )
        expected_directed_values = np.zeros((*config.grid_nodes[:2], 3))
        expected_directed_values[:, :, 2] = -float(config.inlet_velocity_mps)
        np.testing.assert_allclose(directed_values[1], expected_directed_values)
        np.testing.assert_array_equal(enforcement_weights, projection_weights)
        self.assertEqual(
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            ),
            0,
            msg=(
                "a directed physical inlet face may not split the final "
                "internal pressure-cell plane into singleton components"
            ),
        )
        self.assertEqual(
            fluid._prepare_pressure_outlet_nullspace_component_graph(),
            (0, 0),
            msg=(
                "the empty inlet-to-outlet channel must have no unanchored "
                "physical or exact-operator pressure components"
            ),
        )
        self.assertEqual(
            int(fluid.pressure_outlet_operator_raw_component_count[None]),
            1,
            msg="the exact operator must see one outlet-anchored channel root",
        )
        self.assertEqual(int(fluid._pressure_outlet_operator_component_count), 0)

    def test_initial_inlet_flow_preserves_intersecting_external_face_normals(self):
        config = replace(
            VerticalFlapFsiConfig(),
            grid_nodes=(4, 4, 8),
            flow_ymin_no_slip_rows=1,
        )
        fluid = solid_mpm_fsi_runner._build_fluid(
            config,
            TaichiRuntimeConfig(arch="cuda"),
        )

        solid_mpm_fsi_runner._initialize_inlet_flow(fluid, config)

        hard_masks = (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        )
        external_exact_masks = (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        obstacle = fluid.obstacle.to_numpy()
        corner_fluid_mask = obstacle[:, 0, -1] == 0
        np.testing.assert_array_equal(
            hard_masks[:, 0, -1],
            np.where(corner_fluid_mask, 0b111, 0).astype(np.int32),
        )
        np.testing.assert_array_equal(
            external_exact_masks[:, 0, -1],
            np.where(corner_fluid_mask, 0b010, 0).astype(np.int32),
            err_msg=(
                "the compact corner row owns only the physical ymin normal; "
                "zmax remains a separate directed external face"
            ),
        )
        directed_masks = (
            fluid.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        )
        np.testing.assert_array_equal(
            directed_masks[1],
            np.full(config.grid_nodes[:2], 0b111, dtype=np.int32),
        )

        solid_mpm_fsi_runner._refresh_zmax_inlet_boundary(fluid, config)
        refreshed_hard_masks = (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        )
        refreshed_external_exact_masks = (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        np.testing.assert_array_equal(
            refreshed_hard_masks[:, 0, -1],
            np.where(corner_fluid_mask, 0b111, 0).astype(np.int32),
        )
        np.testing.assert_array_equal(
            refreshed_external_exact_masks[:, 0, -1],
            np.where(corner_fluid_mask, 0b010, 0).astype(np.int32),
        )

    def test_flow_reporting_pressure_extrema_do_not_clamp_to_zero(self):
        positive = _fake_flow_report_fluid(
            np.array(
                [
                    [[12.0, 15.0], [21.0, 18.0]],
                    [[33.0, 24.0], [19.0, 27.0]],
                ],
                dtype=np.float32,
            )
        )
        positive_report = solid_mpm_fsi_runner._flow_state_report(positive, {})

        self.assertEqual(positive_report["pressure_min_pa"], 12.0)
        self.assertEqual(positive_report["pressure_max_pa"], 33.0)

        negative = _fake_flow_report_fluid(
            np.array(
                [
                    [[-12.0, -15.0], [-21.0, -18.0]],
                    [[-33.0, -24.0], [-19.0, -27.0]],
                ],
                dtype=np.float32,
            )
        )
        negative_report = solid_mpm_fsi_runner._flow_state_report(negative, {})

        self.assertEqual(negative_report["pressure_min_pa"], -33.0)
        self.assertEqual(negative_report["pressure_max_pa"], -12.0)

    def test_flow_projection_report_fields_flatten_solver_diagnostics(self):
        fields = solid_mpm_fsi_runner._flow_projection_report_fields(
            {
                "projection_report": {
                    "pressure_solver": "fv_cg",
                    "pressure_solve_failed": True,
                    "pressure_solve_failure_action": (
                        "reported_kept_nonconverged_pressure_correction"
                    ),
                    "cg_iterations_max": 1080,
                    "cg_converged_all": False,
                    "cg_breakdown": "residual did not converge",
                    "fsi_pressure_snapshot_updated": False,
                }
            }
        )

        self.assertEqual(fields["flow_projection_pressure_solver"], "fv_cg")
        self.assertTrue(fields["flow_projection_pressure_solve_failed"])
        self.assertEqual(
            fields["flow_projection_pressure_solve_failure_action"],
            "reported_kept_nonconverged_pressure_correction",
        )
        self.assertEqual(fields["flow_projection_cg_iterations_max"], 1080)
        self.assertFalse(fields["flow_projection_cg_converged_all"])
        self.assertEqual(
            fields["flow_projection_cg_breakdown"],
            "residual did not converge",
        )
        self.assertFalse(fields["flow_projection_fsi_pressure_snapshot_updated"])

    def test_obstacle_no_slip_weight_controls_generic_boundary_row_strength(self):
        config = VerticalFlapFsiConfig(
            flow_obstacle_no_slip_layers=1,
            flow_obstacle_no_slip_weight=0.5,
        )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(config)

        obstacle = np.zeros((3, 3, 3), dtype=np.int32)
        obstacle[1, 1, 1] = 1
        active = np.zeros_like(obstacle)
        values = np.ones(obstacle.shape + (3,), dtype=np.float32)
        weights = np.zeros(obstacle.shape, dtype=np.float32)

        count = solid_mpm_fsi_runner._apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )

        self.assertEqual(count, 6)
        self.assertTrue(np.all(active[weights > 0.0] == 1))
        np.testing.assert_allclose(weights[weights > 0.0], 0.5)
        np.testing.assert_allclose(values[weights > 0.0], 0.0)
        with self.assertRaisesRegex(ValueError, "flow_obstacle_no_slip_weight"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(flow_obstacle_no_slip_weight=1.5)
            )
        with self.assertRaisesRegex(ValueError, "flow_obstacle_cap_no_slip_weight"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(flow_obstacle_cap_no_slip_weight=-0.1)
            )

    def test_obstacle_cap_no_slip_weight_keeps_coarse_tip_cells_open(self):
        config = VerticalFlapFsiConfig(
            flow_obstacle_no_slip_layers=1,
            flow_obstacle_no_slip_weight=1.0,
            flow_obstacle_cap_no_slip_weight=0.0,
        )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(config)

        obstacle = np.zeros((3, 4, 5), dtype=np.int32)
        obstacle[1, 0:2, 2] = 1
        active = np.zeros_like(obstacle)
        values = np.ones(obstacle.shape + (3,), dtype=np.float32)
        weights = np.zeros(obstacle.shape, dtype=np.float32)

        solid_mpm_fsi_runner._apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )

        self.assertEqual(active[1, 2, 2], 0)
        self.assertEqual(weights[1, 2, 2], 0.0)
        self.assertEqual(active[1, 1, 1], 1)
        self.assertEqual(weights[1, 1, 1], 1.0)
        self.assertEqual(active[1, 1, 3], 1)
        self.assertEqual(weights[1, 1, 3], 1.0)

    def test_obstacle_wake_no_slip_layers_extend_only_downstream(self):
        config = VerticalFlapFsiConfig(
            flow_obstacle_no_slip_layers=1,
            flow_obstacle_no_slip_weight=1.0,
            flow_obstacle_wake_no_slip_layers=3,
            flow_obstacle_wake_no_slip_weight=0.5,
        )
        solid_mpm_fsi_runner._validate_rectangular_solid_config(config)

        obstacle = np.zeros((3, 3, 7), dtype=np.int32)
        obstacle[1, 1, 3] = 1
        active = np.zeros_like(obstacle)
        values = np.ones(obstacle.shape + (3,), dtype=np.float32)
        weights = np.zeros(obstacle.shape, dtype=np.float32)

        count = solid_mpm_fsi_runner._apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )

        self.assertEqual(count, 8)
        self.assertEqual(active[1, 1, 4], 1)
        self.assertEqual(weights[1, 1, 4], 1.0)
        self.assertEqual(active[1, 1, 2], 1)
        self.assertEqual(active[1, 1, 1], 1)
        self.assertEqual(weights[1, 1, 2], 1.0)
        self.assertEqual(weights[1, 1, 1], 0.5)
        self.assertEqual(active[1, 1, 0], 1)
        self.assertEqual(weights[1, 1, 0], 0.5)
        self.assertEqual(active[0, 1, 2], 0)
        np.testing.assert_allclose(values[active.astype(bool)], 0.0)

    def test_source_strength_factor_supports_constant_and_ramp_profiles(self):
        constant = VerticalFlapFsiConfig(flow_inlet_source_strength=0.4)
        ramp = VerticalFlapFsiConfig(
            flow_inlet_source_strength=0.6,
            flow_inlet_source_profile="linear_ramp",
            flow_inlet_source_ramp_steps=3,
        )

        self.assertAlmostEqual(
            solid_mpm_fsi_runner._flow_inlet_source_factor(constant, 0),
            0.4,
        )
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._flow_inlet_source_factor(ramp, 0),
            0.2,
        )
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._flow_inlet_source_factor(ramp, 2),
            0.6,
        )
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._flow_inlet_source_factor(ramp, 10),
            0.6,
        )

    def test_boundary_predictor_velocity_inlet_does_not_add_projection_source(self):
        class Field:
            def __init__(self, value):
                self.value = np.array(value)

            def to_numpy(self):
                return self.value.copy()

            def from_numpy(self, value):
                self.value = np.array(value)

        class FakeFluid:
            def __init__(self):
                self.velocity_dirichlet_boundary_authority = "legacy"
                shape = (1, 2, 3)
                self.velocity_dirichlet_boundary_authority = "legacy"
                self.obstacle = Field(np.zeros(shape, dtype=np.int32))
                self.velocity_dirichlet_boundary_active = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_value_mps = Field(
                    np.zeros(shape + (3,), dtype=np.float32)
                )
                self.velocity_dirichlet_boundary_projection_weight = Field(
                    np.zeros(shape, dtype=np.float32)
                )
                self.velocity_dirichlet_boundary_enforcement_weight = Field(
                    np.zeros(shape, dtype=np.float32)
                )
                self.velocity_dirichlet_boundary_marker_region_id = Field(
                    np.full(shape, -1, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_hard_fixed_component_mask = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_external_exact_component_mask = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_owned_row = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.source_calls: list[float] = []
                self.predict_viscosities: list[float | None] = []
                self.predict_walls: list[object] = []

            def clear_volume_source(self):
                self.source_calls.clear()

            def add_zmax_velocity_inlet_volume_source(self, *, normal_velocity_mps):
                self.source_calls.append(float(normal_velocity_mps))

            def apply_velocity_dirichlet_boundary_rows(self, *, read_report=True):
                return None

            def predict(
                self,
                dt_s=None,
                *,
                advection_scheme="euler",
                kinematic_viscosity_m2_s=None,
                no_slip_domain_walls=None,
            ):
                self.predict_viscosities.append(kinematic_viscosity_m2_s)
                self.predict_walls.append(no_slip_domain_walls)
                return None

        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 2, 3),
            inlet_velocity_mps=10.0,
            flow_driver_mode="sustained_boundary_predictor",
            flow_solid_boundary_mode="cell_obstacle_layers",
            flow_inlet_source_strength=0.75,
            flow_predictor_substeps=1,
            flow_predictor_no_slip_domain_walls=(),
            flow_predictor_kinematic_viscosity_multiplier=2.0,
        )
        fake_fluid = FakeFluid()
        with patch.object(
            solid_mpm_fsi_runner,
            "_project_current_flow",
            return_value={"projection_report": {}},
        ):
            report = solid_mpm_fsi_runner._flow_advance_current_step(
                fake_fluid,
                config,
                flow_phase="fsi",
                step_index_local=0,
                step_index_global=0,
                preflow_history=[],
                reset_pressure=True,
            )

        self.assertFalse(report["flow_volume_source_applied"])
        self.assertEqual(fake_fluid.source_calls, [])
        self.assertEqual(
            fake_fluid.predict_viscosities,
            [2.0 * config.air_viscosity_pa_s / config.air_density_kgm3],
        )
        self.assertEqual(
            fake_fluid.predict_walls,
            [(False, False, False, False, False, False)],
        )
        self.assertAlmostEqual(report["flow_inlet_source_factor"], 0.0)
        self.assertAlmostEqual(
            report["flow_predictor_kinematic_viscosity_m2_s"],
            2.0 * config.air_viscosity_pa_s / config.air_density_kgm3,
        )
        self.assertAlmostEqual(
            report["flow_inlet_source_normal_velocity_mps"],
            0.0,
        )

    def test_source_predictor_velocity_inlet_adds_projection_source(self):
        class Field:
            def __init__(self, value):
                self.value = np.array(value)

            def to_numpy(self):
                return self.value.copy()

            def from_numpy(self, value):
                self.value = np.array(value)

        class FakeFluid:
            def __init__(self):
                self.velocity_dirichlet_boundary_authority = "legacy"
                shape = (1, 2, 3)
                self.obstacle = Field(np.zeros(shape, dtype=np.int32))
                self.velocity_dirichlet_boundary_active = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_value_mps = Field(
                    np.zeros(shape + (3,), dtype=np.float32)
                )
                self.velocity_dirichlet_boundary_projection_weight = Field(
                    np.zeros(shape, dtype=np.float32)
                )
                self.velocity_dirichlet_boundary_enforcement_weight = Field(
                    np.zeros(shape, dtype=np.float32)
                )
                self.velocity_dirichlet_boundary_marker_region_id = Field(
                    np.full(shape, -1, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_hard_fixed_component_mask = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_external_exact_component_mask = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.velocity_dirichlet_boundary_owned_row = Field(
                    np.zeros(shape, dtype=np.int32)
                )
                self.source_calls: list[float] = []
                self.predict_viscosities: list[float | None] = []
                self.predict_walls: list[object] = []

            def clear_volume_source(self):
                self.source_calls.clear()

            def add_zmax_velocity_inlet_volume_source(self, *, normal_velocity_mps):
                self.source_calls.append(float(normal_velocity_mps))

            def apply_velocity_dirichlet_boundary_rows(self, *, read_report=True):
                return None

            def predict(
                self,
                dt_s=None,
                *,
                advection_scheme="euler",
                kinematic_viscosity_m2_s=None,
                no_slip_domain_walls=None,
            ):
                self.predict_viscosities.append(kinematic_viscosity_m2_s)
                self.predict_walls.append(no_slip_domain_walls)
                return None

        config = VerticalFlapFsiConfig(
            grid_nodes=(1, 2, 3),
            inlet_velocity_mps=10.0,
            flow_driver_mode="sustained_inlet_predictor",
            flow_solid_boundary_mode="cell_obstacle_layers",
            flow_inlet_source_strength=0.75,
            flow_predictor_substeps=1,
            flow_predictor_no_slip_domain_walls=(),
            flow_predictor_kinematic_viscosity_multiplier=3.0,
        )
        fake_fluid = FakeFluid()
        with patch.object(
            solid_mpm_fsi_runner,
            "_project_current_flow",
            return_value={"projection_report": {}},
        ):
            report = solid_mpm_fsi_runner._flow_advance_current_step(
                fake_fluid,
                config,
                flow_phase="fsi",
                step_index_local=0,
                step_index_global=0,
                preflow_history=[],
                reset_pressure=True,
            )

        self.assertTrue(report["flow_volume_source_applied"])
        self.assertEqual(fake_fluid.source_calls, [-7.5])
        self.assertEqual(
            fake_fluid.predict_viscosities,
            [3.0 * config.air_viscosity_pa_s / config.air_density_kgm3],
        )
        self.assertEqual(
            fake_fluid.predict_walls,
            [(False, False, False, False, False, False)],
        )
        self.assertAlmostEqual(report["flow_inlet_source_factor"], 0.75)
        self.assertAlmostEqual(
            report["flow_predictor_kinematic_viscosity_m2_s"],
            3.0 * config.air_viscosity_pa_s / config.air_density_kgm3,
        )
        self.assertAlmostEqual(
            report["flow_inlet_source_normal_velocity_mps"],
            -7.5,
        )

    def test_source_ramp_schedule_continues_from_preflow_by_default(self):
        ramp = VerticalFlapFsiConfig(
            flow_inlet_source_strength=0.75,
            flow_inlet_source_profile="linear_ramp",
            flow_inlet_source_ramp_steps=5,
            flow_inlet_source_schedule_scope="global",
        )
        preflow_history = [{} for _ in range(5)]
        global_step = solid_mpm_fsi_runner._flow_source_schedule_step_index(
            ramp,
            step_index_local=0,
            step_index_global=5,
        )

        self.assertEqual(global_step, 5)
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._flow_inlet_source_factor(ramp, global_step),
            0.75,
        )
        self.assertFalse(
            solid_mpm_fsi_runner._flow_source_ramp_restarted_after_preflow(
                ramp,
                flow_phase="fsi",
                step_index_local=0,
                step_index_global=global_step,
                source_schedule_step_index=global_step,
                preflow_history=preflow_history,
            )
        )

        phase_local = VerticalFlapFsiConfig(
            flow_inlet_source_strength=0.75,
            flow_inlet_source_profile="linear_ramp",
            flow_inlet_source_ramp_steps=5,
            flow_inlet_source_schedule_scope="phase_local",
        )
        local_step = solid_mpm_fsi_runner._flow_source_schedule_step_index(
            phase_local,
            step_index_local=0,
            step_index_global=5,
        )

        self.assertEqual(local_step, 0)
        self.assertAlmostEqual(
            solid_mpm_fsi_runner._flow_inlet_source_factor(phase_local, local_step),
            0.15,
        )
        self.assertTrue(
            solid_mpm_fsi_runner._flow_source_ramp_restarted_after_preflow(
                phase_local,
                flow_phase="fsi",
                step_index_local=0,
                step_index_global=len(preflow_history),
                source_schedule_step_index=local_step,
                preflow_history=preflow_history,
            )
        )

    def test_source_ramp_schedule_uses_contiguous_preflow_indices(self):
        ramp = VerticalFlapFsiConfig(
            flow_inlet_source_strength=0.75,
            flow_inlet_source_profile="linear_ramp",
            flow_inlet_source_ramp_steps=5,
            flow_inlet_source_schedule_scope="global",
        )
        schedule_indices = [
            solid_mpm_fsi_runner._flow_source_schedule_step_index(
                ramp,
                step_index_local=step_index,
                step_index_global=step_index,
            )
            for step_index in range(5)
        ]
        factors = [
            solid_mpm_fsi_runner._flow_inlet_source_factor(ramp, step_index)
            for step_index in schedule_indices
        ]

        self.assertEqual(schedule_indices, [0, 1, 2, 3, 4])
        self.assertEqual(
            [round(factor, 2) for factor in factors],
            [0.15, 0.30, 0.45, 0.60, 0.75],
        )

    def test_thin_wall_probe_reach_tracks_refined_streamwise_spacing(self):
        coarse = VerticalFlapFsiConfig(grid_nodes=(4, 80, 160))
        fine = VerticalFlapFsiConfig(grid_nodes=(4, 224, 448))

        self.assertAlmostEqual(thin_wall_pressure_probe_max_multiplier(coarse), 12.0)
        self.assertGreater(thin_wall_pressure_probe_max_multiplier(fine), 25.0)

    def test_full_domain_runner_passes_full_step_neumann_dt_once(self):
        runner_source = inspect.getsource(
            solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid
        )

        self.assertIn("dt_s=float(config.dt_s)", runner_source)
        self.assertNotIn(
            "dt_s=float(config.dt_s) / float(max(1, int(fluid_substeps)))",
            runner_source,
        )

    def test_full_domain_runner_uses_full_span_flaps(self):
        config = VerticalFlapFsiConfig(grid_nodes=(4, 224, 448))

        lower, upper = solid_mpm_fsi_runner._solid_box(config)

        self.assertAlmostEqual(lower[0], 0.0)
        self.assertAlmostEqual(upper[0], config.span_m)

    def test_canonical_runner_rejects_underresolved_solid_particles_for_fine_grid(
        self,
    ):
        config = VerticalFlapFsiConfig(
            grid_nodes=(4, 224, 448),
            enforce_solid_seeding_limit=True,
        )
        seeding = solid_mpm_fsi_runner.solid_seeding_report(config)

        self.assertFalse(seeding["solid_seeding_guard_satisfied"])
        with self.assertRaisesRegex(ValueError, "too sparse"):
            solid_mpm_fsi_runner._enforce_solid_seeding_limit(config)

    def test_full_domain_runner_uses_local_surface_force_support_radius(self):
        base = VerticalFlapFsiConfig(
            grid_nodes=(4, 224, 448),
            solid_particle_counts=(1, 80, 24),
        )
        expected = surface_force_support_radius_m(base)
        config = vertical_flap_case.with_local_surface_force_support(base)

        self.assertAlmostEqual(config.mpm_support_radius_m, expected)
        self.assertLess(config.mpm_support_radius_m, 0.001)
        self.assertLess(config.mpm_support_radius_m, 0.5 * config.flap_thickness_m)
        self.assertNotAlmostEqual(config.mpm_support_radius_m, 0.006)

    def test_full_domain_runner_persists_solid_substeps_in_process_updates(self):
        runner_source = inspect.getsource(
            solid_mpm_fsi_runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
        )

        self.assertIn(
            'solid_substeps = int(solid_substep_cfl["solid_substeps_selected"])',
            runner_source,
        )
        self.assertGreaterEqual(
            runner_source.count("support_radius_m=config.mpm_support_radius_m"),
            2,
        )
        self.assertIn("solid_substeps=solid_substeps", runner_source)

    def test_full_domain_runner_has_official_style_stationary_preflow_option(self):
        parser_source = inspect.getsource(vertical_flap_case._build_parser)
        preflow_source = inspect.getsource(solid_mpm_fsi_runner._run_fixed_solid_preflow)

        self.assertEqual(VerticalFlapFsiConfig.preflow_steps, 0)
        self.assertIn('"--preflow-steps"', parser_source)
        self.assertIn("_flow_advance_current_step(", preflow_source)
        self.assertIn("_sample_stress_to_marker_forces(", preflow_source)
        self.assertNotIn("solid.step(", preflow_source)
        self.assertNotIn("_advance_solid_substeps_batched(", preflow_source)

    # Closed-loop flow recomputation is now structural, but the 50-step
    # displacement history still fails the official-web physical targets.
    @unittest.expectedFailure
    def test_smoke_fsi_chain_matches_reference_displacement_tolerance(self):
        report = run_vertical_flap_fsi_smoke(
            VerticalFlapFsiConfig(step_count=50, displacement_tolerance=0.05)
        )

        self.assertEqual(report["flow_solution_mode"], "computed_projection")
        self.assertEqual(report["streamwise_axis"], "z")
        self.assertEqual(report["out_of_plane_axis"], "x")
        self.assertEqual(
            report["computed_result_sources"]["pressure_pa"], "fluid.pressure"
        )
        self.assertEqual(
            report["computed_result_sources"]["local_velocity_peak_mps"],
            "max(norm(fluid.velocity))",
        )
        self.assertEqual(
            report["computed_result_sources"]["max_displacement_m"], "solid.x-rest_x"
        )
        self.assertNotIn("pressure_jump_pa", report)
        self.assertNotIn("pressure_scale", report["config"])
        self.assertGreater(report["stress_valid_marker_count"], 0)
        self.assertEqual(report["stress_invalid_marker_count"], 0)
        self.assertEqual(report["scatter_invalid_marker_count"], 0)
        self.assertGreater(report["surface_feedback_updated_marker_count"], 0)
        self.assertLess(report["total_marker_force_n"][2], 0.0)
        self.assertLess(report["tip_mean_displacement_m"][2], 0.0)
        self.assertAlmostEqual(report["root_max_displacement_m"], 0.0, delta=1.0e-8)
        self.assertTrue(
            all(
                step["scatter_invalid_marker_count"] == 0
                and step["feedback_invalid_marker_count"] == 0
                for step in report["history"]
            )
        )
        tip_streamwise_history = [
            step["tip_mean_displacement_m"][2] for step in report["history"]
        ]
        self.assertTrue(all(value <= 0.0 for value in tip_streamwise_history))
        self.assertTrue(
            all(
                later <= earlier + 1.0e-8
                for earlier, later in zip(
                    tip_streamwise_history,
                    tip_streamwise_history[1:],
                )
            )
        )
        self.assertLess(
            max(abs(step["tip_mean_displacement_m"][0]) for step in report["history"]),
            1.0e-6,
        )
        self.assertTrue(math.isfinite(report["max_displacement_m"]))
        self.assertLessEqual(
            report["max_displacement_relative_error"],
            report["displacement_tolerance"],
        )
        self.assertLessEqual(
            report["local_velocity_peak_relative_error"],
            report["velocity_peak_tolerance"],
        )
        self.assertLess(report["scatter_action_reaction_residual_n"], 1.0e-9)


if __name__ == "__main__":
    unittest.main()
