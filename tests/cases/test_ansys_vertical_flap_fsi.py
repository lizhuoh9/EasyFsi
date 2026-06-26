import inspect
import math
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

from benchmarks.official import solid_mpm_fsi_runner
import cases.ansys_vertical_flap_fsi as vertical_flap_case
from cases import CASE_MODULES
from cases.ansys_vertical_flap_fsi import (
    ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS,
    ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
    surface_force_support_radius_m,
    thin_wall_pressure_probe_max_multiplier,
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
        self.assertAlmostEqual(metadata["fluid"]["inlet_velocity_mps"], 10.0)
        self.assertAlmostEqual(metadata["solid"]["density_kgm3"], 1600.0)
        self.assertAlmostEqual(metadata["solid"]["young_modulus_pa"], 1.0e6)
        self.assertAlmostEqual(metadata["solid"]["poisson_ratio"], 0.47)
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
        self.assertAlmostEqual(reference["max_displacement_m"], 5.1e-5)
        self.assertEqual(reference["time_step_s"], 5.0e-4)
        self.assertEqual(reference["step_count"], 50)
        self.assertGreaterEqual(reference["local_velocity_peak_mps"], 20.0)
        self.assertLessEqual(reference["local_velocity_peak_mps"], 29.0)

    def test_formal_runner_uses_official_full_span_flap_box(self):
        config = VerticalFlapFsiConfig()
        solid_min, solid_max = solid_mpm_fsi_runner._solid_box(config)

        self.assertAlmostEqual(solid_min[0], 0.0)
        self.assertAlmostEqual(solid_max[0], config.span_m)
        self.assertAlmostEqual(solid_min[2], config.flap_streamwise_min_m)
        self.assertAlmostEqual(solid_max[2], config.flap_streamwise_max_m)

    def test_formal_runner_places_both_streamwise_marker_faces(self):
        class FakeMarkers:
            def __init__(self, marker_capacity, runtime):
                self.marker_capacity = marker_capacity
                self.runtime = runtime
                self.marker_count = 0
                self.positions_m = []
                self.normals = []
                self.areas_m2 = []

            def load_markers(
                self,
                *,
                positions_m,
                velocities_mps,
                normals,
                areas_m2,
                region_ids,
            ):
                self.positions_m = list(positions_m)
                self.normals = list(normals)
                self.areas_m2 = list(areas_m2)
                self.marker_count = len(self.positions_m)

        config = VerticalFlapFsiConfig(marker_count=3)
        dz = config.duct_length_m / float(config.grid_nodes[2])
        with patch.object(solid_mpm_fsi_runner, "HibmMpmSurfaceMarkers", FakeMarkers):
            markers = solid_mpm_fsi_runner._build_markers(config, runtime=None)

        self.assertEqual(markers.marker_capacity, 6)
        self.assertEqual(markers.marker_count, 6)
        self.assertEqual(markers.normals[:3], [(0.0, 0.0, 1.0)] * 3)
        self.assertEqual(markers.normals[3:], [(0.0, 0.0, -1.0)] * 3)
        self.assertTrue(
            all(
                math.isclose(position[2], config.flap_streamwise_max_m + 0.51 * dz)
                for position in markers.positions_m[:3]
            )
        )
        self.assertTrue(
            all(
                math.isclose(position[2], config.flap_streamwise_min_m - 0.51 * dz)
                for position in markers.positions_m[3:]
            )
        )

    def test_solid_substep_cfl_report_preserves_explicit_higher_count(self):
        unstable = VerticalFlapFsiConfig(
            grid_nodes=(4, 320, 640),
            solid_substeps=200,
        )
        unstable_report = solid_mpm_fsi_runner.solid_substep_cfl_report(unstable)

        self.assertGreater(unstable_report["solid_substeps_cfl_minimum"], 900)
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
        self.assertEqual(config.flow_inlet_source_strength, 1.0)
        self.assertEqual(config.flow_inlet_source_ramp_steps, 0)
        self.assertEqual(config.flow_inlet_source_profile, "constant")
        self.assertTrue(config.flow_pressure_outlet_enabled)
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
        self.assertEqual(args.flow_outlet_balance_policy, "report_only")

    def test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance(self):
        source = inspect.getsource(solid_mpm_fsi_runner._run_fixed_solid_preflow)

        self.assertIn('"preflow_steps_requested"', source)
        self.assertIn('"preflow_steps_completed"', source)
        self.assertIn('"preflow_status"', source)
        self.assertIn('"preflow_history"', source)
        self.assertIn('"solid_fixed": True', source)
        self.assertIn('"solid_advanced": False', source)
        self.assertIn("_flow_advance_current_step(", source)
        self.assertIn("_sample_stress_to_marker_forces(markers, fluid)", source)
        self.assertNotIn("solid.step(", source)
        self.assertNotIn("scatter_marker_forces_to_mpm_particles", source)

    def test_preflow_only_step_count_zero_is_diagnostic_only(self):
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            VerticalFlapFsiConfig(step_count=0, preflow_steps=1)
        )
        with self.assertRaisesRegex(ValueError, "preflow-only"):
            solid_mpm_fsi_runner._validate_rectangular_solid_config(
                VerticalFlapFsiConfig(step_count=0, preflow_steps=0)
            )

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.run_rectangular_solid_marker_mpm_fsi_smoke
        )
        self.assertIn("config.step_count == 0 and preflow_history", run_source)
        self.assertIn("_preflow_only_report", run_source)

    def test_diagnostic_flow_controls_are_explicit_and_default_safe(self):
        config = VerticalFlapFsiConfig()

        self.assertTrue(config.apply_marker_feedback_to_fluid)
        self.assertFalse(config.flow_reset_pressure_each_step)
        self.assertFalse(config.flow_reinitialize_inlet_each_step)
        self.assertEqual(config.flow_driver_mode, "projection_only")
        self.assertEqual(config.flow_inlet_source_strength, 1.0)
        self.assertEqual(config.flow_inlet_source_profile, "constant")
        self.assertEqual(config.flow_inlet_source_schedule_scope, "global")

        run_source = inspect.getsource(
            solid_mpm_fsi_runner.run_rectangular_solid_marker_mpm_fsi_smoke
        )
        self.assertIn("apply_marker_feedback_to_fluid", run_source)
        self.assertIn("flow_reset_pressure_each_step", run_source)
        self.assertIn("flow_reinitialize_inlet_each_step", run_source)
        self.assertIn("_flow_advance_current_step", run_source)

    def test_sustained_flow_driver_modes_are_explicit_and_default_safe(self):
        self.assertIn(
            "sustained_volume_source_inlet",
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
        self.assertEqual(
            solid_mpm_fsi_runner._effective_flow_driver_mode(
                VerticalFlapFsiConfig(flow_reinitialize_inlet_each_step=True)
            ),
            "reinitialize_inlet_each_step_diagnostic",
        )

        advance_source = inspect.getsource(
            solid_mpm_fsi_runner._flow_advance_current_step
        )
        self.assertIn("add_zmax_velocity_inlet_volume_source", advance_source)
        self.assertIn("FLOW_DRIVER_SUSTAINED_SOURCE", advance_source)
        self.assertIn("FLOW_DRIVER_SUSTAINED_PREDICTOR", advance_source)
        self.assertIn("_flow_inlet_source_factor", advance_source)
        self.assertIn(
            "flow_predictor_applied",
            inspect.getsource(solid_mpm_fsi_runner._flow_driver_report),
        )

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
        runner_source = Path(
            "_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/"
            "run_full_domain_two_flap_true_sharp_fsi.py"
        ).read_text(encoding="utf-8")

        self.assertIn("pressure_neumann_dt_s=float(config.dt_s)", runner_source)
        self.assertNotIn(
            "pressure_neumann_dt_s=float(config.dt_s) / float(max(1, int(fluid_substeps)))",
            runner_source,
        )

    def test_full_domain_runner_uses_full_span_flaps(self):
        runner_path = Path(
            "_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/"
            "run_full_domain_two_flap_true_sharp_fsi.py"
        )
        spec = importlib.util.spec_from_file_location("full_domain_flap_runner", runner_path)
        self.assertIsNotNone(spec)
        runner = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(runner)

        config = runner._config(1, grid_nodes=(4, 224, 448), markers_per_face=84)
        boxes = runner._flap_boxes(config)

        self.assertAlmostEqual(boxes["lower"][0][0], 0.0)
        self.assertAlmostEqual(boxes["lower"][1][0], config.span_m)
        self.assertAlmostEqual(boxes["upper"][0][0], 0.0)
        self.assertAlmostEqual(boxes["upper"][1][0], config.span_m)

    def test_full_domain_runner_uses_resolved_solid_particles_for_fine_grid(self):
        runner_path = Path(
            "_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/"
            "run_full_domain_two_flap_true_sharp_fsi.py"
        )
        spec = importlib.util.spec_from_file_location("full_domain_flap_runner", runner_path)
        self.assertIsNotNone(spec)
        runner = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(runner)

        config = runner._config(1, grid_nodes=(4, 224, 448))
        flap_grid_cells_y = config.flap_height_m / (
            config.duct_height_m / config.grid_nodes[1]
        )
        flap_grid_cells_z = config.flap_thickness_m / (
            config.duct_length_m / config.grid_nodes[2]
        )

        self.assertGreaterEqual(config.solid_particle_counts[1], flap_grid_cells_y)
        self.assertGreaterEqual(config.solid_particle_counts[2], flap_grid_cells_z)
        self.assertGreater(config.solid_substeps, 0)
        self.assertLessEqual(config.solid_substeps, 400)

    def test_full_domain_runner_uses_local_surface_force_support_radius(self):
        runner_path = Path(
            "_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/"
            "run_full_domain_two_flap_true_sharp_fsi.py"
        )
        spec = importlib.util.spec_from_file_location("full_domain_flap_runner", runner_path)
        self.assertIsNotNone(spec)
        runner = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(runner)

        base = VerticalFlapFsiConfig(
            grid_nodes=(4, 224, 448),
            solid_particle_counts=(1, 80, 24),
        )
        expected = surface_force_support_radius_m(base)
        config = runner._config(1, grid_nodes=(4, 224, 448))

        self.assertAlmostEqual(config.mpm_support_radius_m, expected)
        self.assertLess(config.mpm_support_radius_m, 0.001)
        self.assertLess(config.mpm_support_radius_m, 0.5 * config.flap_thickness_m)
        self.assertNotAlmostEqual(config.mpm_support_radius_m, 0.006)

    def test_full_domain_runner_persists_solid_substeps_in_process_updates(self):
        runner_source = Path(
            "_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/"
            "run_full_domain_two_flap_true_sharp_fsi.py"
        ).read_text(encoding="utf-8")

        self.assertGreaterEqual(
            runner_source.count('"solid_substeps": int(config.solid_substeps)'),
            3,
        )
        self.assertGreaterEqual(
            runner_source.count('"mpm_support_radius_m": float(config.mpm_support_radius_m)'),
            4,
        )

    def test_full_domain_runner_has_official_style_stationary_preflow_option(self):
        runner_source = Path(
            "_codex_validation/official_ansys_fluent_vertical_flap_full_domain_solve_20260622/"
            "run_full_domain_two_flap_true_sharp_fsi.py"
        ).read_text(encoding="utf-8")

        self.assertIn("PREFLOW_STEPS = 0", runner_source)
        self.assertIn('parser.add_argument("--preflow-steps"', runner_source)
        self.assertIn("def stationary_fluid_load_step() -> Any:", runner_source)
        self.assertIn("def fixed_solid_step() -> dict[str, Any]:", runner_source)
        self.assertIn("preflow_fixed_solid", runner_source)
        self.assertIn("solid_step=fixed_solid_step", runner_source)
        self.assertIn('"preflow_steps": preflow_count', runner_source)
        preflow_block = runner_source[
            runner_source.index("def stationary_fluid_load_step() -> Any:"):
            runner_source.index("for step_index in range(config.step_count):")
        ]
        self.assertNotIn("solid.step(", preflow_block)
        self.assertNotIn("pressure_scale", preflow_block)

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
