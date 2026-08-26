from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.official.solid_mpm_fsi_runner import (
    _apply_marker_feedback_to_fluid,
    _combine_flow_projection_reports,
)


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"


class AnsysVerticalFlapFeedbackConditionedProjectionTests(unittest.TestCase):
    def test_combined_projection_report_sums_all_cg_work_counters(self) -> None:
        combined = _combine_flow_projection_reports(
            [
                {
                    "cg_multigrid_apply_count": 4,
                    "cg_exact_residual_confirmation_count": 2,
                    "cg_restart_count": 1,
                    "cg_unreached_set_mean_projection_count": 3,
                },
                {
                    "cg_multigrid_apply_count": 7,
                    "cg_exact_residual_confirmation_count": 5,
                    "cg_restart_count": 6,
                    "cg_unreached_set_mean_projection_count": 8,
                },
            ]
        )

        self.assertEqual(combined["cg_multigrid_apply_count"], 11)
        self.assertEqual(combined["cg_exact_residual_confirmation_count"], 7)
        self.assertEqual(combined["cg_restart_count"], 7)
        self.assertEqual(combined["cg_unreached_set_mean_projection_count"], 11)

    def test_combined_projection_report_weights_mean_and_sums_momentum(self) -> None:
        combined = _combine_flow_projection_reports(
            [
                {
                    "velocity_dirichlet_boundary_active_cells_total": 2,
                    "velocity_dirichlet_boundary_mean_delta_mps": 3.0,
                    "velocity_dirichlet_boundary_momentum_delta_n_s": (1.0, 2.0, 3.0),
                },
                {
                    "velocity_dirichlet_boundary_active_cells_total": 6,
                    "velocity_dirichlet_boundary_mean_delta_mps": 1.0,
                    "velocity_dirichlet_boundary_momentum_delta_n_s": (4.0, 5.0, 6.0),
                },
            ]
        )

        self.assertEqual(combined["velocity_dirichlet_boundary_active_cells_total"], 8)
        self.assertAlmostEqual(
            combined["velocity_dirichlet_boundary_mean_delta_mps"],
            1.5,
        )
        self.assertEqual(
            combined["velocity_dirichlet_boundary_momentum_delta_n_s"],
            (5.0, 7.0, 9.0),
        )

    def test_combined_projection_report_retains_earlier_closed_component_failure(
        self,
    ) -> None:
        combined = _combine_flow_projection_reports(
            [
                {
                    "pressure_projection_physical_failure": True,
                    "pressure_projection_physical_failure_reason": (
                        "closed_neumann_component_rhs_incompatible"
                    ),
                    "pressure_projection_physical_failure_action": "reported",
                    "cg_componentwise_mean_projection_count": 3,
                    "pressure_nullspace_incompatible_component_count": 2,
                    "pressure_nullspace_component_rhs_mean_max_abs": 4.5,
                    "pressure_nullspace_component_rhs_integral_max_abs": 6.25,
                },
                {
                    "pressure_projection_physical_failure": False,
                    "pressure_projection_physical_failure_reason": "",
                    "pressure_projection_physical_failure_action": "none",
                    "cg_componentwise_mean_projection_count": 5,
                    "pressure_nullspace_incompatible_component_count": 0,
                    "pressure_nullspace_component_rhs_mean_max_abs": 1.25,
                    "pressure_nullspace_component_rhs_integral_max_abs": 2.5,
                },
            ]
        )

        self.assertTrue(combined["pressure_projection_physical_failure"])
        self.assertEqual(
            combined["pressure_projection_physical_failure_reason"],
            "closed_neumann_component_rhs_incompatible",
        )
        self.assertEqual(
            combined["pressure_projection_physical_failure_action"],
            "reported",
        )
        self.assertEqual(combined["cg_componentwise_mean_projection_count"], 8)
        self.assertEqual(
            combined["pressure_nullspace_incompatible_component_count"],
            2,
        )
        self.assertAlmostEqual(
            combined["pressure_nullspace_component_rhs_mean_max_abs"],
            4.5,
        )
        self.assertAlmostEqual(
            combined["pressure_nullspace_component_rhs_integral_max_abs"],
            6.25,
        )

    def test_sharp_feedback_reports_whether_current_rows_consume_feedback(self) -> None:
        class FakeFluid:
            velocity_dirichlet_boundary_authority = "canonical"

        markers = SimpleNamespace(
            x_gamma_m=object(),
            v_gamma_mps=object(),
            region_id=object(),
            marker_count=8,
        )
        for feedback_available in (False, True):
            with self.subTest(feedback_available=feedback_available):
                report = _apply_marker_feedback_to_fluid(
                    markers,
                    FakeFluid(),
                    SimpleNamespace(
                        flow_solid_boundary_mode="hibm_sharp_marker_rows"
                    ),
                    feedback_available=feedback_available,
                )

                self.assertEqual(
                    report["fluid_marker_feedback_enforcement_mode"],
                    "hibm_sharp_reconstructed_rows",
                )
                self.assertIs(
                    report["fluid_projection_consumed_feedback"],
                    feedback_available,
                )
                self.assertEqual(
                    report["fluid_feedback_constraint_marker_count"],
                    8 if feedback_available else 0,
                )

    def test_sharp_feedback_with_no_markers_never_reports_consumption(self) -> None:
        markers = SimpleNamespace(
            x_gamma_m=object(),
            v_gamma_mps=object(),
            region_id=object(),
            marker_count=0,
        )
        fluid = SimpleNamespace(
            velocity_dirichlet_boundary_authority="canonical"
        )
        config = SimpleNamespace(
            flow_solid_boundary_mode="hibm_sharp_marker_rows"
        )

        for feedback_available in (False, True):
            report = _apply_marker_feedback_to_fluid(
                markers,
                fluid,
                config,
                feedback_available=feedback_available,
            )

            self.assertFalse(report["fluid_projection_consumed_feedback"])
            self.assertEqual(
                report["fluid_feedback_constraint_marker_count"],
                0,
            )

    def test_feedback_becomes_available_only_after_post_solid_row_refresh(self) -> None:
        source = _runner_source()
        loop_start = source.index("for step_index in range(config.step_count):")
        loop_body = _fsi_loop_body(source)

        self.assertIn(
            "feedback_available_for_projection = False",
            source[:loop_start],
        )
        availability_gate = loop_body.index(
            "feedback_available_before_projection = ("
        )
        feedback_apply = loop_body.index("_apply_marker_feedback_to_fluid(")
        flow_projection = loop_body.index("_flow_advance_current_step(")
        solid_feedback = loop_body.index(
            "markers.update_surface_feedback_from_mpm_surface_particles("
        )
        row_refresh = loop_body.index(
            "latest_observer_topology_report = (",
            solid_feedback,
        )
        feedback_ready = loop_body.index(
            "feedback_available_for_projection = True",
            row_refresh,
        )
        history_append = loop_body.index("history.append(", feedback_ready)

        self.assertLess(availability_gate, feedback_apply)
        self.assertLess(feedback_apply, flow_projection)
        self.assertLess(flow_projection, solid_feedback)
        self.assertLess(solid_feedback, row_refresh)
        self.assertLess(row_refresh, feedback_ready)
        self.assertLess(feedback_ready, history_append)
        self.assertIn(
            "feedback_available_for_projection and apply_feedback",
            loop_body[availability_gate:feedback_apply],
        )
        self.assertIn(
            "feedback_available=feedback_available_before_projection",
            loop_body[feedback_apply:flow_projection],
        )

        self.assertNotIn(
            "feedback_available_for_projection = False",
            loop_body,
        )
        consumed_condition = loop_body.index(
            'if latest_feedback_constraint_report["fluid_projection_consumed_feedback"]:'
        )
        consumed_increment = loop_body.index(
            "fluid_projection_consumed_feedback_trial_count += 1",
            consumed_condition,
        )
        self.assertLess(consumed_condition, consumed_increment)

    def test_sharp_flow_adds_accumulating_consistency_projection(self) -> None:
        body = _function_body(_runner_source(), "def _flow_advance_current_step_trial(")

        main_project = body.index("main_flow_report = _project_current_flow(")
        reassembly = body.index(
            "_apply_hibm_sharp_marker_boundary_to_fluid(",
            main_project,
        )
        consistency_project = body.index(
            "consistency_flow_report = _project_current_flow(",
            reassembly,
        )

        self.assertLess(main_project, reassembly)
        self.assertLess(reassembly, consistency_project)
        consistency_body = body[consistency_project:]
        self.assertIn("accumulate_pressure_into_previous=True", consistency_body)
        self.assertIn("reset_pressure=False", consistency_body)
        self.assertIn("preserve_velocity_constraints=False", consistency_body)
        self.assertIn("flow_post_dirichlet_consistency_projection_iterations", body)

    def test_post_solid_observer_refresh_rebuilds_current_sharp_rows(self) -> None:
        body = _function_body(
            _runner_source(),
            "def run_hibm_mpm_fsi(",
        )
        solid_update = body.index(
            "latest_feedback_report = markers.update_surface_feedback_from_mpm_surface_particles("
        )
        observer_refresh = body.index(
            "latest_observer_topology_report = (",
            solid_update,
        )
        feedback_ready = body.index(
            "feedback_available_for_projection = True",
            observer_refresh,
        )
        refresh_body = body[observer_refresh:feedback_ready]

        self.assertIn("topology_only=False", refresh_body)

    def test_runner_applies_marker_feedback_to_fluid_before_projection(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())

        apply_index = loop_body.index("_apply_marker_feedback_to_fluid(")
        # Projection now runs inside _flow_advance_current_step(...).
        project_index = loop_body.index("_flow_advance_current_step(")
        stress_index = loop_body.index("_sample_stress_to_marker_forces(")

        self.assertLess(apply_index, project_index)
        self.assertLess(project_index, stress_index)

    def test_runner_tracks_feedback_consumed_projection_count(self) -> None:
        source = _runner_source()

        self.assertIn("fluid_projection_consumed_feedback_count = 0", source)
        self.assertIn(
            "fluid_projection_consumed_feedback_trial_count = 0", source
        )
        self.assertIn("fluid_projection_consumed_feedback_count += 1", source)
        self.assertIn(
            "fluid_projection_consumed_feedback_trial_count += 1", source
        )
        self.assertIn('"fluid_projection_consumed_feedback_count"', source)
        self.assertIn(
            '"fluid_projection_consumed_feedback_trial_count"', source
        )
        self.assertIn('"fluid_projection_consumed_feedback"', source)

    def test_runner_reports_feedback_constraint_metrics_per_step(self) -> None:
        history_body = _history_append_body(_runner_source())

        for field in (
            '"fluid_projection_consumed_feedback"',
            '"fluid_feedback_constraint_marker_count"',
            '"fluid_feedback_constraint_active_cell_count"',
            '"fluid_feedback_constraint_cleared_cell_count"',
            '"fluid_feedback_constraint_obstacle_cell_count"',
            '"fluid_feedback_constraint_non_obstacle_cell_count"',
            '"fluid_feedback_constraint_projection_participating_cell_count"',
            '"no_slip_residual_before_mps"',
            '"no_slip_residual_after_mps"',
            '"no_slip_target_residual_after_assembly_mps"',
            '"no_slip_projected_residual_after_projection_mps"',
        ):
            self.assertIn(field, history_body)

    def test_adapter_requires_canonical_authority_without_a_host_fallback(self) -> None:
        source = _runner_source()
        adapter_body = _function_body(source, "def _apply_marker_feedback_to_fluid(")

        self.assertIn('velocity_dirichlet_boundary_authority', adapter_body)
        self.assertIn('"canonical"', adapter_body)
        self.assertNotIn("host_fallback", source)
        sharp_branch = adapter_body[: adapter_body.index('if authority != "legacy"')]
        self.assertNotIn("apply_marker_feedback_constraints", sharp_branch)


def _runner_source() -> str:
    return RUNNER_SOURCE.read_text(encoding="utf-8")


def _fsi_loop_body(source: str) -> str:
    loop_start = source.index("for step_index in range(config.step_count):")
    loop_end = source.index("    if (\n        latest_stress_report is None", loop_start)
    return source[loop_start:loop_end]


def _history_append_body(source: str) -> str:
    loop_body = _fsi_loop_body(source)
    append_start = loop_body.index("history.append(")
    append_end = loop_body.index("\n        )", append_start) + len("\n        )")
    return loop_body[append_start:append_end]


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    next_function = source.find("\ndef ", start + len(signature))
    if next_function < 0:
        return source[start:]
    return source[start:next_function]


if __name__ == "__main__":
    unittest.main()
