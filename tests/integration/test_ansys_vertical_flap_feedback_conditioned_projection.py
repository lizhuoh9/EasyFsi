from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.official.solid_mpm_fsi_runner import _combine_flow_projection_reports


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"
FLUID_SOLVER_SOURCE = Path("simulation_core") / "fluids" / "solver.py"


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

    def test_sharp_flow_adds_accumulating_consistency_projection(self) -> None:
        body = _function_body(_runner_source(), "def _flow_advance_current_step(")

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
            "def prepare_rectangular_solid_marker_mpm_fsi_runtime(",
        )
        solid_update = body.index(
            "latest_feedback_report = markers.update_surface_feedback_from_mpm_surface_particles("
        )
        observer_refresh = body.index(
            "latest_observer_topology_report = (",
            solid_update,
        )
        trial_return = body.index(
            "return FsiTrialResult(",
            observer_refresh,
        )
        refresh_body = body[observer_refresh:trial_return]

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
        self.assertIn("fluid_projection_consumed_feedback_count += 1", source)
        self.assertIn('"fluid_projection_consumed_feedback_count"', source)
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

    def test_runner_keeps_feedback_ownership_on_the_solver(self) -> None:
        source = _runner_source()

        self.assertNotIn("previous_feedback_constraint_cells", source)
        self.assertNotIn('"_feedback_constraint_cells"', source)

    def test_adapter_reads_marker_feedback_and_updates_fluid_constraints(self) -> None:
        source = _runner_source()
        adapter_body = _function_body(source, "def _apply_marker_feedback_to_fluid(")

        # Device path: marker fields are passed straight into the fused
        # apply_marker_feedback_constraints() kernel dispatch, which reads
        # marker positions/velocities on-device and writes the fluid's
        # velocity-Dirichlet constraint fields without a host round-trip.
        self.assertIn("report = fluid.apply_marker_feedback_constraints(", adapter_body)
        self.assertIn("markers.x_gamma_m,", adapter_body)
        self.assertIn("markers.v_gamma_mps,", adapter_body)
        self.assertIn("markers.region_id,", adapter_body)
        self.assertNotIn("getattr(fluid,", adapter_body)
        self.assertNotIn("_host_fallback", source)
        self.assertNotIn(".to_numpy(", adapter_body)

    def test_solver_clears_only_device_owned_feedback_constraints(self) -> None:
        # Clearing of previously-owned cells happens inside
        # _clear_marker_feedback_constraints_kernel(), gated on the
        # per-cell marker_feedback_owned flag.
        solver_source = _fluid_solver_source()
        clear_kernel_body = _method_body(
            solver_source, "def _clear_marker_feedback_constraints_kernel(self):"
        )
        self.assertIn("if self.marker_feedback_owned[i, j, k] != 0:", clear_kernel_body)
        self.assertIn("self.velocity_dirichlet_boundary_active[i, j, k] = 0", clear_kernel_body)
        self.assertIn("self.velocity_dirichlet_boundary_value_mps[i, j, k] = ti.Vector(", clear_kernel_body)
        self.assertIn("self.velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0", clear_kernel_body)
        self.assertIn("self.marker_feedback_owned[i, j, k] = 0", clear_kernel_body)
        self.assertIn("self.report_marker_feedback_cleared_cell_count[None]", clear_kernel_body)

    def test_runner_computes_post_projection_no_slip_residual(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())

        project_index = loop_body.index("_flow_advance_current_step(")
        residual_index = loop_body.index("_measure_projected_no_slip_residual(")

        self.assertLess(project_index, residual_index)
        self.assertIn(
            '"no_slip_projected_residual_after_projection_mps"',
            _runner_source(),
        )
        self.assertIn(
            '"no_slip_target_residual_after_assembly_mps"',
            _runner_source(),
        )


def _runner_source() -> str:
    return RUNNER_SOURCE.read_text(encoding="utf-8")


def _fluid_solver_source() -> str:
    return FLUID_SOLVER_SOURCE.read_text(encoding="utf-8")


def _fsi_loop_body(source: str) -> str:
    trial_start = source.index("    def evaluate_trial(")
    trial_end = source.index("    def commit_step(", trial_start)
    return source[trial_start:trial_end]


def _history_append_body(source: str) -> str:
    commit_start = source.index("    def commit_step(")
    commit_end = source.index("    class RectangularMarkerVelocityRuntime", commit_start)
    commit_body = source[commit_start:commit_end]
    append_start = commit_body.index("history.append(")
    append_end = commit_body.index("\n        )", append_start) + len("\n        )")
    return commit_body[append_start:append_end]


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    next_function = source.find("\ndef ", start + len(signature))
    if next_function < 0:
        return source[start:]
    return source[start:next_function]


def _method_body(source: str, signature: str) -> str:
    """Extract an indented class-method body (stops at the next sibling
    ``    def``/``    @`` at the same indentation, unlike ``_function_body``
    which only recognizes top-level ``def``)."""

    start = source.index(signature)
    search_from = start + len(signature)
    next_def = source.find("\n    def ", search_from)
    next_decorator = source.find("\n    @", search_from)
    candidates = [index for index in (next_def, next_decorator) if index >= 0]
    if not candidates:
        return source[start:]
    return source[start : min(candidates)]


if __name__ == "__main__":
    unittest.main()
