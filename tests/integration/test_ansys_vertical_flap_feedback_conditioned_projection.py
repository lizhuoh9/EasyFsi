from __future__ import annotations

import unittest
from pathlib import Path


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"
FLUID_SOLVER_SOURCE = Path("simulation_core") / "fluids" / "solver.py"


class AnsysVerticalFlapFeedbackConditionedProjectionTests(unittest.TestCase):
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

    def test_runner_carries_feedback_owned_cells_between_steps(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())

        self.assertIn(
            "feedback_constraint_cells: set[tuple[int, int, int]] = set()",
            _runner_source(),
        )
        self.assertIn(
            "previous_feedback_constraint_cells=feedback_constraint_cells",
            loop_body,
        )
        self.assertIn(
            'feedback_constraint_cells = latest_feedback_constraint_report["_feedback_constraint_cells"]',
            loop_body,
        )

    def test_adapter_reads_marker_feedback_and_updates_fluid_constraints(self) -> None:
        source = _runner_source()
        adapter_body = _function_body(source, "def _apply_marker_feedback_to_fluid(")

        # Device path: marker fields are passed straight into the fused
        # apply_marker_feedback_constraints() kernel dispatch, which reads
        # marker positions/velocities on-device and writes the fluid's
        # velocity-Dirichlet constraint fields without a host round-trip.
        self.assertIn("apply_device = getattr(fluid, \"apply_marker_feedback_constraints\", None)", adapter_body)
        self.assertIn("markers.x_gamma_m,", adapter_body)
        self.assertIn("markers.v_gamma_mps,", adapter_body)
        self.assertIn("report = apply_device(", adapter_body)

        host_fallback_body = _function_body(
            source, "def _apply_marker_feedback_to_fluid_host_fallback("
        )
        self.assertIn("markers.x_gamma_m.to_numpy()", host_fallback_body)
        self.assertIn("markers.v_gamma_mps.to_numpy()", host_fallback_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_active.to_numpy()", host_fallback_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_value_mps.to_numpy()", host_fallback_body)
        self.assertIn(
            "fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()",
            host_fallback_body,
        )
        self.assertIn("fluid.velocity_dirichlet_boundary_active.from_numpy", host_fallback_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_value_mps.from_numpy", host_fallback_body)
        self.assertIn(
            "fluid.velocity_dirichlet_boundary_projection_weight.from_numpy",
            host_fallback_body,
        )

    def test_adapter_clears_only_previous_feedback_owned_constraints(self) -> None:
        source = _runner_source()
        adapter_body = _function_body(source, "def _apply_marker_feedback_to_fluid(")

        # Device path: clearing of previously-owned cells happens inside
        # _clear_marker_feedback_constraints_kernel(), gated on the
        # per-cell marker_feedback_owned flag (only cells this adapter
        # itself claimed last step get reset).
        self.assertIn("apply_device = getattr(fluid, \"apply_marker_feedback_constraints\", None)", adapter_body)
        self.assertIn("report = apply_device(", adapter_body)
        self.assertIn("previous_feedback_constraint_cells", adapter_body)

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

        # Host fallback path (used when the device kernel is unavailable)
        # still clears only the caller-supplied previously-owned cells.
        host_fallback_body = _function_body(
            source, "def _apply_marker_feedback_to_fluid_host_fallback("
        )
        self.assertIn("cleared_cell_count = 0", host_fallback_body)
        self.assertIn("for i, j, k in previous_feedback_constraint_cells:", host_fallback_body)
        self.assertIn("active[i, j, k] = 0", host_fallback_body)
        self.assertIn("values[i, j, k] = 0.0", host_fallback_body)
        self.assertIn("weights[i, j, k] = 0.0", host_fallback_body)
        self.assertIn("cleared_cell_count += 1", host_fallback_body)
        self.assertIn('"fluid_feedback_constraint_cleared_cell_count"', host_fallback_body)

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
