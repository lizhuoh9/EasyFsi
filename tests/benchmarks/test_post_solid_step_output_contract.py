from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import (
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    VerticalFlapFsiConfig,
)
from src.refactored.validation.ansys_vertical_flap_fsi import (
    native_fine_final_contracts,
)


def _sharp_config() -> SimpleNamespace:
    return SimpleNamespace(
        apply_marker_feedback_to_fluid=True,
        flow_solid_boundary_mode=(
            runner.FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS
        ),
        flow_hibm_sharp_interpolate_velocity_rows=True,
        flow_reprojection_iterations=37,
        flow_reprojection_cg_tolerance=2.0e-7,
        flow_projection_iterations=1080,
        flow_cg_tolerance=1.0e-6,
    )


def _projection_state() -> dict[str, object]:
    return {
        "projection_report": {"cg_converged_all": True},
        "local_velocity_peak_mps": 3.0,
        "pressure_min_pa": -2.0,
        "pressure_max_pa": 4.0,
    }


class _TransactionalFluid:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.state = "initial"
        self._saved_state: str | None = None

    def save_state(self) -> None:
        self._events.append("save")
        self._saved_state = self.state

    def restore_state(self) -> None:
        self._events.append("restore")
        assert self._saved_state is not None
        self.state = self._saved_state


def test_post_solid_finalize_preserves_outer_step_base_snapshot() -> None:
    events: list[str] = []
    fluid = _TransactionalFluid(events)
    fluid.state = "physical_step_base"
    fluid.save_state()
    fluid.state = "post_solid_entry"

    def finalize_unchecked(**kwargs):
        kwargs["fluid"].state = "post_solid_finalized"
        return {"status": "completed"}

    with mock.patch.object(
        runner,
        "_finalize_post_solid_kinematic_flow_unchecked",
        side_effect=finalize_unchecked,
    ):
        result = runner._finalize_post_solid_kinematic_flow(
            markers=SimpleNamespace(marker_count=4),
            fluid=fluid,
            config=_sharp_config(),
            sharp_boundary_cache={},
            sharp_boundary_report={},
            step_index_local=0,
            step_index_global=77,
            capture_observer_snapshot=False,
            capture_final_snapshot=False,
        )

    fluid.restore_state()

    assert result == {"status": "completed"}
    assert fluid.state == "physical_step_base"


def test_post_solid_finalize_orders_feedback_projection_gate_then_capture() -> None:
    events: list[str] = []
    projector = object()
    feedback: dict[str, object] = {}
    terminal = {
        "hibm_joint_qp_converged": True,
        "hibm_joint_qp_cycles_used": 1,
    }

    def apply_feedback(*args, **kwargs):
        events.append("feedback")
        assert kwargs["feedback_available"] is True
        return dict(feedback)

    def project(*args, **kwargs):
        events.append("project")
        assert kwargs["reset_pressure"] is False
        assert kwargs["accumulate_pressure_into_previous"] is True
        assert kwargs["homogenize_pressure_interface_rhs_for_increment"] is True
        assert kwargs["pre_projection_velocity_projector"] is projector
        assert kwargs["pressure_velocity_nullspace_projector"] is projector
        assert kwargs["projection_iterations"] == 37
        assert kwargs["cg_tolerance"] == pytest.approx(2.0e-7)
        return _projection_state()

    def require_gate(diagnostics, *, context):
        events.append("gate")
        assert diagnostics == terminal
        assert "post-solid kinematic projection" in context

    with (
        mock.patch.object(
            runner,
            "_apply_marker_feedback_to_fluid",
            side_effect=apply_feedback,
        ),
        mock.patch.object(
            runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=projector,
        ),
        mock.patch.object(runner, "_project_current_flow", side_effect=project),
        mock.patch.object(
            runner,
            "_sample_hibm_no_slip_report",
            side_effect=lambda *args, **kwargs: events.append("no_slip") or {},
        ),
        mock.patch.object(
            runner,
            "_hibm_joint_qp_cycle_diagnostics",
            return_value={"converged": True},
        ),
        mock.patch.object(
            runner,
            "_hibm_joint_qp_terminal_diagnostics",
            return_value=terminal,
        ),
        mock.patch.object(
            runner,
            "_require_hibm_joint_qp_convergence",
            side_effect=require_gate,
        ),
        mock.patch.object(
            runner,
            "_flow_parity_snapshot",
            side_effect=lambda fluid: events.append("observer_snapshot")
            or {"pressure": np.zeros((1, 1, 1), dtype=np.float32)},
        ),
        mock.patch.object(
            runner,
            "_flow_field_snapshot",
            side_effect=lambda fluid: events.append("final_snapshot")
            or {"pressure": np.zeros((1, 1, 1), dtype=np.float32)},
        ),
    ):
        result = runner._finalize_post_solid_kinematic_flow(
            markers=SimpleNamespace(marker_count=4),
            fluid=_TransactionalFluid(events),
            config=_sharp_config(),
            sharp_boundary_cache={},
            sharp_boundary_report={"hibm_sharp_marker_boundary_enabled": True},
            step_index_local=0,
            step_index_global=77,
            capture_observer_snapshot=True,
            capture_final_snapshot=True,
        )

    assert events == [
        "feedback",
        "project",
        "no_slip",
        "gate",
        "observer_snapshot",
        "final_snapshot",
    ]
    assert result["flow_report"]["flow_solution_stage"] == (
        runner.FLOW_STAGE_POST_SOLID_KINEMATIC_PROJECTION
    )
    for snapshot_key in ("observer_flow_snapshot", "final_flow_snapshot"):
        snapshot = result[snapshot_key]
        assert snapshot["flow_solution_stage"].item() == (
            runner.FLOW_STAGE_POST_SOLID_KINEMATIC_PROJECTION
        )
        assert snapshot["boundary_topology_stage"].item() == (
            runner.FLOW_STAGE_POST_SOLID_KINEMATIC_PROJECTION
        )


def test_failed_post_solid_gate_prevents_snapshot_capture() -> None:
    events: list[str] = []

    with (
        mock.patch.object(
            runner,
            "_apply_marker_feedback_to_fluid",
            return_value={},
        ),
        mock.patch.object(
            runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=object(),
        ),
        mock.patch.object(
            runner,
            "_project_current_flow",
            return_value=_projection_state(),
        ),
        mock.patch.object(
            runner,
            "_sample_hibm_no_slip_report",
            return_value={},
        ),
        mock.patch.object(
            runner,
            "_hibm_joint_qp_cycle_diagnostics",
            return_value={"converged": False},
        ),
        mock.patch.object(
            runner,
            "_hibm_joint_qp_terminal_diagnostics",
            return_value={"hibm_joint_qp_converged": False},
        ),
        mock.patch.object(
            runner,
            "_require_hibm_joint_qp_convergence",
            side_effect=runner.HibmJointQpConvergenceError(
                "failed", diagnostics={}
            ),
        ),
        mock.patch.object(
            runner,
            "_flow_parity_snapshot",
            side_effect=lambda fluid: events.append("observer_snapshot"),
        ),
        mock.patch.object(
            runner,
            "_flow_field_snapshot",
            side_effect=lambda fluid: events.append("final_snapshot"),
        ),
        mock.patch.object(
            runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            side_effect=lambda *args, **kwargs: events.append("rebuild") or {},
        ),
    ):
        with pytest.raises(runner.HibmJointQpConvergenceError):
            runner._finalize_post_solid_kinematic_flow(
                markers=SimpleNamespace(marker_count=4),
                fluid=_TransactionalFluid(events),
                config=_sharp_config(),
                sharp_boundary_cache={},
                sharp_boundary_report={},
                step_index_local=0,
                step_index_global=77,
                capture_observer_snapshot=True,
                capture_final_snapshot=True,
            )

    assert events == []


def test_failed_post_solid_projection_propagates_to_outer_step_transaction() -> None:
    events: list[str] = []

    with (
        mock.patch.object(
            runner,
            "_apply_marker_feedback_to_fluid",
            side_effect=lambda *args, **kwargs: events.append("feedback")
            or {},
        ),
        mock.patch.object(
            runner,
            "_hibm_pre_projection_velocity_projector_from_cache",
            return_value=object(),
        ),
        mock.patch.object(
            runner,
            "_project_current_flow",
            side_effect=RuntimeError("pressure solve failed"),
        ),
        mock.patch.object(
            runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            side_effect=lambda *args, **kwargs: events.append("rebuild") or {},
        ),
    ):
        with pytest.raises(RuntimeError, match="pressure solve failed"):
            runner._finalize_post_solid_kinematic_flow(
                markers=SimpleNamespace(marker_count=4),
                fluid=_TransactionalFluid(events),
                config=_sharp_config(),
                sharp_boundary_cache={},
                sharp_boundary_report={},
                step_index_local=0,
                step_index_global=77,
                capture_observer_snapshot=True,
                capture_final_snapshot=True,
            )

    assert events == ["feedback"]


def test_runner_commits_history_and_observer_only_after_synchronized_finalize() -> None:
    source = inspect.getsource(
        runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
    )
    finalize = source.index("_finalize_post_solid_kinematic_flow(")
    history_commit = source.index("history.append(", finalize)
    observer_commit = source.index("step_observer(", history_commit)

    assert "stage=\"pre_solid_projection\"" not in source
    assert finalize < history_commit < observer_commit


def test_unqualified_step_end_diagnostics_use_post_solid_state() -> None:
    source = inspect.getsource(
        runner.prepare_rectangular_solid_marker_mpm_fsi_runtime
    )
    history_start = source.index("history.append(")
    history_block = source[
        history_start : source.index("if step_observer", history_start)
    ]
    final_block = source[source.rindex("return {") :]

    for block in (history_block, final_block):
        assert "latest_feedback_constraint_report[" not in block
        assert (
            "canonical_velocity_dirichlet_report="
            "latest_post_solid_flow_report.get(" in block
        )


def test_step_snapshot_rejects_a_relabelled_pre_solid_flow_state() -> None:
    with pytest.raises(RuntimeError, match="post-solid synchronized flow"):
        runner._step_observer_snapshot(
            {
                "flow_solution_stage": np.asarray("pre_solid_projection"),
                "boundary_topology_stage": np.asarray("pre_solid_projection"),
                "flow_boundary_state_synchronized": np.asarray(True),
            },
            SimpleNamespace(particle_count=0),
            SimpleNamespace(marker_count=0),
            solid_positions_m=np.empty((0, 3), dtype=np.float32),
            solid_rest_positions_m=np.empty((0, 3), dtype=np.float32),
            fixed_mask=np.empty(0, dtype=bool),
            tip_mask=np.empty(0, dtype=bool),
        )


def test_ansys_identity_locks_direct_partitioned_time_layers() -> None:
    assert (
        "flow_post_solid_kinematic_projection_enabled"
        not in VerticalFlapFsiConfig.__dataclass_fields__
    )
    assert ANSYS_VERTICAL_FLAP_CASE_METADATA["coupling_time_layer"] == {
        "scheme": "direct_explicit_partitioned",
        "physical_step_owner": (
            "benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi"
        ),
        "step_end_flow_stage": "pre_solid_projection",
        "step_end_structure_geometry_stage": "post_solid_observer",
        "transport_advanced_by_step_end_projection": False,
        "fail_closed_on_solver_health": True,
    }
    assert (
        "flow_post_solid_kinematic_projection_enabled"
        not in native_fine_final_contracts.FINAL_FINE_CONFIG_IDENTITY
    )
    assert native_fine_final_contracts.FINAL_FINE_TIME_LAYER_IDENTITY == {
        "scheme": "explicit_loose",
        "step_end_flow_stage": "pre_solid_projection",
        "step_end_structure_geometry_stage": "post_solid_observer",
        "transport_advanced_by_step_end_projection": False,
        "fluent_strong_coupling_equivalent": False,
    }
