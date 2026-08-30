from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from simulation_core.coupling.interface_initial_guess_controller import InterfaceInitialGuessController
from simulation_core.coupling.accepted_fsi_checkpoint import load_accepted_fsi_checkpoint
from simulation_core.coupling.hibm_mpm import reports
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmReport
from tests.coupling.test_accepted_fsi_checkpoint import IDENTITY, _state


class _Field:
    def __init__(self, values: np.ndarray):
        self.values = values

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()


def _snapshot_with_production_metadata() -> dict[str, np.ndarray]:
    flow = runner._synchronized_flow_boundary_snapshot(
        {"pressure": np.arange(8, dtype=np.float64).reshape(2, 2, 2)},
        stage="pre_solid_projection",
    )
    solid = SimpleNamespace(particle_count=2, v=_Field(np.full((2, 3), 2.0, np.float32)))
    markers = SimpleNamespace(
        marker_count=2,
        x_gamma_m=_Field(np.full((2, 3), 3.0, np.float32)),
        v_gamma_mps=_Field(np.full((2, 3), 4.0, np.float32)),
        n_gamma=_Field(np.full((2, 3), 5.0, np.float32)),
        A_gamma_m2=_Field(np.full((2,), 6.0, np.float32)),
        region_id=_Field(np.asarray([1, 2], dtype=np.int32)),
    )
    return runner._stage_aware_step_observer_snapshot(
        flow, solid, markers,
        solid_positions_m=np.full((2, 3), 7.0, np.float32),
        solid_rest_positions_m=np.full((2, 3), 8.0, np.float32),
        fixed_mask=np.asarray([True, False]), tip_mask=np.asarray([False, True]),
        expected_flow_stage="pre_solid_projection",
        structure_geometry_stage="post_solid_observer",
        error_message="test stage mismatch",
    )


def _iqn_trial_metadata() -> dict[str, np.ndarray]:
    trace = np.full((1, 2, 3), 0.5, np.float64)
    coupling = runner.FsiCouplingReport(
        iterations=1, converged=True, relative_residual=0.0,
        absolute_residual_mps=0.0, max_marker_residual_mps=0.0,
        relative_residual_history=(0.0,), absolute_residual_history_mps=(0.0,),
        update_modes=("picard",), trial_guess_history_mps=trace,
        trial_candidate_history_mps=trace, trial_residual_history_mps=np.zeros_like(trace),
    )
    return runner._accepted_iqn_trial_vector_arrays(
        coupling,
        context=runner.FsiStepContext(step=1, step_index=0, time_s=0.1, dt_s=0.1),
        layout_sha256="a" * 64,
    )


def _typed_reports():
    return {
        "latest_stress_report": reports.HibmMpmFluidStressSampleReport(0, 0, 0.0),
        "latest_force_report": reports.HibmMpmSurfaceMarkerForceReport(
            (0.0,) * 3, (0.0,) * 3, (0.0,) * 3, 0, 0, 0, (0.0,) * 3, 0.0,
        ),
        "latest_scatter_report": reports.HibmMpmMpmForceScatterReport(
            0, 0, 0, (0.0,) * 3, (0.0,) * 3, 0.0,
        ),
        "latest_solid_report": NeoHookeanMpmReport(
            particle_count=2, active_grid_nodes=0, grid_out_of_bounds_particle_count=0,
            particle_spacing_m=0.1, grid_spacing_m=(0.1,) * 3, total_mass_kg=0.0,
            total_volume_m3=0.0, primary_mean_displacement_m=(0.0,) * 3,
            primary_mean_velocity_mps=(0.0,) * 3, secondary_mean_displacement_m=(0.0,) * 3,
            secondary_mean_velocity_mps=(0.0,) * 3, particle_momentum_kg_mps=(0.0,) * 3,
            grid_momentum_kg_mps=(0.0,) * 3, external_force_n=(0.0,) * 3,
            transfer_relative_error=0.0, max_speed_mps=0.0, max_abs_j=1.0,
            deformation_clamp_count=0, mean_radial_stretch=1.0, max_radial_stretch_error=0.0,
        ),
        "latest_feedback_report": reports.HibmMpmSurfaceUpdateReport(0, 0, 0.0, 0.0),
    }


def _record(step: int) -> dict[str, object]:
    trial = {
        "flow_wall_time_s": 0.0, "hibm_wall_time_s": 0.0, "solid_wall_time_s": 0.0,
        "cg_iterations_total": 0, "flow_momentum_advection_substeps_total": 0,
        "flow_sst_transport_substeps_total": 0, "solid_substeps_executed_total": 0,
        "feedback_consumed": False,
    }
    return {
        "history_row": {"step": step, "time_s": step * 0.1},
        "solid_step_execution_reports": [{}], "solid_trial_execution_reports": [{}],
        "coupling_step_reports": [{"hibm_fsi_coupling_iterations_used": 1}],
        "coupling_trial_work_reports": [trial],
    }


def _state_with_snapshot(step: int, snapshot: dict[str, np.ndarray]):
    base = _state(step=step)
    outbox_snapshot = {**snapshot, **_iqn_trial_metadata()}
    reports_with_metadata = {
        "preflow_report": {
            "nested_snapshot": snapshot,
            "final_flow_field_snapshot": snapshot,
        },
        "latest_solid_step_report": {"stage": np.asarray("post_solid")},
        "latest_flow_report": {"nested_snapshot": snapshot},
        "latest_feedback_constraint_report": {}, "latest_dynamic_obstacle_report": {},
        "solid_substep_cfl": {}, "final_flow_field_snapshot": snapshot,
        "terminal_fields": {
            "computed_pressure_min_pa": 0.0, "computed_pressure_max_pa": 0.0,
            "traction": {}, "projection_boundary": {},
            "final_stress_marker_diagnostics": (), "final_stress_face_diagnostics": {},
        },
        "anchor_install_report": {}, "pressure_pair_anchor_pair_map": {},
    }
    controller = InterfaceInitialGuessController("carry_forward")
    reference = np.zeros((2, 3), np.float32)
    for _ in range(step):
        controller.begin_step(reference, layout_id="a" * 64, dt_s=0.1)
        controller.accept_step(np.full((2, 3), 0.5), layout_id="a" * 64)
    runner_state = {
        "dt_s": 0.1, "coupling_mode": "iqn_ils",
        **{name: 0 for name in runner._FSI_CHECKPOINT_COUNTERS},
        **reports_with_metadata, **_typed_reports(),
        "observer_identity": "metadata-destination",
        "observer_outbox": {
            "step": step, "time_s": step * 0.1,
            "history_row": {"step": step}, "snapshot": outbox_snapshot,
        },
    }
    return replace(base, runner_state=runner_state,
                   initial_guess_state=controller.snapshot(),
                   marker_reference_positions_m=reference)


def _assert_snapshot(actual: dict[str, object], expected: dict[str, np.ndarray]) -> None:
    assert set(actual) == set(expected)
    for name, values in expected.items():
        restored = actual[name]
        assert isinstance(restored, np.ndarray), name
        assert restored.dtype == values.dtype and restored.shape == values.shape, name
        assert not restored.flags.writeable, name
        np.testing.assert_array_equal(restored, values, err_msg=name)
        assert restored.tobytes() == values.tobytes(), name


def test_production_snapshot_metadata_commits_loads_and_replays(tmp_path: Path):
    snapshot = _snapshot_with_production_metadata()
    state = _state_with_snapshot(1, snapshot)
    committed = runner._commit_accepted_fsi_checkpoint(
        tmp_path / "accepted", state=state, identity=IDENTITY, record=_record(1),
        previous_tail=None, expected_generation=None,
    )
    loaded = load_accepted_fsi_checkpoint(
        tmp_path / "accepted", expected_identity=IDENTITY, target_step_count=1,
    )
    _assert_snapshot(loaded.state.runner_state["final_flow_field_snapshot"], snapshot)
    _assert_snapshot(loaded.state.runner_state["preflow_report"]["nested_snapshot"], snapshot)
    _assert_snapshot(loaded.state.runner_state["preflow_report"]["final_flow_field_snapshot"], snapshot)
    _assert_snapshot(loaded.state.runner_state["latest_flow_report"]["nested_snapshot"], snapshot)
    received = []
    def observer(*args):
        received.append(args)
    observer.checkpoint_replay_safe = True
    observer.checkpoint_identity = "metadata-destination"
    observer.record_iqn_trial_vectors = True
    runner._replay_fsi_checkpoint_observer(loaded.state.runner_state, observer)
    _assert_snapshot(received[0][3], state.runner_state["observer_outbox"]["snapshot"])
    replay_path = tmp_path / "replayed_snapshot.npz"
    np.savez(replay_path, **received[0][3])
    with np.load(replay_path, allow_pickle=False) as replayed:
        assert set(replayed.files) == set(received[0][3])
        for name, values in received[0][3].items():
            assert replayed[name].dtype == values.dtype and replayed[name].shape == values.shape
            assert replayed[name].tobytes() == values.tobytes()
    assert committed.generation == loaded.generation


@pytest.mark.parametrize("bad", [np.asarray("physical-string"), np.asarray([object()])])
def test_physical_string_or_object_array_cannot_replace_durable_metadata_parent(tmp_path: Path, bad):
    snapshot = _snapshot_with_production_metadata()
    state = _state_with_snapshot(1, snapshot)
    first = runner._commit_accepted_fsi_checkpoint(
        tmp_path / "accepted", state=state, identity=IDENTITY, record=_record(1),
        previous_tail=None, expected_generation=None,
    )
    second_state = _state_with_snapshot(2, snapshot)
    fields = dict(second_state.macro_state.fluid_fields)
    fields["pressure"] = bad
    second = replace(second_state, macro_state=replace(
        second_state.macro_state, fluid_fields=fields,
    ))
    with pytest.raises(runner.FsiCheckpointCommitError) as error:
        runner._commit_accepted_fsi_checkpoint(
            tmp_path / "accepted", state=second, identity=IDENTITY, record=_record(2),
            previous_tail=first.history_tail, expected_generation=first.generation,
        )
    assert error.value.__cause__ is not None
    assert "fluid pressure must be a finite numeric array" in str(error.value.__cause__)
    loaded = load_accepted_fsi_checkpoint(
        tmp_path / "accepted", expected_identity=IDENTITY, target_step_count=2,
    )
    assert loaded.generation == first.generation and loaded.state.macro_state.accepted_step_index == 1
