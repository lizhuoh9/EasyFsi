import ast
import inspect
import textwrap
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import selected_formulation_solver_config


def test_production_history_time_round_trips_through_accepted_checkpoint(tmp_path: Path):
    from simulation_core.coupling.accepted_fsi_checkpoint import (
        load_accepted_fsi_checkpoint,
        write_accepted_fsi_checkpoint,
    )
    from tests.coupling.test_accepted_fsi_checkpoint import IDENTITY, _state

    tree = ast.parse(textwrap.dedent(inspect.getsource(runner.run_hibm_mpm_fsi)))
    append = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "history"
        and node.func.attr == "append"
    )
    producer_row = append.args[0]
    assert isinstance(producer_row, ast.Dict)
    selected = {"step", "time_s"}
    # Keep the actual producer expressions for physical step/time.  Every
    # unrelated report is a host placeholder, so no solver or CUDA path runs.
    row_expression = ast.Dict(
        keys=producer_row.keys,
        values=[
            value if isinstance(key, ast.Constant) and key.value in selected
            else ast.Dict(keys=[], values=[]) if key is None
            else ast.Constant(value=None)
            for key, value in zip(producer_row.keys, producer_row.values)
        ],
    )
    step_index = 0
    scope = dict(vars(runner))
    scope.update(step_index=step_index, config=SimpleNamespace(dt_s=0.1))
    produced = eval(compile(ast.fix_missing_locations(ast.Expression(row_expression)), "<history-row>", "eval"), scope)
    row = {name: produced[name] for name in selected}
    assert row == {"step": 1, "time_s": 0.1}
    path = tmp_path / "accepted"
    write_accepted_fsi_checkpoint(
        path, state=_state(step=1), identity=IDENTITY,
        record={"history_row": row, "coupling_step_reports": [{"accepted_step": 1}]},
    )
    loaded = load_accepted_fsi_checkpoint(
        path, expected_identity=IDENTITY, target_step_count=1,
    )
    assert loaded.records[0]["history_row"] == row


def test_checkpoint_retained_iqn_signature_must_match_live_algorithm():
    from simulation_core.coupling.iqn_ils import IqnIlsConfig
    config = _config(iqn_reuse_previous_step_history=True)
    history = SimpleNamespace(config_signature=IqnIlsConfig(
        history_limit=config.iqn_history_limit,
        initial_picard_relaxation=config.iqn_initial_picard_relaxation,
        svd_relative_cutoff=config.iqn_svd_relative_cutoff,
    ).signature)
    runner._validate_fsi_checkpoint_iqn_history(history, config)
    changed = SimpleNamespace(config_signature=(*history.config_signature[:3], 100.0, *history.config_signature[4:]))
    with pytest.raises(ValueError, match="IQN.*signature"):
        runner._validate_fsi_checkpoint_iqn_history(changed, config)


def test_checkpoint_retained_iqn_cannot_silently_disappear_when_reuse_disabled():
    history = SimpleNamespace(config_signature=(8, 0.5, 1.0e-10, 1.0e10, None, 2.0))
    with pytest.raises(ValueError, match="IQN.*reuse"):
        runner._validate_fsi_checkpoint_iqn_history(history, _config(iqn_reuse_previous_step_history=False))
    runner._validate_fsi_checkpoint_iqn_history(None, _config(iqn_reuse_previous_step_history=True))


def _config(**changes):
    return SimpleNamespace(**{**asdict(selected_formulation_solver_config(step_count=50)), **changes})


def test_checkpoint_paths_allow_fresh_preflow_and_refuse_implicit_overwrite(tmp_path: Path):
    prefix = tmp_path / "accepted"
    config = _config(fsi_checkpoint_output_path=str(prefix), preflow_snapshot_input_path="preflow")
    assert runner._fsi_checkpoint_paths(config) == (None, prefix.with_suffix(".json"))
    prefix.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="checkpoint"):
        runner._fsi_checkpoint_paths(config)
    resumed = _config(fsi_checkpoint_input_path=str(prefix), step_count=5000)
    assert runner._fsi_checkpoint_paths(resumed) == (prefix.with_suffix(".json"), prefix.with_suffix(".json"))


@pytest.mark.parametrize("changes", [
    {"preflow_snapshot_input_path": "preflow"},
    {"preflow_snapshot_output_path": "preflow"},
    {"fsi_checkpoint_output_path": "unrequested_fork"},
    {"step_count": 0},
    {"iqn_kalman_oracle_interpolation_target_step": 1},
])
def test_checkpoint_paths_reject_ambiguous_startup(tmp_path: Path, changes):
    prefix = tmp_path / "accepted"
    prefix.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        runner._fsi_checkpoint_paths(_config(fsi_checkpoint_input_path=str(prefix), **changes))


def test_checkpoint_paths_reject_export_npz_as_restart(tmp_path: Path):
    with pytest.raises(ValueError, match="npz"):
        runner._fsi_checkpoint_paths(_config(fsi_checkpoint_input_path=str(tmp_path / "step_0001.npz")))


def test_fsi_identity_ignores_only_total_target_and_checkpoint_preflow_paths():
    first = _config(step_count=3, fsi_checkpoint_output_path="state", preflow_snapshot_input_path="preflow")
    second = _config(step_count=5000, fsi_checkpoint_input_path="state.json", preflow_snapshot_input_path=None)
    assert runner._fsi_checkpoint_config_payload(first) == runner._fsi_checkpoint_config_payload(second)


@pytest.mark.parametrize("field,value", [
    ("young_modulus_pa", 2.0e6), ("solid_substeps", 1600),
    ("solid_cfl_target", 0.1), ("dt_s", 0.001),
    ("iqn_history_limit", 4), ("initial_guess_mode", "linear_extrapolation"),
    ("fsi_coupling_relative_tolerance", 0.002),
])
def test_fsi_identity_keeps_physics_and_algorithm_settings(field, value):
    baseline = _config()
    assert runner._fsi_checkpoint_config_payload(baseline) != runner._fsi_checkpoint_config_payload(_config(**{field: value}))


def test_invalid_runner_metadata_cannot_reach_checkpoint_publication(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setattr(runner, "write_accepted_fsi_checkpoint", lambda *a, **k: writes.append(k))
    state = SimpleNamespace(macro_state=SimpleNamespace(accepted_step_index=2), runner_state={})
    with pytest.raises(ValueError, match="runner"):
        runner._commit_accepted_fsi_checkpoint(
            tmp_path / "state", state=state, identity={}, record={},
            previous_tail=None, expected_generation=None,
        )
    assert writes == []


def test_io_failure_reports_memory_and_last_durable_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_validate_fsi_checkpoint_runner_state", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_validate_fsi_checkpoint_records", lambda *a, **k: None)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(runner, "write_accepted_fsi_checkpoint", fail)
    state = SimpleNamespace(macro_state=SimpleNamespace(accepted_step_index=2), runner_state={})
    with pytest.raises(runner.FsiCheckpointCommitError) as failure:
        runner._commit_accepted_fsi_checkpoint(
            tmp_path / "state", state=state, identity={}, record={},
            previous_tail=SimpleNamespace(step=1), expected_generation="previous",
        )
    assert failure.value.diagnostics["in_memory_accepted_step"] == 2
    assert failure.value.diagnostics["durable_accepted_step"] == 1
    assert failure.value.diagnostics["checkpoint_failure_kind"] == "persistence"
    assert isinstance(failure.value.__cause__, OSError)


def _runtime_restore_case(monkeypatch):
    import numpy as np
    events = []
    class Field:
        def __init__(self, value):
            self.value = value
        def to_numpy(self):
            return self.value.copy()
        def from_numpy(self, value):
            events.append("unexpected_field_write")
    vector = np.zeros((2, 3), np.float32)
    scalar = np.zeros((2,), np.float32)
    pressure = np.zeros((2,), np.float64)
    macro = SimpleNamespace(solid_particle_count=2, solid_fields={"v": vector},
        fluid_fields={"velocity": vector, "pressure": pressure}, marker_state={},
        fluid_host_metadata={"sst_wall_distance_valid": False,
                             "hibm_dynamic_solid_volume_enabled": True},
        marker_pressure_neumann_gradient=None)
    state = SimpleNamespace(macro_state=macro,
        fluid_boundary_fields={"last_ledger_field": scalar, "pressure": pressure},
        velocity_boundary_authority="canonical", ledger_generation=8)
    fluid = SimpleNamespace(velocity=Field(vector), last_ledger_field=Field(scalar),
                            pressure=Field(pressure), pressure_tmp=Field(pressure),
                            velocity_dirichlet_boundary_authority="canonical",
                            _sst_wall_distance_valid=True,
                            hibm_dynamic_solid_volume_enabled=False)
    solid = SimpleNamespace(particle_count=2, v=Field(vector))
    monkeypatch.setattr(runner, "validate_accepted_fsi_state", lambda state: None)
    monkeypatch.setattr(runner, "validate_marker_interface_state", lambda *a: events.append("marker_preflight"))
    monkeypatch.setattr(runner, "_canonical_snapshot_restore_prepare_plan", lambda *a: events.append("ledger_preflight"))
    monkeypatch.setattr(runner, "restore_host_macro_step_state", lambda *a, **k: events.append("macro_restore"))
    monkeypatch.setattr(runner, "_restore_preflow_snapshot_fields", lambda *a, **k: events.append("ledger_restore"))
    return state, fluid, solid, events


def test_complete_restore_installs_ledger_after_macro_rollback(monkeypatch):
    state, fluid, solid, events = _runtime_restore_case(monkeypatch)
    runner._restore_accepted_fsi_runtime_state(state, fluid=fluid, solid=solid,
                                             markers=object(), gradient_field=None)
    assert events == ["marker_preflight", "ledger_preflight", "macro_restore", "ledger_restore"]
    assert fluid._sst_wall_distance_valid is False
    assert fluid.hibm_dynamic_solid_volume_enabled is True


def test_bad_last_runtime_destination_has_zero_restore_writes(monkeypatch):
    import numpy as np
    state, fluid, solid, events = _runtime_restore_case(monkeypatch)
    fluid.last_ledger_field.value = np.zeros((3,), np.float32)
    with pytest.raises(ValueError, match="shape/dtype"):
        runner._restore_accepted_fsi_runtime_state(state, fluid=fluid, solid=solid,
                                                 markers=object(), gradient_field=None)
    assert events == []


def test_checkpoint_observer_requires_explicit_idempotent_destination():
    def observer(*args):
        raise AssertionError("must not run")
    with pytest.raises(ValueError, match="replay"):
        runner._fsi_checkpoint_observer_identity(observer)
    observer.checkpoint_replay_safe = True
    with pytest.raises(ValueError, match="identity"):
        runner._fsi_checkpoint_observer_identity(observer)
    observer.checkpoint_identity = "accepted-output-directory"
    assert runner._fsi_checkpoint_observer_identity(observer) == observer.checkpoint_identity
    assert runner._fsi_checkpoint_observer_identity(None) is None


def test_raw_iqn_trial_export_with_checkpoint_reaches_runtime_build(monkeypatch, tmp_path: Path):
    class RuntimeBuildReached(RuntimeError):
        pass

    builds = []
    def stop_before_runtime(*args, **kwargs):
        builds.append((args, kwargs))
        raise RuntimeBuildReached
    monkeypatch.setattr(runner, "_build_fluid", stop_before_runtime)

    def observer(*args):
        raise AssertionError("observer must not run")
    observer.checkpoint_replay_safe = True
    observer.checkpoint_identity = "accepted-output-directory"
    observer.record_iqn_trial_vectors = True

    checkpointed = _config(coupling_mode="iqn_ils", fsi_checkpoint_output_path=str(tmp_path / "accepted"))
    with pytest.raises(RuntimeBuildReached):
        runner.run_hibm_mpm_fsi(
            case_id="host-test", case_metadata={}, boundary_conditions={},
            reference_results={}, config=checkpointed, step_observer=observer,
        )
    assert len(builds) == 1

    observer.record_iqn_trial_vectors = False
    with pytest.raises(RuntimeBuildReached):
        runner.run_hibm_mpm_fsi(
            case_id="host-test", case_metadata={}, boundary_conditions={},
            reference_results={}, config=checkpointed, step_observer=observer,
        )
    assert len(builds) == 2

    observer.record_iqn_trial_vectors = True
    standalone = _config(coupling_mode="iqn_ils")
    with pytest.raises(RuntimeBuildReached):
        runner.run_hibm_mpm_fsi(
            case_id="host-test", case_metadata={}, boundary_conditions={},
            reference_results={}, config=standalone, step_observer=observer,
        )
    assert len(builds) == 3


def test_checkpoint_outbox_replays_accepted_step_before_advancing():
    import numpy as np
    delivered = []
    def observer(*args):
        delivered.append(args)
    observer.checkpoint_replay_safe = True
    observer.checkpoint_identity = "destination"
    payload = {"step": 42, "time_s": 0.021, "history_row": {"step": 42},
               "snapshot": {"pressure": np.asarray([1.0])}}
    state = {"observer_identity": "destination", "observer_outbox": payload}
    runner._replay_fsi_checkpoint_observer(state, observer)
    assert len(delivered) == 1
    assert delivered[0][:3] == (42, 0.021, {"step": 42})
    np.testing.assert_array_equal(delivered[0][3]["pressure"], payload["snapshot"]["pressure"])
    observer.checkpoint_identity = "different-directory"
    with pytest.raises(ValueError, match="identity"):
        runner._replay_fsi_checkpoint_observer(state, observer)
    assert len(delivered) == 1


@pytest.mark.parametrize("damage", ["pressure_tmp_shape", "pressure_tmp_dtype", "derived_api"])
def test_auxiliary_restore_failure_precedes_any_macro_write(monkeypatch, damage):
    import numpy as np
    state, fluid, solid, events = _runtime_restore_case(monkeypatch)
    if damage == "pressure_tmp_shape":
        fluid.pressure_tmp.value = np.zeros((3,), np.float64)
    elif damage == "pressure_tmp_dtype":
        fluid.pressure_tmp.value = np.zeros((2,), np.float32)
    else:
        fluid.hibm_no_slip_component_face_valid_mask = object()
    with pytest.raises(ValueError, match="pressure_tmp|derived"):
        runner._restore_accepted_fsi_runtime_state(state, fluid=fluid, solid=solid,
                                                 markers=object(), gradient_field=None)
    assert "macro_restore" not in events
    assert "ledger_restore" not in events
    assert "unexpected_field_write" not in events
