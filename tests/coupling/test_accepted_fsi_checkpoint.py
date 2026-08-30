from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simulation_core.coupling.accepted_fsi_checkpoint import (
    AcceptedFsiState,
    load_accepted_fsi_checkpoint,
    write_accepted_fsi_checkpoint,
)
from simulation_core.coupling.hibm_mpm.macro_step_state import (
    FLUID_MACRO_STATE_FIELDS,
    SOLID_MACRO_STATE_FIELDS,
    HostMacroStepState,
)
from simulation_core.fluids.preflow_snapshot import PREFLOW_SNAPSHOT_FIELD_NAMES


IDENTITY = {name: letter * 64 for name, letter in (
    ("config_sha256", "a"), ("source_sha256", "b"), ("geometry_sha256", "c"),
)}


def _fluid_fields():
    shape = (2, 2, 2)
    vectors = {"velocity", "velocity_prev", "velocity_dirichlet_boundary_value_mps",
               "velocity_dirichlet_boundary_pressure_mobility",
               "velocity_dirichlet_boundary_component_enforcement_weight",
               "velocity_dirichlet_boundary_component_region_id",
               "external_velocity_boundary_x_face_value_mps",
               "external_velocity_boundary_y_face_value_mps",
               "external_velocity_boundary_z_face_value_mps"}
    floating = {"pressure", "fsi_pressure", "velocity_dirichlet_boundary_projection_weight",
                "velocity_dirichlet_boundary_enforcement_weight", "volume_source_s"}
    result = {}
    for name in set(PREFLOW_SNAPSHOT_FIELD_NAMES) | set(FLUID_MACRO_STATE_FIELDS):
        dtype = np.float32
        if name in {"pressure", "fsi_pressure"}:
            dtype = np.float64
        elif "region_id" in name:
            dtype = np.int32
        elif name not in floating and name not in vectors and not name.startswith("sst_") and not name.endswith("_value_mps"):
            dtype = np.int32
        result[name] = np.zeros(shape + ((3,) if name in vectors else ()), dtype=dtype)
        if "region_id" in name:
            result[name].fill(-1)
        elif name in {"velocity_dirichlet_boundary_pressure_mobility", "sst_specific_dissipation_rate", "sst_wall_distance_m"}:
            result[name].fill(1.0)
    return result


def _state(step=1):
    fields = _fluid_fields()
    solid = {
        name: np.zeros(
            (2, 3, 3) if name in {"C", "F"} else (2, 3),
            np.float64 if name == "F" else np.float32,
        )
        for name in SOLID_MACRO_STATE_FIELDS
    }
    solid["F"][:] = np.eye(3, dtype=np.float64)
    marker = {name: np.zeros((2,) if name == "A_gamma_m2" else (2, 3), np.float32)
              for name in ("x_gamma_m", "pressure_probe_origin_m", "v_gamma_mps", "n_gamma", "A_gamma_m2")}
    marker["_marker_geometry"] = {
        "marker_count": 2, "projection_vertex_count": 2,
        "projection_triangle_count": 0, "projection_segment_count": 1,
        "open_ribbon_tip_cap_binding": None,
    }
    macro = HostMacroStepState(
        accepted_step_index=step, accepted_time_s=step * 0.1,
        feedback_available_for_projection=True,
        fluid_fields={name: fields[name] for name in FLUID_MACRO_STATE_FIELDS},
        fluid_host_metadata={
            "sst_wall_distance_valid": False, "sst_wall_distance_cache_key": None,
            "sst_no_slip_domain_walls": (False,) * 6,
            "sst_no_slip_domain_wall_mask": 0, "hibm_dynamic_solid_volume_enabled": False,
        },
        solid_fields=solid, solid_particle_count=2, marker_state=marker,
        marker_count=2, marker_projection_vertex_count=2,
        marker_pressure_neumann_gradient=None,
    )
    return AcceptedFsiState(
        macro_state=macro,
        fluid_boundary_fields={name: fields[name] for name in PREFLOW_SNAPSHOT_FIELD_NAMES},
        velocity_boundary_authority="canonical", ledger_generation=step + 3,
        marker_reference_positions_m=None, initial_guess_state=None,
        kalman_state=None, iqn_history=None,
        runner_state={"dt_s": 0.1, "coupling_mode": "direct_explicit", "diagnostic": float("nan")},
    )


def _record(step):
    return {"history_row": {"step": step, "time_s": step * 0.1},
            "coupling_step_reports": [{"accepted_step": step}]}


def test_full_physical_state_round_trip_and_contiguous_resume(tmp_path: Path):
    path = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY,
                                         record=_record(1))
    second_state = _state(2)
    expected_f = second_state.macro_state.solid_fields["F"].copy()
    expected_f[0, 0, 0] = 1.0 + np.ldexp(1.0, -35)
    expected_f[0, 0, 1] = np.ldexp(1.0, -35)
    second_state = replace(
        second_state,
        macro_state=replace(
            second_state.macro_state,
            solid_fields={**second_state.macro_state.solid_fields, "F": expected_f},
        ),
    )
    second = write_accepted_fsi_checkpoint(
        path, state=second_state, identity=IDENTITY, record=_record(2),
        previous_tail=first.history_tail, expected_generation=first.generation,
    )
    loaded = load_accepted_fsi_checkpoint(path, expected_identity=IDENTITY, target_step_count=5)
    assert loaded.generation == second.generation
    assert loaded.state.macro_state.accepted_step_index == 2
    assert len(loaded.records) == 2
    assert [r["history_row"]["step"] for r in loaded.records] == [1, 2]
    assert np.isnan(loaded.state.runner_state["diagnostic"])
    assert set(loaded.state.macro_state.solid_fields) == set(SOLID_MACRO_STATE_FIELDS)
    assert loaded.state.macro_state.solid_fields["F"].dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(loaded.state.macro_state.solid_fields["F"], expected_f)
    assert not loaded.state.macro_state.solid_fields["F"].flags.writeable
    # Reaching the same total target is a valid no-op, not another physical step.
    assert load_accepted_fsi_checkpoint(path, expected_identity=IDENTITY, target_step_count=2).state.macro_state.accepted_time_s == 0.2


def test_loader_rejects_expected_generation_mismatch_before_state_decode(tmp_path: Path):
    path = tmp_path / "accepted"
    committed = write_accepted_fsi_checkpoint(
        path, state=_state(), identity=IDENTITY, record=_record(1)
    )
    with pytest.raises(ValueError, match="generation mismatch"):
        load_accepted_fsi_checkpoint(
            path,
            expected_identity=IDENTITY,
            target_step_count=2,
            expected_generation="f" * 32,
        )
    assert load_accepted_fsi_checkpoint(
        path, expected_identity=IDENTITY, target_step_count=2,
        expected_generation=committed.generation,
    ).generation == committed.generation


def test_rejects_legacy_f32_f_before_checkpoint_publication(tmp_path: Path):
    path = tmp_path / "accepted"
    state = _state()
    bad_fields = dict(state.macro_state.solid_fields)
    bad_fields["F"] = bad_fields["F"].astype(np.float32)
    bad_state = replace(
        state,
        macro_state=replace(state.macro_state, solid_fields=bad_fields),
    )

    with pytest.raises(ValueError, match="solid F shape/dtype is invalid"):
        write_accepted_fsi_checkpoint(
            path, state=bad_state, identity=IDENTITY, record=_record(1)
        )

    assert not path.exists()


@pytest.mark.parametrize("field", ["config_sha256", "source_sha256", "geometry_sha256"])
def test_resume_rejects_changed_identity(tmp_path: Path, field):
    path = tmp_path / "accepted"
    write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY, record=_record(1))
    with pytest.raises(ValueError, match="identity"):
        load_accepted_fsi_checkpoint(path, expected_identity={**IDENTITY, field: "d" * 64}, target_step_count=5)


def test_reject_time_drift_missing_state_and_inconsistent_duplicate_fields(tmp_path: Path):
    path = tmp_path / "accepted"
    original = _state()
    bad_states = [
        replace(original, macro_state=replace(original.macro_state, accepted_time_s=0.099)),
        replace(original, macro_state=replace(original.macro_state, solid_fields={"x": original.macro_state.solid_fields["x"]})),
        replace(original, fluid_boundary_fields={**original.fluid_boundary_fields, "velocity": np.ones((2, 2, 2, 3), np.float32)}),
        replace(original, ledger_generation=True),
    ]
    for state in bad_states:
        with pytest.raises((TypeError, ValueError)):
            write_accepted_fsi_checkpoint(path, state=state, identity=IDENTITY, record=_record(1))
    assert not path.with_suffix(".json").exists()


def test_reject_shorter_target_and_physical_nonfinite_without_losing_previous(tmp_path: Path):
    path = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY, record=_record(1))
    bad = _state(2)
    bad.macro_state.solid_fields["C"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        write_accepted_fsi_checkpoint(path, state=bad, identity=IDENTITY, record=_record(2),
                                     previous_tail=first.history_tail, expected_generation=first.generation)
    assert load_accepted_fsi_checkpoint(path, expected_identity=IDENTITY, target_step_count=5).generation == first.generation
    with pytest.raises(ValueError, match="target"):
        load_accepted_fsi_checkpoint(path, expected_identity=IDENTITY, target_step_count=0)


@pytest.mark.parametrize("damage", [
    "missing_fluid_metadata", "string_wall_flag", "wall_mask_mismatch",
    "missing_geometry_key", "boolean_geometry_count", "missing_cap_binding",
    "bad_cap_binding", "wrong_controller_shape",
])
def test_incomplete_metadata_cannot_replace_accepted_commit(tmp_path: Path, damage):
    path = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY,
                                         record=_record(1))
    manifest = path.with_suffix(".json").read_bytes()
    bad = _state(2)
    metadata = dict(bad.macro_state.fluid_host_metadata)
    geometry = dict(bad.macro_state.marker_state["_marker_geometry"])
    if damage == "missing_fluid_metadata":
        metadata = {}
    elif damage == "string_wall_flag":
        metadata["sst_no_slip_domain_walls"] = ("false",) + (False,) * 5
    elif damage == "wall_mask_mismatch":
        metadata["sst_no_slip_domain_wall_mask"] = 1
    elif damage == "missing_geometry_key":
        del geometry["projection_segment_count"]
    elif damage == "boolean_geometry_count":
        geometry["projection_triangle_count"] = True
    elif damage == "missing_cap_binding":
        geometry["projection_vertex_count"] = 6
        bad = replace(bad, macro_state=replace(bad.macro_state, marker_projection_vertex_count=6))
    elif damage == "bad_cap_binding":
        geometry["open_ribbon_tip_cap_binding"] = (0,)
    else:
        from simulation_core.coupling.interface_initial_guess_controller import InterfaceInitialGuessController
        controller = InterfaceInitialGuessController("carry_forward")
        for _ in range(2):
            controller.begin_step(np.zeros((1, 3)), layout_id="test", dt_s=0.1)
            controller.accept_step(np.zeros((1, 3)), layout_id="test")
        bad = replace(bad, initial_guess_state=controller.snapshot(),
                      marker_reference_positions_m=np.zeros((2, 3)),
                      runner_state={"dt_s": 0.1, "coupling_mode": "iqn_ils"})
    bad = replace(bad, macro_state=replace(bad.macro_state,
        fluid_host_metadata=metadata,
        marker_state={**bad.macro_state.marker_state, "_marker_geometry": geometry}))
    with pytest.raises((TypeError, ValueError)):
        write_accepted_fsi_checkpoint(path, state=bad, identity=IDENTITY, record=_record(2),
            previous_tail=first.history_tail, expected_generation=first.generation)
    assert path.with_suffix(".json").read_bytes() == manifest
    assert load_accepted_fsi_checkpoint(path, expected_identity=IDENTITY, target_step_count=3).generation == first.generation


@pytest.mark.parametrize("field", ["config_sha256", "source_sha256", "geometry_sha256"])
def test_new_identity_cannot_append_to_existing_accepted_history(tmp_path: Path, field):
    path = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY,
                                         record=_record(1))
    manifest = path.with_suffix(".json").read_bytes()
    history_files = set(tmp_path.glob("*.history.*.json"))
    with pytest.raises(ValueError, match="identity"):
        write_accepted_fsi_checkpoint(path, state=_state(2),
            identity={**IDENTITY, field: "d" * 64}, record=_record(2),
            previous_tail=first.history_tail, expected_generation=first.generation)
    assert path.with_suffix(".json").read_bytes() == manifest
    assert set(tmp_path.glob("*.history.*.json")) == history_files


@pytest.mark.parametrize("damage", ["missing_generation", "missing_tail", "stale_generation"])
def test_continuation_requires_exact_current_parent_before_journal_append(tmp_path: Path, damage):
    path = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY,
                                         record=_record(1))
    manifest = path.with_suffix(".json").read_bytes()
    history_files = set(tmp_path.glob("*.history.*.json"))
    generation = None if damage == "missing_generation" else ("e" * 32 if damage == "stale_generation" else first.generation)
    tail = None if damage == "missing_tail" else first.history_tail
    with pytest.raises(ValueError):
        write_accepted_fsi_checkpoint(path, state=_state(2), identity=IDENTITY,
            record=_record(2), previous_tail=tail, expected_generation=generation)
    assert path.with_suffix(".json").read_bytes() == manifest
    assert set(tmp_path.glob("*.history.*.json")) == history_files


@pytest.mark.parametrize("initial_mode", ["carry_forward", "linear_extrapolation", "kalman", "oracle_replay"])
@pytest.mark.parametrize("active", [False, True])
def test_disk_round_trip_preserves_real_controller_predictions(tmp_path: Path, initial_mode, active):
    from simulation_core.coupling.interface_initial_guess_controller import InterfaceInitialGuessController
    from simulation_core.coupling.interface_kalman_predictor import InterfaceKalmanConfig
    from simulation_core.coupling.active_kalman_writeback import ActiveKalmanWritebackController
    from simulation_core.coupling.iqn_ils import IqnIlsConfig, IqnIlsSecantHistory

    kalman_config = InterfaceKalmanConfig(
        rate_process_noise_spectral_density=0.1, measurement_variance=0.2,
        initial_value_variance=0.3, initial_rate_variance=0.4,
    )
    options = {}
    if initial_mode == "kalman":
        options["kalman_config"] = kalman_config
    if initial_mode == "oracle_replay":
        options["oracle_replay"] = (np.zeros((2, 3)), np.ones((2, 3)))
    controller = InterfaceInitialGuessController(initial_mode, **options)
    controller.begin_step(np.zeros((2, 3)), dt_s=0.1, layout_id="layout")
    controller.accept_step(np.full((2, 3), 0.5), layout_id="layout")
    active_controller = None
    observations = {
        "interface_marker_velocity": np.zeros((2, 3)),
        "fluid_fsi_pressure_feedback": np.zeros((2, 2, 2)),
        "solid_particle_velocity": np.zeros((2, 3)),
    }
    if active:
        active_controller = ActiveKalmanWritebackController(
            "global", {owner: kalman_config for owner in observations}, observations,
        )
        for owner, values in active_controller.begin_step(0.1).items():
            active_controller.observe(owner, np.ones_like(values))
        active_controller.commit_step()
    secants = IqnIlsSecantHistory(
        delta_residual=np.ones((6, 1)), delta_candidate=np.full((6, 1), 2.0),
        source_step=1, layout_id="layout", dt_s=0.1, marker_shape=(2, 3),
        config_signature=IqnIlsConfig().signature, terminal_residual_norm=0.01,
    )
    state = replace(_state(), initial_guess_state=controller.snapshot(),
        kalman_state=None if active_controller is None else active_controller.snapshot(),
        iqn_history=secants, marker_reference_positions_m=np.zeros((2, 3), np.float32),
        runner_state={"dt_s": 0.1, "coupling_mode": "iqn_ils"})
    path = tmp_path / "accepted"
    write_accepted_fsi_checkpoint(path, state=state, identity=IDENTITY, record=_record(1))
    loaded = load_accepted_fsi_checkpoint(path, expected_identity=IDENTITY, target_step_count=2)
    restored = InterfaceInitialGuessController(initial_mode, **options)
    restored.restore(loaded.state.initial_guess_state)
    np.testing.assert_array_equal(
        restored.begin_step(np.full((2, 3), 0.5), dt_s=0.1, layout_id="layout"),
        controller.begin_step(np.full((2, 3), 0.5), dt_s=0.1, layout_id="layout"),
    )
    np.testing.assert_array_equal(loaded.state.iqn_history.delta_residual, secants.delta_residual)
    if active:
        restored_active = ActiveKalmanWritebackController(
            "global", {owner: kalman_config for owner in observations}, observations,
        )
        restored_active.restore(loaded.state.kalman_state)
        expected = active_controller.begin_step(0.1)
        actual = restored_active.begin_step(0.1)
        for owner in observations:
            np.testing.assert_array_equal(actual[owner], expected[owner])


def test_finite_but_invalid_nested_snapshot_cannot_replace_commit(tmp_path: Path):
    from simulation_core.coupling.interface_initial_guess_controller import InterfaceInitialGuessController
    path = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(path, state=_state(), identity=IDENTITY, record=_record(1))
    manifest = path.with_suffix(".json").read_bytes()
    controller = InterfaceInitialGuessController("carry_forward")
    for _ in range(2):
        controller.begin_step(np.zeros((2, 3)), dt_s=0.1, layout_id="layout")
        controller.accept_step(np.zeros((2, 3)), layout_id="layout")
    snapshot = controller.snapshot()
    # Deliberately emulate a corrupted in-memory object after construction.
    object.__setattr__(snapshot, "begin_count", 9)
    state = replace(_state(2), initial_guess_state=snapshot,
        marker_reference_positions_m=np.zeros((2, 3), np.float32),
        runner_state={"dt_s": 0.1, "coupling_mode": "iqn_ils"})
    with pytest.raises(ValueError, match="counters"):
        write_accepted_fsi_checkpoint(path, state=state, identity=IDENTITY, record=_record(2),
            previous_tail=first.history_tail, expected_generation=first.generation)
    assert path.with_suffix(".json").read_bytes() == manifest
