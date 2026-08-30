"""CPU checkpoint contracts for bound material-surface marker state."""

from dataclasses import replace

import numpy as np
import pytest

from simulation_core.coupling.accepted_fsi_checkpoint import (
    load_accepted_fsi_checkpoint,
    validate_accepted_fsi_state,
    write_accepted_fsi_checkpoint,
)
from simulation_core.coupling.hibm_mpm.interface_state import (
    capture_marker_interface_state,
)
from tests.coupling.test_accepted_fsi_checkpoint import IDENTITY, _record, _state
from tests.solvers.test_material_surface_transfer import _reset, material_surface


def _bound_checkpoint_state(material_surface, step=1):
    """Capture a complete accepted state from the real bound marker fixture."""

    markers, position, velocity, _, _, _, _ = _reset(material_surface)
    particle_positions = position.to_numpy().astype(np.float32, copy=False)
    particle_velocities = velocity.to_numpy().astype(np.float32, copy=False)
    particle_count = particle_positions.shape[0]
    solid_fields = {
        "x": particle_positions.copy(),
        "position_increment_residual_m": np.zeros((particle_count, 3), np.float32),
        "v": particle_velocities.copy(),
        "C": np.zeros((particle_count, 3, 3), np.float32),
        "F": np.broadcast_to(np.eye(3, dtype=np.float64), (particle_count, 3, 3)).copy(),
    }
    base = _state(step)
    marker_state = capture_marker_interface_state(markers)
    macro = replace(
        base.macro_state,
        solid_fields=solid_fields,
        solid_particle_count=particle_count,
        marker_state=marker_state,
        marker_count=markers.marker_count,
        marker_projection_vertex_count=markers.projection_vertex_count,
    )
    return replace(base, macro_state=macro)


def _with_marker_geometry(state, geometry):
    macro = replace(
        state.macro_state,
        marker_state={
            **state.macro_state.marker_state,
            "_marker_geometry": geometry,
        },
    )
    return replace(state, macro_state=macro)


def test_legacy_unbound_marker_geometry_remains_checkpoint_compatible(tmp_path):
    state = _state()

    validate_accepted_fsi_state(state)
    commit = write_accepted_fsi_checkpoint(
        tmp_path / "accepted", state=state, identity=IDENTITY, record=_record(1)
    )
    loaded = load_accepted_fsi_checkpoint(
        tmp_path / "accepted", expected_identity=IDENTITY, target_step_count=1
    )

    assert loaded.generation == commit.generation
    assert set(loaded.state.macro_state.marker_state["_marker_geometry"]) == {
        "marker_count",
        "projection_vertex_count",
        "projection_triangle_count",
        "projection_segment_count",
        "open_ribbon_tip_cap_binding",
    }


def test_bound_material_surface_checkpoint_round_trip_uses_actual_marker_contract(
    tmp_path, material_surface
):
    state = _bound_checkpoint_state(material_surface, step=1)
    markers, position, velocity, _, _, _, _ = _reset(material_surface)

    validate_accepted_fsi_state(state)
    commit = write_accepted_fsi_checkpoint(
        tmp_path / "accepted", state=state, identity=IDENTITY, record=_record(1)
    )
    loaded = load_accepted_fsi_checkpoint(
        tmp_path / "accepted", expected_identity=IDENTITY, target_step_count=1
    )
    validate_accepted_fsi_state(loaded.state)
    markers.validate_accepted_material_surface_state(
        loaded.state.macro_state.marker_state,
        particle_positions_m=position.to_numpy(),
        particle_velocities_mps=velocity.to_numpy(),
    )

    assert loaded.generation == commit.generation
    assert (
        loaded.state.macro_state.marker_state["_marker_geometry"]
        ["material_surface_binding_identity"]
        == markers.material_surface_binding_identity
    )


def test_invalid_bound_material_digest_or_unknown_geometry_key_cannot_replace_parent(
    tmp_path, material_surface
):
    path = tmp_path / "accepted"
    first_state = _bound_checkpoint_state(material_surface, step=1)
    first = write_accepted_fsi_checkpoint(
        path, state=first_state, identity=IDENTITY, record=_record(1)
    )
    manifest = path.with_suffix(".json").read_bytes()
    history_files = set(tmp_path.glob("*.history.*.json"))
    second_state = _bound_checkpoint_state(material_surface, step=2)
    original_geometry = second_state.macro_state.marker_state["_marker_geometry"]
    invalid_values = (None, "", True, "A" * 64, "g" * 64, "a" * 63)
    for value in invalid_values:
        bad = _with_marker_geometry(
            second_state,
            {**original_geometry, "material_surface_binding_identity": value},
        )
        with pytest.raises(ValueError, match="material surface binding identity"):
            write_accepted_fsi_checkpoint(
                path,
                state=bad,
                identity=IDENTITY,
                record=_record(2),
                previous_tail=first.history_tail,
                expected_generation=first.generation,
            )
        assert path.with_suffix(".json").read_bytes() == manifest
        assert set(tmp_path.glob("*.history.*.json")) == history_files
    bad = _with_marker_geometry(
        second_state,
        {**original_geometry, "unexpected": "rejected"},
    )
    with pytest.raises(ValueError, match="marker geometry metadata schema"):
        write_accepted_fsi_checkpoint(
            path,
            state=bad,
            identity=IDENTITY,
            record=_record(2),
            previous_tail=first.history_tail,
            expected_generation=first.generation,
        )
    assert path.with_suffix(".json").read_bytes() == manifest
    assert set(tmp_path.glob("*.history.*.json")) == history_files
    loaded = load_accepted_fsi_checkpoint(
        path, expected_identity=IDENTITY, target_step_count=2
    )
    assert loaded.generation == first.generation
