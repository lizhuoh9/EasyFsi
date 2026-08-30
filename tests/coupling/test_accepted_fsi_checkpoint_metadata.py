"""Complete cap/SST metadata and accepted-time rejection contracts."""

from dataclasses import replace

import numpy as np
import pytest

from simulation_core.coupling.accepted_fsi_checkpoint import (
    load_accepted_fsi_checkpoint,
    write_accepted_fsi_checkpoint,
)
from tests.coupling.test_accepted_fsi_checkpoint import IDENTITY, _record, _state


def _cap_state(step=1):
    base = _state(step)
    marker_state = {
        name: np.ones((4,) if name == "A_gamma_m2" else (4, 3), dtype=np.float32)
        for name in base.macro_state.marker_state if name != "_marker_geometry"
    }
    marker_state["_marker_geometry"] = {
        "marker_count": 4, "projection_vertex_count": 8,
        "projection_triangle_count": 0, "projection_segment_count": 5,
        "open_ribbon_tip_cap_binding": (0, 1, 2, 3, 4, 5, 6, 7, 303, 0, 0.003),
    }
    metadata = {
        **base.macro_state.fluid_host_metadata,
        "sst_wall_distance_valid": True,
        "sst_wall_distance_cache_key": ((False,) * 6, 2, 0, 4, 5, "a" * 64, "b" * 64),
    }
    return replace(base, macro_state=replace(
        base.macro_state, marker_state=marker_state, marker_count=4,
        marker_projection_vertex_count=8, fluid_host_metadata=metadata,
        marker_pressure_neumann_gradient=np.arange(8, dtype=np.float32),
    ))


def test_projection_cap_sst_identity_and_pressure_gradient_survive_disk_round_trip(tmp_path):
    state = _cap_state()
    prefix = tmp_path / "accepted"
    write_accepted_fsi_checkpoint(prefix, state=state, identity=IDENTITY, record=_record(1))
    loaded = load_accepted_fsi_checkpoint(prefix, expected_identity=IDENTITY, target_step_count=5000)
    assert loaded.state.macro_state.marker_count == 4
    assert loaded.state.macro_state.marker_projection_vertex_count == 8
    assert loaded.state.macro_state.marker_state["_marker_geometry"] == state.macro_state.marker_state["_marker_geometry"]
    assert loaded.state.macro_state.fluid_host_metadata == state.macro_state.fluid_host_metadata
    np.testing.assert_array_equal(
        loaded.state.macro_state.marker_pressure_neumann_gradient,
        state.macro_state.marker_pressure_neumann_gradient,
    )


@pytest.mark.parametrize("record", (
    [], {"history_row": []}, {"history_row": {"step": 2}},
    {"history_row": {"step": 0, "time_s": 0.2}},
    {"history_row": {"step": 3, "time_s": 0.2}},
    {"history_row": {"step": True, "time_s": 0.2}},
    {"history_row": {"step": 2, "time_s": True}},
    {"history_row": {"step": 2, "time_s": "0.2"}},
    {"history_row": {"step": 2, "time_s": 0.0}},
    {"history_row": {"step": 2, "time_s": float("nan")}},
    {"history_row": {"step": 2, "time_s": 0.199}},
))
def test_invalid_or_incomplete_history_time_never_changes_durable_parent(tmp_path, record):
    prefix = tmp_path / "accepted"
    first = write_accepted_fsi_checkpoint(prefix, state=_state(), identity=IDENTITY, record=_record(1))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises((TypeError, ValueError)):
        write_accepted_fsi_checkpoint(
            prefix, state=_state(2), identity=IDENTITY, record=record,
            previous_tail=first.history_tail, expected_generation=first.generation,
        )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert load_accepted_fsi_checkpoint(
        prefix, expected_identity=IDENTITY, target_step_count=2,
    ).generation == first.generation
