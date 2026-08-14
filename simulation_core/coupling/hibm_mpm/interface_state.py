from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


MARKER_INTERFACE_STATE_FIELDS = (
    "x_gamma_m",
    "pressure_probe_origin_m",
    "v_gamma_mps",
    "n_gamma",
    "A_gamma_m2",
)
_MARKER_GEOMETRY_STATE_KEY = "_marker_geometry"


def capture_marker_interface_state(markers: Any) -> dict[str, Any]:
    count = int(markers.marker_count)
    state: dict[str, Any] = {
        name: np.asarray(getattr(markers, name).to_numpy())[:count].copy()
        for name in MARKER_INTERFACE_STATE_FIELDS
    }
    binding = getattr(markers, "_open_ribbon_tip_cap_binding", None)
    state[_MARKER_GEOMETRY_STATE_KEY] = {
        "marker_count": count,
        "projection_vertex_count": int(
            getattr(markers, "projection_vertex_count", count)
        ),
        "projection_triangle_count": int(
            getattr(markers, "projection_triangle_count", 0)
        ),
        "projection_segment_count": int(
            getattr(markers, "projection_segment_count", 0)
        ),
        "open_ribbon_tip_cap_binding": (
            None if binding is None else tuple(binding)
        ),
    }
    return state


def marker_velocity_state(state: Mapping[str, Any]) -> np.ndarray:
    velocity = _state_array(state, "v_gamma_mps")
    if velocity.ndim != 2 or velocity.shape[0] == 0 or velocity.shape[1] != 3:
        raise ValueError("marker velocity state must have shape (marker_count, 3)")
    return velocity.copy()


def marker_trial_state(
    step_base_state: Mapping[str, Any],
    marker_velocity_guess_mps: Any,
) -> dict[str, Any]:
    base_velocity = marker_velocity_state(step_base_state)
    guess = np.asarray(marker_velocity_guess_mps, dtype=np.float64)
    if guess.shape != base_velocity.shape or not bool(np.all(np.isfinite(guess))):
        raise ValueError("marker velocity guess must match the finite base velocity")
    trial = {
        name: _state_array(step_base_state, name).copy()
        for name in MARKER_INTERFACE_STATE_FIELDS
    }
    trial["v_gamma_mps"] = guess.astype(base_velocity.dtype, copy=False)
    trial[_MARKER_GEOMETRY_STATE_KEY] = dict(
        _marker_geometry_state(step_base_state)
    )
    return trial


def restore_marker_interface_state(
    markers: Any,
    state: Mapping[str, Any],
) -> None:
    geometry = _marker_geometry_state(state)
    count = int(geometry["marker_count"])
    validated: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
    for name in MARKER_INTERFACE_STATE_FIELDS:
        field = getattr(markers, name)
        full = field.to_numpy()
        array = np.asarray(_state_array(state, name), dtype=full.dtype)
        expected_shape = tuple(full[:count].shape)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"marker interface state {name!r} shape mismatch: "
                f"{tuple(array.shape)} != {expected_shape}"
            )
        validated[name] = (field, full, array)

    markers._begin_marker_geometry_write()
    for field, full, array in validated.values():
        full[:count] = array
        field.from_numpy(full)
    markers.marker_count = count
    markers.projection_vertex_count = int(geometry["projection_vertex_count"])
    markers.projection_triangle_count = int(geometry["projection_triangle_count"])
    markers.projection_segment_count = int(geometry["projection_segment_count"])
    markers._open_ribbon_tip_cap_binding = geometry[
        "open_ribbon_tip_cap_binding"
    ]
    if int(markers.projection_vertex_count) > count:
        refresh_tip_cap = getattr(
            markers,
            "_refresh_open_ribbon_tip_cap_projection_vertices",
            None,
        )
        if callable(refresh_tip_cap):
            # Projection-only cap vertices are derived from physical markers
            # and are not independent fixed-point unknowns. Rebuild them from
            # the restored physical state so a rejected trial cannot leak its
            # geometry or velocity into the next trial.
            refresh_tip_cap()


def _marker_geometry_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    geometry = state.get(_MARKER_GEOMETRY_STATE_KEY)
    if not isinstance(geometry, Mapping):
        raise ValueError("marker interface state is missing geometry metadata")
    required = (
        "marker_count",
        "projection_vertex_count",
        "projection_triangle_count",
        "projection_segment_count",
        "open_ribbon_tip_cap_binding",
    )
    missing = [name for name in required if name not in geometry]
    if missing:
        raise ValueError(
            "marker interface state geometry metadata is missing "
            + ", ".join(repr(name) for name in missing)
        )
    return geometry


def _state_array(state: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in state:
        raise ValueError(f"marker interface state is missing {name!r}")
    array = np.asarray(state[name])
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"marker interface state {name!r} must be finite")
    return array
