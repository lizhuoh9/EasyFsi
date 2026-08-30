from __future__ import annotations

from collections.abc import Mapping
import hashlib
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
    refresh_tip_cap = _tip_cap_projection_refresh_callback(
        markers,
        marker_count=count,
    )
    state = _capture_active_marker_fields(markers, marker_count=count)
    if refresh_tip_cap is not None:
        refresh_tip_cap()
        state = _capture_active_marker_fields(markers, marker_count=count)
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
    material_identity = getattr(markers, "material_surface_binding_identity", None)
    if material_identity is not None:
        state[_MARKER_GEOMETRY_STATE_KEY]["material_surface_binding_identity"] = material_identity
    return state


def marker_velocity_state(state: Mapping[str, Any]) -> np.ndarray:
    velocity = _state_array(state, "v_gamma_mps")
    if velocity.ndim != 2 or velocity.shape[0] == 0 or velocity.shape[1] != 3:
        raise ValueError("marker velocity state must have shape (marker_count, 3)")
    return velocity.copy()


def marker_layout_identity(
    markers: Any,
    *,
    reference_positions_m: Any,
    namespace: str = "hibm_mpm_marker_velocity_v1",
) -> str:
    """Fingerprint immutable physical-marker order and projection topology."""

    marker_count = int(getattr(markers, "marker_count", 0))
    projection_vertex_count = int(
        getattr(markers, "projection_vertex_count", marker_count)
    )
    projection_triangle_count = int(
        getattr(markers, "projection_triangle_count", 0)
    )
    projection_segment_count = int(
        getattr(markers, "projection_segment_count", 0)
    )
    if marker_count <= 0 or projection_vertex_count < marker_count:
        raise ValueError("marker layout counts are invalid")
    if projection_triangle_count < 0 or projection_segment_count < 0:
        raise ValueError("marker projection counts must be non-negative")
    reference = np.asarray(reference_positions_m)
    if reference.shape != (marker_count, 3) or not bool(
        np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "reference_positions_m must be finite with shape (marker_count, 3)"
        )
    regions = np.asarray(markers.region_id.to_numpy())
    if regions.ndim != 1 or regions.shape[0] < projection_vertex_count:
        raise ValueError("marker region layout does not cover projection vertices")
    primitive_count = max(projection_triangle_count, projection_segment_count)
    primitive_indices = np.asarray(markers.projection_triangle_indices.to_numpy())
    if (
        primitive_indices.ndim != 2
        or primitive_indices.shape[0] < primitive_count
        or primitive_indices.shape[1] != 3
    ):
        raise ValueError("marker projection topology has invalid capacity")

    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8"))
    _update_layout_digest(
        digest,
        np.asarray(
            [
                marker_count,
                projection_vertex_count,
                projection_triangle_count,
                projection_segment_count,
                int(getattr(markers, "marker_capacity", regions.shape[0])),
                int(
                    getattr(
                        markers,
                        "projection_triangle_capacity",
                        primitive_indices.shape[0],
                    )
                ),
            ],
            dtype=np.int64,
        ),
    )
    _update_layout_digest(digest, reference)
    _update_layout_digest(digest, regions[:projection_vertex_count])
    _update_layout_digest(digest, primitive_indices[:primitive_count])
    binding = getattr(markers, "_open_ribbon_tip_cap_binding", None)
    digest.update(repr(None if binding is None else tuple(binding)).encode("utf-8"))
    material_identity = getattr(markers, "material_surface_binding_identity", None)
    if material_identity is not None:
        digest.update(b"material_surface_binding_identity")
        digest.update(str(material_identity).encode("ascii"))
    return digest.hexdigest()


def marker_trial_state(
    step_base_state: Mapping[str, Any],
    marker_velocity_guess_mps: Any,
) -> dict[str, Any]:
    base_velocity = marker_velocity_state(step_base_state)
    guess = np.asarray(marker_velocity_guess_mps, dtype=np.float64)
    if guess.shape != base_velocity.shape or not bool(np.all(np.isfinite(guess))):
        raise ValueError("marker velocity guess must match the finite base velocity")
    with np.errstate(over="ignore", invalid="ignore"):
        stored_guess = guess.astype(base_velocity.dtype, copy=False)
    if not bool(np.all(np.isfinite(stored_guess))):
        raise ValueError(
            "marker velocity guess must remain finite in the base velocity dtype"
        )
    trial = {
        name: _state_array(step_base_state, name).copy()
        for name in MARKER_INTERFACE_STATE_FIELDS
    }
    trial["v_gamma_mps"] = stored_guess
    trial[_MARKER_GEOMETRY_STATE_KEY] = dict(
        _marker_geometry_state(step_base_state)
    )
    return trial


def restore_marker_interface_state(
    markers: Any,
    state: Mapping[str, Any],
) -> None:
    geometry, count, refresh_tip_cap, validated = (
        _validated_marker_restore_inputs(markers, state)
    )

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
    if refresh_tip_cap is not None:
        # Projection-only cap vertices are derived from physical markers
        # and are not independent fixed-point unknowns. Rebuild them from
        # the restored physical state so a rejected trial cannot leak its
        # geometry or velocity into the next trial.
        refresh_tip_cap()


def validate_marker_interface_state(
    markers: Any,
    state: Mapping[str, Any],
) -> None:
    _validated_marker_restore_inputs(markers, state)


def _capture_active_marker_fields(
    markers: Any,
    *,
    marker_count: int,
) -> dict[str, np.ndarray]:
    if marker_count <= 0:
        raise ValueError("marker_count must be positive")
    captured: dict[str, np.ndarray] = {}
    for name in MARKER_INTERFACE_STATE_FIELDS:
        full = np.asarray(getattr(markers, name).to_numpy())
        if full.ndim == 0 or full.shape[0] < marker_count:
            raise ValueError(
                f"marker field {name!r} does not have marker_count capacity"
            )
        active = full[:marker_count]
        if not bool(np.all(np.isfinite(active))):
            raise ValueError(f"marker field {name!r} must be finite")
        captured[name] = active.copy()
    return captured


def _validated_marker_restore_inputs(
    markers: Any,
    state: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    int,
    Any | None,
    dict[str, tuple[Any, np.ndarray, np.ndarray]],
]:
    raw_geometry = _marker_geometry_state(state)
    material_identity = getattr(markers, "material_surface_binding_identity", None)
    if raw_geometry.get("material_surface_binding_identity") != material_identity:
        raise ValueError("material surface binding identity differs from restore state")
    try:
        count = int(raw_geometry["marker_count"])
        projection_vertex_count = int(raw_geometry["projection_vertex_count"])
        projection_triangle_count = int(raw_geometry["projection_triangle_count"])
        projection_segment_count = int(raw_geometry["projection_segment_count"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("marker geometry metadata counts must be integers") from exc
    if count <= 0:
        raise ValueError("marker geometry metadata marker_count must be positive")
    if projection_vertex_count < count:
        raise ValueError(
            "marker geometry metadata projection_vertex_count is too small"
        )
    if projection_triangle_count < 0 or projection_segment_count < 0:
        raise ValueError("marker geometry metadata counts must be non-negative")
    geometry = {
        "marker_count": count,
        "projection_vertex_count": projection_vertex_count,
        "projection_triangle_count": projection_triangle_count,
        "projection_segment_count": projection_segment_count,
        "open_ribbon_tip_cap_binding": raw_geometry[
            "open_ribbon_tip_cap_binding"
        ],
    }
    if material_identity is not None:
        transfer = markers._material_surface_transfer
        bound_counts = {
            "marker_count": transfer.marker_count,
            "projection_vertex_count": transfer._projection_vertex_count,
            "projection_triangle_count": 0,
            "projection_segment_count": transfer._projection_segment_count,
        }
        for name, expected in bound_counts.items():
            if geometry[name] != expected:
                raise ValueError("material surface bound topology differs from restore state")
        if geometry["open_ribbon_tip_cap_binding"] != transfer.cap_binding:
            raise ValueError("material surface bound cap differs from restore state")
    refresh_tip_cap = _tip_cap_projection_refresh_callback(
        markers,
        marker_count=count,
        projection_vertex_count=projection_vertex_count,
    )
    validated: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
    for name in MARKER_INTERFACE_STATE_FIELDS:
        field = getattr(markers, name)
        full = np.asarray(field.to_numpy())
        source = _state_array(state, name)
        expected_shape = tuple(full[:count].shape)
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"marker interface state {name!r} shape mismatch: "
                f"{tuple(source.shape)} != {expected_shape}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(source, dtype=full.dtype)
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(
                f"marker interface state {name!r} must remain finite "
                "in the owner dtype"
            )
        if source.dtype != full.dtype:
            raise ValueError(
                f"marker interface state {name!r} dtype mismatch: "
                f"{source.dtype} != {full.dtype}"
            )
        validated[name] = (field, full, array.copy())
    return geometry, count, refresh_tip_cap, validated


def _tip_cap_projection_refresh_callback(
    markers: Any,
    *,
    marker_count: int,
    projection_vertex_count: int | None = None,
) -> Any | None:
    if projection_vertex_count is None:
        projection_vertex_count = int(
            getattr(markers, "projection_vertex_count", marker_count)
        )
    if projection_vertex_count <= marker_count:
        return None
    refresh_tip_cap = getattr(
        markers,
        "refresh_open_ribbon_tip_cap_projection_vertices",
        None,
    )
    if not callable(refresh_tip_cap):
        raise RuntimeError(
            "projection-only tip-cap vertices require callable "
            "markers.refresh_open_ribbon_tip_cap_projection_vertices"
        )
    return refresh_tip_cap


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


def _update_layout_digest(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
