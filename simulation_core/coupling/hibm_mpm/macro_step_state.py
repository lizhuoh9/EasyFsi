from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np

from .interface_state import (
    capture_marker_interface_state,
    restore_marker_interface_state,
    validate_marker_interface_state,
)


FLUID_MACRO_STATE_FIELDS = (
    "velocity",
    "velocity_prev",
    "pressure",
    "fsi_pressure",
    "sst_turbulent_kinetic_energy",
    "sst_specific_dissipation_rate",
    "sst_eddy_viscosity_pa_s",
    "sst_wall_distance_m",
    "obstacle",
    "hibm_air_cell",
    "hibm_dynamic_solid_volume_obstacle",
    "hibm_dynamic_solid_volume_external_carve",
    "hibm_fresh_fluid_cell",
    "volume_source_s",
    "external_velocity_boundary_x_face_active_component_mask",
    "external_velocity_boundary_x_face_value_mps",
    "external_velocity_boundary_y_face_active_component_mask",
    "external_velocity_boundary_y_face_value_mps",
    "external_velocity_boundary_z_face_active_component_mask",
    "external_velocity_boundary_z_face_value_mps",
)

SOLID_MACRO_STATE_FIELDS = (
    "x",
    "position_increment_residual_m",
    "v",
    "C",
    "F",
)


@dataclass(frozen=True)
class HostMacroStepState:
    accepted_step_index: int
    accepted_time_s: float
    feedback_available_for_projection: bool
    fluid_fields: Mapping[str, np.ndarray]
    fluid_host_metadata: Mapping[str, Any]
    solid_fields: Mapping[str, np.ndarray]
    solid_particle_count: int
    marker_state: Mapping[str, Any]
    marker_count: int
    marker_projection_vertex_count: int
    marker_pressure_neumann_gradient: np.ndarray | None


def capture_host_macro_step_state(
    *,
    fluid: Any,
    solid: Any,
    markers: Any,
    accepted_step_index: int,
    accepted_time_s: float,
    feedback_available_for_projection: bool,
    marker_pressure_neumann_gradient_field: Any | None = None,
) -> HostMacroStepState:
    step_index = _non_negative_integer(
        accepted_step_index,
        name="accepted_step_index",
    )
    time_s = float(accepted_time_s)
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("accepted_time_s must be finite and non-negative")

    fluid_fields = _capture_owner_fields(
        fluid,
        FLUID_MACRO_STATE_FIELDS,
        owner_name="fluid",
    )
    solid_fields = _capture_owner_fields(
        solid,
        SOLID_MACRO_STATE_FIELDS,
        owner_name="solid",
    )
    particle_count = _positive_integer(
        getattr(solid, "particle_count", 0),
        name="solid particle_count",
    )
    marker_state = capture_marker_interface_state(markers)
    marker_count = _positive_integer(
        getattr(markers, "marker_count", 0),
        name="marker_count",
    )
    projection_vertex_count = _positive_integer(
        getattr(markers, "projection_vertex_count", marker_count),
        name="projection_vertex_count",
    )
    if projection_vertex_count < marker_count:
        raise ValueError("projection_vertex_count cannot be smaller than marker_count")
    marker_gradient = _capture_marker_gradient(
        marker_pressure_neumann_gradient_field,
        projection_vertex_count=projection_vertex_count,
    )
    return HostMacroStepState(
        accepted_step_index=step_index,
        accepted_time_s=time_s,
        feedback_available_for_projection=bool(
            feedback_available_for_projection
        ),
        fluid_fields=fluid_fields,
        fluid_host_metadata=_capture_fluid_host_metadata(fluid),
        solid_fields=solid_fields,
        solid_particle_count=particle_count,
        marker_state=marker_state,
        marker_count=marker_count,
        marker_projection_vertex_count=projection_vertex_count,
        marker_pressure_neumann_gradient=marker_gradient,
    )


def restore_host_macro_step_state(
    state: HostMacroStepState,
    *,
    fluid: Any,
    solid: Any,
    markers: Any,
    marker_pressure_neumann_gradient_field: Any | None = None,
    record_particle_position_write: Callable[[], None] | None = None,
) -> None:
    if not isinstance(state, HostMacroStepState):
        raise TypeError("state must be a HostMacroStepState")
    if int(getattr(solid, "particle_count", 0)) != state.solid_particle_count:
        raise ValueError("solid particle_count changed during macro transaction")

    validated_solid = _validated_owner_restore(
        solid,
        state.solid_fields,
        SOLID_MACRO_STATE_FIELDS,
        owner_name="solid",
    )
    validated_fluid = _validated_owner_restore(
        fluid,
        state.fluid_fields,
        FLUID_MACRO_STATE_FIELDS,
        owner_name="fluid",
    )
    validated_fluid_metadata = _validated_fluid_host_metadata_restore(
        state.fluid_host_metadata
    )
    validate_marker_interface_state(markers, state.marker_state)
    validated_gradient = _validated_marker_gradient_restore(
        marker_pressure_neumann_gradient_field,
        state,
    )

    abort_guard = getattr(solid, "abort_out_of_bounds_guard_batch", None)
    if callable(abort_guard):
        abort_guard()
    _write_owner_fields(validated_solid)
    solid.save_state()
    solid.restore_state()
    if record_particle_position_write is not None:
        record_particle_position_write()

    _write_owner_fields(validated_fluid)
    _stage_fluid_host_metadata(fluid, validated_fluid_metadata)
    fluid.save_state()
    fluid.restore_state()
    invalidate_pressure_warmstart = getattr(
        fluid,
        "invalidate_pressure_warmstart",
        None,
    )
    if callable(invalidate_pressure_warmstart):
        invalidate_pressure_warmstart()

    restore_marker_interface_state(markers, state.marker_state)
    if validated_gradient is not None:
        field, full, active = validated_gradient
        full[: state.marker_projection_vertex_count] = active
        field.from_numpy(full)


def _capture_owner_fields(
    owner: Any,
    field_names: tuple[str, ...],
    *,
    owner_name: str,
) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    for name in field_names:
        field = getattr(owner, name)
        array = np.asarray(field.to_numpy())
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{owner_name} field {name!r} must be finite")
        captured[name] = array.copy()
    return captured


def _validated_owner_restore(
    owner: Any,
    captured: Mapping[str, np.ndarray],
    field_names: tuple[str, ...],
    *,
    owner_name: str,
) -> tuple[tuple[Any, np.ndarray], ...]:
    validated: list[tuple[Any, np.ndarray]] = []
    for name in field_names:
        if name not in captured:
            raise ValueError(f"{owner_name} state is missing field {name!r}")
        field = getattr(owner, name)
        current = np.asarray(field.to_numpy())
        array = np.asarray(captured[name])
        if array.shape != current.shape:
            raise ValueError(
                f"{owner_name} field {name!r} shape changed: "
                f"{array.shape} != {current.shape}"
            )
        if array.dtype != current.dtype:
            raise ValueError(
                f"{owner_name} field {name!r} dtype changed: "
                f"{array.dtype} != {current.dtype}"
            )
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{owner_name} field {name!r} must be finite")
        validated.append((field, array.copy()))
    return tuple(validated)


def _write_owner_fields(
    validated: tuple[tuple[Any, np.ndarray], ...],
) -> None:
    for field, array in validated:
        field.from_numpy(array)


def _capture_fluid_host_metadata(fluid: Any) -> dict[str, Any]:
    return {
        "sst_wall_distance_valid": bool(
            getattr(fluid, "_sst_wall_distance_valid")
        ),
        "sst_wall_distance_cache_key": getattr(
            fluid,
            "_sst_wall_distance_cache_key",
        ),
        "sst_no_slip_domain_walls": tuple(
            getattr(fluid, "_sst_no_slip_domain_walls")
        ),
        "sst_no_slip_domain_wall_mask": int(
            fluid.sst_no_slip_domain_wall_mask[None]
        ),
        "hibm_dynamic_solid_volume_enabled": bool(
            getattr(fluid, "hibm_dynamic_solid_volume_enabled")
        ),
    }


def _stage_fluid_host_metadata(
    fluid: Any,
    metadata: Mapping[str, Any],
) -> None:
    fluid._sst_wall_distance_valid = bool(
        metadata["sst_wall_distance_valid"]
    )
    fluid._sst_wall_distance_cache_key = metadata[
        "sst_wall_distance_cache_key"
    ]
    fluid._sst_no_slip_domain_walls = tuple(
        metadata["sst_no_slip_domain_walls"]
    )
    fluid.sst_no_slip_domain_wall_mask[None] = int(
        metadata["sst_no_slip_domain_wall_mask"]
    )
    fluid.hibm_dynamic_solid_volume_enabled = bool(
        metadata["hibm_dynamic_solid_volume_enabled"]
    )


def _validated_fluid_host_metadata_restore(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "sst_wall_distance_valid",
        "sst_wall_distance_cache_key",
        "sst_no_slip_domain_walls",
        "sst_no_slip_domain_wall_mask",
        "hibm_dynamic_solid_volume_enabled",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(
            "fluid metadata is missing "
            + ", ".join(repr(name) for name in missing)
        )
    try:
        walls = tuple(bool(value) for value in metadata["sst_no_slip_domain_walls"])
        wall_mask = int(metadata["sst_no_slip_domain_wall_mask"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fluid metadata contains invalid wall values") from exc
    if len(walls) != 6:
        raise ValueError("fluid metadata must contain six domain-wall flags")
    return {
        "sst_wall_distance_valid": bool(metadata["sst_wall_distance_valid"]),
        "sst_wall_distance_cache_key": metadata["sst_wall_distance_cache_key"],
        "sst_no_slip_domain_walls": walls,
        "sst_no_slip_domain_wall_mask": wall_mask,
        "hibm_dynamic_solid_volume_enabled": bool(
            metadata["hibm_dynamic_solid_volume_enabled"]
        ),
    }


def _capture_marker_gradient(
    field: Any | None,
    *,
    projection_vertex_count: int,
) -> np.ndarray | None:
    if field is None:
        return None
    full = np.asarray(field.to_numpy())
    if full.ndim != 1 or full.shape[0] < projection_vertex_count:
        raise ValueError(
            "marker pressure-Neumann normal-gradient field must have scalar "
            "capacity (marker_capacity,)"
        )
    active = full[:projection_vertex_count]
    if not bool(np.all(np.isfinite(active))):
        raise ValueError("marker pressure-Neumann gradient must be finite")
    return active.copy()


def _validated_marker_gradient_restore(
    field: Any | None,
    state: HostMacroStepState,
) -> tuple[Any, np.ndarray, np.ndarray] | None:
    active = state.marker_pressure_neumann_gradient
    if (active is None) != (field is None):
        raise ValueError(
            "marker pressure-Neumann gradient snapshot/field presence mismatch"
        )
    if active is None:
        return None
    assert field is not None
    full = np.asarray(field.to_numpy())
    expected_shape = full[: state.marker_projection_vertex_count].shape
    if active.shape != expected_shape or active.dtype != full.dtype:
        raise ValueError(
            "marker pressure-Neumann gradient shape or dtype changed"
        )
    if not bool(np.all(np.isfinite(active))):
        raise ValueError("marker pressure-Neumann gradient must be finite")
    return field, full, active.copy()


def _non_negative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _positive_integer(value: Any, *, name: str) -> int:
    integer = _non_negative_integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer
