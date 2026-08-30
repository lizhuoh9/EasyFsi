"""Complete accepted-boundary FSI persistence; no runtime field writes here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np

from simulation_core.diagnostics.checkpoint_codec import CheckpointStateCodec
from simulation_core.diagnostics.checkpoint_store import (
    HistoryTail, append_history, load_checkpoint, read_checkpoint_head, save_checkpoint,
)
from simulation_core.fluids.preflow_snapshot import validate_preflow_snapshot_fields
from .active_kalman_writeback import (
    ActiveKalmanOwnerMetricsSnapshot, ActiveKalmanWritebackSnapshot,
)
from .interface_initial_guess_controller import InterfaceInitialGuessSnapshot
from .interface_kalman_predictor import InterfaceKalmanConfig, InterfaceKalmanSnapshot
from .iqn_ils import IqnIlsSecantHistory
from .hibm_mpm.macro_step_state import (
    FLUID_MACRO_STATE_FIELDS, SOLID_MACRO_STATE_FIELDS, HostMacroStepState,
)
from .hibm_mpm.interface_state import MARKER_INTERFACE_STATE_FIELDS


@dataclass(frozen=True)
class AcceptedFsiState:
    macro_state: HostMacroStepState
    fluid_boundary_fields: Mapping[str, np.ndarray]
    velocity_boundary_authority: str
    ledger_generation: int
    marker_reference_positions_m: np.ndarray | None
    initial_guess_state: InterfaceInitialGuessSnapshot | None
    kalman_state: ActiveKalmanWritebackSnapshot | None
    iqn_history: IqnIlsSecantHistory | None
    runner_state: Mapping[str, object]


@dataclass(frozen=True)
class AcceptedFsiCommit:
    generation: str
    history_tail: HistoryTail


@dataclass(frozen=True)
class LoadedAcceptedFsiCheckpoint:
    state: AcceptedFsiState
    records: tuple[Mapping[str, object], ...]
    generation: str
    history_tail: HistoryTail


def _state_codec() -> CheckpointStateCodec:
    # Explicit constructor whitelist, never imports named by a saved payload.
    from .hibm_mpm import reports
    from .hibm_mpm.marker_mac_constraint import (
        HibmMpmMarkerMacConstraintReport, HibmMpmMarkerPressureNullspaceReport,
    )
    from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmReport

    classes = (
        AcceptedFsiState, HostMacroStepState, InterfaceInitialGuessSnapshot,
        ActiveKalmanWritebackSnapshot, ActiveKalmanOwnerMetricsSnapshot,
        InterfaceKalmanConfig, InterfaceKalmanSnapshot, IqnIlsSecantHistory,
        reports.HibmMpmSurfaceMarkerForceReport,
        reports.HibmMpmFluidStressSampleReport,
        reports.HibmMpmNoSlipResidualReport,
        reports.HibmMpmMpmForceScatterReport,
        reports.HibmMpmExternalForceClearReport,
        reports.HibmMpmSurfaceUpdateReport,
        reports.HibmMpmIbNodeSearchReport,
        reports.HibmMpmIbBoundaryConditionReport,
        reports.HibmMpmPressureDisconnectedRegionReport,
        reports.HibmMpmPressureNeumannMatrixReport,
        reports.HibmMpmPressureNeumannGradientReport,
        reports.HibmMpmSharpFluidToMpmLoadReport,
        reports.HibmMpmSharpMpmStepReport,
        reports.HibmMpmSharpNeoHookeanStepReport,
        HibmMpmMarkerMacConstraintReport, HibmMpmMarkerPressureNullspaceReport,
        NeoHookeanMpmReport,
    )
    # Diagnostic reports contain explicit unavailable NaN sentinels. Physical
    # arrays remain finite-only; physical scalars are checked separately below.
    # Snapshot stage/hash tags are zero-dimensional Unicode metadata, not
    # physical arrays. The codec preserves them in explicitly tagged JSON.
    return CheckpointStateCodec(
        {cls.__name__: cls for cls in classes}, allow_nonfinite_scalars=True,
        allow_unicode_scalars=True,
    )


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-boolean integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _identity(value: Mapping[str, object]) -> dict[str, str]:
    required = {"config_sha256", "source_sha256", "geometry_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("checkpoint identity has invalid fields")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
           for v in value.values()):
        raise ValueError("checkpoint identity contains invalid SHA256")
    return dict(value)


def _finite_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite numeric array")
    return array


def _validate_fluid_metadata(metadata: object) -> None:
    required = {
        "sst_wall_distance_valid", "sst_wall_distance_cache_key",
        "sst_no_slip_domain_walls", "sst_no_slip_domain_wall_mask",
        "hibm_dynamic_solid_volume_enabled",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("fluid host metadata schema is incomplete")
    for name in ("sst_wall_distance_valid", "hibm_dynamic_solid_volume_enabled"):
        if type(metadata[name]) is not bool:
            raise TypeError(f"fluid metadata {name} must be boolean")
    walls = metadata["sst_no_slip_domain_walls"]
    if not isinstance(walls, tuple) or len(walls) != 6 or any(type(v) is not bool for v in walls):
        raise ValueError("fluid metadata must contain six boolean wall flags")
    mask = _integer(metadata["sst_no_slip_domain_wall_mask"], "wall mask")
    if mask != sum(1 << index for index, active in enumerate(walls) if active):
        raise ValueError("fluid wall mask differs from domain-wall flags")
    key = metadata["sst_wall_distance_cache_key"]
    if key is not None:
        if not isinstance(key, tuple) or len(key) != 7 or key[0] != walls:
            raise ValueError("SST wall-distance cache identity is invalid")
        _integer(key[1], "SST obstacle revision")
        if _integer(key[2], "SST inactive axis", -1) not in (-1, 0, 1, 2):
            raise ValueError("SST inactive axis is invalid")
        _integer(key[3], "SST marker count")
        _integer(key[4], "SST segment count")
        for digest in key[5:]:
            if not isinstance(digest, str) or (digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                raise ValueError("SST geometry digest is invalid")


def _validate_marker_geometry(geometry: object, marker_count: int, projection_count: int) -> None:
    required = {
        "marker_count", "projection_vertex_count", "projection_triangle_count",
        "projection_segment_count", "open_ribbon_tip_cap_binding",
    }
    material_identity_key = "material_surface_binding_identity"
    valid_schemas = (required, required | {material_identity_key})
    if not isinstance(geometry, Mapping) or set(geometry) not in valid_schemas:
        raise ValueError("marker geometry metadata schema is incomplete")
    if material_identity_key in geometry:
        material_identity = geometry[material_identity_key]
        if (
            not isinstance(material_identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", material_identity) is None
        ):
            raise ValueError("material surface binding identity is invalid")
    if _integer(geometry["marker_count"], "geometry marker count", 1) != marker_count:
        raise ValueError("marker geometry count differs")
    if _integer(geometry["projection_vertex_count"], "geometry projection count", marker_count) != projection_count:
        raise ValueError("projection geometry count differs")
    _integer(geometry["projection_triangle_count"], "projection triangle count")
    _integer(geometry["projection_segment_count"], "projection segment count")
    binding = geometry["open_ribbon_tip_cap_binding"]
    if binding is None:
        if projection_count != marker_count:
            raise ValueError("projection-only vertices require an explicit cap binding")
        return
    if not isinstance(binding, tuple) or len(binding) != 11:
        raise ValueError("open-ribbon cap binding is invalid")
    indices = tuple(_integer(v, "cap vertex index") for v in binding[:8])
    if len(set(indices[:4])) != 4 or any(v >= marker_count for v in indices[:4]):
        raise ValueError("cap source indices must be four physical markers")
    if indices[4:] != tuple(range(marker_count, marker_count + 4)) or projection_count != marker_count + 4:
        raise ValueError("cap derived vertex layout differs")
    _integer(binding[8], "cap region")
    if _integer(binding[9], "cap inactive axis") not in (0, 1, 2):
        raise ValueError("cap inactive axis is invalid")
    _positive(binding[10], "cap area per length")


def validate_accepted_fsi_state(state: AcceptedFsiState) -> None:
    """Validate physical/counter consistency without touching a runtime field."""
    if not isinstance(state, AcceptedFsiState) or not isinstance(state.macro_state, HostMacroStepState):
        raise TypeError("state must contain a typed HostMacroStepState")
    macro = state.macro_state
    if not isinstance(state.runner_state, Mapping):
        raise TypeError("runner state must be a mapping")
    _validate_fluid_metadata(macro.fluid_host_metadata)
    step = _integer(macro.accepted_step_index, "accepted step", 1)
    dt = _positive(state.runner_state.get("dt_s"), "dt_s")
    accepted_time = _positive(macro.accepted_time_s, "accepted time")
    expected_time = step * dt
    if abs(accepted_time - expected_time) > 4 * max(math.ulp(expected_time), math.ulp(accepted_time)):
        raise ValueError("accepted time does not match completed full macro steps")
    if macro.feedback_available_for_projection is not True:
        raise ValueError("an accepted FSI state must retain its feedback flag")
    _integer(state.ledger_generation, "ledger generation")
    if set(macro.fluid_fields) != set(FLUID_MACRO_STATE_FIELDS):
        raise ValueError("fluid macro state is incomplete")
    if set(macro.solid_fields) != set(SOLID_MACRO_STATE_FIELDS):
        raise ValueError("solid macro state is incomplete")
    boundary = validate_preflow_snapshot_fields(
        state.fluid_boundary_fields,
        velocity_dirichlet_boundary_authority=state.velocity_boundary_authority,
    )
    for name, value in macro.fluid_fields.items():
        array = _finite_array(value, f"fluid {name}")
        if name in boundary and (array.dtype != boundary[name].dtype or not np.array_equal(array, boundary[name])):
            raise ValueError(f"fluid macro/boundary state differs for {name}")
        if name not in boundary:
            dtype = np.dtype(np.float32 if name == "volume_source_s" else np.int32)
            if array.shape != boundary["pressure"].shape or array.dtype != dtype:
                raise ValueError(f"fluid {name} shape/dtype is invalid")
            if name != "volume_source_s" and np.any((array != 0) & (array != 1)):
                raise ValueError(f"fluid {name} must be a binary mask")
    count = _integer(macro.solid_particle_count, "particle count", 1)
    positions = _finite_array(macro.solid_fields["x"], "solid x")
    if positions.ndim != 2:
        raise ValueError("solid x shape is invalid")
    capacity = positions.shape[0]
    if capacity < count:
        raise ValueError("solid state does not cover active particles")
    for name, value in macro.solid_fields.items():
        array = _finite_array(value, f"solid {name}")
        expected = (capacity, 3, 3) if name in {"C", "F"} else (capacity, 3)
        dtype = np.dtype(np.float64 if name == "F" else np.float32)
        if array.shape != expected or array.dtype != dtype:
            raise ValueError(f"solid {name} shape/dtype is invalid")
    marker_count = _integer(macro.marker_count, "marker count", 1)
    projection_count = _integer(macro.marker_projection_vertex_count, "projection vertex count", marker_count)
    if set(macro.marker_state) != {*MARKER_INTERFACE_STATE_FIELDS, "_marker_geometry"}:
        raise ValueError("marker state is incomplete")
    geometry = macro.marker_state["_marker_geometry"]
    _validate_marker_geometry(geometry, marker_count, projection_count)
    for name in MARKER_INTERFACE_STATE_FIELDS:
        array = _finite_array(macro.marker_state[name], f"marker {name}")
        shape = (marker_count,) if name == "A_gamma_m2" else (marker_count, 3)
        if array.shape != shape or array.dtype != np.dtype(np.float32):
            raise ValueError(f"marker {name} shape/dtype is invalid")
    if macro.marker_pressure_neumann_gradient is not None:
        gradient = _finite_array(macro.marker_pressure_neumann_gradient, "marker pressure gradient")
        if gradient.shape != (projection_count,) or gradient.dtype != np.dtype(np.float32):
            raise ValueError("marker pressure gradient shape/dtype is invalid")
    mode = state.runner_state.get("coupling_mode")
    if mode not in {"direct_explicit", "iqn_ils"}:
        raise ValueError("checkpoint coupling mode is invalid")
    if mode == "iqn_ils":
        if not isinstance(state.initial_guess_state, InterfaceInitialGuessSnapshot):
            raise ValueError("IQN checkpoint requires initial-guess state")
        if state.initial_guess_state.accepted_step_count != step:
            raise ValueError("initial-guess accepted count differs from physical state")
        if state.initial_guess_state.shape != (marker_count, 3):
            raise ValueError("initial-guess shape differs from physical markers")
        reference = _finite_array(state.marker_reference_positions_m, "marker reference")
        if reference.shape != (marker_count, 3):
            raise ValueError("marker reference shape is invalid")
    elif any(value is not None for value in (state.initial_guess_state, state.iqn_history, state.marker_reference_positions_m)):
        raise ValueError("direct-explicit checkpoint contains unrelated IQN state")
    if state.iqn_history is not None:
        if not isinstance(state.iqn_history, IqnIlsSecantHistory) or state.iqn_history.source_step != step:
            raise ValueError("IQN secants do not belong to the last accepted step")
        if state.iqn_history.dt_s != dt or state.iqn_history.marker_shape != (marker_count, 3):
            raise ValueError("IQN secant time/shape identity differs")
        if state.iqn_history.layout_id != state.initial_guess_state.layout_id:
            raise ValueError("IQN and initial-guess layout identities differ")
    if state.kalman_state is not None:
        if not isinstance(state.kalman_state, ActiveKalmanWritebackSnapshot):
            raise TypeError("active-Kalman state has invalid type")
        for _, metrics in state.kalman_state.owner_metrics:
            if metrics.commit_count != step:
                raise ValueError("active-Kalman accepted count differs from physical state")
        owner_shapes = {
            "interface_marker_velocity": (marker_count, 3),
            "fluid_fsi_pressure_feedback": boundary["fsi_pressure"].shape,
            "solid_particle_velocity": (count, 3),
        }
        for owner, predictor in state.kalman_state.predictor_snapshots:
            if owner not in owner_shapes or predictor.values.shape != owner_shapes[owner]:
                raise ValueError("active-Kalman owner shape differs from physical state")


def _validate_record(record: object, step: int, dt_s: float) -> Mapping[str, object]:
    if not isinstance(record, Mapping) or not isinstance(record.get("history_row"), Mapping):
        raise ValueError("accepted history record must contain history_row")
    row = record["history_row"]
    if _integer(row.get("step"), "history step", 1) != step:
        raise ValueError("history step is not contiguous")
    recorded_time = _positive(row.get("time_s"), "history time")
    if abs(recorded_time - step * dt_s) > 4 * max(math.ulp(recorded_time), math.ulp(step * dt_s)):
        raise ValueError("history time does not match full macro steps")
    return record


def _validate_continuation_parent(
    path: str | Path, *, step: int, identity: Mapping[str, str],
    previous_tail: HistoryTail | None, expected_generation: str | None,
) -> None:
    # O(1) manifest read, not an O(step) walk of the immutable history chain.
    # A resumed process has already checked the entire prefix during load.
    head = read_checkpoint_head(path)
    if head is None:
        if step != 1 or previous_tail is not None or expected_generation is not None:
            raise ValueError("new accepted checkpoint must start at step one without a parent")
        return
    if set(head.metadata) != {"identity", "state"}:
        raise ValueError("parent accepted checkpoint metadata schema is invalid")
    if _identity(head.metadata["identity"]) != identity:
        raise ValueError("accepted checkpoint continuation identity mismatch")
    if head.generation != expected_generation or head.history_tail != previous_tail:
        raise ValueError("accepted checkpoint continuation must reference the current parent")
    if head.accepted_step != step - 1:
        raise ValueError("accepted checkpoint continuation must advance exactly one step")


def write_accepted_fsi_checkpoint(
    path: str | Path, *, state: AcceptedFsiState,
    identity: Mapping[str, str], record: Mapping[str, object],
    previous_tail: HistoryTail | None = None,
    expected_generation: str | None = None,
) -> AcceptedFsiCommit:
    validate_accepted_fsi_state(state)
    identity_payload = _identity(identity)
    step = state.macro_state.accepted_step_index
    _validate_record(record, step, float(state.runner_state["dt_s"]))
    _validate_continuation_parent(
        path, step=step, identity=identity_payload,
        previous_tail=previous_tail, expected_generation=expected_generation,
    )
    codec = _state_codec()
    encoded = codec.encode(state)
    # Run the exact load-time constructors before replacing a good manifest.
    # This rechecks nested immutable snapshots, including finite-but-invalid data.
    validate_accepted_fsi_state(codec.decode(encoded.metadata, encoded.arrays))
    journal = codec.encode(record)
    if journal.arrays:
        raise ValueError("accepted history journal must not contain physical arrays")
    # Validate/encode everything before publishing either journal or checkpoint.
    tail = append_history(path, step=step, record=journal.metadata, previous=previous_tail)
    generation = save_checkpoint(
        path, accepted_step=step,
        metadata={"identity": identity_payload, "state": encoded.metadata},
        arrays=encoded.arrays, history_tail=tail,
        expected_generation=expected_generation,
    )
    return AcceptedFsiCommit(generation, tail)


def load_accepted_fsi_checkpoint(
    path: str | Path, *, expected_identity: Mapping[str, str], target_step_count: int,
    expected_generation: str | None = None,
) -> LoadedAcceptedFsiCheckpoint:
    target = _integer(target_step_count, "target step count")
    if expected_generation is not None and (
        not isinstance(expected_generation, str)
        or re.fullmatch(r"[0-9a-f]{32}", expected_generation) is None
    ):
        raise ValueError("expected checkpoint generation is invalid")
    stored = load_checkpoint(path)
    if (
        expected_generation is not None
        and stored.generation != expected_generation
    ):
        raise ValueError("accepted FSI checkpoint generation mismatch")
    if set(stored.metadata) != {"identity", "state"}:
        raise ValueError("accepted FSI checkpoint metadata schema is invalid")
    stored_identity = _identity(stored.metadata["identity"])
    requested_identity = _identity(expected_identity)
    if stored_identity != requested_identity:
        mismatches = "; ".join(
            f"{field}: stored={stored_identity[field]}, "
            f"expected={requested_identity[field]}"
            for field in (
                "config_sha256",
                "source_sha256",
                "geometry_sha256",
            )
            if stored_identity[field] != requested_identity[field]
        )
        raise ValueError(
            "accepted FSI checkpoint identity mismatch: " + mismatches
        )
    if stored.accepted_step > target:
        raise ValueError("target step count precedes the accepted checkpoint")
    codec = _state_codec()
    state = codec.decode(stored.metadata["state"], stored.arrays)
    validate_accepted_fsi_state(state)
    if state.macro_state.accepted_step_index != stored.accepted_step:
        raise ValueError("manifest and physical accepted steps differ")
    records = tuple(codec.decode(record, {}) for record in stored.history)
    dt = float(state.runner_state["dt_s"])
    for step, record in enumerate(records, start=1):
        _validate_record(record, step, dt)
    if len(records) != stored.accepted_step or stored.history_tail is None:
        raise ValueError("accepted history prefix is incomplete")
    return LoadedAcceptedFsiCheckpoint(state, records, stored.generation, stored.history_tail)
