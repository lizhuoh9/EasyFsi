"""Capture canonical HIBM claim scratch immediately before a failed commit.

This wrapper deliberately leaves the solver's fail-closed validator in place.
It only records the transient Taichi fields that the transaction clears while
unwinding, then delegates to the original validator and preserves its exit.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core.coupling.hibm_mpm.core import (
    HIBM_COMPONENT_FACE_TARGET_CONFLICT_AUTHOR_CARDINALITY,
    HIBM_COMPONENT_FACE_TARGET_CONFLICT_PREPARE_PAIR,
    HIBM_COMPONENT_FACE_TARGET_CONFLICT_SOURCE_NAMES,
    HibmMpmIbBoundaryConditions,
)

_ASSEMBLY_CONTEXT: dict[str, Any] | None = None
_MAX_CAPTURED_WITNESS_LANES = 64
_FACE_PROJECTED_SEGMENT_CLASSIFICATION = (
    "same_segment_distinct_anchor_face_projected_bracket"
)


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _row(values: np.ndarray) -> list[int]:
    return [int(value) for value in values]


def _source_from_key(key: int, *, ny: int, nz: int) -> list[int]:
    plane = ny * nz
    i, remainder = divmod(int(key), plane)
    j, k = divmod(remainder, nz)
    return [i, j, k]


def _snapshot_vector(
    snapshot: dict[str, np.ndarray],
    name: str,
    index: tuple[int, ...] | int,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    return np.asarray(snapshot[name][index], dtype=dtype).reshape(-1).copy()


def _host_array(field: Any, *, name: str) -> np.ndarray:
    to_numpy = getattr(field, "to_numpy", None)
    if not callable(to_numpy):
        raise TypeError(f"{name} must provide to_numpy() for diagnostic capture")
    return np.asarray(to_numpy()).copy()


def _capture_host_snapshot(
    boundary: HibmMpmIbBoundaryConditions,
    context: dict[str, Any],
    diagnostic_context: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Read failure-only lane data from Taichi exactly once per field."""

    search = diagnostic_context["search"]
    fields: dict[str, Any] = {
        "first_author_witness": (
            boundary.velocity_dirichlet_component_face_segment_first_author_linear_key
        ),
        "second_author_witness": (
            boundary.velocity_dirichlet_component_face_segment_second_author_linear_key
        ),
        "claim_count": boundary.velocity_dirichlet_component_face_claim_count,
        "claim_target": boundary.velocity_dirichlet_component_face_claim_target_mps,
        "claim_region": boundary.velocity_dirichlet_component_face_claim_region_id,
        "claim_alpha": boundary.velocity_dirichlet_component_face_claim_alpha,
        "active_ib_node": boundary.active_ib_node,
        "obstacle": context["obstacle_field"],
        "actual_sample_valid": (
            boundary.velocity_dirichlet_component_face_actual_sample_valid
        ),
        "actual_sample_point": (
            boundary.velocity_dirichlet_component_face_actual_sample_point_m
        ),
        "actual_sample_velocity": (
            boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps
        ),
        "normal": boundary.pressure_neumann_normal_field,
        "serialized_target": boundary.velocity_dirichlet_mps_field,
        "projection_indices": search.node_projection_marker_indices,
        "projection_weights": search.node_projection_marker_weights,
        "nearest_marker": search.nearest_marker,
        "boundary_point": search.node_boundary_point_m,
        "interior_point": search.node_interior_fluid_point_m,
        "marker_region": diagnostic_context["marker_region_id"],
        "face_x": context["cell_face_x_m"],
        "face_y": context["cell_face_y_m"],
        "face_z": context["cell_face_z_m"],
        "center_x": context["cell_center_x_m"],
        "center_y": context["cell_center_y_m"],
        "center_z": context["cell_center_z_m"],
        "target_conflict_report": (
            boundary.report_velocity_dirichlet_component_face_target_conflict_count
        ),
        "region_conflict_report": (
            boundary.report_velocity_dirichlet_component_face_region_conflict_count
        ),
        "alpha_conflict_report": (
            boundary.report_velocity_dirichlet_component_face_alpha_conflict_count
        ),
        "claim_conflict_report": (
            boundary.report_velocity_dirichlet_component_face_conflict_count
        ),
        "duplicate_claim_report": (
            boundary.report_velocity_dirichlet_component_face_duplicate_claim_count
        ),
    }
    if bool(diagnostic_context["marker_geometry_available"]):
        fields["marker_position"] = diagnostic_context["marker_position_m"]
        fields["marker_velocity"] = diagnostic_context["marker_velocity_mps"]
    return {
        name: _host_array(field, name=name)
        for name, field in fields.items()
    }


def _decode_conflict_author_witnesses(
    raw_author_witnesses: tuple[int, int] | list[int],
    *,
    grid_nodes: tuple[int, int, int],
) -> dict[str, Any]:
    """Decode failure-only author provenance without guessing source rows."""

    nx, ny, nz = (int(value) for value in grid_nodes)
    node_count = nx * ny * nz
    raw_values = [int(value) for value in raw_author_witnesses]
    payloads = [-value - 2 for value in raw_values]
    errors: list[str] = []
    if node_count <= 0:
        errors.append("grid node count must be positive")
    for slot, (raw_value, payload) in enumerate(zip(raw_values, payloads)):
        if raw_value > -2 or payload < 0:
            errors.append(f"slot {slot} is not a negative conflict witness")

    path_code: int | None = None
    author_keys: list[int] = []
    decode_strategy = "invalid"
    if not errors and all(payload < 4 * node_count for payload in payloads):
        path_codes = [3 - payload // node_count for payload in payloads]
        if all(0 <= code < len(HIBM_COMPONENT_FACE_TARGET_CONFLICT_SOURCE_NAMES) for code in path_codes):
            if len(set(path_codes)) == 1:
                path_code = int(path_codes[0])
                decode_strategy = "path_tagged_single_author_witnesses"
                for payload in payloads:
                    reverse_key = payload % node_count
                    author_keys.append(node_count - 1 - reverse_key)
            else:
                errors.append(
                    "path-tagged author witnesses disagree on conflict path"
                )
        else:
            errors.append("path-tagged author witness has an invalid path code")
    elif not errors:
        path_code = HIBM_COMPONENT_FACE_TARGET_CONFLICT_AUTHOR_CARDINALITY
        decode_strategy = "packed_author_pair_witnesses"
        author_base = node_count + 1
        for slot, payload in enumerate(payloads):
            packed_values = (payload // author_base, payload % author_base)
            for packed_value in packed_values:
                author_key = packed_value - 1
                if 0 <= author_key < node_count:
                    author_keys.append(int(author_key))
                else:
                    errors.append(
                        f"slot {slot} contains an invalid packed author key"
                    )

    if len(set(author_keys)) != len(author_keys):
        errors.append("decoded author witnesses are not unique")
    rows = [
        _source_from_key(key, ny=ny, nz=nz)
        for key in author_keys
        if 0 <= key < node_count
    ]
    return {
        "raw_author_witnesses": raw_values,
        "conflict_path_code": path_code,
        "conflict_source": (
            HIBM_COMPONENT_FACE_TARGET_CONFLICT_SOURCE_NAMES[path_code]
            if path_code is not None
            and 0 <= path_code < len(HIBM_COMPONENT_FACE_TARGET_CONFLICT_SOURCE_NAMES)
            else None
        ),
        "author_linear_keys": author_keys,
        "author_source_rows": rows,
        "decode_strategy": decode_strategy,
        "decode_errors": errors,
    }


def _segment_parameter(
    point: np.ndarray,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    *,
    inactive_axis: int,
) -> tuple[float, float, float]:
    active = np.ones(3, dtype=bool)
    if 0 <= inactive_axis < 3:
        active[inactive_axis] = False
    delta = endpoint_b[active] - endpoint_a[active]
    offset = point[active] - endpoint_a[active]
    length_squared = float(np.dot(delta, delta))
    if not np.isfinite(length_squared) or length_squared <= 1.0e-20:
        return float("nan"), float("inf"), length_squared
    parameter = float(np.dot(offset, delta) / length_squared)
    residual = offset - parameter * delta
    return parameter, float(np.linalg.norm(residual)), length_squared


def _component_coordinate_bracketed(
    face_center: np.ndarray,
    boundary_points: list[np.ndarray],
    component_axis: int,
) -> bool:
    if len(boundary_points) != 2 or component_axis not in (0, 1, 2):
        return False
    values = [float(point[component_axis]) for point in boundary_points]
    face_value = float(face_center[component_axis])
    return min(values) <= face_value <= max(values)


def _classify_face_projected_segment_pair(lane: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed classification for the proposed finite-segment invariant."""

    failures: list[str] = []
    authors = list(lane.get("authors", ()))
    component_axis = int(lane.get("component_axis", -1))
    inactive_axis = int(lane.get("inactive_axis", -1))
    face_center = np.asarray(lane.get("face_center_m", ()), dtype=np.float64)
    cell_widths = np.asarray(lane.get("target_cell_width_m", ()), dtype=np.float64)
    claim_region = int(lane.get("claim_region_id", -1))

    if len(authors) != 2:
        failures.append("requires exactly two decoded authors")
    if face_center.shape != (3,) or not np.all(np.isfinite(face_center)):
        failures.append("face center is not a finite 3-vector")
    if cell_widths.shape != (3,) or not np.all(np.isfinite(cell_widths)):
        failures.append("cell widths are not a finite 3-vector")
    if component_axis not in (0, 1, 2):
        failures.append("component axis is invalid")
    if inactive_axis not in (0, 1, 2):
        failures.append("inactive projection axis is unavailable")
    if component_axis == inactive_axis:
        failures.append("component axis cannot be the inactive projection axis")
    if int(lane.get("conflict_path_code", -1)) != HIBM_COMPONENT_FACE_TARGET_CONFLICT_PREPARE_PAIR:
        failures.append("conflict did not originate in prepare-pair arbitration")
    if int(lane.get("claim_count", -1)) != 2:
        failures.append("claim cardinality is not two")
    if not bool(lane.get("interpolate_interior_velocity", False)):
        failures.append("interior-velocity interpolation is disabled")

    target = [int(value) for value in lane.get("component_face", ())]
    source_rows = [list(author.get("source_row", ())) for author in authors]
    if len(target) == 3 and component_axis in (0, 1, 2):
        minus = list(target)
        minus[component_axis] -= 1
        if source_rows != [minus, target]:
            failures.append("decoded authors are not the exact adjacent face pair")
    else:
        failures.append("component face is unavailable")

    segment_indices: list[list[int]] = []
    segment_weights: list[np.ndarray] = []
    boundary_points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    serialized_targets: list[np.ndarray] = []
    actual_sample_points: list[np.ndarray] = []
    endpoint_positions_by_author: list[np.ndarray] = []
    endpoint_velocities_by_author: list[np.ndarray] = []
    for author_index, author in enumerate(authors):
        if not bool(author.get("active_ib_node", False)):
            failures.append(f"author {author_index} is not an active IB row")
        if bool(author.get("obstacle", True)):
            failures.append(f"author {author_index} is an obstacle row")
        if not bool(author.get("actual_sample_valid", False)):
            failures.append(f"author {author_index} lacks an actual sample")
        indices = [int(value) for value in author.get("projection_marker_indices", ())]
        weights = np.asarray(
            author.get("projection_marker_weights", ()), dtype=np.float64
        )
        segment_indices.append(indices)
        segment_weights.append(weights)
        if len(indices) != 3 or indices[0] < 0 or indices[1] < 0 or indices[0] == indices[1] or indices[2] != -1:
            failures.append(f"author {author_index} lacks exact (a,b,-1) provenance")
        if weights.shape != (3,) or not np.all(np.isfinite(weights)):
            failures.append(f"author {author_index} has invalid segment weights")
        elif (
            np.min(weights) < -1.0e-6
            or np.max(weights) > 1.0 + 1.0e-6
            or abs(float(np.sum(weights)) - 1.0) > 2.0e-6
            or abs(float(weights[2])) > 1.0e-6
        ):
            failures.append(f"author {author_index} has non-affine segment weights")
        elif float(np.min(weights[:2])) <= 1.0e-6:
            failures.append(
                f"author {author_index} uses an endpoint-clamped segment weight"
            )
        regions = [int(value) for value in author.get("projection_marker_regions", ())]
        if claim_region < 0 or regions != [claim_region, claim_region]:
            failures.append(f"author {author_index} crosses a marker-region seam")
        if int(author.get("nearest_marker_region_id", -1)) != claim_region:
            failures.append(f"author {author_index} nearest marker region differs")
        if int(author.get("nearest_marker", -1)) not in indices[:2]:
            failures.append(f"author {author_index} nearest marker is not a segment endpoint")

        for key, destination in (
            ("boundary_point_m", boundary_points),
            ("normal", normals),
            ("serialized_target_mps", serialized_targets),
        ):
            vector = np.asarray(author.get(key, ()), dtype=np.float64)
            destination.append(vector)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                failures.append(f"author {author_index} has invalid {key}")
        for key in (
            "nominal_interior_point_m",
            "actual_sample_point_m",
            "actual_sample_velocity_mps",
        ):
            vector = np.asarray(author.get(key, ()), dtype=np.float64)
            if key == "actual_sample_point_m":
                actual_sample_points.append(vector)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                failures.append(f"author {author_index} has invalid {key}")

        endpoint_positions = np.asarray(
            author.get("projection_marker_positions_m", ()), dtype=np.float64
        )
        endpoint_velocities = np.asarray(
            author.get("projection_marker_velocities_mps", ()), dtype=np.float64
        )
        endpoint_positions_by_author.append(endpoint_positions)
        endpoint_velocities_by_author.append(endpoint_velocities)
        if endpoint_positions.shape != (2, 3) or not np.all(np.isfinite(endpoint_positions)):
            failures.append(f"author {author_index} has invalid endpoint positions")
        if endpoint_velocities.shape != (2, 3) or not np.all(np.isfinite(endpoint_velocities)):
            failures.append(f"author {author_index} has invalid endpoint velocities")

    same_segment = len(segment_indices) == 2 and segment_indices[0] == segment_indices[1]
    if not same_segment:
        failures.append("authors do not use the same ordered segment")
    geometry_consistent = (
        len(endpoint_positions_by_author) == 2
        and endpoint_positions_by_author[0].shape == (2, 3)
        and endpoint_positions_by_author[1].shape == (2, 3)
        and np.array_equal(endpoint_positions_by_author[0], endpoint_positions_by_author[1])
        and endpoint_velocities_by_author[0].shape == (2, 3)
        and endpoint_velocities_by_author[1].shape == (2, 3)
        and np.array_equal(endpoint_velocities_by_author[0], endpoint_velocities_by_author[1])
    )
    if not geometry_consistent:
        failures.append("authors do not share identical endpoint state")

    normal_alignment = float("nan")
    if len(normals) == 2 and all(vector.shape == (3,) for vector in normals):
        normal_norms = [float(np.linalg.norm(vector)) for vector in normals]
        if min(normal_norms) > 1.0e-12:
            normal_alignment = float(
                np.dot(normals[0], normals[1]) / (normal_norms[0] * normal_norms[1])
            )
    if not np.isfinite(normal_alignment) or normal_alignment < 0.999999:
        failures.append("author normals are not aligned")

    probe_ray_normal_alignments: list[float] = []
    face_probe_ray_progresses: list[float] = []
    active_coordinates = np.ones(3, dtype=bool)
    if 0 <= inactive_axis < 3:
        active_coordinates[inactive_axis] = False
    if (
        face_center.shape == (3,)
        and len(boundary_points) == 2
        and len(actual_sample_points) == 2
        and len(normals) == 2
    ):
        for boundary_point, sample_point, normal in zip(
            boundary_points,
            actual_sample_points,
            normals,
        ):
            ray = sample_point[active_coordinates] - boundary_point[active_coordinates]
            active_normal = normal[active_coordinates]
            ray_norm = float(np.linalg.norm(ray))
            normal_norm = float(np.linalg.norm(active_normal))
            alignment = float("nan")
            progress = float("nan")
            if ray_norm > 1.0e-12 and normal_norm > 1.0e-12:
                alignment = float(
                    np.dot(ray, active_normal) / (ray_norm * normal_norm)
                )
                face_offset = (
                    face_center[active_coordinates]
                    - boundary_point[active_coordinates]
                )
                progress = float(np.dot(face_offset, ray) / (ray_norm * ray_norm))
            probe_ray_normal_alignments.append(alignment)
            face_probe_ray_progresses.append(progress)
    rays_are_valid = bool(
        len(probe_ray_normal_alignments) == 2
        and all(
            np.isfinite(value) and value >= 0.999999
            for value in probe_ray_normal_alignments
        )
        and len(face_probe_ray_progresses) == 2
        and all(
            np.isfinite(value) and 1.0e-6 < value <= 1.0 + 1.0e-6
            for value in face_probe_ray_progresses
        )
    )
    if not rays_are_valid:
        failures.append("actual probe rays are degenerate, reversed, or normal-inconsistent")

    geometric_author_parameters: list[float] = []
    face_parameter = float("nan")
    face_distance = float("inf")
    segment_length_squared = float("nan")
    if geometry_consistent and face_center.shape == (3,):
        endpoints = endpoint_positions_by_author[0]
        face_parameter, face_distance, segment_length_squared = _segment_parameter(
            face_center,
            endpoints[0],
            endpoints[1],
            inactive_axis=inactive_axis,
        )
        for boundary_point in boundary_points:
            parameter, _distance, _length_squared = _segment_parameter(
                boundary_point,
                endpoints[0],
                endpoints[1],
                inactive_axis=inactive_axis,
            )
            geometric_author_parameters.append(parameter)
    if not np.isfinite(segment_length_squared) or segment_length_squared <= 1.0e-20:
        failures.append("active-plane segment is degenerate")

    author_parameters = [
        float(weights[1])
        for weights in segment_weights
        if weights.shape == (3,) and np.all(np.isfinite(weights))
    ]
    parameters_finite = (
        len(author_parameters) == 2
        and np.isfinite(face_parameter)
        and all(np.isfinite(value) for value in author_parameters)
    )
    parameter_tolerance = 2.0e-6
    weights_match_geometry = (
        parameters_finite
        and len(geometric_author_parameters) == 2
        and all(np.isfinite(value) for value in geometric_author_parameters)
    )
    if weights_match_geometry:
        weights_match_geometry = all(
            abs(geometric_parameter - parameter) <= parameter_tolerance
            for geometric_parameter, parameter in zip(
                geometric_author_parameters,
                author_parameters,
            )
        )
    if not weights_match_geometry:
        failures.append("serialized weights do not match geometric anchors")

    serialized_targets_match = geometry_consistent and len(serialized_targets) == 2
    if serialized_targets_match:
        endpoint_velocities = endpoint_velocities_by_author[0]
        for weights, serialized in zip(segment_weights, serialized_targets):
            if weights.shape != (3,) or serialized.shape != (3,):
                serialized_targets_match = False
                break
            reconstructed = (
                float(weights[0]) * endpoint_velocities[0]
                + float(weights[1]) * endpoint_velocities[1]
            )
            if float(np.max(np.abs(reconstructed - serialized))) > 1.0e-6:
                serialized_targets_match = False
                break
    if not serialized_targets_match:
        failures.append("serialized target is not endpoint-velocity interpolation")

    anchor_separation = (
        abs(author_parameters[0] - author_parameters[1])
        if parameters_finite
        else 0.0
    )
    if anchor_separation <= parameter_tolerance:
        failures.append("author anchors are not distinct")
    bracket_margin = (
        min(face_parameter - min(author_parameters), max(author_parameters) - face_parameter)
        if parameters_finite
        else float("-inf")
    )
    segment_parameter_bracketed = bool(
        parameters_finite
        and bracket_margin > parameter_tolerance
        and -parameter_tolerance <= face_parameter <= 1.0 + parameter_tolerance
        and all(-parameter_tolerance <= value <= 1.0 + parameter_tolerance for value in author_parameters)
    )
    if not segment_parameter_bracketed:
        failures.append("face projection is not strictly bracketed by author anchors")

    support_distance = float("nan")
    if cell_widths.shape == (3,) and np.all(np.isfinite(cell_widths)):
        active_widths = cell_widths.copy()
        if 0 <= inactive_axis < 3:
            active_widths[inactive_axis] = 0.0
        support_distance = float(np.linalg.norm(active_widths))
    face_within_local_support = bool(
        np.isfinite(face_distance)
        and np.isfinite(support_distance)
        and support_distance > 0.0
        and face_distance <= support_distance * (1.0 + 1.0e-6)
    )
    if not face_within_local_support:
        failures.append("face projection lies outside local cell support")

    component_bracketed = (
        _component_coordinate_bracketed(face_center, boundary_points, component_axis)
        if face_center.shape == (3,)
        else False
    )
    if component_bracketed:
        failures.append("face axis is already bracketed by the two author points")
    classification = (
        _FACE_PROJECTED_SEGMENT_CLASSIFICATION
        if not failures
        else "does_not_match_face_projected_segment_invariant"
    )
    return {
        "classification": classification,
        "classification_failures": failures,
        "same_ordered_projection_segment": same_segment,
        "normal_alignment_cosine": normal_alignment,
        "probe_ray_normal_alignment_cosines": probe_ray_normal_alignments,
        "face_probe_ray_progresses": face_probe_ray_progresses,
        "probe_rays_are_valid": rays_are_valid,
        "serialized_targets_match_endpoint_interpolation": serialized_targets_match,
        "author_weights_match_geometric_parameters": weights_match_geometry,
        "author_segment_parameters": author_parameters,
        "author_geometric_segment_parameters": geometric_author_parameters,
        "face_segment_parameter": face_parameter,
        "author_parameter_separation": anchor_separation,
        "segment_parameter_bracket_margin": bracket_margin,
        "segment_parameter_bracketed": segment_parameter_bracketed,
        "component_axis_coordinate_bracketed": component_bracketed,
        "face_to_segment_distance_m": face_distance,
        "active_plane_segment_length_squared_m2": segment_length_squared,
        "local_active_support_distance_m": support_distance,
        "face_projection_within_local_support": face_within_local_support,
    }


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            _strict_json_value(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _attach_diagnostic_capture_error(
    solver_error: RuntimeError,
    diagnostic_error: BaseException,
) -> None:
    message = (
        "HIBM component-claim diagnostic capture failed: "
        f"{type(diagnostic_error).__name__}: {diagnostic_error}"
    )
    add_note = getattr(solver_error, "add_note", None)
    if callable(add_note):
        add_note(message)
    else:
        solver_error.diagnostic_capture_error = message


def _capture_face_geometry(
    snapshot: dict[str, np.ndarray],
    target: tuple[int, int, int],
    component_axis: int,
) -> dict[str, Any]:
    faces = tuple(
        snapshot[name]
        for name in ("face_x", "face_y", "face_z")
    )
    centers = tuple(
        snapshot[name]
        for name in ("center_x", "center_y", "center_z")
    )
    face_center = np.asarray(
        [float(centers[axis][target[axis]]) for axis in range(3)],
        dtype=np.float64,
    )
    face_center[component_axis] = float(
        faces[component_axis][target[component_axis]]
    )
    widths = [
        float(
            abs(
                float(faces[axis][target[axis] + 1])
                - float(faces[axis][target[axis]])
            )
        )
        for axis in range(3)
    ]
    preceding_width = 0.0
    if target[component_axis] > 0:
        preceding_width = float(
            abs(
                float(faces[component_axis][target[component_axis]])
                - float(faces[component_axis][target[component_axis] - 1])
            )
        )
    return {
        "face_center_m": _vector(face_center),
        "target_cell_width_m": widths,
        "preceding_component_cell_width_m": preceding_width,
    }


def _capture_author(
    diagnostic_context: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    source_row: list[int],
) -> dict[str, Any]:
    source = tuple(int(value) for value in source_row)
    marker_geometry_available = bool(
        diagnostic_context["marker_geometry_available"]
    )
    marker_capacity = int(snapshot["marker_region"].shape[0])
    projection_indices = _row(
        _snapshot_vector(
            snapshot,
            "projection_indices",
            source,
            dtype=np.int64,
        )
    )
    projection_weights = _snapshot_vector(
        snapshot,
        "projection_weights",
        source,
    )
    valid_projection_indices = [
        index for index in projection_indices if 0 <= index < marker_capacity
    ]
    nearest_marker = int(snapshot["nearest_marker"][source])
    return {
        "source_row": list(source),
        "active_ib_node": bool(snapshot["active_ib_node"][source]),
        "obstacle": bool(snapshot["obstacle"][source]),
        "actual_sample_valid": bool(snapshot["actual_sample_valid"][source]),
        "boundary_point_m": _vector(
            _snapshot_vector(snapshot, "boundary_point", source)
        ),
        "nominal_interior_point_m": _vector(
            _snapshot_vector(snapshot, "interior_point", source)
        ),
        "actual_sample_point_m": _vector(
            _snapshot_vector(
                snapshot,
                "actual_sample_point",
                source,
            )
        ),
        "actual_sample_velocity_mps": _vector(
            _snapshot_vector(
                snapshot,
                "actual_sample_velocity",
                source,
            )
        ),
        "normal": _vector(
            _snapshot_vector(snapshot, "normal", source)
        ),
        "nearest_marker": nearest_marker,
        "nearest_marker_region_id": (
            int(snapshot["marker_region"][nearest_marker])
            if 0 <= nearest_marker < marker_capacity
            else -1
        ),
        "projection_marker_indices": projection_indices,
        "projection_marker_weights": _vector(projection_weights),
        "projection_marker_regions": [
            int(snapshot["marker_region"][index])
            for index in valid_projection_indices
        ],
        "projection_marker_positions_m": (
            [
                _vector(_snapshot_vector(snapshot, "marker_position", index))
                for index in valid_projection_indices
            ]
            if marker_geometry_available
            else []
        ),
        "projection_marker_velocities_mps": (
            [
                _vector(_snapshot_vector(snapshot, "marker_velocity", index))
                for index in valid_projection_indices
            ]
            if marker_geometry_available
            else []
        ),
        "serialized_target_mps": _vector(
            _snapshot_vector(snapshot, "serialized_target", source)
        ),
    }


def _capture_witness_lane(
    context: dict[str, Any],
    diagnostic_context: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    grid_nodes: tuple[int, int, int],
    witness_index: tuple[int, int, int, int],
    raw_first_author_witness: int,
) -> dict[str, Any]:
    i, j, k, component_axis = witness_index
    target = (i, j, k)
    raw_second_author_witness = int(
        snapshot["second_author_witness"][witness_index]
    )
    decoded = _decode_conflict_author_witnesses(
        (raw_first_author_witness, raw_second_author_witness),
        grid_nodes=grid_nodes,
    )
    claim_count = _snapshot_vector(
        snapshot,
        "claim_count",
        target,
        dtype=np.int64,
    )
    claim_target = _snapshot_vector(
        snapshot,
        "claim_target",
        target,
    )
    claim_region = _snapshot_vector(
        snapshot,
        "claim_region",
        target,
        dtype=np.int64,
    )
    claim_alpha = _snapshot_vector(
        snapshot,
        "claim_alpha",
        target,
    )
    lane: dict[str, Any] = {
        "component_face": list(target),
        "component_axis": component_axis,
        "claim_count": int(claim_count[component_axis]),
        "claim_target_mps": float(claim_target[component_axis]),
        "claim_region_id": int(claim_region[component_axis]),
        "claim_alpha": float(claim_alpha[component_axis]),
        "interpolate_interior_velocity": bool(
            context.get("interpolate_interior_velocity", False)
        ),
        "inactive_axis": int(
            diagnostic_context.get(
                "inactive_axis",
                context.get("surface_projection_inactive_axis", -1),
            )
        ),
        **decoded,
        **_capture_face_geometry(snapshot, target, component_axis),
    }
    lane["authors"] = [
        _capture_author(
            diagnostic_context,
            snapshot,
            source_row,
        )
        for source_row in decoded["author_source_rows"]
    ]
    lane["derived"] = _classify_face_projected_segment_pair(lane)
    return lane


def _snapshot_int(snapshot: dict[str, np.ndarray], name: str) -> int:
    return int(np.asarray(snapshot[name]).item())


def _summarize_witness_capture(
    *,
    target_conflict_event_count: int,
    witness_indices: list[tuple[int, int, int, int]],
    selected_witness_count: int,
    lanes: list[dict[str, Any]],
    lane_capture_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_witness_lane_count = len(witness_indices)
    unique_witness_lane_count = len(set(witness_indices))
    captured_witness_lane_count = len(lanes)
    truncated_witness_lane_count = max(
        0,
        raw_witness_lane_count - int(selected_witness_count),
    )
    decode_error_lane_count = sum(
        bool(lane.get("decode_errors")) for lane in lanes
    )
    classification_counts: dict[str, int] = {}
    for lane in lanes:
        classification = str(lane["derived"]["classification"])
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
    capture_complete = bool(
        target_conflict_event_count > 0
        and target_conflict_event_count == raw_witness_lane_count
        and raw_witness_lane_count == unique_witness_lane_count
        and unique_witness_lane_count == int(selected_witness_count)
        and int(selected_witness_count) == captured_witness_lane_count
        and truncated_witness_lane_count == 0
        and decode_error_lane_count == 0
        and not lane_capture_errors
    )
    all_match = bool(
        capture_complete
        and classification_counts.get(
            _FACE_PROJECTED_SEGMENT_CLASSIFICATION, 0
        )
        == target_conflict_event_count
    )
    return {
        "raw_witness_lane_count": raw_witness_lane_count,
        "unique_witness_lane_count": unique_witness_lane_count,
        "captured_witness_lane_count": captured_witness_lane_count,
        "truncated_witness_lane_count": truncated_witness_lane_count,
        "decode_error_lane_count": decode_error_lane_count,
        "lane_capture_error_count": len(lane_capture_errors),
        "capture_complete": capture_complete,
        "classification_counts": classification_counts,
        "all_target_conflicts_match_face_projected_segment_invariant": all_match,
    }


def _capture(boundary: HibmMpmIbBoundaryConditions) -> dict[str, Any]:
    context = _ASSEMBLY_CONTEXT
    diagnostic_context = boundary.__dict__.get(
        "_canonical_velocity_dirichlet_precommit_diagnostic_context"
    )
    if not isinstance(context, dict):
        raise RuntimeError("component-face assembly context is unavailable")
    if not isinstance(diagnostic_context, dict):
        raise RuntimeError("core precommit diagnostic context is unavailable")
    snapshot = _capture_host_snapshot(
        boundary,
        context,
        diagnostic_context,
    )
    first_author_witness = snapshot["first_author_witness"]
    witness_indices = [
        tuple(int(value) for value in index)
        for index in np.argwhere(first_author_witness <= -2)
    ]
    selected_indices = witness_indices[:_MAX_CAPTURED_WITNESS_LANES]
    grid_nodes = tuple(int(value) for value in boundary.grid_nodes)

    lanes: list[dict[str, Any]] = []
    lane_capture_errors: list[dict[str, Any]] = []
    for witness_index in selected_indices:
        raw_first_author_witness = int(first_author_witness[witness_index])
        try:
            lanes.append(
                _capture_witness_lane(
                    context,
                    diagnostic_context,
                    snapshot,
                    grid_nodes,
                    witness_index,
                    raw_first_author_witness,
                )
            )
        except Exception as error:
            lane_capture_errors.append(
                {
                    "witness_index": list(witness_index),
                    "raw_first_author_witness": raw_first_author_witness,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    target_conflict_event_count = _snapshot_int(
        snapshot,
        "target_conflict_report",
    )
    summary = _summarize_witness_capture(
        target_conflict_event_count=target_conflict_event_count,
        witness_indices=witness_indices,
        selected_witness_count=len(selected_indices),
        lanes=lanes,
        lane_capture_errors=lane_capture_errors,
    )
    return {
        "schema_version": 2,
        "capture_policy": "failure_only_bounded_bulk_host_witness_capture",
        "witness_capture_limit": _MAX_CAPTURED_WITNESS_LANES,
        "target_conflict_event_count": target_conflict_event_count,
        "target_conflict_count": target_conflict_event_count,
        **summary,
        "region_conflict_count": _snapshot_int(
            snapshot,
            "region_conflict_report",
        ),
        "alpha_conflict_count": _snapshot_int(
            snapshot,
            "alpha_conflict_report",
        ),
        "claim_conflict_count": _snapshot_int(
            snapshot,
            "claim_conflict_report",
        ),
        "duplicate_claim_component_count": _snapshot_int(
            snapshot,
            "duplicate_claim_report",
        ),
        "interpolate_interior_velocity": bool(
            context.get("interpolate_interior_velocity", False)
        ),
        "witness_lanes": lanes,
        "lane_capture_errors": lane_capture_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-output", required=True)
    parser.add_argument("--runner-script", required=True)
    args, runner_args = parser.parse_known_args()
    output_path = Path(args.diagnostic_output).resolve()
    runner_script = Path(args.runner_script).resolve()
    original_argv = sys.argv
    captured = False
    original_validate = (
        HibmMpmIbBoundaryConditions
        ._validate_canonical_velocity_dirichlet_target_conflict_precommit
    )
    original_assemble = (
        HibmMpmIbBoundaryConditions
        .assemble_velocity_dirichlet_component_face_ledger
    )

    def capture_context_then_assemble(
        self: HibmMpmIbBoundaryConditions,
        *call_args: Any,
        **call_kwargs: Any,
    ) -> dict[str, object]:
        global _ASSEMBLY_CONTEXT
        _ASSEMBLY_CONTEXT = call_kwargs
        try:
            return original_assemble(self, *call_args, **call_kwargs)
        finally:
            _ASSEMBLY_CONTEXT = None

    def capture_then_validate(
        self: HibmMpmIbBoundaryConditions,
    ) -> None:
        nonlocal captured
        try:
            original_validate(self)
        except RuntimeError as solver_error:
            if not captured:
                captured = True
                try:
                    payload = _capture(self)
                    payload["interpolate_interior_velocity"] = bool(
                        (_ASSEMBLY_CONTEXT or {}).get(
                            "interpolate_interior_velocity",
                            False,
                        )
                    )
                    _write_json_atomic(output_path, payload)
                except BaseException as diagnostic_error:
                    _attach_diagnostic_capture_error(
                        solver_error,
                        diagnostic_error,
                    )
            raise

    HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_target_conflict_precommit = (  # type: ignore[method-assign]
        capture_then_validate
    )
    HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger = (  # type: ignore[method-assign]
        capture_context_then_assemble
    )
    sys.argv = [str(runner_script), *runner_args]
    try:
        runpy.run_path(str(runner_script), run_name="__main__")
    finally:
        sys.argv = original_argv
        HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_target_conflict_precommit = (  # type: ignore[method-assign]
            original_validate
        )
        HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger = (  # type: ignore[method-assign]
            original_assemble
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
