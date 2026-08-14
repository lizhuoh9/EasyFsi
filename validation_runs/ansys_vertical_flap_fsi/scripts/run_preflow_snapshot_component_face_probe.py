"""Capture process-local component-face conflict state during a diagnostic replay.

This tool is intentionally non-authoritative.  It delegates snapshot identity,
one-step execution, output isolation, and before/after snapshot hashing to the
existing diagnostic replay runner, while temporarily instrumenting two HIBM
boundary methods in this process.  No production source is modified.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation_runs.ansys_vertical_flap_fsi.scripts._preflow_snapshot_diagnostic_contracts import (
    json_safe,
    sha256_file,
    write_json_exclusive,
)
from validation_runs.ansys_vertical_flap_fsi.scripts.run_preflow_snapshot_one_step_diagnostic import (
    DEFAULT_ALLOWED_SOURCE_DIFFS,
    run_diagnostic_replay,
)


PROBE_FILENAME = "component_face_probe.json"
PAIR_MODE_NAMES = (
    (1, "projection_only_seam"),
    (2, "exact_author_cohort"),
    (4, "face_first_finite_segment_pair"),
    (8, "same_storage_direct_relocation_face_first"),
    (16, "inactive_axis_extrusion_direct_pair"),
    (32, "component_axis_direct_face_relocation_shadow"),
)


def _load_boundary_type() -> type[Any]:
    from simulation_core.coupling.hibm_mpm.core import (
        HibmMpmIbBoundaryConditions,
    )

    return HibmMpmIbBoundaryConditions


def run_component_face_probe(
    *,
    snapshot_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    allowed_source_diffs: Sequence[str] = DEFAULT_ALLOWED_SOURCE_DIFFS,
) -> dict[str, Any]:
    """Run the existing replay with temporary conflict-capture instrumentation."""

    boundary_type = _load_boundary_type()
    assemble_name = "assemble_velocity_dirichlet_component_face_ledger"
    validator_name = (
        "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
    )
    original_assemble = getattr(boundary_type, assemble_name)
    original_validator = getattr(boundary_type, validator_name)
    output_path = Path(output_dir)
    probe_path = output_path / PROBE_FILENAME
    assembly_by_instance: dict[int, dict[str, Any]] = {}
    probe_written = False

    def instrumented_assemble(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError(
                "component-face probe requires the keyword-only assembly contract"
            )
        assembly_by_instance[id(self)] = {
            name: kwargs[name]
            for name in (
                "obstacle_field",
                "velocity_field",
                "search",
                "cell_face_x_m",
                "cell_face_y_m",
                "cell_face_z_m",
                "cell_center_x_m",
                "cell_center_y_m",
                "cell_center_z_m",
                "grid_nodes",
            )
            if name in kwargs
        }
        return original_assemble(self, **kwargs)

    def instrumented_validator(self: Any) -> Any:
        nonlocal probe_written
        count = _int_field(
            self.report_velocity_dirichlet_component_face_target_conflict_count,
            None,
        )
        capture_problem: Exception | None = None
        if count > 0 and not probe_written:
            try:
                diagnostic = (
                    self._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
                )
                payload = _build_probe_payload(
                    boundary=self,
                    assembly=assembly_by_instance.get(id(self)),
                    first_conflict=diagnostic,
                    target_conflict_count=count,
                    output_dir=output_path,
                )
            except Exception as exc:
                capture_problem = exc
                payload = {
                    **_base_probe_payload(output_path),
                    "target_conflict_count": count,
                    "capture_error_type": type(exc).__name__,
                    "capture_error": str(exc),
                }
            try:
                write_json_exclusive(probe_path, payload)
                probe_written = True
            except Exception as exc:
                capture_problem = exc
        try:
            return original_validator(self)
        except BaseException as exc:
            if capture_problem is not None:
                note = (
                    "component-face probe capture failed: "
                    f"{type(capture_problem).__name__}: {capture_problem}"
                )
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(note)
                else:
                    exc.__dict__.setdefault("__notes__", []).append(note)
            raise

    setattr(boundary_type, assemble_name, instrumented_assemble)
    try:
        setattr(boundary_type, validator_name, instrumented_validator)
    except BaseException:
        setattr(boundary_type, assemble_name, original_assemble)
        raise
    try:
        return run_diagnostic_replay(
            snapshot_path=snapshot_path,
            config_path=config_path,
            source_manifest_path=source_manifest_path,
            output_dir=output_path,
            allowed_source_diffs=allowed_source_diffs,
        )
    finally:
        try:
            setattr(boundary_type, validator_name, original_validator)
        finally:
            setattr(boundary_type, assemble_name, original_assemble)


def _base_probe_payload(output_dir: Path) -> dict[str, Any]:
    tool_path = Path(__file__).resolve()
    return {
        "component_face_probe": True,
        "diagnostic_replay": True,
        "evidence_class": "diagnostic_only",
        "formal_validation_eligible": False,
        "parity_claimed": False,
        "fluent_parity_claimed": False,
        "fresh_preflow": False,
        "production_identity_valid": False,
        "output_dir": str(output_dir),
        "snapshot_integrity_enforced_by": (
            "diagnostic_replay.json:snapshot_artifacts_unchanged"
        ),
        "probe_tool": {
            "path": tool_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(tool_path),
        },
    }


def _build_probe_payload(
    *,
    boundary: Any,
    assembly: Mapping[str, Any] | None,
    first_conflict: Any,
    target_conflict_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    if not isinstance(assembly, Mapping):
        raise RuntimeError("component-face assembly fields were not captured")
    if not isinstance(first_conflict, Mapping):
        raise RuntimeError("first target conflict diagnostic is unavailable")
    face = _index3(first_conflict.get("component_face"))
    axis = int(first_conflict["component_axis"])
    pair_index = (*face, axis)
    nodes = tuple(int(value) for value in assembly["grid_nodes"])
    if len(nodes) != 3 or nodes != tuple(int(value) for value in boundary.grid_nodes):
        raise RuntimeError("captured component-face grid does not match boundary")

    pair = _pair_state(boundary, pair_index)
    arrays = _coordinate_arrays(assembly, nodes)
    runtime_obstacle = _host_field_snapshot(assembly["obstacle_field"])
    runtime_velocity = _host_field_snapshot(assembly["velocity_field"])
    sampler = _HostCanonicalSampler(
        obstacle_field=runtime_obstacle,
        velocity_field=runtime_velocity,
        faces=arrays["faces"],
        centers=arrays["centers"],
        nodes=nodes,
    )
    authors = _author_state(
        boundary=boundary,
        search=assembly["search"],
        velocity_field=runtime_velocity,
        first_conflict=first_conflict,
        pair=pair,
        nodes=nodes,
        sampler=sampler,
    )
    walk_candidates = _canonical_walk_candidates(
        pair=pair,
        nodes=nodes,
        faces=arrays["faces"],
        sampler=sampler,
    )
    return {
        **_base_probe_payload(output_dir),
        "target_conflict_count": int(target_conflict_count),
        "first_conflict": json_safe(first_conflict),
        "global_counters": {
            "scope": "global_not_face_local",
            "actual_sample_evaluation_count": _int_field(
                boundary.report_velocity_dirichlet_component_face_actual_sample_evaluation_count,
                None,
            ),
            "missing_actual_sample_count": _int_field(
                boundary.report_velocity_dirichlet_component_face_missing_actual_sample_count,
                None,
            ),
        },
        "pair_reconstruction_state": pair,
        "authors": authors,
        "runtime_obstacle_stencil": _obstacle_stencil(
            obstacle_field=runtime_obstacle,
            face=face,
            nodes=nodes,
            faces=arrays["faces"],
            centers=arrays["centers"],
        ),
        "canonical_walk_candidates": walk_candidates,
    }


def _pair_state(boundary: Any, pair_index: tuple[int, int, int, int]) -> dict[str, Any]:
    mode = _int_field(
        boundary.velocity_dirichlet_component_face_segment_projection_only_seam,
        pair_index,
    )
    return {
        "mode": mode,
        "mode_names": [name for bit, name in PAIR_MODE_NAMES if mode & bit],
        "author_linear_keys": [
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_first_author_linear_key,
                pair_index,
            ),
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_second_author_linear_key,
                pair_index,
            ),
        ],
        "author_kinds": [
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_first_author_kind,
                pair_index,
            ),
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_second_author_kind,
                pair_index,
            ),
        ],
        "admission_valid": bool(
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_admission_valid,
                pair_index,
            )
        ),
        "full_valid": bool(
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_full_valid,
                pair_index,
            )
        ),
        "boundary_point_m": _vector_field(
            boundary.velocity_dirichlet_component_face_segment_pair_boundary_point_m,
            pair_index,
        ),
        "normal": _vector_field(
            boundary.velocity_dirichlet_component_face_segment_pair_normal,
            pair_index,
        ),
        "nominal_probe_m": _vector_field(
            boundary.velocity_dirichlet_component_face_segment_pair_nominal_probe_m,
            pair_index,
        ),
        "boundary_target_mps": _float_field(
            boundary.velocity_dirichlet_component_face_segment_pair_boundary_target_mps,
            pair_index,
        ),
        "endpoint_clamped": bool(
            _int_field(
                boundary.velocity_dirichlet_component_face_segment_pair_endpoint_clamped,
                pair_index,
            )
        ),
        "clamp_support_ratio": _float_field(
            boundary.velocity_dirichlet_component_face_segment_pair_clamp_support_ratio,
            pair_index,
        ),
        "geometry_tolerance": _float_field(
            boundary.velocity_dirichlet_component_face_segment_pair_geometry_tolerance,
            pair_index,
        ),
    }


def _author_state(
    *,
    boundary: Any,
    search: Any,
    velocity_field: Any,
    first_conflict: Mapping[str, Any],
    pair: Mapping[str, Any],
    nodes: tuple[int, int, int],
    sampler: "_HostCanonicalSampler",
) -> list[dict[str, Any]]:
    diagnostic_authors = first_conflict.get("authors", ())
    diagnostic_by_row = {
        _index3(row["source_row"]): row
        for row in diagnostic_authors
        if isinstance(row, Mapping) and "source_row" in row
    }
    payloads: list[dict[str, Any]] = []
    stored_keys = [int(value) for value in pair["author_linear_keys"]]
    stored_kinds = [int(value) for value in pair["author_kinds"]]
    kind_by_key = dict(zip(stored_keys, stored_kinds))
    conflict_keys = [
        int(value) for value in first_conflict.get("author_linear_keys", ())
    ]
    keys = conflict_keys if len(conflict_keys) == 2 else stored_keys
    for key in keys:
        kind = kind_by_key.get(key)
        row = _row_from_linear_key(key, nodes)
        nominal_point = _vector_field(search.node_interior_fluid_point_m, row)
        actual_valid = bool(
            _int_field(
                boundary.velocity_dirichlet_component_face_actual_sample_valid,
                row,
            )
        )
        actual_point = _vector_field(
            boundary.velocity_dirichlet_component_face_actual_sample_point_m,
            row,
        )
        actual_velocity = _vector_field(
            boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps,
            row,
        )
        live_velocity = _vector_field(velocity_field, row)
        payloads.append(
            {
                "author_linear_key": key,
                "author_kind": kind,
                "source_row": list(row),
                "base_diagnostic": json_safe(diagnostic_by_row.get(row)),
                "raw_node_boundary_point_m": _vector_field(
                    search.node_boundary_point_m,
                    row,
                ),
                "raw_node_interior_fluid_point_m": nominal_point,
                "node_boundary_normal": _vector_field(
                    boundary.pressure_neumann_normal_field,
                    row,
                ),
                "nominal_sample": sampler.sample(nominal_point),
                "actual_sample": {
                    "valid": actual_valid,
                    "point_m": actual_point,
                    "velocity_mps": actual_velocity,
                    "velocity_finite": _vector_is_finite(actual_velocity),
                },
                "live_storage_velocity_mps": live_velocity,
                "live_storage_velocity_finite": _vector_is_finite(live_velocity),
            }
        )
    return payloads


def _coordinate_arrays(
    assembly: Mapping[str, Any],
    nodes: tuple[int, int, int],
) -> dict[str, list[list[float]]]:
    face_fields = (
        assembly["cell_face_x_m"],
        assembly["cell_face_y_m"],
        assembly["cell_face_z_m"],
    )
    center_fields = (
        assembly["cell_center_x_m"],
        assembly["cell_center_y_m"],
        assembly["cell_center_z_m"],
    )
    return {
        "faces": [
            [_float_field(snapshot, index) for index in range(count + 1)]
            for snapshot, count in zip(
                (_host_field_snapshot(field) for field in face_fields), nodes
            )
        ],
        "centers": [
            [_float_field(snapshot, index) for index in range(count)]
            for snapshot, count in zip(
                (_host_field_snapshot(field) for field in center_fields), nodes
            )
        ],
    }


def _obstacle_stencil(
    *,
    obstacle_field: Any,
    face: tuple[int, int, int],
    nodes: tuple[int, int, int],
    faces: list[list[float]],
    centers: list[list[float]],
) -> dict[str, Any]:
    radii = (1, 1, 8)
    ranges = [
        range(max(0, value - radius), min(count, value + radius + 1))
        for value, radius, count in zip(face, radii, nodes)
    ]
    cells = [
        {
            "index": [i, j, k],
            "obstacle": _int_field(obstacle_field, (i, j, k)),
        }
        for i in ranges[0]
        for j in ranges[1]
        for k in ranges[2]
    ]
    axis_coordinates = {}
    for name, indices, axis_faces, axis_centers in zip(
        ("x", "y", "z"), ranges, faces, centers
    ):
        axis_coordinates[name] = [
            {
                "index": index,
                "lower_face_m": axis_faces[index],
                "center_m": axis_centers[index],
                "upper_face_m": axis_faces[index + 1],
            }
            for index in indices
        ]
    return {
        "target": list(face),
        "offsets": {"i": [-1, 1], "j": [-1, 1], "k": [-8, 8]},
        "cells": cells,
        "axis_coordinates": axis_coordinates,
    }


class _HostCanonicalSampler:
    """Host mirror of core.py canonical backward-MAC sampling (13294-13488)."""

    def __init__(
        self,
        *,
        obstacle_field: Any,
        velocity_field: Any,
        faces: list[list[float]],
        centers: list[list[float]],
        nodes: tuple[int, int, int],
    ) -> None:
        self.obstacle = obstacle_field
        self.velocity = velocity_field
        self.faces = faces
        self.centers = centers
        self.nodes = nodes

    def sample(self, position: Sequence[float]) -> dict[str, Any]:
        point = [float(value) for value in position]
        inside = _vector_is_finite(point) and all(
            self.faces[axis][0] <= point[axis] < self.faces[axis][count]
            for axis, count in enumerate(self.nodes)
        )
        if not inside:
            return {
                "valid": False,
                "inside_physical_domain": False,
                "point_m": point,
                "velocity_mps": [0.0, 0.0, 0.0],
                "component_weights": [0.0, 0.0, 0.0],
                "component_weight_by_axis": {"x": 0.0, "y": 0.0, "z": 0.0},
                "minimum_component_weight": 0.0,
            }
        center_g = [
            _center_grid_coordinate(
                point[axis], self.faces[axis], self.centers[axis], count
            )
            for axis, count in enumerate(self.nodes)
        ]
        face_g = [
            _face_grid_coordinate(point[axis], self.faces[axis], count)
            for axis, count in enumerate(self.nodes)
        ]
        values: list[float] = []
        weights: list[float] = []
        for component_axis in range(3):
            grid = list(center_g)
            grid[component_axis] = face_g[component_axis]
            bases = [
                min(max(math.floor(grid[axis]), 0), self.nodes[axis] - 2)
                for axis in range(3)
            ]
            fractions = [
                min(max(grid[axis] - bases[axis], 0.0), 1.0)
                for axis in range(3)
            ]
            value = 0.0
            component_weight = 0.0
            for oi in range(2):
                for oj in range(2):
                    for ok in range(2):
                        offsets = (oi, oj, ok)
                        weight = math.prod(
                            fraction if offset else 1.0 - fraction
                            for fraction, offset in zip(fractions, offsets)
                        )
                        row = tuple(
                            bases[axis] + offsets[axis] for axis in range(3)
                        )
                        if self._support_is_fluid_fluid(row, component_axis):
                            value += weight * _vector_field(
                                self.velocity, row
                            )[component_axis]
                            component_weight += weight
            if component_weight > 1.0e-12:
                value /= component_weight
            values.append(value)
            weights.append(component_weight)
        minimum_weight = min(weights)
        return {
            "valid": minimum_weight > 1.0e-12,
            "inside_physical_domain": True,
            "point_m": point,
            "velocity_mps": values,
            "component_weights": weights,
            "component_weight_by_axis": dict(zip(("x", "y", "z"), weights)),
            "minimum_component_weight": minimum_weight,
        }

    def _support_is_fluid_fluid(
        self,
        row: tuple[int, int, int],
        axis: int,
    ) -> bool:
        if not all(0 <= row[index] < self.nodes[index] for index in range(3)):
            return False
        if row[axis] <= 0:
            return False
        minus_row = list(row)
        minus_row[axis] -= 1
        return (
            _int_field(self.obstacle, row) == 0
            and _int_field(self.obstacle, tuple(minus_row)) == 0
        )


def _canonical_walk_candidates(
    *,
    pair: Mapping[str, Any],
    nodes: tuple[int, int, int],
    faces: list[list[float]],
    sampler: _HostCanonicalSampler,
) -> list[dict[str, Any]]:
    boundary = [float(value) for value in pair["boundary_point_m"]]
    nominal = [float(value) for value in pair["nominal_probe_m"]]
    segment = [nominal[axis] - boundary[axis] for axis in range(3)]
    distance_squared = sum(value * value for value in segment)
    if (
        not _vector_is_finite(boundary)
        or not _vector_is_finite(nominal)
        or not math.isfinite(distance_squared)
        or distance_squared <= 1.0e-24
    ):
        return []
    nominal_distance = math.sqrt(distance_squared)
    walk_normal = [value / nominal_distance for value in segment]
    geometry_base = _row_from_linear_key(
        int(pair["author_linear_keys"][0]), nodes
    )
    widths = [
        faces[axis][geometry_base[axis] + 1]
        - faces[axis][geometry_base[axis]]
        for axis in range(3)
    ]
    denominator = max(
        sum(
            abs(walk_normal[axis]) / max(widths[axis], 1.0e-12)
            for axis in range(3)
        ),
        1.0e-12,
    )
    walk_step_m = 0.5 / denominator
    rows: list[dict[str, Any]] = []
    accepted = False
    for step_index in range(5):
        distance = nominal_distance + walk_step_m * step_index
        point = [
            boundary[axis] + walk_normal[axis] * distance for axis in range(3)
        ]
        sampled = sampler.sample(point)
        would_execute = not accepted
        first_accepted = would_execute and bool(sampled["valid"])
        rows.append(
            {
                "step_index": step_index,
                "distance_m": distance,
                "walk_step_m": walk_step_m,
                "would_execute_before_core_short_circuit": would_execute,
                "first_accepted": first_accepted,
                **sampled,
            }
        )
        accepted = accepted or first_accepted
    return rows


def _center_grid_coordinate(
    value: float,
    faces: Sequence[float],
    centers: Sequence[float],
    count: int,
) -> float:
    if value <= centers[0]:
        half_width = max(centers[0] - faces[0], 1.0e-18)
        return -0.5 * (centers[0] - value) / half_width
    if value >= centers[count - 1]:
        half_width = max(faces[count] - centers[count - 1], 1.0e-18)
        return count - 1 + 0.5 * (value - centers[count - 1]) / half_width
    lower = 0
    upper = count - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if value >= centers[middle]:
            lower = middle
        else:
            upper = middle
    distance = max(centers[lower + 1] - centers[lower], 1.0e-18)
    return lower + (value - centers[lower]) / distance


def _face_grid_coordinate(
    value: float,
    faces: Sequence[float],
    count: int,
) -> float:
    if value <= faces[0]:
        return (value - faces[0]) / max(faces[1] - faces[0], 1.0e-18)
    if value >= faces[count - 1]:
        return count - 1 + (value - faces[count - 1]) / max(
            faces[count - 1] - faces[count - 2], 1.0e-18
        )
    lower = 0
    upper = count - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if value >= faces[middle]:
            lower = middle
        else:
            upper = middle
    return lower + (value - faces[lower]) / max(
        faces[lower + 1] - faces[lower], 1.0e-18
    )


def _row_from_linear_key(
    linear_key: int,
    nodes: tuple[int, int, int],
) -> tuple[int, int, int]:
    nx, ny, nz = nodes
    if linear_key < 0 or linear_key >= nx * ny * nz:
        raise ValueError(f"author linear key is outside the grid: {linear_key}")
    node_plane = ny * nz
    i = linear_key // node_plane
    remainder = linear_key - i * node_plane
    j = remainder // nz
    k = remainder - j * nz
    return int(i), int(j), int(k)


def _index3(value: Any) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"expected a three-axis index, got {result!r}")
    return result


def _int_field(field: Any, index: Any) -> int:
    return int(field[index])


def _float_field(field: Any, index: Any) -> float:
    return float(field[index])


def _vector_field(field: Any, index: Any) -> list[float]:
    value = field[index]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    return [float(item) for item in value]


def _vector_is_finite(value: Sequence[float]) -> bool:
    return all(math.isfinite(float(item)) for item in value)


def _host_field_snapshot(field: Any) -> Any:
    to_numpy = getattr(field, "to_numpy", None)
    return to_numpy() if callable(to_numpy) else field


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic-only ANSYS vertical-flap one-step replay and "
            "capture component-face conflict state before validation raises."
        )
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--source-manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-source-diff", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    allowed = tuple(args.allow_source_diff or DEFAULT_ALLOWED_SOURCE_DIFFS)
    try:
        payload = run_component_face_probe(
            snapshot_path=args.snapshot,
            config_path=args.config_json,
            source_manifest_path=args.source_manifest_json,
            output_dir=args.output_dir,
            allowed_source_diffs=allowed,
        )
    except Exception as exc:  # pragma: no cover - command-line failure path.
        print(f"[preflow_snapshot_component_face_probe] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[preflow_snapshot_component_face_probe] wrote diagnostic-only evidence "
        f"with status {payload['status']!r} to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
