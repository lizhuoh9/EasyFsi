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
    HIBM_RELOCATION_NO_WINNER_SOURCE_LINEAR_KEY,
    HibmMpmIbBoundaryConditions,
)

_ASSEMBLY_CONTEXT: dict[str, Any] | None = None


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _row(values: np.ndarray) -> list[int]:
    return [int(value) for value in values]


def _source_from_key(key: int, *, ny: int, nz: int) -> list[int]:
    plane = ny * nz
    i, remainder = divmod(int(key), plane)
    j, k = divmod(remainder, nz)
    return [i, j, k]


def _capture(boundary: HibmMpmIbBoundaryConditions) -> dict[str, Any]:
    claim_count = boundary.velocity_dirichlet_component_face_claim_count.to_numpy()
    claim_target = boundary.velocity_dirichlet_component_face_claim_target_mps.to_numpy()
    claim_region = boundary.velocity_dirichlet_component_face_claim_region_id.to_numpy()
    claim_alpha = boundary.velocity_dirichlet_component_face_claim_alpha.to_numpy()
    active = boundary.active_ib_node.to_numpy()
    velocity = boundary.velocity_dirichlet_mps_field.to_numpy()
    normal = boundary.pressure_neumann_normal_field.to_numpy()
    shadow_valid = (
        boundary.velocity_dirichlet_relocation_shadow_claim_valid.to_numpy()
    )
    shadow_source = (
        boundary.velocity_dirichlet_relocation_shadow_source_row.to_numpy()
    )
    shadow_storage = (
        boundary.velocity_dirichlet_relocation_shadow_storage_base_row.to_numpy()
    )
    winner_key = (
        boundary.velocity_dirichlet_relocation_winner_source_linear_key.to_numpy()
    )
    ny, nz = int(boundary.grid_nodes[1]), int(boundary.grid_nodes[2])

    duplicate_lanes: list[dict[str, Any]] = []
    for i, j, k, axis in np.argwhere(claim_count > 1):
        row = (int(i), int(j), int(k))
        duplicate_lanes.append(
            {
                "storage_row": list(row),
                "axis": int(axis),
                "claim_count": int(claim_count[row][axis]),
                "published_target_mps": float(claim_target[row][axis]),
                "published_region_id": int(claim_region[row][axis]),
                "published_alpha": float(claim_alpha[row][axis]),
            }
        )

    direct_direct_lanes: list[dict[str, Any]] = []
    direct_direct_target_conflicts: list[dict[str, Any]] = []
    context = _ASSEMBLY_CONTEXT
    if context is not None:
        obstacle = context["obstacle_field"].to_numpy()
        search = context["search"]
        boundary_point = search.node_boundary_point_m.to_numpy()
        interior_point = search.node_interior_fluid_point_m.to_numpy()
        nearest_marker = search.nearest_marker.to_numpy()
        projection_indices = search.node_projection_marker_indices.to_numpy()
        projection_weights = search.node_projection_marker_weights.to_numpy()
        marker_region_id = context["marker_region_id"].to_numpy()
        marker_position_field = context.get("marker_position_m")
        marker_velocity_field = context.get("marker_velocity_mps")
        markers = context.get("markers")
        if marker_position_field is None and markers is not None:
            marker_position_field = markers.x_gamma_m
        if marker_velocity_field is None and markers is not None:
            marker_velocity_field = markers.v_gamma_mps
        marker_position = (
            marker_position_field.to_numpy()
            if marker_position_field is not None
            else None
        )
        marker_velocity = (
            marker_velocity_field.to_numpy()
            if marker_velocity_field is not None
            else None
        )
        inactive_axis = int(context.get("surface_projection_inactive_axis", -1))
        faces = tuple(
            context[name].to_numpy()
            for name in ("cell_face_x_m", "cell_face_y_m", "cell_face_z_m")
        )
        centers = tuple(
            context[name].to_numpy()
            for name in (
                "cell_center_x_m",
                "cell_center_y_m",
                "cell_center_z_m",
            )
        )

        for lane in duplicate_lanes:
            target = tuple(int(value) for value in lane["storage_row"])
            axis = int(lane["axis"])
            minus = list(target)
            minus[axis] -= 1
            sources = (tuple(minus), target)
            if not all(
                min(source) >= 0
                and active[source] != 0
                and obstacle[source] == 0
                for source in sources
            ):
                continue
            targets = tuple(float(velocity[source][axis]) for source in sources)
            face_center = np.asarray(
                [float(centers[index][target[index]]) for index in range(3)],
                dtype=np.float64,
            )
            face_center[axis] = float(faces[axis][target[axis]])
            authors: list[dict[str, Any]] = []
            for source, target_value in zip(sources, targets):
                boundary_value = boundary_point[source].astype(np.float64)
                interior_value = interior_point[source].astype(np.float64)
                segment = interior_value - boundary_value
                distance_squared = float(np.dot(segment, segment))
                progress = float(
                    np.dot(face_center - boundary_value, segment)
                    / distance_squared
                )
                marker = int(nearest_marker[source])
                author_projection_indices = _row(projection_indices[source])
                valid_projection_indices = tuple(
                    index
                    for index in author_projection_indices
                    if 0 <= index < marker_region_id.shape[0]
                )
                authors.append(
                    {
                        "source_row": list(source),
                        "target_mps": target_value,
                        "boundary_point_m": _vector(boundary_value),
                        "interior_point_m": _vector(interior_value),
                        "normal": _vector(normal[source]),
                        "face_distance_from_boundary_m": float(
                            np.linalg.norm(face_center - boundary_value)
                        ),
                        "segment_progress": progress,
                        "nearest_marker": marker,
                        "nearest_marker_region_id": (
                            int(marker_region_id[marker])
                            if 0 <= marker < marker_region_id.shape[0]
                            else -1
                        ),
                        "projection_marker_indices": author_projection_indices,
                        "projection_marker_weights": _vector(
                            projection_weights[source]
                        ),
                        "projection_marker_positions_m": (
                            [
                                _vector(marker_position[index])
                                for index in valid_projection_indices
                            ]
                            if marker_position is not None
                            else []
                        ),
                        "projection_marker_velocities_mps": (
                            [
                                _vector(marker_velocity[index])
                                for index in valid_projection_indices
                            ]
                            if marker_velocity is not None
                            else []
                        ),
                    }
                )
            lane_payload = {
                "storage_row": list(target),
                "axis": axis,
                "face_center_m": _vector(face_center),
                "inactive_axis": inactive_axis,
                "target_cell_width_m": [
                    float(abs(faces[index][target[index] + 1] - faces[index][target[index]]))
                    for index in range(3)
                ],
                "preceding_component_cell_width_m": (
                    float(
                        abs(
                            faces[axis][target[axis]]
                            - faces[axis][target[axis] - 1]
                        )
                    )
                    if target[axis] > 0
                    else 0.0
                ),
                "absolute_target_delta_mps": abs(targets[0] - targets[1]),
                "authors": authors,
            }
            direct_direct_lanes.append(lane_payload)
            if abs(targets[0] - targets[1]) > 1.0e-6:
                direct_direct_target_conflicts.append(lane_payload)

    relocation_winners: list[dict[str, Any]] = []
    for i, j, k in np.argwhere(
        winner_key != HIBM_RELOCATION_NO_WINNER_SOURCE_LINEAR_KEY
    ):
        destination = (int(i), int(j), int(k))
        key = int(winner_key[destination])
        relocation_winners.append(
            {
                "destination_row": list(destination),
                "source_key": key,
                "source_row": _source_from_key(key, ny=ny, nz=nz),
                "shadow_valid": bool(shadow_valid[destination]),
                "shadow_source_row": _row(shadow_source[destination]),
                "shadow_storage_base_row": _row(shadow_storage[destination]),
            }
        )

    scalar = lambda field: int(field[None])
    return {
        "target_conflict_count": scalar(
            boundary.report_velocity_dirichlet_component_face_target_conflict_count
        ),
        "region_conflict_count": scalar(
            boundary.report_velocity_dirichlet_component_face_region_conflict_count
        ),
        "alpha_conflict_count": scalar(
            boundary.report_velocity_dirichlet_component_face_alpha_conflict_count
        ),
        "claim_conflict_count": scalar(
            boundary.report_velocity_dirichlet_component_face_conflict_count
        ),
        "duplicate_claim_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_duplicate_claim_count
        ),
        "direct_geometry_reconstructed_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_direct_geometry_reconstructed_count
        ),
        "direct_geometry_one_sided_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_direct_geometry_one_sided_count
        ),
        "segment_identical_provenance_merged_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_segment_identical_provenance_merged_count
        ),
        "segment_endpoint_clamped_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_segment_endpoint_clamped_count
        ),
        "max_segment_endpoint_clamp_overrun_support_ratio": float(
            boundary.report_velocity_dirichlet_component_face_max_segment_endpoint_clamp_overrun_support_ratio[
                None
            ]
        ),
        "relocation_merged_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_relocation_merged_count
        ),
        "relocated_claim_component_count": scalar(
            boundary.report_velocity_dirichlet_component_face_relocated_claim_count
        ),
        "duplicate_claim_lanes": duplicate_lanes,
        "active_ib_row_count": int(np.count_nonzero(active)),
        "direct_direct_lanes": direct_direct_lanes,
        "direct_direct_target_conflicts": direct_direct_target_conflicts,
        "relocation_winners": relocation_winners,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-output", required=True)
    parser.add_argument("--runner-script", required=True)
    args, runner_args = parser.parse_known_args()
    output_path = Path(args.diagnostic_output).resolve()
    runner_script = Path(args.runner_script).resolve()
    original_validate = (
        HibmMpmIbBoundaryConditions
        ._validate_canonical_velocity_dirichlet_relocation_precommit
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
        *,
        interpolate_interior_velocity: bool,
    ) -> None:
        payload = _capture(self)
        payload["interpolate_interior_velocity"] = bool(
            interpolate_interior_velocity
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        original_validate(
            self,
            interpolate_interior_velocity=interpolate_interior_velocity,
        )

    HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_relocation_precommit = (  # type: ignore[method-assign]
        capture_then_validate
    )
    HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger = (  # type: ignore[method-assign]
        capture_context_then_assemble
    )
    sys.argv = [str(runner_script), *runner_args]
    try:
        runpy.run_path(str(runner_script), run_name="__main__")
    finally:
        HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_relocation_precommit = (  # type: ignore[method-assign]
            original_validate
        )
        HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger = (  # type: ignore[method-assign]
            original_assemble
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
