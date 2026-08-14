import json
import runpy
from pathlib import Path

import numpy as np
import taichi as ti

from simulation_core.coupling.hibm_mpm.core import (
    HibmMpmIbBoundaryConditions,
    HibmMpmSurfaceMarkers,
)


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_our_solver_vertical_flap.py")
TRACE_PATH = CAMPAIGN_ROOT / "diagnostic_traces" / "closure_matrix_r18.json"
TARGET_TRACE_PATH = CAMPAIGN_ROOT / "diagnostic_traces" / "target_conflict_r18.json"
_original_close = HibmMpmIbBoundaryConditions._close_owned_hard_targets_to_marker_constraints
_original_validate_target = HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_target_conflict_precommit
_original_assemble = HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
_original_refresh_tip = HibmMpmSurfaceMarkers._refresh_open_ribbon_tip_cap_projection_vertices


@ti.kernel
def _copy_tip_velocity_to_projection_vertices(
    marker_velocity_mps: ti.template(),
    primary_tip: ti.i32,
    secondary_tip: ti.i32,
    primary_edge: ti.i32,
    secondary_edge: ti.i32,
    primary_cap: ti.i32,
    secondary_cap: ti.i32,
):
    marker_velocity_mps[primary_edge] = marker_velocity_mps[primary_tip]
    marker_velocity_mps[secondary_edge] = marker_velocity_mps[secondary_tip]
    marker_velocity_mps[primary_cap] = marker_velocity_mps[primary_tip]
    marker_velocity_mps[secondary_cap] = marker_velocity_mps[secondary_tip]


def _refresh_tip_with_piecewise_constant_velocity(self):
    _original_refresh_tip(self)
    binding = getattr(self, "_open_ribbon_tip_cap_binding", None)
    if binding is None:
        return
    _copy_tip_velocity_to_projection_vertices(
        self.v_gamma_mps,
        int(binding[1]),
        int(binding[3]),
        int(binding[4]),
        int(binding[5]),
        int(binding[6]),
        int(binding[7]),
    )


def _center_coordinate(value, faces, centers, count):
    if value <= centers[0]:
        return -0.5 * (centers[0] - value) / max(centers[0] - faces[0], 1.0e-18)
    if value >= centers[count - 1]:
        return (count - 1) + 0.5 * (value - centers[count - 1]) / max(
            faces[count] - centers[count - 1], 1.0e-18
        )
    lower = int(np.searchsorted(centers[:count], value, side="right") - 1)
    upper = min(lower + 1, count - 1)
    return lower + (value - centers[lower]) / max(
        centers[upper] - centers[lower], 1.0e-18
    )


def _face_coordinate(value, faces, count):
    if value <= faces[0]:
        return (value - faces[0]) / max(faces[1] - faces[0], 1.0e-18)
    if value >= faces[count - 1]:
        return (count - 1) + (value - faces[count - 1]) / max(
            faces[count - 1] - faces[count - 2], 1.0e-18
        )
    lower = int(np.searchsorted(faces[:count], value, side="right") - 1)
    upper = min(lower + 1, count - 1)
    return lower + (value - faces[lower]) / max(
        faces[upper] - faces[lower], 1.0e-18
    )


def _stencil(position, axis, faces, centers, shape):
    coordinate = np.asarray(
        [
            _center_coordinate(position[index], faces[index], centers[index], shape[index])
            for index in range(3)
        ],
        dtype=np.float64,
    )
    coordinate[axis] = _face_coordinate(position[axis], faces[axis], shape[axis])
    base = np.clip(np.floor(coordinate).astype(np.int64), 0, np.asarray(shape) - 2)
    fraction = np.clip(coordinate - base, 0.0, 1.0)
    entries = []
    for oi in range(2):
        for oj in range(2):
            for ok in range(2):
                offset = (oi, oj, ok)
                weight = float(
                    np.prod(
                        [
                            1.0 - fraction[index]
                            if offset[index] == 0
                            else fraction[index]
                            for index in range(3)
                        ]
                    )
                )
                entries.append((tuple((base + offset).tolist()), weight))
    return tuple(base.tolist()), tuple(fraction.tolist()), entries


def _diagnose(self, kwargs, error):
    markers = kwargs["markers"]
    marker_count = int(markers.projection_vertex_count)
    physical_marker_count = int(markers.marker_count)
    primary_region = int(kwargs["primary_region_id"])
    secondary_region = int(kwargs["secondary_region_id"])
    shape = tuple(int(value) for value in self.grid_nodes)
    faces = tuple(
        np.asarray(kwargs[name].to_numpy(), dtype=np.float64)
        for name in ("cell_face_x_m", "cell_face_y_m", "cell_face_z_m")
    )
    centers = tuple(
        np.asarray(kwargs[name].to_numpy(), dtype=np.float64)
        for name in ("cell_center_x_m", "cell_center_y_m", "cell_center_z_m")
    )
    positions = np.asarray(
        self.velocity_dirichlet_marker_target_closure_sample_position_m.to_numpy(),
        dtype=np.float64,
    )
    sample_valid = np.asarray(
        self.velocity_dirichlet_marker_target_closure_sample_valid.to_numpy(),
        dtype=np.int32,
    )
    values = np.asarray(
        self.velocity_dirichlet_marker_target_closure_value_mps.to_numpy(),
        dtype=np.float64,
    )
    valid_mask = np.asarray(
        self.velocity_dirichlet_marker_target_closure_component_face_valid_mask.to_numpy(),
        dtype=np.int32,
    )
    adjustable_mask = np.asarray(
        self.velocity_dirichlet_marker_target_closure_adjustable_component_mask.to_numpy(),
        dtype=np.int32,
    )
    hard_mask = np.asarray(
        self.velocity_dirichlet_marker_target_closure_hard_fixed_component_mask.to_numpy(),
        dtype=np.int32,
    )
    external_mask = np.asarray(
        self.velocity_dirichlet_marker_target_closure_external_exact_component_mask.to_numpy(),
        dtype=np.int32,
    )
    marker_velocity = np.asarray(markers.v_gamma_mps.to_numpy(), dtype=np.float64)
    marker_region = np.asarray(markers.region_id.to_numpy(), dtype=np.int32)
    sample_source = np.asarray(
        self.velocity_dirichlet_marker_target_closure_sample_source_code.to_numpy(),
        dtype=np.int32,
    )

    rows = []
    adjustable_keys = set()
    for marker in range(marker_count):
        constrained = (
            marker_region[marker] in (primary_region, secondary_region)
            or marker >= physical_marker_count
        )
        if not constrained or sample_valid[marker] == 0:
            continue
        for axis in range(3):
            base, fraction, entries = _stencil(
                positions[marker], axis, faces, centers, shape
            )
            valid_entries = [
                (index, weight)
                for index, weight in entries
                if valid_mask[index] & (1 << axis)
            ]
            valid_weight = sum(weight for _index, weight in valid_entries)
            if valid_weight <= 1.0e-12:
                continue
            normalized = [
                (index, weight / valid_weight) for index, weight in valid_entries
            ]
            sampled = sum(weight * values[index][axis] for index, weight in normalized)
            residual = marker_velocity[marker][axis] - sampled
            adjustable = [
                (index, weight)
                for index, weight in normalized
                if weight > 1.0e-15 and adjustable_mask[index] & (1 << axis)
            ]
            q_free = [
                (index, weight)
                for index, weight in normalized
                if weight > 1.0e-15
                and not (hard_mask[index] & (1 << axis))
                and not (external_mask[index] & (1 << axis))
            ]
            if q_free or not adjustable:
                continue
            fixed_value = sum(
                weight * values[index][axis]
                for index, weight in normalized
                if not (adjustable_mask[index] & (1 << axis))
            )
            for index, _weight in adjustable:
                adjustable_keys.add((*index, axis))
            rows.append(
                {
                    "marker": marker,
                    "axis": axis,
                    "region": int(marker_region[marker]),
                    "projection_only": marker >= physical_marker_count,
                    "sample_source": int(sample_source[marker]),
                    "position_m": positions[marker].tolist(),
                    "target_mps": float(marker_velocity[marker][axis]),
                    "sampled_mps": float(sampled),
                    "residual_mps": float(residual),
                    "base": base,
                    "fraction": fraction,
                    "adjustable": [((*index, axis), float(weight)) for index, weight in adjustable],
                    "fixed_value_mps": float(fixed_value),
                    "support": [
                        {
                            "index": (*index, axis),
                            "weight": float(weight),
                            "value_mps": float(values[index][axis]),
                            "adjustable": bool(adjustable_mask[index] & (1 << axis)),
                            "hard": bool(hard_mask[index] & (1 << axis)),
                            "external": bool(external_mask[index] & (1 << axis)),
                        }
                        for index, weight in normalized
                    ],
                }
            )

    keys = sorted(adjustable_keys)
    key_to_column = {key: column for column, key in enumerate(keys)}
    matrix = np.zeros((len(rows), len(keys)), dtype=np.float64)
    rhs = np.zeros(len(rows), dtype=np.float64)
    current = np.zeros(len(keys), dtype=np.float64)
    for column, key in enumerate(keys):
        current[column] = values[key[:3]][key[3]]
    for row_index, row in enumerate(rows):
        for key, weight in row["adjustable"]:
            matrix[row_index, key_to_column[key]] += weight
        rhs[row_index] = row["target_mps"] - row["fixed_value_mps"]
    current_residual = rhs - matrix @ current
    solution, _sum_sq, rank, singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
    optimal_residual = rhs - matrix @ solution
    for index, row in enumerate(rows):
        row["current_linear_residual_mps"] = float(current_residual[index])
        row["least_squares_residual_mps"] = float(optimal_residual[index])
        row.pop("adjustable")
    top_current = sorted(rows, key=lambda row: abs(row["current_linear_residual_mps"]), reverse=True)[:20]
    top_optimal = sorted(rows, key=lambda row: abs(row["least_squares_residual_mps"]), reverse=True)[:20]
    payload = {
        "error": str(error),
        "constraint_count": len(rows),
        "adjustable_dof_count": len(keys),
        "matrix_rank": int(rank),
        "matrix_nullity": int(len(keys) - rank),
        "minimum_singular_value": float(singular_values[-1]) if singular_values.size else None,
        "maximum_singular_value": float(singular_values[0]) if singular_values.size else None,
        "current_max_abs_residual_mps": float(np.max(np.abs(current_residual))),
        "least_squares_max_abs_residual_mps": float(np.max(np.abs(optimal_residual))),
        "least_squares_l2_residual_mps": float(np.linalg.norm(optimal_residual)),
        "top_current_residuals": top_current,
        "top_least_squares_residuals": top_optimal,
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _traced_close(self, **kwargs):
    try:
        return _original_close(self, **kwargs)
    except RuntimeError as error:
        if "marker compatibility closure did not converge" in str(error):
            _diagnose(self, kwargs, error)
        raise


def _traced_validate_target(self):
    try:
        return _original_validate_target(self)
    except RuntimeError as error:
        if "conflicting canonical component-face claims (target)" in str(error):
            first_conflict = (
                self._canonical_velocity_dirichlet_first_target_conflict_diagnostic()
            )
            payload = {
                "error": str(error),
                "target_conflict_count": int(
                    self.report_velocity_dirichlet_component_face_target_conflict_count[
                        None
                    ]
                ),
                "first_conflict": first_conflict,
            }
            runtime = self.__dict__.get("_diagnostic_assemble_arguments", {})
            obstacle = runtime.get("obstacle_field")
            if first_conflict is not None and obstacle is not None:
                target = tuple(first_conflict["component_face"])
                axis = int(first_conflict["component_axis"])
                pair = (*target, axis)
                source_rows = [list(target), list(target)]
                source_rows[0][axis] -= 1
                payload["pair_state"] = {
                    "admission_valid": int(
                        self.velocity_dirichlet_component_face_segment_pair_admission_valid[
                            pair
                        ]
                    ),
                    "full_valid": int(
                        self.velocity_dirichlet_component_face_segment_pair_full_valid[
                            pair
                        ]
                    ),
                    "adjacent_direct_pair_target_valid": int(
                        self.velocity_dirichlet_component_face_adjacent_direct_pair_target_valid[
                            pair
                        ]
                    ),
                    "first_author_linear_key": int(
                        self.velocity_dirichlet_component_face_segment_pair_first_author_linear_key[
                            pair
                        ]
                    ),
                    "second_author_linear_key": int(
                        self.velocity_dirichlet_component_face_segment_pair_second_author_linear_key[
                            pair
                        ]
                    ),
                    "first_author_kind": int(
                        self.velocity_dirichlet_component_face_segment_pair_first_author_kind[
                            pair
                        ]
                    ),
                    "second_author_kind": int(
                        self.velocity_dirichlet_component_face_segment_pair_second_author_kind[
                            pair
                        ]
                    ),
                    "segment_mode": int(
                        self.velocity_dirichlet_component_face_segment_projection_only_seam[
                            pair
                        ]
                    ),
                }
                source_state = []
                for raw_row in source_rows:
                    row = tuple(raw_row)
                    source_state.append(
                        {
                            "row": row,
                            "active_ib_node": int(self.active_ib_node[row]),
                            "obstacle": int(obstacle[row]),
                            "normal": tuple(
                                float(value)
                                for value in self.pressure_neumann_normal_field[row]
                            ),
                            "direct_selected_storage_offset": tuple(
                                int(value)
                                for value in self.velocity_dirichlet_component_face_direct_selected_storage_offset[
                                    row
                                ]
                            ),
                            "actual_sample_valid": int(
                                self.velocity_dirichlet_component_face_actual_sample_valid[
                                    row
                                ]
                            ),
                            "actual_sample_point_m": tuple(
                                float(value)
                                for value in self.velocity_dirichlet_component_face_actual_sample_point_m[
                                    row
                                ]
                            ),
                            "actual_sample_velocity_mps": tuple(
                                float(value)
                                for value in self.velocity_dirichlet_component_face_actual_sample_velocity_mps[
                                    row
                                ]
                            ),
                            "shadow_valid": int(
                                self.velocity_dirichlet_relocation_shadow_claim_valid[
                                    row
                                ]
                            ),
                            "shadow_source_row": tuple(
                                int(value)
                                for value in self.velocity_dirichlet_relocation_shadow_source_row[
                                    row
                                ]
                            ),
                            "shadow_storage_base_row": tuple(
                                int(value)
                                for value in self.velocity_dirichlet_relocation_shadow_storage_base_row[
                                    row
                                ]
                            ),
                            "shadow_selected_storage_offset": tuple(
                                int(value)
                                for value in self.velocity_dirichlet_relocation_shadow_selected_storage_offset[
                                    row
                                ]
                            ),
                            "shadow_sample_point_m": tuple(
                                float(value)
                                for value in self.velocity_dirichlet_relocation_shadow_sample_point_m[
                                    row
                                ]
                            ),
                            "shadow_sample_velocity_mps": tuple(
                                float(value)
                                for value in self.velocity_dirichlet_relocation_shadow_sample_velocity_mps[
                                    row
                                ]
                            ),
                            "shadow_alpha": float(
                                self.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
                                    row
                                ]
                            ),
                        }
                    )
                payload["source_state"] = source_state
                search = runtime.get("search")
                if search is not None:
                    payload["search_support"] = {
                        "radius_xyz_m": tuple(
                            float(value)
                            for value in search._last_search_support_radius_xyz_m
                        ),
                        "anisotropic": bool(
                            search._last_search_support_anisotropic
                        ),
                        "inactive_axis": int(search._last_search_inactive_axis),
                    }
            TARGET_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            TARGET_TRACE_PATH.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        raise


def _traced_assemble(self, *args, **kwargs):
    self.__dict__["_diagnostic_assemble_arguments"] = kwargs
    return _original_assemble(self, *args, **kwargs)


HibmMpmIbBoundaryConditions._close_owned_hard_targets_to_marker_constraints = (
    _traced_close
)
HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_target_conflict_precommit = (
    _traced_validate_target
)
HibmMpmSurfaceMarkers._refresh_open_ribbon_tip_cap_projection_vertices = (
    _refresh_tip_with_piecewise_constant_velocity
)
HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger = (
    _traced_assemble
)
runpy.run_path(str(RUNNER), run_name="__main__")
