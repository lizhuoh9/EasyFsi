import ast
import inspect
import textwrap
import unittest

import numpy as np
import taichi as ti

from simulation_core import (
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)


@ti.kernel
def _short_f32_segment_pair_geometry_probe(
    boundary: ti.template(),
    marker_position_m: ti.template(),
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    first_anchor_z_m: ti.f32,
    second_anchor_z_m: ti.f32,
    result: ti.template(),
):
    (
        admission_valid,
        valid,
        canonical_boundary_point,
        canonical_normal,
        canonical_nominal_probe,
        face_segment_parameter,
        author_interpolation_weight,
        maximum_geometry_tolerance,
    ) = (
        boundary._canonical_component_face_distinct_finite_segment_pair_geometry(
            ti.Vector([0, 1, 1]),
            ti.Vector([0.125, 0.25, 0.375]),
            ti.Vector([0.125, 0.125, first_anchor_z_m]),
            ti.Vector([0.125, 0.125, second_anchor_z_m]),
            ti.Vector([0.125, 0.375, first_anchor_z_m]),
            ti.Vector([0.125, 0.375, second_anchor_z_m]),
            ti.Vector([0.125, 0.375, first_anchor_z_m]),
            ti.Vector([0.125, 0.375, second_anchor_z_m]),
            ti.Vector([0.0, 1.0, 0.0]),
            ti.Vector([0.0, 1.0, 0.0]),
            ti.Vector([0, 1, -1]),
            ti.Vector([0.7, 0.3, 0.0]),
            ti.Vector([0.3, 0.7, 0.0]),
            0,
            marker_position_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
        )
    )
    result[0] = admission_valid
    result[1] = valid


@ti.kernel
def _captured_finite_segment_union_admission_probe(
    boundary: ti.template(),
    marker_position_m: ti.template(),
    marker_velocity_mps: ti.template(),
    marker_region_id: ti.template(),
    face_center_m: ti.template(),
    component_axis: ti.i32,
    author_source_center_m: ti.template(),
    author_boundary_point_m: ti.template(),
    author_nominal_probe_m: ti.template(),
    author_actual_probe_m: ti.template(),
    author_normal: ti.template(),
    author_projection_indices: ti.template(),
    author_projection_weights: ti.template(),
    projection_segment_indices: ti.template(),
    projection_segment_count: ti.i32,
    configured_source_support_xyz_m: ti.template(),
    configured_source_support_available: ti.i32,
    configured_source_support_anisotropic: ti.i32,
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    claim_region: ti.i32,
    allow_inactive_axis_extrusion_direct_pair: ti.i32,
    allow_inactive_axis_double_relocation_face_transport: ti.i32,
    result_i32: ti.template(),
    result_f64: ti.template(),
):
    # The topology arguments deliberately accompany the direct contract even
    # before the production helper consumes them.  This makes the current
    # terminal false-positive an assertion RED, while preserving the exact
    # topology input that the production fix must wire into admission.
    target = ti.Vector([0, 1, 2])
    face_center = face_center_m[None]
    first_projection_indices = ti.Vector(
        [
            author_projection_indices[0].x,
            author_projection_indices[0].y,
            -1,
        ]
    )
    second_projection_indices = ti.Vector(
        [
            author_projection_indices[1].x,
            author_projection_indices[1].y,
            -1,
        ]
    )
    first_nearest_marker = ti.select(
        author_projection_weights[0].y > author_projection_weights[0].x,
        first_projection_indices.y,
        first_projection_indices.x,
    )
    second_nearest_marker = ti.select(
        author_projection_weights[1].y > author_projection_weights[1].x,
        second_projection_indices.y,
        second_projection_indices.x,
    )
    # Direct tests may encode explicit nearest-marker provenance in the
    # otherwise unused input z lane.  Production segment provenance remains
    # the canonical (a, b, -1) vector passed to the helper below.
    if author_projection_indices[0].z >= 0:
        first_nearest_marker = author_projection_indices[0].z
    if author_projection_indices[1].z >= 0:
        second_nearest_marker = author_projection_indices[1].z
    (
        first_valid,
        first_target,
        first_distance_squared,
        first_closest_point,
        first_endpoint_clamped,
        first_clamp_support_ratio,
    ) = boundary._canonical_component_face_segment_projection_target(
        target,
        component_axis,
        face_center,
        first_projection_indices.x,
        first_projection_indices.y,
        claim_region,
        0,
        1,
        marker_position_m,
        marker_velocity_mps,
        marker_region_id,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
    )
    (
        second_valid,
        second_target,
        second_distance_squared,
        second_closest_point,
        second_endpoint_clamped,
        second_clamp_support_ratio,
    ) = boundary._canonical_component_face_segment_projection_target(
        target,
        component_axis,
        face_center,
        second_projection_indices.x,
        second_projection_indices.y,
        claim_region,
        0,
        1,
        marker_position_m,
        marker_velocity_mps,
        marker_region_id,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
    )
    (
        pair_admission_valid,
        pair_full_valid,
        pair_boundary_point,
        pair_normal,
        pair_nominal_probe,
        pair_boundary_target,
        pair_endpoint_clamped,
        pair_clamp_support_ratio,
        pair_geometry_tolerance,
        _pair_direct_face_owner_shadow,
    ) = boundary._canonical_component_face_finite_segment_union_owner_geometry(
        target,
        component_axis,
        face_center,
        author_source_center_m[0],
        author_source_center_m[1],
        author_source_center_m[0],
        author_source_center_m[1],
        author_boundary_point_m[0],
        author_boundary_point_m[1],
        author_nominal_probe_m[0],
        author_nominal_probe_m[1],
        author_actual_probe_m[0],
        author_actual_probe_m[1],
        author_normal[0],
        author_normal[1],
        first_projection_indices,
        second_projection_indices,
        author_projection_weights[0],
        author_projection_weights[1],
        first_nearest_marker,
        second_nearest_marker,
        claim_region,
        configured_source_support_available,
        configured_source_support_anisotropic,
        configured_source_support_xyz_m[None],
        projection_segment_indices,
        projection_segment_count,
        projection_segment_count > 0,
        0,
        allow_inactive_axis_extrusion_direct_pair,
        allow_inactive_axis_double_relocation_face_transport,
        -1,
        marker_position_m,
        marker_velocity_mps,
        marker_region_id,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
    )
    result_i32[0] = first_valid
    result_i32[1] = second_valid
    result_i32[2] = first_endpoint_clamped
    result_i32[3] = second_endpoint_clamped
    result_i32[4] = pair_admission_valid
    result_i32[5] = pair_full_valid
    result_i32[6] = pair_endpoint_clamped
    result_f64[0] = first_distance_squared
    result_f64[1] = second_distance_squared
    result_f64[2] = ti.cast(first_clamp_support_ratio, ti.f64)
    result_f64[3] = ti.cast(second_clamp_support_ratio, ti.f64)
    result_f64[4] = ti.cast(pair_clamp_support_ratio, ti.f64)
    result_f64[5] = ti.cast(pair_geometry_tolerance, ti.f64)
    result_f64[6] = ti.cast(pair_boundary_target, ti.f64)
    for axis in ti.static(range(3)):
        result_f64[7 + axis] = ti.cast(pair_boundary_point[axis], ti.f64)
        result_f64[10 + axis] = ti.cast(pair_normal[axis], ti.f64)
        result_f64[13 + axis] = ti.cast(pair_nominal_probe[axis], ti.f64)


@ti.kernel
def _same_storage_direct_relocation_geometry_probe(
    boundary: ti.template(),
    marker_position_m: ti.template(),
    marker_velocity_mps: ti.template(),
    marker_region_id: ti.template(),
    target_index: ti.template(),
    face_center_m: ti.template(),
    component_axis: ti.i32,
    surface_projection_inactive_axis: ti.i32,
    author_source_center_m: ti.template(),
    author_boundary_point_m: ti.template(),
    author_nominal_probe_m: ti.template(),
    author_actual_probe_m: ti.template(),
    author_normal: ti.template(),
    author_projection_indices: ti.template(),
    author_projection_weights: ti.template(),
    projection_segment_indices: ti.template(),
    projection_segment_count: ti.i32,
    configured_source_support_xyz_m: ti.template(),
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    claim_region: ti.i32,
    marker_capacity: ti.i32,
    result_i32: ti.template(),
    result_f64: ti.template(),
):
    target = target_index[None]
    direct_indices = ti.Vector(
        [
            author_projection_indices[0].x,
            author_projection_indices[0].y,
            -1,
        ]
    )
    relocation_indices = ti.Vector(
        [
            author_projection_indices[1].x,
            author_projection_indices[1].y,
            -1,
        ]
    )
    direct_nearest = ti.select(
        author_projection_weights[0].y > author_projection_weights[0].x,
        direct_indices.y,
        direct_indices.x,
    )
    relocation_nearest = ti.select(
        author_projection_weights[1].y > author_projection_weights[1].x,
        relocation_indices.y,
        relocation_indices.x,
    )
    if author_projection_indices[0].z >= 0:
        direct_nearest = author_projection_indices[0].z
    if author_projection_indices[1].z >= 0:
        relocation_nearest = author_projection_indices[1].z
    direct_target = (
        author_projection_weights[0].x
        * marker_velocity_mps[direct_indices.x][component_axis]
        + author_projection_weights[0].y
        * marker_velocity_mps[direct_indices.y][component_axis]
    )
    relocation_target = (
        author_projection_weights[1].x
        * marker_velocity_mps[relocation_indices.x][component_axis]
        + author_projection_weights[1].y
        * marker_velocity_mps[relocation_indices.y][component_axis]
    )
    (
        admission_valid,
        full_valid,
        canonical_boundary,
        canonical_normal,
        canonical_probe,
        canonical_target,
        geometry_tolerance,
    ) = boundary._canonical_component_face_same_storage_direct_relocation_geometry(
        target,
        component_axis,
        face_center_m[None],
        author_source_center_m[0],
        author_source_center_m[1],
        author_boundary_point_m[0],
        author_nominal_probe_m[0],
        author_actual_probe_m[0],
        author_boundary_point_m[1],
        author_actual_probe_m[1],
        author_normal[0],
        author_normal[1],
        direct_indices,
        relocation_indices,
        author_projection_weights[0],
        author_projection_weights[1],
        direct_nearest,
        relocation_nearest,
        direct_target,
        relocation_target,
        claim_region,
        1,
        1,
        configured_source_support_xyz_m[None],
        projection_segment_indices,
        projection_segment_count,
        projection_segment_count > 0,
        marker_position_m,
        marker_velocity_mps,
        marker_region_id,
        marker_capacity,
        surface_projection_inactive_axis,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
    )
    result_i32[0] = admission_valid
    result_i32[1] = full_valid
    result_f64[5] = ti.cast(geometry_tolerance, ti.f64)
    result_f64[6] = ti.cast(canonical_target, ti.f64)
    for axis in ti.static(range(3)):
        result_f64[7 + axis] = ti.cast(canonical_boundary[axis], ti.f64)
        result_f64[10 + axis] = ti.cast(canonical_normal[axis], ti.f64)
        result_f64[13 + axis] = ti.cast(canonical_probe[axis], ti.f64)


@ti.kernel
def _same_storage_segment_mode_validation_probe(
    boundary: ti.template(),
    result: ti.template(),
):
    result[0] = boundary._canonical_component_face_same_storage_segment_mode_is_valid(
        12
    )
    result[1] = boundary._canonical_component_face_same_storage_segment_mode_is_valid(
        12 | 16
    )
    result[2] = boundary._canonical_component_face_same_storage_segment_mode_is_valid(
        12 | 64
    )
    result[3] = boundary._canonical_component_face_same_storage_segment_mode_is_valid(
        4
    )
    result[4] = boundary._canonical_component_face_same_storage_segment_mode_is_valid(
        8
    )
    result[5] = boundary._canonical_component_face_same_storage_segment_mode_is_valid(
        0
    )


class HibmMpmSegmentPairGeometryTests(unittest.TestCase):
    def test_same_storage_segment_mode_accepts_only_exact_bit8_pair(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=2,
            runtime=runtime,
        )
        result = ti.field(dtype=ti.i32, shape=6)

        _same_storage_segment_mode_validation_probe(boundary, result)

        self.assertEqual(
            tuple(int(result[index]) for index in range(6)),
            (1, 0, 0, 0, 0, 0),
        )

    def test_storage_classifier_caches_direct_and_shadow_component_offsets(
        self,
    ) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=2,
            runtime=runtime,
        )
        obstacle = ti.field(dtype=ti.i32, shape=(4, 4, 4))
        node_boundary = ti.Vector.field(3, dtype=ti.f32, shape=(4, 4, 4))
        cell_face_x = ti.field(dtype=ti.f32, shape=5)
        cell_face_y = ti.field(dtype=ti.f32, shape=5)
        cell_face_z = ti.field(dtype=ti.f32, shape=5)
        cell_center_x = ti.field(dtype=ti.f32, shape=4)
        cell_center_y = ti.field(dtype=ti.f32, shape=4)
        cell_center_z = ti.field(dtype=ti.f32, shape=4)
        cell_face_x.from_numpy(
            np.asarray((0.0, 0.1, 0.4, 0.7, 1.0), dtype=np.float32)
        )
        cell_center_x.from_numpy(
            np.asarray((0.05, 0.25, 0.55, 0.85), dtype=np.float32)
        )
        for face_field in (cell_face_y, cell_face_z):
            face_field.from_numpy(
                np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float32)
            )
        for center_field in (cell_center_y, cell_center_z):
            center_field.from_numpy(
                np.asarray((0.125, 0.375, 0.625, 0.875), dtype=np.float32)
            )

        lower_direct = (1, 1, 1)
        target_direct = (2, 2, 2)
        shadow_source = (2, 1, 2)
        boundary.active_ib_node[lower_direct] = 1
        boundary.velocity_dirichlet_component_face_actual_sample_valid[
            lower_direct
        ] = 1
        node_boundary[lower_direct] = (0.25, 0.375, 0.375)
        boundary.velocity_dirichlet_component_face_actual_sample_point_m[
            lower_direct
        ] = (0.75, 0.375, 0.375)

        boundary.active_ib_node[target_direct] = 1
        boundary.velocity_dirichlet_component_face_actual_sample_valid[
            target_direct
        ] = 1
        node_boundary[target_direct] = (0.54, 0.375, 0.625)
        boundary.velocity_dirichlet_component_face_actual_sample_point_m[
            target_direct
        ] = (0.54, 0.75, 0.625)
        boundary.active_ib_node[shadow_source] = 1
        obstacle[shadow_source] = 1
        node_boundary[shadow_source] = (0.54, 0.375, 0.625)
        boundary.velocity_dirichlet_relocation_shadow_claim_valid[target_direct] = 1
        boundary.velocity_dirichlet_relocation_shadow_source_row[target_direct] = (
            shadow_source
        )
        boundary.velocity_dirichlet_relocation_shadow_storage_base_row[
            target_direct
        ] = target_direct
        boundary.velocity_dirichlet_relocation_shadow_sample_point_m[
            target_direct
        ] = (0.54, 0.75, 0.625)

        boundary._classify_canonical_component_face_storage_kernel(
            obstacle,
            node_boundary,
            cell_face_x,
            cell_face_y,
            cell_face_z,
            cell_center_x,
            cell_center_y,
            cell_center_z,
            4,
            4,
            4,
        )

        self.assertEqual(
            tuple(
                int(value)
                for value in
                boundary.velocity_dirichlet_component_face_direct_selected_storage_offset[
                    lower_direct
                ]
            ),
            (1, 0, 0),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in
                boundary.velocity_dirichlet_component_face_direct_selected_storage_offset[
                    target_direct
                ]
            ),
            (0, 1, 0),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in
                boundary.velocity_dirichlet_relocation_shadow_selected_storage_offset[
                    target_direct
                ]
            ),
            (0, 1, 0),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in
                boundary.velocity_dirichlet_component_face_direct_selected_storage_offset[
                    (3, 3, 3)
                ]
            ),
            (-1, -1, -1),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in
                boundary.velocity_dirichlet_relocation_shadow_selected_storage_offset[
                    (3, 3, 3)
                ]
            ),
            (-1, -1, -1),
        )

    @staticmethod
    def _run_direct_finite_segment_union_case(
        case: dict[str, object],
    ) -> tuple[tuple[int, ...], tuple[float, ...], tuple[tuple[int, int], ...]]:
        """Evaluate one compact pair fixture without ledger integration."""

        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        positions = np.asarray(case["positions"], dtype=np.float32)
        velocities = np.asarray(case["velocities"], dtype=np.float32)
        marker_count = int(positions.shape[0])
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=marker_count,
            runtime=runtime,
        )
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=marker_count,
            projection_triangle_capacity=max(marker_count, 1),
            runtime=runtime,
        )
        markers.load_markers(
            positions_m=tuple(tuple(row) for row in positions),
            velocities_mps=tuple(tuple(row) for row in velocities),
            normals=((0.0, 0.0, 1.0),) * marker_count,
            areas_m2=(1.0 / marker_count,) * marker_count,
            region_ids=tuple(
                int(region)
                for region in case.get(
                    "region_ids",
                    (int(case["region"]),) * marker_count,
                )
            ),
        )
        projection_segments = tuple(
            tuple(int(value) for value in segment)
            for segment in case.get("projection_segments", ())
        )
        if projection_segments:
            if case.get("raw_projection_segments", False):
                for segment_index, (marker_a, marker_b) in enumerate(
                    projection_segments
                ):
                    markers.projection_triangle_indices[segment_index] = (
                        marker_a,
                        marker_b,
                        -1,
                    )
                markers.projection_segment_count = len(projection_segments)
            else:
                markers.set_projection_segments(projection_segments)

        target_index = ti.Vector.field(3, dtype=ti.i32, shape=())
        face_center_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        author_source_center_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_boundary_point_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_nominal_probe_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_actual_probe_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_normal = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_projection_indices = ti.Vector.field(3, dtype=ti.i32, shape=2)
        author_projection_weights = ti.Vector.field(3, dtype=ti.f32, shape=2)
        configured_source_support_xyz_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        cell_face_x_m = ti.field(dtype=ti.f32, shape=5)
        cell_face_y_m = ti.field(dtype=ti.f32, shape=5)
        cell_face_z_m = ti.field(dtype=ti.f32, shape=5)
        result_i32 = ti.field(dtype=ti.i32, shape=7)
        result_f64 = ti.field(dtype=ti.f64, shape=16)

        target_index[None] = case.get("target", (0, 1, 2))
        face_center_m[None] = case["face"]
        author_source_center_m.from_numpy(
            np.asarray(case["source_centers"], dtype=np.float32)
        )
        author_boundary_point_m.from_numpy(
            np.asarray(case["boundaries"], dtype=np.float32)
        )
        author_nominal_probe_m.from_numpy(
            np.asarray(case["probes"], dtype=np.float32)
        )
        author_actual_probe_m.from_numpy(
            np.asarray(case["probes"], dtype=np.float32)
        )
        author_normal.from_numpy(np.asarray(case["normals"], dtype=np.float32))
        projection_indices = np.asarray(case["indices"], dtype=np.int32)
        if "nearest_markers" in case:
            projection_indices = projection_indices.copy()
            projection_indices[:, 2] = np.asarray(
                case["nearest_markers"], dtype=np.int32
            )
        author_projection_indices.from_numpy(projection_indices)
        author_projection_weights.from_numpy(
            np.asarray(case["weights"], dtype=np.float32)
        )
        configured_source_support_xyz_m[None] = case.get(
            "configured_source_support_xyz_m",
            (1.0e30, 1.0e30, 1.0e30),
        )
        cell_face_x_m.from_numpy(
            np.asarray(case["cell_face_x_m"], dtype=np.float32)
        )
        cell_face_y_m.from_numpy(
            np.asarray(case["cell_face_y_m"], dtype=np.float32)
        )
        cell_face_z_m.from_numpy(
            np.asarray(case["cell_face_z_m"], dtype=np.float32)
        )
        result_i32.fill(-1)
        result_f64.fill(np.nan)

        if case.get("same_storage_direct_relocation", False):
            _same_storage_direct_relocation_geometry_probe(
                boundary,
                markers.x_gamma_m,
                markers.v_gamma_mps,
                markers.region_id,
                target_index,
                face_center_m,
                int(case.get("component_axis", 2)),
                int(case.get("surface_projection_inactive_axis", 0)),
                author_source_center_m,
                author_boundary_point_m,
                author_nominal_probe_m,
                author_actual_probe_m,
                author_normal,
                author_projection_indices,
                author_projection_weights,
                markers.projection_triangle_indices,
                int(markers.projection_segment_count),
                configured_source_support_xyz_m,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                int(case["region"]),
                marker_count,
                result_i32,
                result_f64,
            )
        else:
            _captured_finite_segment_union_admission_probe(
                boundary,
                markers.x_gamma_m,
                markers.v_gamma_mps,
                markers.region_id,
                face_center_m,
                int(case.get("component_axis", 2)),
                author_source_center_m,
                author_boundary_point_m,
                author_nominal_probe_m,
                author_actual_probe_m,
                author_normal,
                author_projection_indices,
                author_projection_weights,
                markers.projection_triangle_indices,
                int(markers.projection_segment_count),
                configured_source_support_xyz_m,
                int(case.get("configured_source_support_available", 1)),
                int(case.get("configured_source_support_anisotropic", 1)),
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                int(case["region"]),
                int(case.get("allow_inactive_axis_extrusion_direct_pair", 0)),
                int(
                    case.get(
                        "allow_inactive_axis_double_relocation_face_transport",
                        0,
                    )
                ),
                result_i32,
                result_f64,
            )
        installed_segment_array = markers.projection_triangle_indices.to_numpy()
        installed_segments = tuple(
            tuple(int(value) for value in installed_segment_array[index, :2])
            for index in range(int(markers.projection_segment_count))
        )
        return (
            tuple(int(result_i32[index]) for index in range(7)),
            tuple(float(result_f64[index]) for index in range(16)),
            installed_segments,
        )

    @staticmethod
    def _inactive_axis_offset_anchor_transport_case(
        *,
        allow_double_relocation_transport: bool,
    ) -> dict[str, object]:
        """Return one equal-author anchor offset from its canonical face point."""

        return {
            "component_axis": 0,
            "face": (0.1, 0.625, 0.375),
            "region": 303,
            "positions": ((0.1, 0.375, 0.125), (0.1, 0.875, 0.125)),
            "velocities": ((2.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
            "projection_segments": ((0, 1),),
            "source_centers": ((0.05, 0.625, 0.375), (0.15, 0.625, 0.375)),
            "boundaries": ((0.05, 0.635, 0.125), (0.15, 0.635, 0.125)),
            "probes": ((0.05, 0.635, 0.625), (0.15, 0.635, 0.625)),
            "normals": ((0.0, 0.0, 1.0),) * 2,
            "indices": ((0, 1, -1),) * 2,
            "weights": ((0.48, 0.52, 0.0),) * 2,
            "cell_face_x_m": (0.1, 0.2, 0.5, 0.8, 1.1),
            "cell_face_y_m": (0.0, 0.5, 0.75, 1.0, 1.25),
            "cell_face_z_m": (-0.25, 0.0, 0.25, 0.5, 0.75),
            "allow_inactive_axis_extrusion_direct_pair": 1,
            "allow_inactive_axis_double_relocation_face_transport": int(
                allow_double_relocation_transport
            ),
        }

    def test_inactive_axis_direct_pair_cannot_transport_offset_anchor(self) -> None:
        """Bit16's direct proof cannot consume bit64-only anchor transport."""

        case = self._inactive_axis_offset_anchor_transport_case(
            allow_double_relocation_transport=False
        )
        integer_result, _, _ = self._run_direct_finite_segment_union_case(case)

        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(integer_result[4:6], (0, 0))

    def test_inactive_axis_double_relocation_transports_offset_anchor(self) -> None:
        """The bit64 proof reconstructs canonical geometry at the MAC face."""

        case = self._inactive_axis_offset_anchor_transport_case(
            allow_double_relocation_transport=True
        )
        integer_result, floating_result, installed_segments = (
            self._run_direct_finite_segment_union_case(case)
        )

        self.assertEqual(integer_result, (1, 1, 0, 0, 1, 1, 0))
        self.assertEqual(installed_segments, ((0, 1),))
        self.assertAlmostEqual(floating_result[6], 3.0, places=6)
        for start, expected in (
            (7, (0.1, 0.625, 0.125)),
            (10, (0.0, 0.0, 1.0)),
            (13, (0.1, 0.625, 0.625)),
        ):
            for actual, expected_component in zip(
                floating_result[start : start + 3],
                expected,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected_component, places=6)

    @staticmethod
    def _same_storage_component_axis_case(
        *,
        component_axis: int,
        shadow_anchor_delta_m: float = 0.0,
    ) -> dict[str, object]:
        """Return the compact kind0/kind1 witness from production replay Z."""

        marker_z = (0.0498947069626969, 0.0471474417218531)
        direct_t = 0.3598177433013916
        production_shadow_t = 0.35980960726737976
        direct_anchor_z = marker_z[0] + direct_t * (marker_z[1] - marker_z[0])
        shadow_anchor_z = marker_z[0] + production_shadow_t * (
            marker_z[1] - marker_z[0]
        )
        if shadow_anchor_delta_m:
            shadow_anchor_z = direct_anchor_z + shadow_anchor_delta_m
            production_shadow_t = (shadow_anchor_z - marker_z[0]) / (
                marker_z[1] - marker_z[0]
            )

        x_faces = (0.0, 0.00075, 0.0015, 0.00225, 0.003)
        y_faces = (
            0.010000000,
            0.010078125,
            0.010156250,
            0.010234375,
            0.010312500,
        )
        if component_axis == 0:
            z_faces = (
                0.0,
                0.025,
                direct_anchor_z - 0.00015625,
                direct_anchor_z + 0.00015625,
                0.1,
            )
            source_z = direct_anchor_z
            face = (x_faces[2], 0.0101953125, source_z)
        elif component_axis == 2:
            z_faces = (
                0.0,
                0.025,
                direct_anchor_z,
                direct_anchor_z + 0.0003125,
                0.1,
            )
            source_z = direct_anchor_z + 0.00015625
            face = (0.5 * (x_faces[2] + x_faces[3]), 0.0101953125, z_faces[2])
        else:
            raise ValueError("component_axis must be zero or two")
        direct_source = (0.001875, 0.0101953125, source_z)
        relocation_source = (0.001875, 0.0101171875, source_z)
        direct_boundary = (0.001875, 0.01, direct_anchor_z)
        relocation_boundary = (0.001875, 0.01, shadow_anchor_z)
        direct_probe = (0.001875, 0.0103125, direct_anchor_z)
        relocation_probe = (0.001875, 0.010234375, shadow_anchor_z)
        marker_velocities = ((0.0, 0.0, 0.0),) * 2
        if component_axis == 0 and shadow_anchor_delta_m == 0.0:
            marker_velocities = ((2.0, 0.0, 0.0), (4.0, 0.0, 0.0))
        return {
            "same_storage_direct_relocation": True,
            "target": (2, 2, 2),
            "component_axis": component_axis,
            "surface_projection_inactive_axis": 0,
            "face": face,
            "region": 303,
            "positions": (
                (0.0015, 0.01, marker_z[0]),
                (0.0015, 0.01, marker_z[1]),
            ),
            "velocities": marker_velocities,
            "projection_segments": ((0, 1),),
            "source_centers": (direct_source, relocation_source),
            "boundaries": (direct_boundary, relocation_boundary),
            "probes": (direct_probe, relocation_probe),
            "normals": ((0.0, 1.0, 0.0),) * 2,
            "indices": ((0, 1, -1),) * 2,
            "weights": (
                (1.0 - direct_t, direct_t, 0.0),
                (1.0 - production_shadow_t, production_shadow_t, 0.0),
            ),
            "nearest_markers": (0, 0),
            "configured_source_support_xyz_m": (
                0.0012,
                0.000390625,
                0.00046875,
            ),
            "cell_face_x_m": x_faces,
            "cell_face_y_m": y_faces,
            "cell_face_z_m": z_faces,
        }

    def test_same_storage_direct_relocation_transverse_axis_remains_admitted(
        self,
    ) -> None:
        case = self._same_storage_component_axis_case(component_axis=2)
        integer_result, floating_result, installed_segments = (
            self._run_direct_finite_segment_union_case(case)
        )

        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(installed_segments, ((0, 1),))
        self.assertAlmostEqual(floating_result[6], 0.0, places=12)

    def test_same_storage_direct_relocation_component_axis_is_admitted(
        self,
    ) -> None:
        case = self._same_storage_component_axis_case(component_axis=0)
        anchor_delta = abs(case["boundaries"][1][2] - case["boundaries"][0][2])
        coordinate_scale = float(
            np.max(np.abs(np.asarray(case["positions"], dtype=np.float32)))
        )
        geometry_tolerance = (
            2.0 * float(np.finfo(np.float32).eps) * coordinate_scale
        )
        self.assertGreater(anchor_delta, geometry_tolerance)
        self.assertLessEqual(anchor_delta, 4.0 * geometry_tolerance)
        marker_component_velocities = tuple(
            velocity[0] for velocity in case["velocities"]
        )
        direct_serialized_target = sum(
            weight * velocity
            for weight, velocity in zip(
                case["weights"][0][:2],
                marker_component_velocities,
                strict=True,
            )
        )
        relocation_serialized_target = sum(
            weight * velocity
            for weight, velocity in zip(
                case["weights"][1][:2],
                marker_component_velocities,
                strict=True,
            )
        )
        self.assertNotEqual(direct_serialized_target, relocation_serialized_target)
        integer_result, floating_result, installed_segments = (
            self._run_direct_finite_segment_union_case(case)
        )

        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(installed_segments, ((0, 1),))
        stored_marker_z = np.asarray(
            [position[2] for position in case["positions"]],
            dtype=np.float32,
        ).astype(np.float64)
        stored_face_z = float(np.float32(case["face"][2]))
        face_interpolation_weight = (
            stored_face_z - stored_marker_z[0]
        ) / (stored_marker_z[1] - stored_marker_z[0])
        stored_marker_component_velocities = np.asarray(
            marker_component_velocities,
            dtype=np.float32,
        ).astype(np.float64)
        expected_face_target = float(
            np.float32(
                (1.0 - face_interpolation_weight)
                * stored_marker_component_velocities[0]
                + face_interpolation_weight
                * stored_marker_component_velocities[1]
            )
        )
        self.assertEqual(floating_result[6], expected_face_target)
        for start, expected in (
            (7, (0.0015, 0.01, case["face"][2])),
            (10, (0.0, 1.0, 0.0)),
            (13, (0.0015, 0.0103125, case["face"][2])),
        ):
            for actual, expected_component in zip(
                floating_result[start : start + 3],
                expected,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected_component, places=8)

    def test_same_storage_component_axis_rejects_anchor_beyond_four_tolerances(
        self,
    ) -> None:
        case = self._same_storage_component_axis_case(
            component_axis=0,
            shadow_anchor_delta_m=8.0e-8,
        )
        anchor_delta = abs(case["boundaries"][1][2] - case["boundaries"][0][2])
        coordinate_scale = float(
            np.max(np.abs(np.asarray(case["positions"], dtype=np.float32)))
        )
        geometry_tolerance = (
            2.0 * float(np.finfo(np.float32).eps) * coordinate_scale
        )
        self.assertGreater(anchor_delta, 4.0 * geometry_tolerance)
        self.assertEqual(case["velocities"], ((0.0, 0.0, 0.0),) * 2)
        integer_result, _, _ = self._run_direct_finite_segment_union_case(case)

        self.assertEqual(integer_result[0:2], (0, 0))

    @staticmethod
    def _vf48g_internal_c0_vertex_case() -> dict[str, object]:
        """Return the compact one-ULP shared-vertex witness from vf48g."""

        face = (
            0.000375000003259629,
            0.00039062497671693563,
            0.04671875014901161,
        )
        surface_z = 0.04699999839067459
        boundaries = (
            (face[0], 0.0003515625139698386, surface_z),
            (face[0], 0.00042968749767169356, surface_z),
        )
        probe_offset_m = 0.001125
        return {
            "component_axis": 1,
            "face": face,
            "region": 202,
            "positions": (
                (
                    0.001500000013038516,
                    0.00023437499476131052,
                    surface_z,
                ),
                (
                    0.001500000013038516,
                    0.0003906250058207661,
                    surface_z,
                ),
                (
                    0.001500000013038516,
                    0.0005468750023283064,
                    surface_z,
                ),
            ),
            # A velocity gradient proves that canonicalization reads the
            # shared marker directly instead of averaging author targets.
            "velocities": (
                (0.0, 0.05, 0.0),
                (0.0, 0.125, 0.0),
                (0.0, 0.2, 0.0),
            ),
            "projection_segments": ((0, 1), (1, 2)),
            "source_centers": (
                (face[0], boundaries[0][1], face[2]),
                (face[0], boundaries[1][1], face[2]),
            ),
            "boundaries": boundaries,
            "probes": tuple(
                (boundary[0], boundary[1], boundary[2] - probe_offset_m)
                for boundary in boundaries
            ),
            "normals": ((0.0, 0.0, -1.0),) * 2,
            "indices": ((0, 1, -1), (1, 2, -1)),
            "weights": (
                (0.24999988079071045, 0.7500001192092896, 0.0),
                (0.7500000596046448, 0.24999994039535522, 0.0),
            ),
            "configured_source_support_xyz_m": (
                0.0012,
                0.000390625,
                0.00046875,
            ),
            "configured_source_support_available": 1,
            "configured_source_support_anisotropic": 1,
            "cell_face_x_m": (0.0, 0.00075, 0.0015, 0.00225, 0.003),
            "cell_face_y_m": (
                0.0003125,
                face[1],
                0.00046875,
                0.000546875,
                0.000625,
            ),
            "cell_face_z_m": (
                0.04609375,
                0.04640625,
                0.0465625,
                0.046875,
                0.0471875,
            ),
        }

    @staticmethod
    def _vf48i_strict_interior_owner_case() -> dict[str, object]:
        """Return the compact adjacent-segment witness from vf48i."""

        face = (
            0.000375000003259629,
            0.0072656250558793545,
            0.04671875014901161,
        )
        boundaries = (
            (
                face[0],
                0.007226593792438507,
                0.04698072746396065,
            ),
            (
                face[0],
                0.007304662372916937,
                0.04698072746396065,
            ),
        )
        source_centers = (
            (face[0], 0.0072265625931322575, face[2]),
            (face[0], 0.0073046875186264515, face[2]),
        )
        probe_offset_m = 0.001125
        return {
            "component_axis": 1,
            "face": face,
            "region": 202,
            "positions": (
                (
                    0.001500000013038516,
                    0.007109076250344515,
                    0.04698074236512184,
                ),
                (
                    0.001500000013038516,
                    0.0072654546238482,
                    0.04698072373867035,
                ),
                (
                    0.001500000013038516,
                    0.00742181995883584,
                    0.04698073863983154,
                ),
            ),
            "velocities": (
                (0.0, -0.0005975029780529439, -0.038512006402015686),
                (0.0, -0.00034041042090393603, -0.038546670228242874),
                (0.0, -0.00011021857790183276, -0.038520630449056625),
            ),
            "projection_segments": ((0, 1), (1, 2)),
            "source_centers": source_centers,
            "boundaries": boundaries,
            "probes": tuple(
                (boundary[0], boundary[1], boundary[2] - probe_offset_m)
                for boundary in boundaries
            ),
            "normals": ((0.0, 0.0, -1.0),) * 2,
            "indices": ((0, 1, -1), (1, 2, -1)),
            "nearest_markers": (1, 1),
            "weights": (
                (0.2485051155090332, 0.7514948844909668, 0.0),
                (0.7492543458938599, 0.2507456839084625, 0.0),
            ),
            "configured_source_support_xyz_m": (
                0.0012,
                0.000390625,
                0.00046875,
            ),
            "configured_source_support_available": 1,
            "configured_source_support_anisotropic": 1,
            "cell_face_x_m": (0.0, 0.00075, 0.0015, 0.00225, 0.003),
            "cell_face_y_m": tuple(
                face[1] + offset * 7.8125e-5 for offset in (-1, 0, 1, 2, 3)
            ),
            "cell_face_z_m": (
                0.04609375,
                0.04640625,
                0.0465625,
                0.046875,
                0.0471875,
            ),
        }

    def test_vf48i_strict_interior_owner_survives_distance_band(self) -> None:
        """A common normal offset cannot hide a unique finite-segment owner."""

        case = self._vf48i_strict_interior_owner_case()
        positions = np.asarray(case["positions"], dtype=np.float32).astype(
            np.float64
        )
        velocities = np.asarray(case["velocities"], dtype=np.float32).astype(
            np.float64
        )
        face = np.asarray(case["face"], dtype=np.float32).astype(np.float64)
        active_face = face[1:]
        raw_parameters = []
        distance_squared = []
        for marker_a, marker_b in ((0, 1), (1, 2)):
            segment = positions[marker_b, 1:] - positions[marker_a, 1:]
            raw_parameter = float(
                np.dot(active_face - positions[marker_a, 1:], segment)
                / np.dot(segment, segment)
            )
            raw_parameters.append(raw_parameter)
            closest = positions[marker_a, 1:] + np.clip(
                raw_parameter, 0.0, 1.0
            ) * segment
            delta = active_face - closest
            distance_squared.append(float(np.dot(delta, delta)))

        old_tie_band = (
            4.0
            * float(np.finfo(np.float32).eps)
            * max(
                *distance_squared,
                (7.8125e-5) ** 2,
                (3.125e-4) ** 2,
                1.0e-24,
            )
        )
        self.assertGreater(raw_parameters[0], 1.0)
        self.assertGreater(raw_parameters[1], 2.0e-6)
        self.assertLess(raw_parameters[1], 1.0 - 2.0e-6)
        self.assertGreater(distance_squared[0] - distance_squared[1], 0.0)
        self.assertLessEqual(
            distance_squared[0] - distance_squared[1], old_tie_band
        )

        integer_result, float_result, _ = (
            self._run_direct_finite_segment_union_case(case)
        )
        self.assertEqual(integer_result[0:4], (1, 1, 1, 0))
        self.assertEqual(
            integer_result[4:6],
            (1, 1),
            msg="strict interior primitive must own the adjacent finite union",
        )
        expected_target = velocities[1, 1] + raw_parameters[1] * (
            velocities[2, 1] - velocities[1, 1]
        )
        self.assertAlmostEqual(float_result[6], expected_target, places=10)
        self.assertNotAlmostEqual(float_result[6], velocities[1, 1], places=10)

        swapped = {
            **case,
            **{
                key: tuple(reversed(case[key]))
                for key in (
                    "source_centers",
                    "boundaries",
                    "probes",
                    "normals",
                    "indices",
                    "weights",
                    "nearest_markers",
                )
            },
        }
        swapped_integer, swapped_float, _ = (
            self._run_direct_finite_segment_union_case(swapped)
        )
        self.assertEqual(swapped_integer[4:], integer_result[4:])
        self.assertEqual(
            np.asarray(swapped_float[4:], dtype=np.float64).tobytes(),
            np.asarray(float_result[4:], dtype=np.float64).tobytes(),
        )

    def test_vf48i_strict_interior_owner_is_not_a_fixed_segment_side(
        self,
    ) -> None:
        """Reflecting the chain must reflect ownership without changing state."""

        base = self._vf48i_strict_interior_owner_case()
        face_y = float(base["face"][1])

        def reflected(point: tuple[float, float, float]):
            return (point[0], 2.0 * face_y - point[1], point[2])

        mirrored = {
            **base,
            "positions": tuple(
                reflected(point) for point in reversed(base["positions"])
            ),
            "velocities": tuple(reversed(base["velocities"])),
            "source_centers": tuple(
                reflected(point) for point in reversed(base["source_centers"])
            ),
            "boundaries": tuple(
                reflected(point) for point in reversed(base["boundaries"])
            ),
            "probes": tuple(
                reflected(point) for point in reversed(base["probes"])
            ),
            "normals": tuple(reversed(base["normals"])),
            "weights": tuple(
                (weight[1], weight[0], weight[2])
                for weight in reversed(base["weights"])
            ),
        }
        base_integer, base_float, _ = (
            self._run_direct_finite_segment_union_case(base)
        )
        mirror_integer, mirror_float, _ = (
            self._run_direct_finite_segment_union_case(mirrored)
        )
        self.assertEqual(base_integer[0:4], (1, 1, 1, 0))
        self.assertEqual(mirror_integer[0:4], (1, 1, 0, 1))
        self.assertEqual(base_integer[4:6], (1, 1))
        self.assertEqual(mirror_integer[4:6], (1, 1))
        self.assertAlmostEqual(mirror_float[6], base_float[6], places=10)

    def test_vf48i_strict_interior_bypass_requires_degree_two_topology(
        self,
    ) -> None:
        """The interiority proof cannot bypass missing or nonmanifold topology."""

        base = self._vf48i_strict_interior_owner_case()
        missing_topology = {**base, "projection_segments": ()}
        missing_integer, _, _ = self._run_direct_finite_segment_union_case(
            missing_topology
        )
        self.assertEqual(missing_integer[0:2], (1, 1))
        self.assertEqual(missing_integer[4:6], (0, 0))

        shared = base["positions"][1]
        degree_three = {
            **base,
            "positions": (
                *base["positions"],
                (shared[0], shared[1], shared[2] + 0.00015625),
            ),
            "velocities": (*base["velocities"], base["velocities"][1]),
            "projection_segments": ((0, 1), (1, 2), (1, 3)),
            "raw_projection_segments": True,
        }
        degree_three_integer, _, _ = (
            self._run_direct_finite_segment_union_case(degree_three)
        )
        self.assertEqual(degree_three_integer[0:2], (1, 1))
        self.assertEqual(degree_three_integer[4:6], (0, 0))

    def test_vf48g_internal_degree2_c0_vertex_tie_is_canonical(self) -> None:
        """A physical shared vertex has one state despite two primitive owners."""

        case = self._vf48g_internal_c0_vertex_case()
        positions = np.asarray(case["positions"], dtype=np.float32).astype(
            np.float64
        )
        face = np.asarray(case["face"], dtype=np.float32).astype(np.float64)
        active_face = face[1:]
        closest_points = []
        raw_parameters = []
        for marker_a, marker_b in ((0, 1), (1, 2)):
            segment = positions[marker_b, 1:] - positions[marker_a, 1:]
            raw_parameter = float(
                np.dot(active_face - positions[marker_a, 1:], segment)
                / np.dot(segment, segment)
            )
            raw_parameters.append(raw_parameter)
            closest_points.append(
                positions[marker_a, 1:]
                + np.clip(raw_parameter, 0.0, 1.0) * segment
            )

        self.assertLess(raw_parameters[0], 1.0)
        self.assertLess(raw_parameters[1], 0.0)
        self.assertEqual(
            np.float32(case["face"][1]),
            np.nextafter(
                np.float32(case["positions"][1][1]),
                np.float32(-np.inf),
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            closest_points,
            np.repeat(positions[1:2, 1:], 2, axis=0),
            rtol=0.0,
            atol=4.0e-11,
        )
        distance_squared = tuple(
            float(np.dot(active_face - point, active_face - point))
            for point in closest_points
        )
        distance_difference = abs(distance_squared[0] - distance_squared[1])
        local_width_squared = max(
            (
                float(
                    np.float32(case["cell_face_y_m"][2])
                    - np.float32(case["cell_face_y_m"][1])
                )
            )
            ** 2,
            (
                float(
                    np.float32(case["cell_face_z_m"][3])
                    - np.float32(case["cell_face_z_m"][2])
                )
            )
            ** 2,
        )
        tie_tolerance_squared = (
            4.0
            * float(np.finfo(np.float32).eps)
            * max(*distance_squared, local_width_squared, 1.0e-24)
        )
        self.assertGreater(distance_difference, 0.0)
        self.assertLessEqual(distance_difference, tie_tolerance_squared)
        self.assertEqual(
            sum(1 for segment in case["projection_segments"] if 1 in segment),
            2,
        )

        integer_result, float_result, installed_segments = (
            self._run_direct_finite_segment_union_case(case)
        )
        self.assertEqual(installed_segments, ((0, 1), (1, 2)))
        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(
            integer_result[4:6],
            (1, 1),
            msg="internal degree-2 C0 shared vertex must form one finite union",
        )
        self.assertAlmostEqual(float_result[6], 0.125, places=7)
        np.testing.assert_allclose(
            float_result[7:10],
            (case["face"][0], *positions[1, 1:]),
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            float_result[10:13],
            (0.0, 0.0, -1.0),
            rtol=0.0,
            atol=1.0e-7,
        )

        swapped = {
            **case,
            **{
                key: tuple(reversed(case[key]))
                for key in (
                    "source_centers",
                    "boundaries",
                    "probes",
                    "normals",
                    "indices",
                    "weights",
                )
            },
        }
        swapped_integer, swapped_float, _ = (
            self._run_direct_finite_segment_union_case(swapped)
        )
        self.assertEqual(swapped_integer[4:], integer_result[4:])
        self.assertEqual(swapped_float[4:], float_result[4:])

    def test_internal_near_c0_vertex_canonical_state_is_order_independent(
        self,
    ) -> None:
        """A tolerated near-C0 equivalence class cannot select an author normal."""

        base = self._vf48g_internal_c0_vertex_case()
        base_positions = np.asarray(base["positions"], dtype=np.float32)
        segment_length_m = float(base_positions[1, 1] - base_positions[0, 1])
        bend_height_m = segment_length_m * 5.0e-4
        positions = (
            (
                base["positions"][0][0],
                base["positions"][0][1],
                base["positions"][0][2] + bend_height_m,
            ),
            base["positions"][1],
            (
                base["positions"][2][0],
                base["positions"][2][1],
                base["positions"][2][2] + bend_height_m,
            ),
        )

        boundaries = []
        for author_index, (marker_a, marker_b, _) in enumerate(base["indices"]):
            marker_a_position = np.asarray(positions[marker_a], dtype=np.float32)
            marker_b_position = np.asarray(positions[marker_b], dtype=np.float32)
            weight_b = np.float32(base["weights"][author_index][1])
            anchor = marker_a_position + weight_b * (
                marker_b_position - marker_a_position
            )
            boundaries.append(
                (base["face"][0], float(anchor[1]), float(anchor[2]))
            )
        near_c0 = {
            **base,
            "positions": positions,
            "boundaries": tuple(boundaries),
            "source_centers": tuple(
                (boundary[0], boundary[1], base["face"][2])
                for boundary in boundaries
            ),
            "probes": tuple(
                (boundary[0], boundary[1], boundary[2] - 0.001125)
                for boundary in boundaries
            ),
        }
        first_outward = np.asarray(positions[0], dtype=np.float64)[1:] - np.asarray(
            positions[1], dtype=np.float64
        )[1:]
        second_outward = np.asarray(positions[2], dtype=np.float64)[1:] - np.asarray(
            positions[1], dtype=np.float64
        )[1:]
        tangent_dot = float(
            np.dot(first_outward, second_outward)
            / (np.linalg.norm(first_outward) * np.linalg.norm(second_outward))
        )
        self.assertGreaterEqual(tangent_dot, -1.0)
        self.assertLessEqual(tangent_dot, -0.999999)

        integer_result, float_result, _ = (
            self._run_direct_finite_segment_union_case(near_c0)
        )
        self.assertEqual(integer_result[4:6], (1, 1))
        swapped = {
            **near_c0,
            **{
                key: tuple(reversed(near_c0[key]))
                for key in (
                    "source_centers",
                    "boundaries",
                    "probes",
                    "normals",
                    "indices",
                    "weights",
                )
            },
        }
        swapped_integer, swapped_float, _ = (
            self._run_direct_finite_segment_union_case(swapped)
        )
        self.assertEqual(swapped_integer[4:], integer_result[4:])
        self.assertEqual(
            np.asarray(swapped_float[4:], dtype=np.float64).tobytes(),
            np.asarray(float_result[4:], dtype=np.float64).tobytes(),
        )

    def test_vf48g_internal_c0_vertex_union_remains_fail_closed(self) -> None:
        """Only a complete smooth degree-two shared state may co-own a tie."""

        base = self._vf48g_internal_c0_vertex_case()

        def assert_pair_rejected(
            name: str,
            case: dict[str, object],
            *,
            primitive_valid: bool = True,
        ) -> None:
            with self.subTest(fail_closed=name):
                integer_result, _, _ = self._run_direct_finite_segment_union_case(
                    case
                )
                if primitive_valid:
                    self.assertEqual(integer_result[0:2], (1, 1))
                self.assertEqual(integer_result[4:6], (0, 0))

        missing_topology = {**base, "projection_segments": ()}
        assert_pair_rejected("missing_topology", missing_topology)

        degree_three = {
            **base,
            "positions": (
                *base["positions"],
                (
                    base["positions"][1][0],
                    base["positions"][1][1],
                    base["positions"][1][2] + 0.00015625,
                ),
            ),
            "velocities": (*base["velocities"], (0.0, 0.125, 0.0)),
            "projection_segments": ((0, 1), (1, 2), (1, 3)),
            "raw_projection_segments": True,
        }
        assert_pair_rejected("degree_three", degree_three)

        kink_height_m = float(
            np.float32(base["positions"][2][1])
            - np.float32(base["positions"][1][1])
        ) * 2.0e-3
        kink_positions = (
            base["positions"][0],
            base["positions"][1],
            (
                base["positions"][2][0],
                base["positions"][2][1],
                base["positions"][2][2] + kink_height_m,
            ),
        )
        second_weight = base["weights"][1][1]
        kink_second_boundary = (
            base["boundaries"][1][0],
            base["boundaries"][1][1],
            base["positions"][1][2] + second_weight * kink_height_m,
        )
        kink = {
            **base,
            "positions": kink_positions,
            "boundaries": (base["boundaries"][0], kink_second_boundary),
            "source_centers": (
                base["source_centers"][0],
                (
                    kink_second_boundary[0],
                    kink_second_boundary[1],
                    base["face"][2],
                ),
            ),
            "probes": (
                base["probes"][0],
                (
                    kink_second_boundary[0],
                    kink_second_boundary[1],
                    kink_second_boundary[2] - 0.001125,
                ),
            ),
        }
        assert_pair_rejected("kink", kink)

        normal_discontinuity = {
            **base,
            "normals": ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            "probes": (
                base["probes"][0],
                (
                    base["boundaries"][1][0],
                    base["boundaries"][1][1],
                    base["boundaries"][1][2] + 0.001125,
                ),
            ),
        }
        assert_pair_rejected("normal_discontinuity", normal_discontinuity)

        midpoint = tuple(
            0.5 * (left + right)
            for left, right in zip(base["positions"][0], base["positions"][1])
        )
        half_weight = {
            **base,
            "boundaries": (midpoint, base["boundaries"][1]),
            "source_centers": (
                (midpoint[0], midpoint[1], base["face"][2]),
                base["source_centers"][1],
            ),
            "probes": (
                (midpoint[0], midpoint[1], midpoint[2] - 0.001125),
                base["probes"][1],
            ),
            "weights": ((0.5, 0.5, 0.0), base["weights"][1]),
            "nearest_markers": (1, 1),
        }
        assert_pair_rejected("shared_weight_not_strictly_nearest", half_weight)

        cross_region = {
            **base,
            "region_ids": (202, 101, 202),
            "raw_projection_segments": True,
        }
        assert_pair_rejected(
            "cross_region_shared_vertex",
            cross_region,
            primitive_valid=False,
        )

    def test_internal_c0_vertex_rejects_malformed_author_weights(self) -> None:
        """NaN in a non-shared weight cannot bypass anchor provenance."""

        base = self._vf48g_internal_c0_vertex_case()
        malformed = {
            **base,
            "weights": (
                (float("nan"), base["weights"][0][1], 0.0),
                base["weights"][1],
            ),
            "nearest_markers": (1, 1),
        }
        integer_result, _, _ = self._run_direct_finite_segment_union_case(
            malformed
        )
        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(integer_result[4:6], (0, 0))

    def test_component_axis_pair_has_no_legacy_author_target_bypass(
        self,
    ) -> None:
        """Every component-axis pair must consume only its bound owner payload."""

        source = textwrap.dedent(
            inspect.getsource(
                HibmMpmIbBoundaryConditions._reconstruct_velocity_dirichlet_component_face_segment_claims_kernel
            )
        )
        syntax_tree = ast.parse(source)

        def names(nodes: list[ast.stmt] | ast.expr) -> set[str]:
            body = [ast.Expr(value=nodes)] if isinstance(nodes, ast.expr) else nodes
            subtree = ast.Module(body=body, type_ignores=[])
            return {
                node.id
                for node in ast.walk(subtree)
                if isinstance(node, ast.Name)
            }

        pair_branches = []
        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.If):
                continue
            test_names = names(node.test)
            body_attributes = {
                child.attr
                for statement in node.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Attribute)
            }
            if (
                "face_first_finite_segment_pair" in test_names
                and "authors_are_component_axis_pair" in test_names
                and "velocity_dirichlet_component_face_segment_pair_admission_valid"
                in body_attributes
            ):
                pair_branches.append((node, test_names, names(node.body), body_attributes))

        self.assertEqual(
            len(pair_branches),
            1,
            msg="could not identify the finite-segment-union scratch branch",
        )
        _, test_names, body_names, body_attributes = pair_branches[0]
        contract_violations = []
        if "_canonical_component_face_finite_segment_union_owner_geometry" in source:
            contract_violations.append(
                "reconstruction must not cold-JIT the finite-segment-union helper"
            )
        required_scratch = {
            "velocity_dirichlet_component_face_segment_pair_admission_valid",
            "velocity_dirichlet_component_face_segment_pair_full_valid",
            "velocity_dirichlet_component_face_segment_pair_first_author_linear_key",
            "velocity_dirichlet_component_face_segment_pair_second_author_linear_key",
            "velocity_dirichlet_component_face_segment_pair_first_author_kind",
            "velocity_dirichlet_component_face_segment_pair_second_author_kind",
        }
        missing_scratch = sorted(required_scratch - body_attributes)
        if missing_scratch:
            contract_violations.append(
                "component-axis pair does not consume required scratch: "
                + ", ".join(missing_scratch)
            )
        if "pair_author_match" not in body_names:
            contract_violations.append(
                "component-axis pair does not bind scratch to exact authors"
            )
        if "distinct_pair_admission_valid = 0" not in ast.unparse(pair_branches[0][0]):
            contract_violations.append(
                "author-key or kind mismatch does not fail closed"
            )
        forbidden_author_local_names = {
            "legacy_face_axis_bracketed",
            "first_target",
            "second_target",
        }
        mixed_names = sorted(forbidden_author_local_names & body_names)
        if mixed_names:
            contract_violations.append(
                "unified-helper branch still mixes author-local state: "
                + ", ".join(mixed_names)
            )
        self.assertEqual(contract_violations, [])

    def test_finite_segment_union_owner_geometry_is_cold_jit_precomputed(
        self,
    ) -> None:
        """Prepare/reconstruct consume one transaction-local owner payload."""

        helper_name = "_canonical_component_face_finite_segment_union_owner_geometry"
        precompute = textwrap.dedent(
            inspect.getsource(
                HibmMpmIbBoundaryConditions._precompute_velocity_dirichlet_component_face_segment_pair_geometry_kernel
            )
        )
        prepare = textwrap.dedent(
            inspect.getsource(
                HibmMpmIbBoundaryConditions._prepare_velocity_dirichlet_component_face_claims_kernel
            )
        )
        reconstruct = textwrap.dedent(
            inspect.getsource(
                HibmMpmIbBoundaryConditions._reconstruct_velocity_dirichlet_component_face_segment_claims_kernel
            )
        )

        self.assertEqual(precompute.count(helper_name), 1)
        self.assertNotIn(helper_name, prepare)
        self.assertNotIn(helper_name, reconstruct)
        for source in (prepare, reconstruct):
            self.assertIn(
                "velocity_dirichlet_component_face_segment_pair_admission_valid",
                source,
            )
            self.assertIn(
                "velocity_dirichlet_component_face_segment_pair_first_author_linear_key",
                source,
            )
            self.assertIn(
                "velocity_dirichlet_component_face_segment_pair_first_author_kind",
                source,
            )

    def test_cold_jit_owner_payload_matches_direct_probe_outputs(self) -> None:
        """The nine precomputed owner outputs retain direct-probe parity."""

        case = self._vf48g_internal_c0_vertex_case()
        direct_i32, direct_f64, _ = self._run_direct_finite_segment_union_case(case)
        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4), marker_capacity=3, runtime=runtime
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=3, runtime=runtime)
        markers.load_markers(
            positions_m=case["positions"],
            velocities_mps=case["velocities"],
            normals=((0.0, 0.0, 1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(int(case["region"]),) * 3,
        )
        markers.set_projection_segments(case["projection_segments"])
        nodes = (4, 4, 4)
        node_boundary = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        node_probe = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        node_indices = ti.Vector.field(3, dtype=ti.i32, shape=nodes)
        node_weights = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        nearest = ti.field(dtype=ti.i32, shape=nodes)
        center_x = ti.field(dtype=ti.f32, shape=4)
        center_y = ti.field(dtype=ti.f32, shape=4)
        center_z = ti.field(dtype=ti.f32, shape=4)
        face_x = ti.field(dtype=ti.f32, shape=5)
        face_y = ti.field(dtype=ti.f32, shape=5)
        face_z = ti.field(dtype=ti.f32, shape=5)
        face_x.from_numpy(np.asarray(case["cell_face_x_m"], dtype=np.float32))
        face_y.from_numpy(np.asarray(case["cell_face_y_m"], dtype=np.float32))
        face_z.from_numpy(np.asarray(case["cell_face_z_m"], dtype=np.float32))
        face = np.asarray(case["face"], dtype=np.float32)
        sources = np.asarray(case["source_centers"], dtype=np.float32)
        center_x.from_numpy(np.full(4, face[0], dtype=np.float32))
        center_y.from_numpy(
            np.asarray((sources[0, 1], sources[1, 1], sources[1, 1], sources[1, 1]), dtype=np.float32)
        )
        center_z.from_numpy(np.full(4, face[2], dtype=np.float32))
        first_author, second_author, target = (0, 0, 2), (0, 1, 2), (0, 1, 2)
        for author, boundary_point, probe, indices, weights, nearest_marker in zip(
            (first_author, second_author),
            case["boundaries"],
            case["probes"],
            case["indices"],
            case["weights"],
            (1, 1),
            strict=True,
        ):
            boundary.active_ib_node[author] = 1
            boundary.velocity_dirichlet_component_face_actual_sample_valid[author] = 1
            boundary.velocity_dirichlet_component_face_actual_sample_point_m[author] = probe
            boundary.pressure_neumann_normal_field[author] = case["normals"][0]
            node_boundary[author] = boundary_point
            node_probe[author] = probe
            node_indices[author] = indices
            node_weights[author] = weights
            nearest[author] = nearest_marker

        support = tuple(float(value) for value in case["configured_source_support_xyz_m"])
        obstacle = ti.field(dtype=ti.i32, shape=nodes)
        boundary._precompute_velocity_dirichlet_component_face_segment_pair_geometry_kernel(
            obstacle,
            node_boundary,
            node_probe,
            node_indices,
            node_weights,
            nearest,
            markers.x_gamma_m,
            markers.v_gamma_mps,
            markers.region_id,
            markers.projection_triangle_indices,
            int(markers.projection_segment_count),
            1,
            int(case["configured_source_support_available"]),
            int(case["configured_source_support_anisotropic"]),
            *support,
            face_x,
            face_y,
            face_z,
            center_x,
            center_y,
            center_z,
            *nodes,
            3,
            0,
        )
        index = (*target, int(case["component_axis"]))
        self.assertEqual(
            tuple(
                int(field[index])
                for field in (
                    boundary.velocity_dirichlet_component_face_segment_pair_admission_valid,
                    boundary.velocity_dirichlet_component_face_segment_pair_full_valid,
                    boundary.velocity_dirichlet_component_face_segment_pair_endpoint_clamped,
                )
            ),
            direct_i32[4:7],
        )
        np.testing.assert_array_equal(
            np.asarray(
                (
                    boundary.velocity_dirichlet_component_face_segment_pair_boundary_target_mps[index],
                    boundary.velocity_dirichlet_component_face_segment_pair_clamp_support_ratio[index],
                    boundary.velocity_dirichlet_component_face_segment_pair_geometry_tolerance[index],
                    *boundary.velocity_dirichlet_component_face_segment_pair_boundary_point_m[index],
                    *boundary.velocity_dirichlet_component_face_segment_pair_normal[index],
                    *boundary.velocity_dirichlet_component_face_segment_pair_nominal_probe_m[index],
                ),
                dtype=np.float32,
            ),
            np.asarray((direct_f64[6], direct_f64[4], direct_f64[5], *direct_f64[7:]), dtype=np.float32),
        )
        self.assertEqual(
            (
                int(
                    boundary.velocity_dirichlet_component_face_segment_pair_first_author_linear_key[
                        index
                    ]
                ),
                int(
                    boundary.velocity_dirichlet_component_face_segment_pair_second_author_linear_key[
                        index
                    ]
                ),
            ),
            (2, 6),
        )
        self.assertEqual(
            (
                int(
                    boundary.velocity_dirichlet_component_face_segment_pair_first_author_kind[
                        index
                    ]
                ),
                int(
                    boundary.velocity_dirichlet_component_face_segment_pair_second_author_kind[
                        index
                    ]
                ),
            ),
            (0, 0),
        )

        boundary.active_ib_node[second_author] = 0
        boundary._precompute_velocity_dirichlet_component_face_segment_pair_geometry_kernel(
            obstacle,
            node_boundary,
            node_probe,
            node_indices,
            node_weights,
            nearest,
            markers.x_gamma_m,
            markers.v_gamma_mps,
            markers.region_id,
            markers.projection_triangle_indices,
            int(markers.projection_segment_count),
            1,
            int(case["configured_source_support_available"]),
            int(case["configured_source_support_anisotropic"]),
            *support,
            face_x,
            face_y,
            face_z,
            center_x,
            center_y,
            center_z,
            *nodes,
            3,
            0,
        )
        neutral_i32 = (
            boundary.velocity_dirichlet_component_face_segment_pair_admission_valid,
            boundary.velocity_dirichlet_component_face_segment_pair_full_valid,
            boundary.velocity_dirichlet_component_face_segment_pair_endpoint_clamped,
        )
        neutral_vectors = (
            boundary.velocity_dirichlet_component_face_segment_pair_boundary_point_m,
            boundary.velocity_dirichlet_component_face_segment_pair_normal,
            boundary.velocity_dirichlet_component_face_segment_pair_nominal_probe_m,
        )
        neutral_scalars = (
            boundary.velocity_dirichlet_component_face_segment_pair_boundary_target_mps,
            boundary.velocity_dirichlet_component_face_segment_pair_clamp_support_ratio,
            boundary.velocity_dirichlet_component_face_segment_pair_geometry_tolerance,
        )
        neutral_keys_and_kinds = (
            boundary.velocity_dirichlet_component_face_segment_pair_first_author_linear_key,
            boundary.velocity_dirichlet_component_face_segment_pair_second_author_linear_key,
            boundary.velocity_dirichlet_component_face_segment_pair_first_author_kind,
            boundary.velocity_dirichlet_component_face_segment_pair_second_author_kind,
        )
        self.assertEqual(sum(int(field[index]) == 0 for field in neutral_i32), 3)
        self.assertEqual(
            sum(
                np.array_equal(
                    np.asarray(field[index], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                )
                for field in neutral_vectors
            ),
            3,
        )
        self.assertEqual(sum(float(field[index]) == 0.0 for field in neutral_scalars), 3)
        self.assertEqual(
            sum(int(field[index]) == -1 for field in neutral_keys_and_kinds),
            4,
        )

    def test_component_axis_pair_flag_is_initialized_outside_computation_scope(
        self,
    ) -> None:
        """Taichi must see an outer flag before the guarded pair computation."""

        source = textwrap.dedent(
            inspect.getsource(
                HibmMpmIbBoundaryConditions._reconstruct_velocity_dirichlet_component_face_segment_claims_kernel
            )
        )
        lines = source.splitlines()
        computed_assignment = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == "authors_are_component_axis_pair = ("
        )
        dominating_initializations = [
            index
            for index, line in enumerate(lines[:computed_assignment])
            if (
                line.strip() == "authors_are_component_axis_pair = 0"
                and len(line) - len(line.lstrip())
                < len(lines[computed_assignment])
                - len(lines[computed_assignment].lstrip())
            )
        ]
        self.assertEqual(
            len(dominating_initializations),
            1,
            msg=(
                "authors_are_component_axis_pair must be initialized outside "
                "its conditional computation scope"
            ),
        )

    def test_search_support_provenance_survives_build_and_invalidates_on_error(
        self,
    ) -> None:
        """Real node search owns the support metadata consumed downstream."""

        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=2,
            projection_triangle_capacity=1,
            runtime=runtime,
        )
        inverse_sqrt_two = 1.0 / np.sqrt(2.0)
        markers.load_markers(
            positions_m=((0.5, 0.2, 0.2), (0.5, 0.8, 0.8)),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((0.0, inverse_sqrt_two, -inverse_sqrt_two),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        markers.set_projection_segments(((0, 1),))
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 2),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
            runtime=runtime,
        )
        center_x_m = ti.field(dtype=ti.f32, shape=1)
        center_y_m = ti.field(dtype=ti.f32, shape=1)
        center_z_m = ti.field(dtype=ti.f32, shape=2)
        center_x_m[0] = 0.5
        center_y_m[0] = 0.7
        center_z_m[0] = 0.2
        center_z_m[1] = 0.4

        search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=center_x_m,
            cell_center_y_m=center_y_m,
            cell_center_z_m=center_z_m,
            search_radius_m=0.62,
            search_radius_xyz_m=(1.0, 0.62, 0.62),
            interior_probe_distance_m=0.05,
            search_inactive_axis=0,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(1, 1, 2),
            marker_capacity=2,
            runtime=runtime,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(0.0, 0.0),
        )
        self.assertEqual(search._last_search_inactive_axis, 0)
        self.assertEqual(search._last_search_support_radius_xyz_m, (1.0, 0.62, 0.62))
        self.assertIs(search._last_search_support_anisotropic, True)

        with self.assertRaisesRegex(ValueError, "search_radius_m"):
            search.search_and_classify_grid_fields(
                markers,
                cell_center_x_m=center_x_m,
                cell_center_y_m=center_y_m,
                cell_center_z_m=center_z_m,
                search_radius_m=0.0,
                interior_probe_distance_m=0.05,
                search_inactive_axis=0,
            )
        self.assertEqual(search._last_search_inactive_axis, -1)
        self.assertIsNone(search._last_search_support_radius_xyz_m)
        self.assertIsNone(search._last_search_support_anisotropic)

        search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=center_x_m,
            cell_center_y_m=center_y_m,
            cell_center_z_m=center_z_m,
            search_radius_m=0.62,
            interior_probe_distance_m=0.05,
            search_inactive_axis=0,
        )
        self.assertEqual(search._last_search_support_radius_xyz_m, (0.62,) * 3)
        self.assertIs(search._last_search_support_anisotropic, False)

    def test_component_face_assembly_forwards_search_support_to_both_passes(
        self,
    ) -> None:
        """Host assembly must gate and forward one search contract twice."""

        source = textwrap.dedent(
            inspect.getsource(
                HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
            )
        )
        syntax_tree = ast.parse(source)
        expected_arguments = {
            "source_search_support_available",
            "source_search_support_anisotropic",
            "source_search_support_radius_xyz_m",
        }
        target_calls = {}
        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {
                "_prepare_velocity_dirichlet_component_face_claims_kernel",
                "_reconstruct_velocity_dirichlet_component_face_segment_claims_kernel",
            }:
                continue
            argument_names = {
                child.id
                for argument in node.args
                for child in ast.walk(argument)
                if isinstance(child, ast.Name)
            }
            target_calls[node.func.attr] = argument_names

        self.assertEqual(len(target_calls), 2)
        for call_name, argument_names in target_calls.items():
            self.assertTrue(
                expected_arguments <= argument_names,
                msg=f"{call_name} dropped search support provenance",
            )
        self.assertIn(
            "int(search._last_search_inactive_axis) == inactive_axis",
            source,
        )

    def test_internal_same_region_endpoint_is_not_terminal(self) -> None:
        """A second incident edge makes the clamped endpoint non-terminal."""

        case = {
            "face": (0.125, 1.1, 0.1),
            "region": 202,
            "positions": (
                (0.125, 0.0, 0.0),
                (0.125, 1.0, 0.0),
                (0.125, 1.5, 0.5),
            ),
            "velocities": ((0.0, 0.0, 0.0),) * 3,
            "projection_segments": ((0, 1), (1, 2)),
            "source_centers": ((0.125, 1.1, 0.0), (0.125, 1.1, 0.2)),
            "boundaries": ((0.125, 1.0, 0.0),) * 2,
            "probes": ((0.125, 1.0, 0.5), (0.125, 1.0, 0.7)),
            "normals": ((0.0, 0.0, 1.0),) * 2,
            "indices": ((0, 1, -1),) * 2,
            "weights": ((0.0, 1.0, 0.0),) * 2,
            "cell_face_x_m": (0.0, 0.25, 0.5, 0.75, 1.0),
            "cell_face_y_m": (0.0, 0.5, 1.0, 1.5, 2.0),
            "cell_face_z_m": (-0.3, -0.1, 0.1, 0.3, 0.5),
        }
        integer_result, floating_result, installed_segments = (
            self._run_direct_finite_segment_union_case(case)
        )

        endpoint_marker = 1
        incident_segments = tuple(
            segment for segment in installed_segments if endpoint_marker in segment
        )
        self.assertEqual(installed_segments, ((0, 1), (1, 2)))
        self.assertEqual(len(incident_segments), 2)
        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(integer_result[2:4], (1, 1))
        self.assertLess(floating_result[2], 1.0)
        self.assertLess(floating_result[3], 1.0)
        self.assertEqual(
            integer_result[4:6],
            (0, 0),
            msg=(
                "same-region marker 1 has two incident segments and is an "
                "internal kink, not a closed terminal endpoint"
            ),
        )

    def test_vf48e_exact_segment_endpoint_requires_terminal_topology(self) -> None:
        """A physical face on a closed terminal endpoint is not an interior bracket."""

        endpoint_y_m = 7.812499825377017e-05
        face_z_m = 0.04671875
        case = {
            "component_axis": 1,
            "face": (0.000375000003259629, endpoint_y_m, face_z_m),
            "region": 202,
            "positions": (
                (0.001500000013038516, endpoint_y_m, 0.04699999839067459),
                (
                    0.001500000013038516,
                    0.00023437499476131052,
                    0.04699999839067459,
                ),
            ),
            "velocities": ((0.0, 0.0, 0.0),) * 2,
            "projection_segments": ((0, 1),),
            "source_centers": (
                (0.000375000003259629, 3.90625e-05, face_z_m),
                (0.000375000003259629, 0.0001171875, face_z_m),
            ),
            "boundaries": (
                (0.000375000003259629, endpoint_y_m, 0.04699999839067459),
                (
                    0.000375000003259629,
                    0.00011718749738065526,
                    0.04699999839067459,
                ),
            ),
            "probes": (
                (0.000375000003259629, endpoint_y_m, 0.04649999839067459),
                (
                    0.000375000003259629,
                    0.00011718749738065526,
                    0.04649999839067459,
                ),
            ),
            "normals": ((0.0, 0.0, -1.0),) * 2,
            "indices": ((0, 1, -1),) * 2,
            "weights": ((1.0, 0.0, 0.0), (0.75, 0.25, 0.0)),
            "cell_face_x_m": (0.0, 0.00075, 0.0015, 0.00225, 0.003),
            "cell_face_y_m": tuple(
                float(index) * 7.8125e-05 for index in range(5)
            ),
            "cell_face_z_m": (
                0.04609375,
                0.04640625,
                0.0465625,
                0.046875,
                0.0471875,
            ),
        }

        integer_result, floating_result, installed_segments = (
            self._run_direct_finite_segment_union_case(case)
        )
        positions_y = np.asarray(case["positions"], dtype=np.float64)[:, 1]
        raw_parameter = (
            float(case["face"][1]) - positions_y[0]
        ) / (positions_y[1] - positions_y[0])
        author_parameters = tuple(float(weights[1]) for weights in case["weights"])
        endpoint_degree = sum(0 in segment for segment in installed_segments)

        self.assertAlmostEqual(raw_parameter, 0.0, places=12)
        self.assertEqual(author_parameters, (0.0, 0.25))
        self.assertEqual(installed_segments, ((0, 1),))
        self.assertEqual(endpoint_degree, 1)
        self.assertEqual(integer_result[0:2], (1, 1))
        self.assertEqual(integer_result[2:4], (0, 0))
        self.assertEqual(
            integer_result[4:6],
            (1, 1),
            msg=(
                "vf48e's exact closed endpoint was incorrectly subjected to "
                "the strict interior author bracket"
            ),
        )

        swapped_case = {
            **case,
            **{
                key: tuple(reversed(case[key]))
                for key in (
                    "source_centers",
                    "boundaries",
                    "probes",
                    "normals",
                    "indices",
                    "weights",
                )
            },
        }
        swapped_integer, swapped_floating, _ = (
            self._run_direct_finite_segment_union_case(swapped_case)
        )
        self.assertEqual(swapped_integer[4:], integer_result[4:])
        np.testing.assert_allclose(
            swapped_floating[4:],
            floating_result[4:],
            rtol=0.0,
            atol=0.0,
        )

        internal_endpoint_case = {
            **case,
            "positions": (
                *case["positions"],
                (
                    0.001500000013038516,
                    -7.812499825377017e-05,
                    0.04699999839067459,
                ),
            ),
            "velocities": ((0.0, 0.0, 0.0),) * 3,
            "projection_segments": ((0, 1), (2, 0)),
        }
        internal_integer, _, internal_segments = (
            self._run_direct_finite_segment_union_case(internal_endpoint_case)
        )
        self.assertEqual(sum(0 in segment for segment in internal_segments), 2)
        self.assertEqual(internal_integer[0:2], (1, 1))
        self.assertEqual(internal_integer[4:6], (0, 0))

        no_topology_integer, _, no_topology_segments = (
            self._run_direct_finite_segment_union_case(
                {**case, "projection_segments": ()}
            )
        )
        self.assertEqual(no_topology_segments, ())
        self.assertEqual(no_topology_integer[0:2], (1, 1))
        self.assertEqual(no_topology_integer[4:6], (0, 0))

        short_start_y_m = np.float32(endpoint_y_m)
        short_end_y_m = np.float32(short_start_y_m + np.float32(1.0e-8))
        short_segment_y_m = np.float32(short_end_y_m - short_start_y_m)
        short_face_y_m = np.float32(
            short_start_y_m + np.float32(0.25) * short_segment_y_m
        )
        short_half_y_m = np.float32(
            short_start_y_m + np.float32(0.5) * short_segment_y_m
        )
        short_interior_case = {
            **case,
            "face": (case["face"][0], float(short_face_y_m), face_z_m),
            "positions": (
                (case["positions"][0][0], float(short_start_y_m), case["positions"][0][2]),
                (case["positions"][1][0], float(short_end_y_m), case["positions"][1][2]),
            ),
            "projection_segments": (),
            "boundaries": (
                (
                    case["boundaries"][0][0],
                    float(short_start_y_m),
                    case["boundaries"][0][2],
                ),
                (
                    case["boundaries"][1][0],
                    float(short_half_y_m),
                    case["boundaries"][1][2],
                ),
            ),
            "probes": (
                (case["probes"][0][0], float(short_start_y_m), case["probes"][0][2]),
                (case["probes"][1][0], float(short_half_y_m), case["probes"][1][2]),
            ),
            "weights": ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
            "cell_face_y_m": tuple(
                float(short_face_y_m + np.float32(index - 1) * np.float32(7.8125e-05))
                for index in range(5)
            ),
        }
        short_integer, _, short_segments = (
            self._run_direct_finite_segment_union_case(short_interior_case)
        )
        stored_raw_parameter = (
            float(short_face_y_m) - float(short_start_y_m)
        ) / float(short_segment_y_m)
        self.assertGreater(stored_raw_parameter, 0.0)
        self.assertLess(stored_raw_parameter, 0.5)
        self.assertEqual(short_segments, ())
        self.assertEqual(short_integer[0:4], (1, 1, 0, 0))
        self.assertEqual(
            short_integer[4:6],
            (1, 1),
            msg=(
                "a resolvable short-segment interior face was misclassified "
                "as an exact endpoint by an absolute-coordinate tolerance"
            ),
        )

    def test_vf48f_one_and_half_cell_terminal_reach_uses_configured_support(
        self,
    ) -> None:
        """Search-admitted source support, not one local diagonal, bounds reach."""

        dy_m = 7.8125e-05
        dz_m = 3.125e-04
        endpoint_y_m = 7.812499825377017e-05
        boundary_z_m = 0.05000000074505806
        face_z_m = 0.05046875
        face = (0.000375000003259629, endpoint_y_m, face_z_m)
        case = {
            "component_axis": 1,
            "face": face,
            "region": 101,
            "positions": (
                (0.001500000013038516, endpoint_y_m, boundary_z_m),
                (
                    0.001500000013038516,
                    0.00023437499476131052,
                    boundary_z_m,
                ),
            ),
            "velocities": ((0.0, 0.0, 0.0),) * 2,
            "projection_segments": ((0, 1),),
            "source_centers": (
                (face[0], 3.90625e-05, face_z_m),
                (face[0], 0.0001171875, face_z_m),
            ),
            "boundaries": (
                (face[0], endpoint_y_m, boundary_z_m),
                (face[0], 0.00011718749738065526, boundary_z_m),
            ),
            "probes": (
                (face[0], endpoint_y_m, boundary_z_m + 0.001125),
                (face[0], 0.00011718749738065526, boundary_z_m + 0.001125),
            ),
            "normals": ((0.0, 0.0, 1.0),) * 2,
            "indices": ((0, 1, -1),) * 2,
            "weights": ((1.0, 0.0, 0.0), (0.75, 0.25, 0.0)),
            "cell_face_x_m": (0.0, 0.00075, 0.0015, 0.00225, 0.003),
            "cell_face_y_m": tuple(float(index) * dy_m for index in range(5)),
            "cell_face_z_m": (
                0.0496875,
                0.05,
                0.0503125,
                0.050625,
                0.0509375,
            ),
        }

        positions_y = np.asarray(case["positions"], dtype=np.float64)[:, 1]
        raw_parameter = (face[1] - positions_y[0]) / (
            positions_y[1] - positions_y[0]
        )
        face_reach_m = face_z_m - boundary_z_m
        stored_f32_face_reach_m = float(
            np.float32(np.float32(face_z_m) - np.float32(boundary_z_m))
        )
        local_diagonal_m = float(np.hypot(dy_m, dz_m))
        self.assertAlmostEqual(raw_parameter, 0.0, places=12)
        self.assertEqual(
            sum(0 in segment for segment in case["projection_segments"]),
            1,
        )
        self.assertGreater(face_reach_m, local_diagonal_m)

        support_cases = {
            "within_anisotropic_support": (0.00046875, True, (1, 1)),
            "outside_isotropic_ball": (0.00046875, False, (0, 0)),
            "at_strict_search_boundary": (
                stored_f32_face_reach_m,
                True,
                (0, 0),
            ),
            "beyond_configured_support": (0.000390625, True, (0, 0)),
        }
        for case_name, (
            normal_support_m,
            anisotropic_support,
            expected_pair_valid,
        ) in support_cases.items():
            with self.subTest(case_name):
                integer_result, _, _ = self._run_direct_finite_segment_union_case(
                    {
                        **case,
                        "configured_source_support_xyz_m": (
                            normal_support_m if not anisotropic_support else 0.00075,
                            normal_support_m if not anisotropic_support else 0.0001171875,
                            normal_support_m,
                        ),
                        "configured_source_support_anisotropic": anisotropic_support,
                    }
                )
                self.assertEqual(integer_result[0:2], (1, 1))
                self.assertEqual(
                    integer_result[4:6],
                    expected_pair_valid,
                    msg=(
                        "the finite-union owner must consume the configured "
                        "search support instead of a local-width multiplier"
                    ),
                )

        unavailable_integer, _, _ = self._run_direct_finite_segment_union_case(
            {
                **case,
                "configured_source_support_xyz_m": (
                    0.00075,
                    0.0001171875,
                    0.00046875,
                ),
                "configured_source_support_available": False,
            }
        )
        self.assertEqual(
            unavailable_integer[4:6],
            (0, 0),
            msg="finite-union reconstruction must fail closed without search provenance",
        )

        shared_face_case = {
            "component_axis": 1,
            "face": (0.125, 0.5, 0.1),
            "region": 101,
            "positions": ((0.125, 0.0, 0.0), (0.125, 1.0, 0.0)),
            "velocities": ((0.0, 0.0, 0.0),) * 2,
            "projection_segments": ((0, 1),),
            "source_centers": ((0.125, 0.0, 0.1), (0.125, 1.0, 0.1)),
            "boundaries": ((0.125, 0.0, 0.0), (0.125, 1.0, 0.0)),
            "probes": ((0.125, 0.0, 0.5), (0.125, 1.0, 0.5)),
            "normals": ((0.0, 0.0, 1.0),) * 2,
            "indices": ((0, 1, -1),) * 2,
            "weights": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            "configured_source_support_xyz_m": (1.0, 0.25, 0.2),
            "configured_source_support_anisotropic": True,
            "cell_face_x_m": (0.0, 0.25, 0.5, 0.75, 1.0),
            "cell_face_y_m": (-0.5, 0.5, 1.5, 2.5, 3.5),
            "cell_face_z_m": (-0.1, 0.0, 0.1, 0.2, 0.3),
        }
        shared_integer, _, _ = self._run_direct_finite_segment_union_case(
            shared_face_case
        )
        self.assertGreater(
            abs(
                shared_face_case["face"][1]
                - shared_face_case["boundaries"][0][1]
            ),
            shared_face_case["configured_source_support_xyz_m"][1],
        )
        self.assertEqual(
            shared_integer[4:6],
            (1, 1),
            msg=(
                "a shared component MAC face must inherit its two admitted "
                "source rows without becoming a third node-search query"
            ),
        )

    def test_adjacent_interior_winner_ignores_loser_endpoint_support(
        self,
    ) -> None:
        """Loser support cannot veto a strict-nearest interior primitive."""

        case = {
            "face": (0.125, 0.25, 0.1),
            "region": 202,
            "positions": (
                (0.125, 0.0, -1.0),
                (0.125, 0.0, 0.0),
                (0.125, 1.0, 0.0),
            ),
            "velocities": ((0.0, 0.0, 0.0),) * 3,
            "projection_segments": ((0, 1), (1, 2)),
            "source_centers": ((0.125, 0.25, 0.05), (0.125, 0.25, 0.15)),
            "boundaries": ((0.125, 0.2, 0.0), (0.125, 0.0, -0.1)),
            "probes": ((0.125, 0.2, 0.55), (0.125, 0.0, 0.65)),
            "normals": ((0.0, 0.0, 1.0),) * 2,
            "indices": ((1, 2, -1), (0, 1, -1)),
            "weights": ((0.8, 0.2, 0.0), (0.1, 0.9, 0.0)),
            "cell_face_x_m": (0.0, 0.25, 0.5, 0.75, 1.0),
            "cell_face_y_m": (-0.125, 0.125, 0.375, 0.625, 0.875),
            "cell_face_z_m": (-0.1, 0.0, 0.1, 0.2, 0.3),
        }
        integer_result, floating_result, _ = (
            self._run_direct_finite_segment_union_case(case)
        )

        positions_yz = np.asarray(case["positions"], dtype=np.float64)[:, 1:]
        face_yz = np.asarray(case["face"], dtype=np.float64)[1:]

        def projection(segment_indices: tuple[int, int]):
            marker_a, marker_b = segment_indices
            segment = positions_yz[marker_b] - positions_yz[marker_a]
            raw_parameter = float(
                np.dot(face_yz - positions_yz[marker_a], segment)
                / np.dot(segment, segment)
            )
            parameter = min(max(raw_parameter, 0.0), 1.0)
            closest = positions_yz[marker_a] + parameter * segment
            distance_squared = float(np.dot(face_yz - closest, face_yz - closest))
            return raw_parameter, distance_squared

        winner_raw, winner_distance_squared = projection((1, 2))
        loser_raw, loser_distance_squared = projection((0, 1))
        local_width_squared = max(0.25**2, 0.1**2)
        tie_tolerance_squared = (
            4.0
            * float(np.finfo(np.float32).eps)
            * max(winner_distance_squared, loser_distance_squared, local_width_squared)
        )
        self.assertGreater(winner_raw, 0.0)
        self.assertLess(winner_raw, 1.0)
        self.assertGreater(loser_raw, 1.0)
        self.assertLess(winner_distance_squared, loser_distance_squared)
        self.assertGreater(
            loser_distance_squared - winner_distance_squared,
            tie_tolerance_squared,
        )
        self.assertEqual(integer_result[0], 1)
        self.assertEqual(integer_result[3], 1)
        self.assertGreater(floating_result[3], 1.0)
        self.assertEqual(
            integer_result[4:6],
            (1, 1),
            msg=(
                "strict-nearest interior winner was vetoed only because the "
                "farther loser endpoint exceeds its local support"
            ),
        )

    def test_short_f32_segment_pair_admission_uses_physical_anchor(self) -> None:
        """Admission validates the serialized anchor in metric geometry."""

        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        marker_a_z_m = np.float32(0.37496)
        marker_b_z_m = np.float32(0.37504)
        segment_z_m = np.float32(marker_b_z_m - marker_a_z_m)
        first_weight = np.float32(0.3)
        second_weight = np.float32(0.7)
        first_anchor_z_m = np.float32(
            marker_a_z_m + first_weight * segment_z_m
        )
        second_anchor_z_m = np.float32(
            marker_a_z_m + second_weight * segment_z_m
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=2, runtime=runtime)
        markers.load_markers(
            positions_m=(
                (0.125, 0.125, float(marker_a_z_m)),
                (0.125, 0.125, float(marker_b_z_m)),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((0.0, 1.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=2,
            runtime=runtime,
        )
        cell_face_x_m = ti.field(dtype=ti.f32, shape=5)
        cell_face_y_m = ti.field(dtype=ti.f32, shape=5)
        cell_face_z_m = ti.field(dtype=ti.f32, shape=5)
        unit_faces = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        cell_face_x_m.from_numpy(unit_faces)
        cell_face_y_m.from_numpy(unit_faces)
        cell_face_z_m.from_numpy(unit_faces)
        result = ti.field(dtype=ti.i32, shape=2)

        stored_marker_a_z_m = float(markers.x_gamma_m[0][2])
        stored_marker_b_z_m = float(markers.x_gamma_m[1][2])
        stored_segment_z_m = stored_marker_b_z_m - stored_marker_a_z_m
        geometric_parameters = tuple(
            (float(anchor) - stored_marker_a_z_m) / stored_segment_z_m
            for anchor in (first_anchor_z_m, second_anchor_z_m)
        )
        parameter_errors = tuple(
            abs(geometric - float(serialized))
            for geometric, serialized in zip(
                geometric_parameters,
                (first_weight, second_weight),
                strict=True,
            )
        )
        physical_anchor_errors_m = tuple(
            abs(
                float(anchor)
                - (
                    stored_marker_a_z_m
                    + float(serialized) * stored_segment_z_m
                )
            )
            for anchor, serialized in zip(
                (first_anchor_z_m, second_anchor_z_m),
                (first_weight, second_weight),
                strict=True,
            )
        )
        geometry_tolerance_m = 2.0 * np.finfo(np.float32).eps * 0.375
        self.assertGreater(min(parameter_errors), 2.0e-6)
        self.assertLess(max(physical_anchor_errors_m), geometry_tolerance_m)

        result.fill(-1)
        _short_f32_segment_pair_geometry_probe(
            boundary,
            markers.x_gamma_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            float(first_anchor_z_m),
            float(second_anchor_z_m),
            result,
        )
        physically_matching_result = tuple(int(result[index]) for index in range(2))
        result.fill(-1)
        _short_f32_segment_pair_geometry_probe(
            boundary,
            markers.x_gamma_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            float(first_anchor_z_m),
            float(np.float32(second_anchor_z_m + np.float32(1.0e-6))),
            result,
        )
        physically_drifted_result = tuple(int(result[index]) for index in range(2))

        markers.load_markers(
            positions_m=(
                (100.0, 0.125, float(marker_a_z_m)),
                (100.0, 0.125, float(marker_b_z_m)),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((0.0, 1.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        result.fill(-1)
        _short_f32_segment_pair_geometry_probe(
            boundary,
            markers.x_gamma_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            float(first_anchor_z_m),
            float(second_anchor_z_m),
            result,
        )
        translated_matching_result = tuple(
            int(result[index]) for index in range(2)
        )
        result.fill(-1)
        _short_f32_segment_pair_geometry_probe(
            boundary,
            markers.x_gamma_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            float(first_anchor_z_m),
            float(np.float32(second_anchor_z_m + np.float32(1.0e-6))),
            result,
        )
        translated_drifted_result = tuple(
            int(result[index]) for index in range(2)
        )

        self.assertEqual(physically_drifted_result, (0, 0))
        self.assertEqual(physically_matching_result, (1, 1))
        self.assertEqual(translated_matching_result, physically_matching_result)
        self.assertEqual(translated_drifted_result, physically_drifted_result)

    @staticmethod
    def _vf48c_captured_pair_case(case_name: str) -> dict[str, object]:
        """Return one compact-index copy of a full vf48c failure witness."""

        if case_name == "adjacent_strict_nearest":
            return {
                "face": (0.000375000003259629, 0.006601562723517418, 0.046562500298023224),
                "dy": 7.81253911554813e-05,
                "dz": 0.0003124997019767761,
                "region": 202,
                # Captured production markers 105, 106, 107.
                "positions": (
                    (0.001500000013038516, 0.0064510987140238285, 0.04687691479921341),
                    (0.001500000013038516, 0.006606992334127426, 0.04687266796827316),
                    (0.001500000013038516, 0.006762884557247162, 0.04686838760972023),
                ),
                "velocities": (
                    (-2.68781635837101e-10, -0.019029440358281136, -0.0852612853050232),
                    (-2.84329837452191e-10, -0.019266681745648384, -0.08784008026123047),
                    (-2.97122881853795e-10, -0.019509775564074516, -0.09044352918863297),
                ),
                "boundaries": (
                    (0.000375000003259629, 0.006614363752305508, 0.046872466802597046),
                    (0.000375000003259629, 0.006605756469070911, 0.04687270149588585),
                ),
                "probes": (
                    (0.000375000003259629, 0.006570685189217329, 0.04528167471289635),
                    (0.000375000003259629, 0.006570926867425442, 0.045594166964292526),
                ),
                "normals": (
                    (0.0, -0.027446772903203964, -0.9996232986450195),
                    (0.0, -0.027231713756918907, -0.9996291995048523),
                ),
                "indices": ((1, 2, -1), (0, 1, -1)),
                "weights": (
                    (0.9527151584625244, 0.0472848117351532, 0.0),
                    (0.007926464080810547, 0.9920735359191895, 0.0),
                ),
            }
        if case_name == "same_segment_endpoint_author":
            return {
                "face": (0.000375000003259629, 0.009023437276482582, 0.05000000074505806),
                "dy": 7.8124925494194e-05,
                "dz": 0.0003124997019767761,
                "region": 101,
                # Captured production segment (56,57).
                "positions": (
                    (0.001500000013038516, 0.008865741081535816, 0.049810655415058136),
                    (0.001500000013038516, 0.009022136218845844, 0.049805741757154465),
                ),
                "velocities": (
                    (8.59934679020569e-10, 0.025367803871631622, -0.12640249729156494),
                    (6.95789759141974e-10, 0.02545979619026184, -0.1294480711221695),
                ),
                "boundaries": (
                    (0.000375000003259629, 0.009022136218845844, 0.049805741757154465),
                    (0.000375000003259629, 0.009012434631586075, 0.04980604723095894),
                ),
                "probes": (
                    (0.000375000003259629, 0.009058658964931965, 0.05096819996833801),
                    (0.000375000003259629, 0.009058765135705471, 0.05128069594502449),
                ),
                "normals": (
                    (0.0, 0.03140304982662201, 0.9995068311691284),
                    (0.0, 0.031402502208948135, 0.9995068907737732),
                ),
                "indices": ((0, 1, -1), (0, 1, -1)),
                "weights": (
                    (0.0, 1.0, 0.0),
                    (0.06203341484069824, 0.9379665851593018, 0.0),
                ),
            }
        if case_name == "terminal_endpoint_clamp":
            return {
                "face": (0.000375000003259629, 0.009960937313735485, 0.046562500298023224),
                "dy": 7.8124925494194e-05,
                "dz": 0.0003124997019767761,
                "region": 202,
                # Captured terminal production segment (127,129).
                "positions": (
                    (0.001500000013038516, 0.009883272461593151, 0.0467824749648571),
                    (0.001500000013038516, 0.009961388073861599, 0.046781234443187714),
                ),
                "velocities": (
                    (3.35948699414779e-11, -0.022780701518058777, -0.14448581635951996),
                    (3.6318132529134e-11, -0.02278713881969452, -0.14526715874671936),
                ),
                "boundaries": (
                    (0.000375000003259629, 0.009961388073861599, 0.046781234443187714),
                    (0.000375000003259629, 0.009961388073861599, 0.046781234443187714),
                ),
                "probes": (
                    (0.000375000003259629, 0.009937571361660957, 0.04528148099780083),
                    (0.000375000003259629, 0.009942532517015934, 0.045593902468681335),
                ),
                "normals": (
                    (0.0, -0.01587841659784317, -0.9998739361763),
                    (0.0, -0.01587860845029354, -0.9998738765716553),
                ),
                "indices": ((0, 1, -1), (0, 1, -1)),
                "weights": ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            }
        raise ValueError(case_name)

    def test_vf48c_captured_pairs_require_finite_segment_union_admission(
        self,
    ) -> None:
        """Three production-resolvable pair shapes must enter reconstruction."""

        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=3,
            runtime=runtime,
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=3, runtime=runtime)
        face_center_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        author_source_center_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_boundary_point_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_nominal_probe_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_actual_probe_m = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_normal = ti.Vector.field(3, dtype=ti.f32, shape=2)
        author_projection_indices = ti.Vector.field(3, dtype=ti.i32, shape=2)
        author_projection_weights = ti.Vector.field(3, dtype=ti.f32, shape=2)
        configured_source_support_xyz_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        cell_face_x_m = ti.field(dtype=ti.f32, shape=5)
        cell_face_y_m = ti.field(dtype=ti.f32, shape=5)
        cell_face_z_m = ti.field(dtype=ti.f32, shape=5)
        result_i32 = ti.field(dtype=ti.i32, shape=7)
        result_f64 = ti.field(dtype=ti.f64, shape=16)
        cell_face_x_m.from_numpy(np.linspace(0.0, 1.0, 5, dtype=np.float32))
        configured_source_support_xyz_m[None] = (1.0e30, 1.0e30, 1.0e30)

        for case_name in (
            "adjacent_strict_nearest",
            "same_segment_endpoint_author",
            "terminal_endpoint_clamp",
        ):
            with self.subTest(captured_case=case_name):
                case = self._vf48c_captured_pair_case(case_name)
                positions = np.asarray(case["positions"], dtype=np.float32)
                velocities = np.asarray(case["velocities"], dtype=np.float32)
                marker_count = int(positions.shape[0])
                markers.load_markers(
                    positions_m=tuple(tuple(row) for row in positions),
                    velocities_mps=tuple(tuple(row) for row in velocities),
                    normals=((0.0, 0.0, 1.0),) * marker_count,
                    areas_m2=(1.0 / marker_count,) * marker_count,
                    region_ids=(int(case["region"]),) * marker_count,
                )
                projection_segments = tuple(
                    dict.fromkeys(
                        tuple(sorted((int(indices[0]), int(indices[1]))))
                        for indices in case["indices"]
                    )
                )
                markers.set_projection_segments(projection_segments)
                face_center_m[None] = case["face"]
                author_boundary_point_m.from_numpy(
                    np.asarray(case["boundaries"], dtype=np.float32)
                )
                probes = np.asarray(case["probes"], dtype=np.float32)
                author_nominal_probe_m.from_numpy(probes)
                author_actual_probe_m.from_numpy(probes)
                author_normal.from_numpy(np.asarray(case["normals"], dtype=np.float32))
                author_projection_indices.from_numpy(
                    np.asarray(case["indices"], dtype=np.int32)
                )
                author_projection_weights.from_numpy(
                    np.asarray(case["weights"], dtype=np.float32)
                )
                dy_m = float(case["dy"])
                dz_m = float(case["dz"])
                face = np.asarray(case["face"], dtype=np.float32)
                source_centers = np.asarray(
                    (
                        (face[0], face[1], face[2] - 0.5 * dz_m),
                        (face[0], face[1], face[2] + 0.5 * dz_m),
                    ),
                    dtype=np.float32,
                )
                author_source_center_m.from_numpy(source_centers)
                cell_face_y_m.from_numpy(
                    np.asarray(
                        [face[1] + offset * dy_m for offset in (-1.5, -0.5, 0.5, 1.5, 2.5)],
                        dtype=np.float32,
                    )
                )
                cell_face_z_m.from_numpy(
                    np.asarray(
                        [face[2] + offset * dz_m for offset in (-8.0, -1.0, 0.0, 1.0, 8.0)],
                        dtype=np.float32,
                    )
                )
                result_i32.fill(-1)
                result_f64.fill(np.nan)

                _captured_finite_segment_union_admission_probe(
                    boundary,
                    markers.x_gamma_m,
                    markers.v_gamma_mps,
                    markers.region_id,
                    face_center_m,
                    2,
                    author_source_center_m,
                    author_boundary_point_m,
                    author_nominal_probe_m,
                    author_actual_probe_m,
                    author_normal,
                    author_projection_indices,
                    author_projection_weights,
                    markers.projection_triangle_indices,
                    int(markers.projection_segment_count),
                    configured_source_support_xyz_m,
                    1,
                    1,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    int(case["region"]),
                    result_i32,
                    result_f64,
                )
                integer_result = tuple(int(result_i32[index]) for index in range(7))
                floating_result = tuple(float(result_f64[index]) for index in range(16))

                self.assertEqual(integer_result[0:2], (1, 1))
                if case_name == "adjacent_strict_nearest":
                    self.assertEqual(sum(integer_result[2:4]), 1)
                    local_width_squared = max(dy_m * dy_m, dz_m * dz_m)
                    tie_tolerance = (
                        4.0
                        * float(np.finfo(np.float32).eps)
                        * max(floating_result[0], floating_result[1], local_width_squared)
                    )
                    self.assertGreater(
                        abs(floating_result[0] - floating_result[1]),
                        tie_tolerance,
                    )
                elif case_name == "same_segment_endpoint_author":
                    segment = positions[1, 1:].astype(np.float64) - positions[0, 1:].astype(np.float64)
                    raw_t = float(
                        np.dot(face[1:].astype(np.float64) - positions[0, 1:].astype(np.float64), segment)
                        / np.dot(segment, segment)
                    )
                    author_t = tuple(float(row[1]) for row in case["weights"])
                    self.assertLess(min(author_t), raw_t)
                    self.assertLess(raw_t, max(author_t))
                    self.assertIn(1.0, author_t)
                else:
                    self.assertEqual(integer_result[2:4], (1, 1))
                    self.assertAlmostEqual(floating_result[2], 0.07276335, delta=0.01)
                    self.assertAlmostEqual(floating_result[3], 0.07276335, delta=0.01)

                    endpoint = positions[1, 1:].astype(np.float64)
                    tangent = positions[1, 1:].astype(np.float64) - positions[0, 1:].astype(np.float64)
                    tangent /= np.linalg.norm(tangent)
                    chord_normal = np.asarray((-tangent[1], tangent[0]))
                    if chord_normal[1] > 0.0:
                        chord_normal = -chord_normal
                    q = face[1:].astype(np.float64) - endpoint
                    beta_m = float(np.dot(q, tangent))
                    alpha_m = float(np.dot(q, chord_normal))
                    dual_support_m = 0.5 * (
                        abs(float(tangent[0])) * dy_m
                        + abs(float(tangent[1])) * dz_m
                    )
                    self.assertGreaterEqual(beta_m, 0.0)
                    self.assertLessEqual(beta_m, dual_support_m)
                    self.assertGreater(alpha_m, 0.0)
                    canonical_ray = q / np.linalg.norm(q)
                    margins = []
                    source_centers_yz = (
                        (float(face[1]), float(face[2]) - 0.5 * dz_m),
                        (float(face[1]), float(face[2]) + 0.5 * dz_m),
                    )
                    for probe, source_center, normal in zip(
                        probes[:, 1:].astype(np.float64),
                        source_centers_yz,
                        np.asarray(case["normals"], dtype=np.float32)[:, 1:].astype(np.float64),
                        strict=True,
                    ):
                        normal /= np.linalg.norm(normal)
                        self.assertGreater(float(np.dot(normal, chord_normal)), 0.999999)
                        margins.append(
                            float(
                                np.dot(probe - endpoint, normal)
                                - np.dot(
                                    np.asarray(source_center, dtype=np.float64)
                                    - endpoint,
                                    normal,
                                )
                            )
                        )
                    self.assertGreater(min(margins), 0.0)
                    self.assertLess(max(margins) - min(margins), 1.0e-8)
                    canonical_probe = endpoint + (
                        np.linalg.norm(q) + min(margins)
                    ) * canonical_ray
                    self.assertGreater(
                        float(np.dot(canonical_probe - face[1:], canonical_ray)),
                        0.0,
                    )

                self.assertEqual(
                    integer_result[4],
                    1,
                    msg=(
                        f"{case_name} has valid finite primitives but the current "
                        "pair admission rejects the production-resolvable union"
                    ),
                )
                self.assertEqual(integer_result[5], 1)

                canonical_integer_payload = result_i32.to_numpy()[4:].tobytes(
                    order="C"
                )
                canonical_float_payload = result_f64.to_numpy()[4:].tobytes(
                    order="C"
                )
                for field, values in (
                    (author_source_center_m, source_centers),
                    (
                        author_boundary_point_m,
                        np.asarray(case["boundaries"], dtype=np.float32),
                    ),
                    (author_nominal_probe_m, probes),
                    (author_actual_probe_m, probes),
                    (
                        author_normal,
                        np.asarray(case["normals"], dtype=np.float32),
                    ),
                    (
                        author_projection_indices,
                        np.asarray(case["indices"], dtype=np.int32),
                    ),
                    (
                        author_projection_weights,
                        np.asarray(case["weights"], dtype=np.float32),
                    ),
                ):
                    field.from_numpy(values[::-1].copy())
                result_i32.fill(-1)
                result_f64.fill(np.nan)
                _captured_finite_segment_union_admission_probe(
                    boundary,
                    markers.x_gamma_m,
                    markers.v_gamma_mps,
                    markers.region_id,
                    face_center_m,
                    2,
                    author_source_center_m,
                    author_boundary_point_m,
                    author_nominal_probe_m,
                    author_actual_probe_m,
                    author_normal,
                    author_projection_indices,
                    author_projection_weights,
                    markers.projection_triangle_indices,
                    int(markers.projection_segment_count),
                    configured_source_support_xyz_m,
                    1,
                    1,
                    cell_face_x_m,
                    cell_face_y_m,
                    cell_face_z_m,
                    int(case["region"]),
                    result_i32,
                    result_f64,
                )
                self.assertEqual(
                    result_i32.to_numpy()[4:].tobytes(order="C"),
                    canonical_integer_payload,
                )
                self.assertEqual(
                    result_f64.to_numpy()[4:].tobytes(order="C"),
                    canonical_float_payload,
                )

        def run_fail_closed_case(case: dict[str, object]):
            positions = np.asarray(case["positions"], dtype=np.float32)
            velocities = np.asarray(case["velocities"], dtype=np.float32)
            marker_count = int(positions.shape[0])
            markers.load_markers(
                positions_m=tuple(tuple(row) for row in positions),
                velocities_mps=tuple(tuple(row) for row in velocities),
                normals=((0.0, 0.0, 1.0),) * marker_count,
                areas_m2=(1.0 / marker_count,) * marker_count,
                region_ids=(int(case["region"]),) * marker_count,
            )
            projection_segments = tuple(
                dict.fromkeys(
                    tuple(sorted((int(indices[0]), int(indices[1]))))
                    for indices in case["indices"]
                )
            )
            markers.set_projection_segments(projection_segments)
            face = np.asarray(case["face"], dtype=np.float32)
            face_center_m[None] = face
            author_boundary_point_m.from_numpy(
                np.asarray(case["boundaries"], dtype=np.float32)
            )
            probes = np.asarray(case["probes"], dtype=np.float32)
            author_nominal_probe_m.from_numpy(probes)
            author_actual_probe_m.from_numpy(probes)
            author_normal.from_numpy(np.asarray(case["normals"], dtype=np.float32))
            author_projection_indices.from_numpy(
                np.asarray(case["indices"], dtype=np.int32)
            )
            author_projection_weights.from_numpy(
                np.asarray(case["weights"], dtype=np.float32)
            )
            dy_m = float(case["dy"])
            dz_m = float(case["dz"])
            source_centers = np.asarray(
                (
                    (face[0], face[1], face[2] - 0.5 * dz_m),
                    (face[0], face[1], face[2] + 0.5 * dz_m),
                ),
                dtype=np.float32,
            )
            author_source_center_m.from_numpy(source_centers)
            cell_face_y_m.from_numpy(
                np.asarray(
                    [face[1] + offset * dy_m for offset in (-1.5, -0.5, 0.5, 1.5, 2.5)],
                    dtype=np.float32,
                )
            )
            cell_face_z_m.from_numpy(
                np.asarray(
                    [face[2] + offset * dz_m for offset in (-8.0, -1.0, 0.0, 1.0, 8.0)],
                    dtype=np.float32,
                )
            )
            immutable_inputs_before = tuple(
                field.to_numpy().tobytes(order="C")
                for field in (
                    markers.x_gamma_m,
                    markers.v_gamma_mps,
                    markers.region_id,
                    author_source_center_m,
                    author_boundary_point_m,
                    author_nominal_probe_m,
                    author_actual_probe_m,
                    author_normal,
                    author_projection_indices,
                    author_projection_weights,
                )
            )
            result_i32.fill(-1)
            result_f64.fill(np.nan)
            _captured_finite_segment_union_admission_probe(
                boundary,
                markers.x_gamma_m,
                markers.v_gamma_mps,
                markers.region_id,
                face_center_m,
                2,
                author_source_center_m,
                author_boundary_point_m,
                author_nominal_probe_m,
                author_actual_probe_m,
                author_normal,
                author_projection_indices,
                author_projection_weights,
                markers.projection_triangle_indices,
                int(markers.projection_segment_count),
                configured_source_support_xyz_m,
                1,
                1,
                cell_face_x_m,
                cell_face_y_m,
                cell_face_z_m,
                int(case["region"]),
                result_i32,
                result_f64,
            )
            immutable_inputs_after = tuple(
                field.to_numpy().tobytes(order="C")
                for field in (
                    markers.x_gamma_m,
                    markers.v_gamma_mps,
                    markers.region_id,
                    author_source_center_m,
                    author_boundary_point_m,
                    author_nominal_probe_m,
                    author_actual_probe_m,
                    author_normal,
                    author_projection_indices,
                    author_projection_weights,
                )
            )
            self.assertEqual(
                immutable_inputs_after,
                immutable_inputs_before,
                msg="rejected pair geometry mutated an author or primitive input",
            )
            return (
                tuple(int(result_i32[index]) for index in range(7)),
                tuple(float(result_f64[index]) for index in range(16)),
            )

        # A genuine equal-distance corner with two distinct closest points has
        # no unique primitive owner.  Unlike a C0 shared-vertex snap, it must
        # remain fail closed.
        unresolved_tie = {
            "face": (0.125, 0.25, 0.625),
            "dy": 0.25,
            "dz": 0.25,
            "region": 202,
            "positions": (
                (0.125, 0.125, 0.5),
                (0.125, 0.375, 0.5),
                (0.125, 0.375, 0.75),
            ),
            "velocities": ((0.0, 0.0, 0.0),) * 3,
            "boundaries": (
                (0.125, 0.25, 0.5),
                (0.125, 0.375, 0.625),
            ),
            "probes": (
                (0.125, 0.25, 0.75),
                (0.125, 0.25, 0.625),
            ),
            "normals": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            "indices": ((0, 1, -1), (1, 2, -1)),
            "weights": ((0.5, 0.5, 0.0), (0.5, 0.5, 0.0)),
        }
        with self.subTest(fail_closed="adjacent_unresolved_tie"):
            tie_integer, tie_float = run_fail_closed_case(unresolved_tie)
            self.assertEqual(tie_integer[0:2], (1, 1))
            self.assertAlmostEqual(tie_float[0], tie_float[1], places=12)
            self.assertEqual(tie_integer[4], 0)

        terminal = self._vf48c_captured_pair_case("terminal_endpoint_clamp")
        terminal_positions = np.asarray(terminal["positions"], dtype=np.float32)
        endpoint = terminal_positions[1].astype(np.float64)
        tangent_yz = (
            terminal_positions[1, 1:].astype(np.float64)
            - terminal_positions[0, 1:].astype(np.float64)
        )
        tangent_yz /= np.linalg.norm(tangent_yz)
        chord_normal_yz = np.asarray((-tangent_yz[1], tangent_yz[0]))
        if chord_normal_yz[1] > 0.0:
            chord_normal_yz = -chord_normal_yz
        terminal_support_m = 0.5 * (
            abs(float(tangent_yz[0])) * float(terminal["dy"])
            + abs(float(tangent_yz[1])) * float(terminal["dz"])
        )

        support_outside_q = (
            1.25 * terminal_support_m * tangent_yz
            + 0.5 * float(terminal["dz"]) * chord_normal_yz
        )
        support_outside = {
            **terminal,
            "face": (
                float(endpoint[0]),
                float(endpoint[1] + support_outside_q[0]),
                float(endpoint[2] + support_outside_q[1]),
            ),
        }
        with self.subTest(fail_closed="terminal_support_outside"):
            support_integer, _ = run_fail_closed_case(support_outside)
            self.assertEqual(support_integer[0:2], (0, 0))
            self.assertEqual(support_integer[2:4], (1, 1))
            self.assertEqual(support_integer[4], 0)

        backside_q = (
            0.25 * terminal_support_m * tangent_yz
            - 0.5 * float(terminal["dz"]) * chord_normal_yz
        )
        backside = {
            **terminal,
            "face": (
                float(endpoint[0]),
                float(endpoint[1] + backside_q[0]),
                float(endpoint[2] + backside_q[1]),
            ),
        }
        with self.subTest(fail_closed="terminal_backside"):
            backside_integer, _ = run_fail_closed_case(backside)
            self.assertEqual(backside_integer[0:2], (1, 1))
            self.assertLess(float(np.dot(backside_q, chord_normal_yz)), 0.0)
            self.assertEqual(backside_integer[4], 0)

        normal_cone_failure = {
            **terminal,
            "normals": (
                terminal["normals"][0],
                tuple(-float(value) for value in terminal["normals"][1]),
            ),
        }
        with self.subTest(fail_closed="terminal_normal_cone"):
            cone_integer, _ = run_fail_closed_case(normal_cone_failure)
            self.assertEqual(cone_integer[0:2], (1, 1))
            self.assertEqual(cone_integer[4], 0)

if __name__ == "__main__":
    unittest.main()
