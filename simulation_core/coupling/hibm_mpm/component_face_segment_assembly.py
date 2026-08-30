"""Device-resident registry-owned component-face segment assembly.

This class deliberately owns source-by-axis transaction scratch.  It never
publishes a canonical ledger field; callers must retain the existing sole
eight-field commit after all three passes accept.
"""

import math
import operator
from typing import Sequence

import taichi as ti

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi

from .component_face_segment_audit import RegisteredSegmentAudit
from .component_face_candidate_geometry import RegisteredCandidateGeometry
from .component_face_segment_geometry import (
    build_registered_segment_topology,
    finite_segment_projection_2d,
    registered_endpoint_in_strict_face_support_2d,
    registered_normal_in_active_plane_2d,
)


@ti.data_oriented
class RegisteredComponentFaceSegmentAssembler(RegisteredCandidateGeometry, RegisteredSegmentAudit):
    """Scratch and F64 owner selection for the 2-D finite-segment route."""

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        marker_capacity: int,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        if len(grid_nodes) != 3 or any(int(value) <= 0 for value in grid_nodes):
            raise ValueError("grid_nodes must contain three positive integers")
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        init_taichi(runtime)
        self.grid_nodes = tuple(int(value) for value in grid_nodes)
        self.marker_capacity = int(marker_capacity)
        self.source_axis_record_capacity = math.prod(self.grid_nodes) * 3
        source_shape = self.grid_nodes + (3,)
        self.raw_route_valid = ti.field(dtype=ti.i32, shape=source_shape)
        self.raw_route_kind = ti.field(dtype=ti.i32, shape=source_shape)
        self.raw_route_target = ti.Vector.field(3, dtype=ti.i32, shape=source_shape)
        self.raw_route_region = ti.field(dtype=ti.i32, shape=source_shape)
        self.raw_route_primitive = ti.Vector.field(3, dtype=ti.i32, shape=source_shape)
        self.raw_route_weights = ti.Vector.field(3, dtype=ti.f32, shape=source_shape)
        self.raw_route_boundary_target_mps = ti.field(dtype=ti.f32, shape=source_shape)
        self.raw_route_anchor_m = ti.Vector.field(3, dtype=ti.f32, shape=source_shape)
        self.raw_route_nominal_sample_m = ti.Vector.field(3, dtype=ti.f32, shape=source_shape)
        self.raw_route_actual_sample_m = ti.Vector.field(3, dtype=ti.f32, shape=source_shape)
        self.raw_route_normal = ti.Vector.field(3, dtype=ti.f32, shape=source_shape)
        self.raw_route_sample_valid = ti.field(dtype=ti.i32, shape=source_shape)
        self.raw_route_generation = ti.field(dtype=ti.i32, shape=source_shape)
        self.face_raw_count = ti.Vector.field(3, dtype=ti.i32, shape=self.grid_nodes)
        self.candidate_face_requested = ti.Vector.field(3, dtype=ti.i32, shape=self.grid_nodes)
        self.candidate_owner_permission = ti.field(dtype=ti.i32, shape=source_shape)
        self.candidate_owner_failure = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_valid = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_segment = ti.Vector.field(2, dtype=ti.i32, shape=source_shape)
        self.owner_weight = ti.field(dtype=ti.f32, shape=source_shape)
        self.owner_target_mps = ti.field(dtype=ti.f32, shape=source_shape)
        self.owner_region = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_segment_index = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_vertex_tie = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_vertex = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_point_m = ti.Vector.field(3, dtype=ti.f32, shape=source_shape)
        self.owner_ambiguous = ti.field(dtype=ti.i32, shape=source_shape)
        self.owner_blocked = ti.field(dtype=ti.i32, shape=source_shape)
        self.certificate_valid = ti.field(dtype=ti.i32, shape=source_shape)
        self.certificate_blocked = ti.field(dtype=ti.i32, shape=source_shape)
        self.certificate_path_edge_count = ti.field(dtype=ti.i32, shape=source_shape)
        self.audit_valid = ti.field(dtype=ti.i32, shape=source_shape)
        self.audit_raw_count = ti.field(dtype=ti.i32, shape=source_shape)
        self.audit_failure = ti.field(dtype=ti.i32, shape=source_shape)
        self.audit_rejection_count = ti.field(dtype=ti.i32, shape=())
        self.raw_route_audit_failure = ti.field(dtype=ti.i32, shape=source_shape)
        self.audit_first_rejected_source_key = ti.field(dtype=ti.i32, shape=())
        self.audit_first_rejected_face_key = ti.field(dtype=ti.i32, shape=())
        self._audited_owner_valid = ti.field(dtype=ti.i32, shape=source_shape)
        self._audited_owner_failure = ti.field(dtype=ti.i32, shape=source_shape)
        self._registered_edge_valid = ti.field(dtype=ti.i32, shape=self.marker_capacity)
        self._registered_edge_normal = ti.Vector.field(3, dtype=ti.f64, shape=self.marker_capacity)
        self._registered_alias_valid = ti.field(dtype=ti.i32, shape=self.marker_capacity)
        self.registered_topology_ready = ti.field(dtype=ti.i32, shape=())
        self.registered_topology_vertex_count = ti.field(dtype=ti.i32, shape=())
        self.registered_topology_segment_count = ti.field(dtype=ti.i32, shape=())
        self.registered_vertex_degree = ti.field(dtype=ti.i32, shape=self.marker_capacity)
        self.registered_vertex_adjacency = ti.Vector.field(
            2, dtype=ti.i32, shape=self.marker_capacity
        )
        self.explicit_endpoint_alias = ti.field(
            dtype=ti.i32, shape=self.marker_capacity
        )
        self.explicit_alias_expected_role = ti.field(
            dtype=ti.i32, shape=self.marker_capacity
        )
        self.certificate_path_edge = ti.field(dtype=ti.i32, shape=self.marker_capacity)
        self.certificate_candidate_path_edge = ti.field(
            dtype=ti.i32, shape=self.marker_capacity
        )
        self.registered_topology_ready[None] = 0
        self.registered_topology_vertex_count[None] = 0
        self.registered_topology_segment_count[None] = 0
        for marker in range(self.marker_capacity):
            self.registered_vertex_degree[marker] = 0
            self.registered_vertex_adjacency[marker] = (-1, -1)
            self.explicit_endpoint_alias[marker] = -1
            self.explicit_alias_expected_role[marker] = -1
            self.certificate_path_edge[marker] = 0
            self.certificate_candidate_path_edge[marker] = 0

    @ti.kernel
    def _clear_device_transaction_kernel(self):
        for i, j, k, axis in ti.ndrange(
            self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2], 3
        ):
            key = (i, j, k, axis)
            self.raw_route_valid[key] = 0
            self.candidate_face_requested[i, j, k][axis] = 0
            self.candidate_owner_permission[key] = 0
            self.candidate_owner_failure[key] = 0
            self.raw_route_kind[key] = -1
            self.raw_route_target[key] = ti.Vector([-1, -1, -1])
            self.raw_route_region[key] = -1
            self.raw_route_primitive[key] = ti.Vector([-1, -1, -1])
            self.raw_route_weights[key] = ti.Vector([0.0, 0.0, 0.0])
            self.raw_route_boundary_target_mps[key] = 0.0
            self.raw_route_anchor_m[key] = ti.Vector([0.0, 0.0, 0.0])
            self.raw_route_nominal_sample_m[key] = ti.Vector([0.0, 0.0, 0.0])
            self.raw_route_actual_sample_m[key] = ti.Vector([0.0, 0.0, 0.0])
            self.raw_route_normal[key] = ti.Vector([0.0, 0.0, 0.0])
            self.raw_route_sample_valid[key] = 0
            self.raw_route_generation[key] = -1
            self.owner_valid[key] = 0
            self.owner_segment[key] = ti.Vector([-1, -1])
            self.owner_weight[key] = 0.0
            self.owner_target_mps[key] = 0.0
            self.owner_region[key] = -1
            self.owner_segment_index[key] = -1
            self.owner_vertex_tie[key] = 0
            self.owner_vertex[key] = -1
            self.owner_point_m[key] = ti.Vector([0.0, 0.0, 0.0])
            self.owner_ambiguous[key] = 0
            self.owner_blocked[key] = 0
            self.certificate_valid[key] = 0
            self.certificate_blocked[key] = 0
            self.certificate_path_edge_count[key] = 0
            self.audit_valid[key] = 0
            self.audit_raw_count[key] = 0
            self.audit_failure[key] = 0
        self.audit_rejection_count[None] = 0
        for node in ti.grouped(self.face_raw_count):
            self.face_raw_count[node] = ti.Vector([0, 0, 0])

    def clear_device_transaction(self) -> None:
        self._clear_device_transaction_kernel()

    def install_registered_topology(
        self,
        segment_indices: Sequence[Sequence[int]],
        *,
        vertex_count: int,
    ) -> None:
        """Install immutable integer adjacency for one registered segment registry."""

        topology = build_registered_segment_topology(
            segment_indices,
            vertex_count=int(vertex_count),
        )
        if int(vertex_count) > self.marker_capacity:
            raise ValueError("registered topology exceeds marker_capacity")
        if len(topology.segments) > self.marker_capacity:
            raise ValueError("registered topology exceeds certificate edge capacity")
        self.registered_topology_ready[None] = 0
        for marker in range(self.marker_capacity):
            self.registered_vertex_degree[marker] = 0
            self.registered_vertex_adjacency[marker] = (-1, -1)
        for marker, degree in enumerate(topology.degree):
            self.registered_vertex_degree[marker] = int(degree)
            self.registered_vertex_adjacency[marker] = topology.adjacency[marker]
        self.registered_topology_vertex_count[None] = int(vertex_count)
        self.registered_topology_segment_count[None] = len(topology.segments)
        self.registered_topology_ready[None] = 1

    def install_explicit_endpoint_aliases(
        self,
        bindings: Sequence[Sequence[int]],
        *,
        expected_role_pairs: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Install only declared cap endpoint aliases; geometry proves them later."""

        aliases = [-1] * self.marker_capacity
        expected_roles = [-1] * self.marker_capacity
        if expected_role_pairs is not None and len(expected_role_pairs) != len(bindings):
            raise ValueError("expected_role_pairs must match endpoint alias bindings")
        for binding_index, binding in enumerate(bindings):
            if len(binding) != 2:
                raise ValueError("endpoint alias must contain two marker indices")
            first, second = int(binding[0]), int(binding[1])
            if first == second or first < 0 or second < 0:
                raise ValueError("endpoint alias markers must be distinct")
            if first >= self.marker_capacity or second >= self.marker_capacity:
                raise ValueError("endpoint alias marker is out of range")
            if aliases[first] != -1 or aliases[second] != -1:
                raise ValueError("endpoint alias marker has multiple bindings")
            aliases[first] = second
            aliases[second] = first
            if expected_role_pairs is not None:
                pair = expected_role_pairs[binding_index]
                if len(pair) != 2:
                    raise ValueError("endpoint alias roles must contain two pressure owners")
                first_role, second_role = operator.index(pair[0]), operator.index(pair[1])
                if first_role < 0 or second_role < 0:
                    raise ValueError("endpoint alias expected pressure owners must be nonnegative")
                expected_roles[first] = first_role
                expected_roles[second] = second_role
        for marker, alias in enumerate(aliases):
            self.explicit_endpoint_alias[marker] = alias
            self.explicit_alias_expected_role[marker] = expected_roles[marker]

    @ti.func
    def _scan_registered_owner_device(
        self,
        face_i: ti.i32,
        face_j: ti.i32,
        face_k: ti.i32,
        component_axis: ti.i32,
        inactive_axis: ti.i32,
        face_x: ti.f32,
        face_y: ti.f32,
        face_z: ti.f32,
        projection_segment_indices: ti.template(),
        projection_segment_count: ti.i32,
        marker_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(),
        projection_vertex_count: ti.i32,
    ):
        face = ti.Vector([face_i, face_j, face_k])
        self.owner_valid[face, component_axis] = 0
        self.owner_segment[face, component_axis] = ti.Vector([-1, -1])
        self.owner_segment_index[face, component_axis] = -1
        self.owner_region[face, component_axis] = -1
        self.owner_target_mps[face, component_axis] = 0.0
        self.owner_weight[face, component_axis] = 0.0
        self.owner_point_m[face, component_axis] = ti.Vector([0.0, 0.0, 0.0])
        best_distance = ti.cast(1.0e30, ti.f64)
        best_point = ti.Vector([
            ti.cast(0.0, ti.f64),
            ti.cast(0.0, ti.f64),
            ti.cast(0.0, ti.f64),
        ])
        best_weight = ti.cast(0.0, ti.f64)
        best_segment = ti.Vector([-1, -1])
        best_segment_index = -1
        best_vertex = -1
        vertex_tie = 0
        nearest_invalid = 0
        registry_invalid = 0
        best_region = -1
        ambiguous = 0
        blocked = 0
        face_center = ti.Vector([face_x, face_y, face_z])
        segment_index = 0
        while segment_index < projection_segment_count:
            segment = projection_segment_indices[segment_index]
            marker_a = ti.min(segment.x, segment.y)
            marker_b = ti.max(segment.x, segment.y)
            valid = ti.cast(
                marker_a >= 0
                and marker_b >= 0
                and marker_a < projection_vertex_count
                and marker_b < projection_vertex_count
                and marker_a != marker_b
                and segment.z == -1
                , ti.i32
            )
            if valid:
                projected, weight, point, distance = finite_segment_projection_2d(
                    face_center,
                    marker_position_m[marker_a],
                    marker_position_m[marker_b],
                    inactive_axis,
                )
                valid = ti.cast(projected, ti.i32)
                if valid != 0:
                    candidate_invalid = ti.cast(
                        marker_region_id[marker_a] < 0
                        or marker_region_id[marker_a] != marker_region_id[marker_b], ti.i32,
                    )
                    vertex = -1
                    if weight == 0.0:
                        vertex = marker_a
                    elif weight == 1.0:
                        vertex = marker_b
                    # Compare in F64 before any F32 storage.  A strict tie is
                    # only resolved later by a shared C0-vertex certificate.
                    if distance < best_distance:
                        best_distance = distance
                        best_point = point
                        best_weight = weight
                        best_segment = ti.Vector([marker_a, marker_b])
                        best_segment_index = segment_index
                        best_vertex = vertex
                        vertex_tie = 0
                        nearest_invalid = candidate_invalid
                        best_region = marker_region_id[marker_a]
                        ambiguous = 0
                    elif distance == best_distance:
                        delta = point - best_point
                        delta[inactive_axis] = 0.0
                        nearest_invalid = ti.max(nearest_invalid, candidate_invalid)
                        if delta.dot(delta) > 1.0e-24:
                            ambiguous = 1
                        else:
                            vertex_tie = 1
                            shared = ti.cast(vertex >= 0 and best_vertex >= 0 and vertex == best_vertex, ti.i32)
                            if vertex >= 0 and best_vertex >= 0:
                                if self.explicit_endpoint_alias[vertex] == best_vertex:
                                    shared = 1
                            if shared == 0:
                                ambiguous = 1
                            # Canonical endpoint IDs, not registration order,
                            # select the representative of a certified C0 tie.
                            if marker_a < best_segment.x or (marker_a == best_segment.x and marker_b < best_segment.y):
                                best_weight = weight
                                best_segment = ti.Vector([marker_a, marker_b])
                                best_segment_index = segment_index
                                best_vertex = vertex
                                best_region = marker_region_id[marker_a]
            if valid == 0:
                registry_invalid = 1
            segment_index += 1
        blocked = ti.max(nearest_invalid, registry_invalid)
        if best_segment.x >= 0:
            self.owner_segment[face, component_axis] = best_segment
            self.owner_segment_index[face, component_axis] = best_segment_index
            self.owner_weight[face, component_axis] = ti.cast(best_weight, ti.f32)
            self.owner_point_m[face, component_axis] = ti.cast(best_point, ti.f32)
            self.owner_region[face, component_axis] = best_region
            if (
                best_region < 0
                or marker_region_id[best_segment.y] != best_region
            ):
                blocked = 1
            target = (
                (1.0 - best_weight)
                * marker_velocity_mps[best_segment.x][component_axis]
                + best_weight
                * marker_velocity_mps[best_segment.y][component_axis]
            )
            if ti.math.isnan(target) or ti.math.isinf(target):
                blocked = 1
            if ambiguous != 0:
                blocked = 1
            if blocked == 0:
                self.owner_valid[face, component_axis] = 1
            self.owner_target_mps[face, component_axis] = ti.cast(target, ti.f32)
        else:
            blocked = 1
        self.owner_vertex_tie[face, component_axis] = vertex_tie
        self.owner_vertex[face, component_axis] = best_vertex
        self.owner_ambiguous[face, component_axis] = ambiguous
        self.owner_blocked[face, component_axis] = blocked

    @ti.kernel
    def _scan_registered_owner_kernel(
        self, face_i: ti.i32, face_j: ti.i32, face_k: ti.i32,
        component_axis: ti.i32, inactive_axis: ti.i32,
        face_x: ti.f32, face_y: ti.f32, face_z: ti.f32,
        projection_segment_indices: ti.template(), projection_segment_count: ti.i32,
        marker_position_m: ti.template(), marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(), projection_vertex_count: ti.i32,
    ):
        self._scan_registered_owner_device(
            face_i, face_j, face_k, component_axis, inactive_axis,
            face_x, face_y, face_z, projection_segment_indices, projection_segment_count,
            marker_position_m, marker_velocity_mps, marker_region_id, projection_vertex_count,
        )

    @ti.kernel
    def _scan_registered_active_faces_kernel(
        self, inactive_axis: ti.i32,
        cell_face_x_m: ti.template(), cell_face_y_m: ti.template(), cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(), cell_center_y_m: ti.template(), cell_center_z_m: ti.template(),
        projection_segment_indices: ti.template(), projection_segment_count: ti.i32,
        marker_position_m: ti.template(), marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(), projection_vertex_count: ti.i32,
    ):
        for i, j, k, axis in ti.ndrange(self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2], 3):
            if self.face_raw_count[i, j, k][axis] > 0:
                point = ti.Vector([cell_center_x_m[i], cell_center_y_m[j], cell_center_z_m[k]])
                if axis == 0:
                    point.x = cell_face_x_m[i]
                elif axis == 1:
                    point.y = cell_face_y_m[j]
                else:
                    point.z = cell_face_z_m[k]
                self._scan_registered_owner_device(
                    i, j, k, axis, inactive_axis, point.x, point.y, point.z,
                    projection_segment_indices, projection_segment_count,
                    marker_position_m, marker_velocity_mps, marker_region_id, projection_vertex_count,
                )

    def scan_registered_active_faces_device(
        self, *, inactive_axis: int, cell_face_x_m, cell_face_y_m, cell_face_z_m,
        cell_center_x_m, cell_center_y_m, cell_center_z_m,
        projection_segment_indices, projection_segment_count: int,
        marker_position_m, marker_velocity_mps, marker_region_id, projection_vertex_count: int,
    ) -> None:
        if int(inactive_axis) not in (0, 1, 2):
            raise ValueError("registered batch requires a 2-D inactive axis")
        if projection_segment_count <= 0 or not 0 < projection_vertex_count <= self.marker_capacity:
            raise ValueError("registered batch requires active finite-segment topology")
        self._validate_registered_field_shapes(
            (cell_face_x_m, cell_face_y_m, cell_face_z_m, cell_center_x_m, cell_center_y_m, cell_center_z_m),
            projection_segment_indices, projection_segment_count,
            (marker_position_m, marker_velocity_mps, marker_region_id), projection_vertex_count,
        )
        self._scan_registered_active_faces_kernel(
            int(inactive_axis), cell_face_x_m, cell_face_y_m, cell_face_z_m,
            cell_center_x_m, cell_center_y_m, cell_center_z_m,
            projection_segment_indices, int(projection_segment_count),
            marker_position_m, marker_velocity_mps, marker_region_id, int(projection_vertex_count),
        )

    def scan_registered_owner_device(
        self,
        *,
        face: tuple[int, int, int],
        component_axis: int,
        inactive_axis: int,
        face_center: tuple[float, float, float],
        projection_segment_indices,
        projection_segment_count: int,
        marker_position_m,
        marker_velocity_mps,
        marker_region_id,
        projection_vertex_count: int,
    ) -> None:
        if int(component_axis) not in (0, 1, 2):
            raise ValueError("component_axis must be 0, 1, or 2")
        if int(inactive_axis) not in (0, 1, 2):
            raise ValueError("inactive_axis must be 0, 1, or 2")
        if len(face) != 3 or len(face_center) != 3:
            raise ValueError("face and face_center must have length three")
        if any(index < 0 or index >= limit for index, limit in zip(face, self.grid_nodes)):
            raise IndexError("face is out of grid range")
        self._scan_registered_owner_kernel(
            int(face[0]), int(face[1]), int(face[2]), int(component_axis),
            int(inactive_axis), float(face_center[0]), float(face_center[1]),
            float(face_center[2]), projection_segment_indices,
            int(projection_segment_count), marker_position_m, marker_velocity_mps,
            marker_region_id, int(projection_vertex_count),
        )

    @ti.func
    def _certificate_edge_is_local_and_finite(
        self,
        edge_index: ti.i32,
        face_center,
        inactive_axis: ti.i32,
        strict_support_radius_m: ti.f64,
        projection_segment_indices: ti.template(),
        marker_position_m: ti.template(),
        projection_vertex_count: ti.i32,
    ):
        edge = projection_segment_indices[edge_index]
        valid = ti.cast(
            edge.x >= 0
            and edge.y >= 0
            and edge.x < projection_vertex_count
            and edge.y < projection_vertex_count
            and edge.x != edge.y
            and edge.z == -1,
            ti.i32,
        )
        if valid != 0:
            valid = (
                registered_endpoint_in_strict_face_support_2d(
                    face_center,
                    marker_position_m[edge.x],
                    inactive_axis,
                    strict_support_radius_m,
                )
                and registered_endpoint_in_strict_face_support_2d(
                    face_center,
                    marker_position_m[edge.y],
                    inactive_axis,
                    strict_support_radius_m,
                )
            )
        return ti.cast(valid, ti.i32)

    @ti.func
    def _certificate_path_has_open_normal_halfplane(
        self,
        projection_segment_normals: ti.template(),
        projection_segment_count: ti.i32,
        inactive_axis: ti.i32,
    ):
        valid = 0
        for first_index in range(projection_segment_count):
            if self.certificate_candidate_path_edge[first_index] != 0:
                first_normal_valid, first_normal = registered_normal_in_active_plane_2d(
                    projection_segment_normals[first_index], inactive_axis
                )
                for second_index in range(projection_segment_count):
                    if (
                        self.certificate_candidate_path_edge[second_index] != 0
                        and first_normal_valid != 0
                    ):
                        second_normal_valid, second_normal = registered_normal_in_active_plane_2d(
                            projection_segment_normals[second_index], inactive_axis
                        )
                        axis = first_normal + second_normal
                        axis_squared = axis.dot(axis)
                        candidate_valid = ti.cast(
                            second_normal_valid != 0 and axis_squared > 1.0e-24,
                            ti.i32,
                        )
                        for test_index in range(projection_segment_count):
                            if self.certificate_candidate_path_edge[test_index] != 0:
                                test_normal_valid, test_normal = registered_normal_in_active_plane_2d(
                                    projection_segment_normals[test_index], inactive_axis
                                )
                                if (
                                    test_normal_valid == 0
                                    or axis.dot(test_normal) <= 1.0e-12
                                ):
                                    candidate_valid = 0
                        if candidate_valid != 0:
                            valid = 1
        return valid

    @ti.func
    def _certificate_explicit_alias_is_valid(
        self, vertex: ti.i32, projection_vertex_count: ti.i32,
        marker_position_m: ti.template(), marker_velocity_mps: ti.template(),
        marker_role: ti.template(),
    ):
        alias = self.explicit_endpoint_alias[vertex]
        valid = 0
        # Taichi boolean expressions need not short-circuit: range checks must
        # precede every alias-indexed field access, including the absent (-1) case.
        if alias >= 0 and alias < projection_vertex_count and alias != vertex:
            valid = ti.cast(
                self.explicit_endpoint_alias[alias] == vertex
                and self.registered_vertex_degree[vertex] == 1
                and self.registered_vertex_degree[alias] == 1
                and marker_role[vertex] >= 0, ti.i32,
            )
            if valid != 0:
                expected_vertex_role = self.explicit_alias_expected_role[vertex]
                expected_alias_role = self.explicit_alias_expected_role[alias]
                if expected_vertex_role >= 0 or expected_alias_role >= 0:
                    valid = ti.cast(marker_role[vertex] == expected_vertex_role
                                    and marker_role[alias] == expected_alias_role, ti.i32)
                else:
                    valid = ti.cast(marker_role[vertex] == marker_role[alias], ti.i32)
            if valid != 0:
                first_position = ti.cast(marker_position_m[vertex], ti.f64)
                second_position = ti.cast(marker_position_m[alias], ti.f64)
                first_velocity = ti.cast(marker_velocity_mps[vertex], ti.f64)
                second_velocity = ti.cast(marker_velocity_mps[alias], ti.f64)
                position_delta = first_position - second_position
                velocity_delta = first_velocity - second_velocity
                valid = ti.cast(
                    self._audit_finite(first_position) != 0 and self._audit_finite(second_position) != 0
                    and self._audit_finite(first_velocity) != 0 and self._audit_finite(second_velocity) != 0
                    and position_delta.dot(position_delta) <= 1.0e-18
                    and velocity_delta.dot(velocity_delta) <= 1.0e-12, ti.i32,
                )
        return valid, alias

    @ti.kernel
    def _certify_registered_local_path_kernel(
        self,
        face_i: ti.i32,
        face_j: ti.i32,
        face_k: ti.i32,
        component_axis: ti.i32,
        inactive_axis: ti.i32,
        source_segment_index: ti.i32,
        owner_segment_index: ti.i32,
        face_x: ti.f32,
        face_y: ti.f32,
        face_z: ti.f32,
        strict_support_radius_m: ti.f64,
        projection_segment_indices: ti.template(),
        projection_segment_normals: ti.template(),
        projection_segment_count: ti.i32,
        marker_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        marker_role: ti.template(),
        projection_vertex_count: ti.i32,
    ):
        face = ti.Vector([face_i, face_j, face_k])
        key = (face_i, face_j, face_k, component_axis)
        face_center = ti.Vector([face_x, face_y, face_z])
        self.certificate_valid[key] = 0
        self.certificate_blocked[key] = 1
        self.certificate_path_edge_count[key] = 0
        for edge_index in range(self.marker_capacity):
            self.certificate_path_edge[edge_index] = 0
            self.certificate_candidate_path_edge[edge_index] = 0
        topology_valid = ti.cast(
            self.registered_topology_ready[None] != 0
            and self.registered_topology_vertex_count[None] == projection_vertex_count
            and self.registered_topology_segment_count[None] == projection_segment_count
            and projection_vertex_count > 0
            and projection_vertex_count <= self.marker_capacity
            and projection_segment_count > 0
            and projection_segment_count <= self.marker_capacity
            and source_segment_index >= 0
            and source_segment_index < projection_segment_count
            and owner_segment_index >= 0
            and owner_segment_index < projection_segment_count,
            ti.i32,
        )
        if topology_valid != 0:
            for vertex in range(projection_vertex_count):
                if self.registered_vertex_degree[vertex] > 2:
                    topology_valid = 0
            for edge_index in range(projection_segment_count):
                edge = projection_segment_indices[edge_index]
                canonical_first = ti.min(edge.x, edge.y)
                canonical_second = ti.max(edge.x, edge.y)
                adjacency_first = self.registered_vertex_adjacency[canonical_first]
                adjacency_second = self.registered_vertex_adjacency[canonical_second]
                if (
                    edge.z != -1
                    or canonical_first < 0
                    or canonical_second >= projection_vertex_count
                    or canonical_first == canonical_second
                    or (adjacency_first.x != edge_index and adjacency_first.y != edge_index)
                    or (adjacency_second.x != edge_index and adjacency_second.y != edge_index)
                ):
                    topology_valid = 0
        accepted = 0
        if topology_valid != 0:
            for start_side in range(2):
                for edge_index in range(self.marker_capacity):
                    self.certificate_candidate_path_edge[edge_index] = 0
                active = ti.cast(accepted == 0, ti.i32)
                reached = ti.cast(source_segment_index == owner_segment_index, ti.i32)
                path_valid = active
                current_edge = source_segment_index
                source_edge = projection_segment_indices[source_segment_index]
                current_vertex = source_edge.x
                if start_side != 0:
                    current_vertex = source_edge.y
                self.certificate_candidate_path_edge[source_segment_index] = active
                if self._certificate_edge_is_local_and_finite(
                    source_segment_index,
                    face_center,
                    inactive_axis,
                    strict_support_radius_m,
                    projection_segment_indices,
                    marker_position_m,
                    projection_vertex_count,
                ) == 0:
                    path_valid = 0
                    active = 0
                for _ in range(self.marker_capacity):
                    if active != 0:
                        neighbors = self.registered_vertex_adjacency[current_vertex]
                        next_edge = -1
                        if neighbors.x >= 0 and neighbors.x != current_edge:
                            next_edge = neighbors.x
                        if neighbors.y >= 0 and neighbors.y != current_edge:
                            if next_edge != -1:
                                path_valid = 0
                            next_edge = neighbors.y
                        if next_edge < 0 or next_edge >= projection_segment_count:
                            alias_valid, alias = self._certificate_explicit_alias_is_valid(
                                current_vertex,
                                projection_vertex_count,
                                marker_position_m,
                                marker_velocity_mps,
                                marker_role,
                            )
                            if alias_valid != 0:
                                alias_neighbors = self.registered_vertex_adjacency[alias]
                                next_edge = alias_neighbors.x
                                if next_edge < 0:
                                    next_edge = alias_neighbors.y
                                current_vertex = alias
                            else:
                                path_valid = 0
                                active = 0
                        elif next_edge == source_segment_index:
                            path_valid = 0
                            active = 0
                        else:
                            self.certificate_candidate_path_edge[next_edge] = 1
                            if self._certificate_edge_is_local_and_finite(
                                next_edge,
                                face_center,
                                inactive_axis,
                                strict_support_radius_m,
                                projection_segment_indices,
                                marker_position_m,
                                projection_vertex_count,
                            ) == 0:
                                path_valid = 0
                                active = 0
                            elif next_edge == owner_segment_index:
                                reached = 1
                                active = 0
                            else:
                                next_segment = projection_segment_indices[next_edge]
                                if next_segment.x == current_vertex:
                                    current_vertex = next_segment.y
                                elif next_segment.y == current_vertex:
                                    current_vertex = next_segment.x
                                else:
                                    path_valid = 0
                                    active = 0
                                current_edge = next_edge
                if (
                    accepted == 0
                    and reached != 0
                    and path_valid != 0
                    and self._certificate_path_has_open_normal_halfplane(
                        projection_segment_normals,
                        projection_segment_count,
                        inactive_axis,
                    ) != 0
                ):
                    for edge_index in range(projection_segment_count):
                        if self.certificate_candidate_path_edge[edge_index] != 0:
                            self.certificate_path_edge[edge_index] = 1
                            ti.atomic_add(self.certificate_path_edge_count[key], 1)
                    accepted = 1
        if accepted != 0:
            self.certificate_valid[key] = 1
            self.certificate_blocked[key] = 0

    def certify_registered_local_path_device(
        self,
        *,
        face: tuple[int, int, int],
        component_axis: int,
        inactive_axis: int,
        source_segment_index: int,
        owner_segment_index: int,
        face_center: tuple[float, float, float],
        strict_support_radius_m: float,
        projection_segment_indices,
        projection_segment_normals,
        projection_segment_count: int,
        marker_position_m,
        marker_velocity_mps,
        marker_role,
        projection_vertex_count: int,
    ) -> None:
        """Certify one source-owner pair by a bounded local path and normal fan."""

        if int(component_axis) not in (0, 1, 2) or int(inactive_axis) not in (0, 1, 2):
            raise ValueError("component_axis and inactive_axis must be 0, 1, or 2")
        if len(face) != 3 or len(face_center) != 3:
            raise ValueError("face and face_center must have length three")
        if any(index < 0 or index >= limit for index, limit in zip(face, self.grid_nodes)):
            raise IndexError("face is out of grid range")
        if not math.isfinite(strict_support_radius_m) or strict_support_radius_m <= 0.0:
            raise ValueError("strict_support_radius_m must be finite and positive")
        self._certify_registered_local_path_kernel(
            int(face[0]), int(face[1]), int(face[2]), int(component_axis),
            int(inactive_axis), int(source_segment_index), int(owner_segment_index),
            float(face_center[0]), float(face_center[1]), float(face_center[2]),
            float(strict_support_radius_m), projection_segment_indices,
            projection_segment_normals, int(projection_segment_count), marker_position_m,
            marker_velocity_mps, marker_role,
            int(projection_vertex_count),
        )
