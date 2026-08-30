"""Live full-source certificates for registered two-dimensional MAC geometry.

Only transaction scratch is written here.  Ownership comes from the independent
nearest-finite-segment pass; a failed certificate never chooses another owner.
"""

import math
import operator

import taichi as ti

from .component_face_segment_geometry import (
    finite_segment_projection_2d,
    registered_normal_in_active_plane_2d,
)


class RegisteredSegmentAudit:
    """Device audit methods used by RegisteredComponentFaceSegmentAssembler."""

    @ti.func
    def _audit_finite(self, value):
        valid = 1
        for axis in ti.static(range(3)):
            if ti.math.isnan(value[axis]) or ti.math.isinf(value[axis]):
                valid = 0
        return valid

    @ti.func
    def _audit_active_delta(self, first, second, inactive_axis: ti.i32):
        delta = ti.cast(first, ti.f64) - ti.cast(second, ti.f64)
        delta[inactive_axis] = 0.0
        return delta

    @ti.func
    def _audit_position_roundoff(self, first, second, inactive_axis: ti.i32):
        # Positions originate in F32 fields.  This bounds coordinate arithmetic,
        # not a physical target mismatch or an admissible geometry displacement.
        scale = ti.cast(0.0, ti.f64)
        for axis in ti.static(range(3)):
            if axis != inactive_axis:
                scale = ti.max(scale, ti.max(ti.abs(first[axis]), ti.abs(second[axis])))
        return ti.max(1.0e-12, 4.0 * 1.1920928955078125e-7 * scale)

    @ti.func
    def _audit_in_support(self, point, center, inactive_axis: ti.i32, anisotropic: ti.i32, radius):
        valid = ti.cast(self._audit_finite(point) != 0 and self._audit_finite(center) != 0, ti.i32)
        delta = self._audit_active_delta(point, center, inactive_axis)
        if anisotropic != 0:
            for axis in ti.static(range(3)):
                if axis != inactive_axis and ti.abs(delta[axis]) >= radius[axis]:
                    valid = 0
        elif delta.dot(delta) >= radius.x * radius.x:
            valid = 0
        return valid

    @ti.func
    def _audit_in_aggregation_support(self, point, center, inactive_axis: ti.i32,
                                      anisotropic: ti.i32, radius):
        """Strict face-global disk containing the original source support.

        The Euclidean radius is the anisotropic box's active-plane diagonal;
        scalar support keeps its original radius. Only owner and connected-path
        geometry use this disk. All raw source checks retain _audit_in_support.
        """
        valid = 0
        if self._audit_finite(point) != 0 and self._audit_finite(center) != 0:
            delta = self._audit_active_delta(point, center, inactive_axis)
            bound_squared = radius.x * radius.x
            if anisotropic != 0:
                bound_squared = ti.cast(0.0, ti.f64)
                for axis in ti.static(range(3)):
                    if axis != inactive_axis:
                        bound_squared += radius[axis] * radius[axis]
            valid = ti.cast(delta.dot(delta) < bound_squared, ti.i32)
        return valid

    @ti.kernel
    def _clear_registered_segment_audit_kernel(self):
        for key in ti.grouped(self.audit_valid):
            self.audit_valid[key] = 0
            self.audit_raw_count[key] = 0
            self.audit_failure[key] = 0
            self.raw_route_audit_failure[key] = 0
            self._audited_owner_valid[key] = 0
            self._audited_owner_failure[key] = 0
        self.audit_rejection_count[None] = 0
        self.audit_first_rejected_source_key[None] = self.source_axis_record_capacity
        self.audit_first_rejected_face_key[None] = self.source_axis_record_capacity

    @ti.kernel
    def _prepare_registered_audit_geometry_kernel(
        self, inactive_axis: ti.i32, segments: ti.template(), segment_count: ti.i32,
        positions: ti.template(), velocities: ti.template(), normals: ti.template(),
        regions: ti.template(), roles: ti.template(), vertex_count: ti.i32,
    ):
        # Compute each live segment normal once, not once per source/path edge.
        for edge_index in range(self.marker_capacity):
            valid = 0
            normal = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            if edge_index < segment_count:
                edge = segments[edge_index]
                valid = ti.cast(
                    edge.x >= 0 and edge.y >= 0 and edge.x < vertex_count
                    and edge.y < vertex_count and edge.x != edge.y and edge.z == -1,
                    ti.i32,
                )
                if valid != 0:
                    chord = self._audit_active_delta(positions[edge.y], positions[edge.x], inactive_axis)
                    if inactive_axis == 0:
                        normal = ti.Vector([0.0, -chord.z, chord.y], dt=ti.f64)
                    elif inactive_axis == 1:
                        normal = ti.Vector([chord.z, 0.0, -chord.x], dt=ti.f64)
                    else:
                        normal = ti.Vector([-chord.y, chord.x, 0.0], dt=ti.f64)
                    first_ok, first = registered_normal_in_active_plane_2d(normals[edge.x], inactive_axis)
                    second_ok, second = registered_normal_in_active_plane_2d(normals[edge.y], inactive_axis)
                    length_squared = normal.dot(normal)
                    valid = ti.cast(
                        self._audit_finite(positions[edge.x]) != 0
                        and self._audit_finite(positions[edge.y]) != 0
                        and self._audit_finite(velocities[edge.x]) != 0
                        and self._audit_finite(velocities[edge.y]) != 0
                        and self._audit_finite(normals[edge.x]) != 0
                        and self._audit_finite(normals[edge.y]) != 0
                        and first_ok != 0 and second_ok != 0 and length_squared > 1.0e-24
                        and regions[edge.x] >= 0 and regions[edge.x] == regions[edge.y],
                        ti.i32,
                    )
                    if valid != 0:
                        normal /= ti.sqrt(length_squared)
                        alignment_first = normal.dot(first)
                        alignment_second = normal.dot(second)
                        if (ti.abs(alignment_first) <= 1.0e-6 or ti.abs(alignment_second) <= 1.0e-6
                                or alignment_first * alignment_second <= 0.0):
                            valid = 0
                        elif alignment_first + alignment_second < 0.0:
                            normal = -normal
                        for endpoint_side in ti.static(range(2)):
                            vertex = edge[endpoint_side]
                            degree = self.registered_vertex_degree[vertex]
                            adjacent = self.registered_vertex_adjacency[vertex]
                            if (degree < 1 or degree > 2
                                    or (adjacent.x != edge_index and adjacent.y != edge_index)):
                                valid = 0
            self._registered_edge_valid[edge_index] = valid
            self._registered_edge_normal[edge_index] = normal
        for vertex in range(self.marker_capacity):
            valid = 0
            if vertex < vertex_count:
                valid, _ = self._certificate_explicit_alias_is_valid(
                    vertex, vertex_count, positions, velocities, roles,
                )
            self._registered_alias_valid[vertex] = valid

    @ti.func
    def _audit_next_edge(self, vertex: ti.i32, current_edge: ti.i32):
        """Return the unique adjacent edge, crossing only a proved explicit alias."""
        next_vertex = vertex
        next_edge = -1
        degree = self.registered_vertex_degree[vertex]
        adjacent = self.registered_vertex_adjacency[vertex]
        if degree == 2:
            if adjacent.x == current_edge:
                next_edge = adjacent.y
            elif adjacent.y == current_edge:
                next_edge = adjacent.x
        elif degree == 1 and self._registered_alias_valid[vertex] != 0:
            next_vertex = self.explicit_endpoint_alias[vertex]
            alias_adjacent = self.registered_vertex_adjacency[next_vertex]
            next_edge = alias_adjacent.x
        return next_vertex, next_edge

    @ti.func
    def _audit_owner_corner(self, owner_edge: ti.i32, vertex: ti.i32, tied: ti.i32,
                            point, center, inactive_axis: ti.i32, anisotropic: ti.i32,
                            radius, segments: ti.template(), positions: ti.template(), segment_count: ti.i32):
        offset = self._audit_active_delta(center, point, inactive_axis)
        tolerance = self._audit_position_roundoff(center, point, inactive_axis)
        first_normal = self._registered_edge_normal[owner_edge]
        valid = 0
        other_vertex, other_edge = self._audit_next_edge(vertex, owner_edge)
        if other_edge >= 0 and other_edge < segment_count and other_edge != owner_edge:
            if (self._registered_edge_valid[other_edge] != 0
                    and self._audit_in_aggregation_support(positions[vertex], center, inactive_axis, anisotropic, radius) != 0
                    and self._audit_in_aggregation_support(positions[other_vertex], center, inactive_axis, anisotropic, radius) != 0):
                first_segment = segments[owner_edge]
                second_segment = segments[other_edge]
                first_other = first_segment.x
                if first_other == vertex:
                    first_other = first_segment.y
                second_other = second_segment.x
                incident = ti.cast(second_segment.x == other_vertex or second_segment.y == other_vertex, ti.i32)
                if second_other == other_vertex:
                    second_other = second_segment.y
                first_chord = self._audit_active_delta(positions[first_other], positions[vertex], inactive_axis)
                second_chord = self._audit_active_delta(positions[second_other], positions[other_vertex], inactive_axis)
                first_chord /= ti.sqrt(first_chord.dot(first_chord))
                second_chord /= ti.sqrt(second_chord.dot(second_chord))
                second_normal = self._registered_edge_normal[other_edge]
                # Endpoint Voronoi conditions plus outward convexity.  A cone
                # combination is used; positive dot with every normal is wrong
                # for an obtuse convex corner.
                valid = ti.cast(
                    incident != 0 and offset.dot(first_chord) <= tolerance
                    and offset.dot(second_chord) <= tolerance
                    and first_chord.dot(second_normal) <= 1.0e-6
                    and second_chord.dot(first_normal) <= 1.0e-6,
                    ti.i32,
                )
                determinant = first_normal.cross(second_normal)[inactive_axis]
                if ti.abs(determinant) > 1.0e-12:
                    first_weight = offset.cross(second_normal)[inactive_axis] / determinant
                    second_weight = first_normal.cross(offset)[inactive_axis] / determinant
                    if first_weight < -tolerance or second_weight < -tolerance:
                        valid = 0
                else:
                    lateral = offset - offset.dot(first_normal) * first_normal
                    if (first_normal.dot(second_normal) <= 0.0
                            or offset.dot(first_normal) < -tolerance
                            or lateral.dot(lateral) > tolerance * tolerance):
                        valid = 0
        elif (self.registered_vertex_degree[vertex] == 1
                and self.explicit_endpoint_alias[vertex] < 0 and tied == 0):
            # An actual open finite endpoint has one outward half-space.  An
            # invalid declared seam must not silently become an open endpoint.
            valid = ti.cast(offset.dot(first_normal) >= -tolerance, ti.i32)
        return valid

    @ti.func
    def _audit_owner(self, key, center, inactive_axis: ti.i32, anisotropic: ti.i32,
                     radius, segments: ti.template(), segment_count: ti.i32,
                     positions: ti.template(), velocities: ti.template(), regions: ti.template()):
        failure = 1
        edge_index = self.owner_segment_index[key]
        if (self.owner_valid[key] == 1 and self.owner_ambiguous[key] == 0
                and self.owner_blocked[key] == 0 and edge_index >= 0 and edge_index < segment_count
                and self._audit_finite(center) != 0):
            if self._registered_edge_valid[edge_index] != 0:
                edge = segments[edge_index]
                first, second = ti.min(edge.x, edge.y), ti.max(edge.x, edge.y)
                projected, weight, point, _ = finite_segment_projection_2d(
                    center, positions[first], positions[second], inactive_axis,
                )
                target = (1.0 - weight) * velocities[first][key.w] + weight * velocities[second][key.w]
                vertex = -1
                if weight == 0.0:
                    vertex = first
                elif weight == 1.0:
                    vertex = second
                stored_point = self.owner_point_m[key]
                metadata_ok = ti.cast(
                    projected != 0 and self.owner_segment[key].x == first
                    and self.owner_segment[key].y == second
                    and self.owner_region[key] == regions[first]
                    and self.owner_weight[key] == ti.cast(weight, ti.f32)
                    and self.owner_target_mps[key] == ti.cast(target, ti.f32)
                    and self.owner_vertex[key] == vertex
                    and self._audit_finite(stored_point) != 0,
                    ti.i32,
                )
                for axis in ti.static(range(3)):
                    if stored_point[axis] != ti.cast(point[axis], ti.f32):
                        metadata_ok = 0
                if metadata_ok != 0:
                    failure = 2
                    # The owner is the finite projection already selected in B.
                    # Keep that global nearest point; bound it without requiring
                    # the anisotropic source box to be closed under projection.
                    if self._audit_in_aggregation_support(point, center, inactive_axis, anisotropic, radius) != 0:
                        failure = 3
                        side_ok = 0
                        if vertex >= 0:
                            side_ok = self._audit_owner_corner(
                                edge_index, vertex, self.owner_vertex_tie[key], point, center,
                                inactive_axis, anisotropic, radius, segments, positions, segment_count,
                            )
                        elif self.owner_vertex_tie[key] == 0:
                            offset = self._audit_active_delta(center, point, inactive_axis)
                            tolerance = self._audit_position_roundoff(center, point, inactive_axis)
                            side_ok = ti.cast(offset.dot(self._registered_edge_normal[edge_index]) >= -tolerance, ti.i32)
                        if side_ok != 0:
                            failure = 0
        return failure

    @ti.kernel
    def _certify_registered_owners_kernel(
        self, inactive_axis: ti.i32, anisotropic: ti.i32, radius: ti.types.vector(3, ti.f64),
        face_x: ti.template(), face_y: ti.template(), face_z: ti.template(),
        center_x: ti.template(), center_y: ti.template(), center_z: ti.template(),
        segments: ti.template(), segment_count: ti.i32, positions: ti.template(),
        velocities: ti.template(), regions: ti.template(),
    ):
        for i, j, k, axis in ti.ndrange(self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2], 3):
            if self.face_raw_count[i, j, k][axis] > 0:
                center = ti.Vector([center_x[i], center_y[j], center_z[k]], dt=ti.f64)
                if axis == 0:
                    center.x = face_x[i]
                elif axis == 1:
                    center.y = face_y[j]
                else:
                    center.z = face_z[k]
                key = ti.Vector([i, j, k, axis])
                failure = self._audit_owner(key, center, inactive_axis, anisotropic, radius,
                                            segments, segment_count, positions, velocities, regions)
                if axis == inactive_axis:
                    failure = 1
                self._audited_owner_valid[key] = ti.cast(failure == 0, ti.i32)
                self._audited_owner_failure[key] = failure

    @ti.func
    def _audit_path(self, source_edge: ti.i32, owner_edge: ti.i32, center,
                    inactive_axis: ti.i32, anisotropic: ti.i32, radius,
                    segments: ti.template(), positions: ti.template(), segment_count: ti.i32):
        accepted = 0
        if source_edge == owner_edge:
            accepted = 1
        else:
            first_normal = self._registered_edge_normal[source_edge]
            source_segment = segments[source_edge]
            for start_side in range(2):
                current_edge = source_edge
                vertex = source_segment[start_side]
                minimum_angle = ti.cast(0.0, ti.f64)
                maximum_angle = ti.cast(0.0, ti.f64)
                active = 1
                edge_visits = 0
                while active != 0 and edge_visits < segment_count:
                    edge_visits += 1
                    next_vertex, next_edge = self._audit_next_edge(vertex, current_edge)
                    active = ti.cast(next_edge >= 0 and next_edge < segment_count and next_edge != source_edge, ti.i32)
                    if active != 0:
                        # Every actual connector, including both sides of an
                        # explicit alias, is local.  Together with the source
                        # anchor and owner projection this proves a connected
                        # path inside the same strict face-global disk.
                        active = ti.cast(
                            self._registered_edge_valid[next_edge] != 0
                            and self._audit_in_aggregation_support(positions[vertex], center, inactive_axis, anisotropic, radius) != 0
                            and self._audit_in_aggregation_support(positions[next_vertex], center, inactive_axis, anisotropic, radius) != 0,
                            ti.i32,
                        )
                    if active != 0:
                        next_segment = segments[next_edge]
                        if next_segment.x != next_vertex and next_segment.y != next_vertex:
                            active = 0
                    if active != 0:
                        normal = self._registered_edge_normal[next_edge]
                        angle = ti.atan2(first_normal.cross(normal)[inactive_axis], first_normal.dot(normal))
                        minimum_angle = ti.min(minimum_angle, angle)
                        maximum_angle = ti.max(maximum_angle, angle)
                        if maximum_angle - minimum_angle >= 3.141591653589793:
                            active = 0
                        elif next_edge == owner_edge:
                            accepted += 1
                            active = 0
                        else:
                            vertex = segments[next_edge].x
                            if vertex == next_vertex:
                                vertex = segments[next_edge].y
                            current_edge = next_edge
        return ti.cast(accepted == 1, ti.i32)

    @ti.func
    def _audit_ray(self, key, anchor, geometric_normal, inactive_axis: ti.i32):
        nominal = self.raw_route_nominal_sample_m[key]
        actual = self.raw_route_actual_sample_m[key]
        raw_normal = self.raw_route_normal[key]
        valid = ti.cast(self._audit_finite(nominal) != 0 and self._audit_finite(actual) != 0
                        and self._audit_finite(raw_normal) != 0, ti.i32)
        if valid != 0:
            nominal_ray = self._audit_active_delta(nominal, anchor, inactive_axis)
            actual_ray = self._audit_active_delta(actual, anchor, inactive_axis)
            normal_ok, normal = registered_normal_in_active_plane_2d(raw_normal, inactive_axis)
            nominal_progress = nominal_ray.dot(geometric_normal)
            actual_progress = actual_ray.dot(geometric_normal)
            nominal_lateral = nominal_ray - nominal_progress * geometric_normal
            actual_lateral = actual_ray - actual_progress * geometric_normal
            nominal_tolerance = self._audit_position_roundoff(nominal, anchor, inactive_axis)
            actual_tolerance = self._audit_position_roundoff(actual, anchor, inactive_axis)
            valid = ti.cast(normal_ok != 0 and nominal_progress > 1.0e-12 and actual_progress > 1.0e-12
                            and nominal_lateral.dot(nominal_lateral) <= nominal_tolerance * nominal_tolerance
                            and actual_lateral.dot(actual_lateral) <= actual_tolerance * actual_tolerance, ti.i32)
            if valid != 0:
                normal_error = normal - nominal_ray / ti.sqrt(nominal_ray.dot(nominal_ray))
                if normal_error.dot(normal_error) > 4.0e-12:
                    valid = 0
            if self.raw_route_kind[key] == 0:
                difference = self._audit_active_delta(nominal, actual, inactive_axis)
                if difference.dot(difference) > nominal_tolerance * nominal_tolerance:
                    valid = 0
        return valid

    @ti.func
    def _audit_raw_record(self, key, center, source_center, inactive_axis: ti.i32,
                          anisotropic: ti.i32, radius, generation: ti.i32,
                          segments: ti.template(), segment_count: ti.i32,
                          positions: ti.template(), velocities: ti.template(), regions: ti.template(), vertex_count: ti.i32):
        failure = 1
        primitive = self.raw_route_primitive[key]
        weights = self.raw_route_weights[key]
        anchor = self.raw_route_anchor_m[key]
        if (self.raw_route_valid[key] == 1 and self.raw_route_kind[key] >= 0 and self.raw_route_kind[key] <= 1
                and self.raw_route_sample_valid[key] == 1 and self.raw_route_generation[key] == generation
                and key.w != inactive_axis):
            failure = 2
            if (primitive.x >= 0 and primitive.y >= 0 and primitive.x < vertex_count
                    and primitive.y < vertex_count and primitive.x != primitive.y and primitive.z == -1):
                source_edge = -1
                matches = 0
                for edge_index in range(segment_count):
                    edge = segments[edge_index]
                    if ((edge.x == primitive.x and edge.y == primitive.y)
                            or (edge.x == primitive.y and edge.y == primitive.x)):
                        source_edge = edge_index
                        matches += 1
                if matches == 1:
                    failure = 3
                    # Match the original F32 interpolation and its existing
                    # two-ULP coordinate tolerance; velocity tolerance is unchanged.
                    expected_anchor = weights.x * positions[primitive.x] + weights.y * positions[primitive.y]
                    residual = self._audit_active_delta(anchor, expected_anchor, inactive_axis)
                    tolerance = 0.5 * self._audit_position_roundoff(anchor, expected_anchor, inactive_axis)
                    tolerance = ti.max(tolerance, 1.0e-12)
                    target = weights.x * velocities[primitive.x][key.w] + weights.y * velocities[primitive.y][key.w]
                    if (self._audit_finite(weights) != 0 and weights.x >= -1.0e-6 and weights.y >= -1.0e-6
                            and ti.abs(weights.z) <= 1.0e-6 and ti.abs(weights.x + weights.y - 1.0) <= 1.0e-6
                            and self._audit_finite(anchor) != 0 and self._audit_finite(expected_anchor) != 0
                            and residual.dot(residual) <= tolerance * tolerance
                            and self._registered_edge_valid[source_edge] != 0
                            and regions[primitive.x] == self.raw_route_region[key]
                            and ti.abs(target - self.raw_route_boundary_target_mps[key]) <= 1.0e-6):
                        failure = 4
                        if (self._audit_in_support(anchor, source_center, inactive_axis, anisotropic, radius) != 0
                                and self._audit_in_support(source_center, center, inactive_axis, anisotropic, radius) != 0
                                and self._audit_in_support(anchor, center, inactive_axis, anisotropic, radius) != 0):
                            failure = 5
                            if self._audit_ray(key, anchor, self._registered_edge_normal[source_edge], inactive_axis) != 0:
                                destination = self.raw_route_target[key]
                                owner_key = ti.Vector([destination.x, destination.y, destination.z, key.w])
                                failure = 6
                                if self._audited_owner_valid[owner_key] != 0:
                                    failure = 7
                                    owner_index = self.owner_segment_index[owner_key]
                                    if self._audit_path(
                                        source_edge, owner_index, center, inactive_axis, anisotropic,
                                        radius, segments, positions, segment_count,
                                    ) != 0:
                                        failure = 0
        return failure

    @ti.kernel
    def _certify_active_raw_routes_kernel(
        self, inactive_axis: ti.i32, anisotropic: ti.i32, radius: ti.types.vector(3, ti.f64), generation: ti.i32,
        face_x: ti.template(), face_y: ti.template(), face_z: ti.template(),
        center_x: ti.template(), center_y: ti.template(), center_z: ti.template(),
        segments: ti.template(), segment_count: ti.i32, positions: ti.template(),
        velocities: ti.template(), regions: ti.template(), vertex_count: ti.i32,
    ):
        for i, j, k, axis in ti.ndrange(self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2], 3):
            key = ti.Vector([i, j, k, axis])
            if self.raw_route_valid[key] != 0:
                destination = self.raw_route_target[key]
                failure = 8
                if (destination.x >= 0 and destination.y >= 0 and destination.z >= 0
                        and destination.x < self.grid_nodes[0] and destination.y < self.grid_nodes[1]
                        and destination.z < self.grid_nodes[2]):
                    center = ti.Vector([center_x[destination.x], center_y[destination.y], center_z[destination.z]], dt=ti.f64)
                    if axis == 0:
                        center.x = face_x[destination.x]
                    elif axis == 1:
                        center.y = face_y[destination.y]
                    else:
                        center.z = face_z[destination.z]
                    source_center = ti.Vector([center_x[i], center_y[j], center_z[k]], dt=ti.f64)
                    failure = self._audit_raw_record(key, center, source_center, inactive_axis, anisotropic, radius,
                                                     generation, segments, segment_count, positions, velocities, regions, vertex_count)
                    ti.atomic_add(self.audit_raw_count[destination, axis], 1)
                    if failure != 0:
                        ti.atomic_max(self.audit_failure[destination, axis], failure)
                self.raw_route_audit_failure[key] = failure
                if failure != 0:
                    ti.atomic_add(self.audit_rejection_count[None], 1)
                    encoded = (((i * self.grid_nodes[1]) + j) * self.grid_nodes[2] + k) * 3 + axis
                    ti.atomic_min(self.audit_first_rejected_source_key[None], encoded)

    @ti.kernel
    def _finalize_registered_segment_audit_kernel(self):
        for i, j, k, axis in ti.ndrange(self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2], 3):
            expected_count = self.face_raw_count[i, j, k][axis]
            actual_count = self.audit_raw_count[i, j, k, axis]
            if expected_count != 0 or actual_count != 0:
                if (expected_count > 0 and actual_count == expected_count
                        and self.audit_failure[i, j, k, axis] == 0
                        and self._audited_owner_valid[i, j, k, axis] == 1):
                    self.audit_valid[i, j, k, axis] = 1
                else:
                    ti.atomic_max(self.audit_failure[i, j, k, axis], 9)
                    ti.atomic_add(self.audit_rejection_count[None], 1)
                    encoded = (((i * self.grid_nodes[1]) + j) * self.grid_nodes[2] + k) * 3 + axis
                    ti.atomic_min(self.audit_first_rejected_face_key[None], encoded)

    def _validate_registered_field_shapes(self, coordinates, segments, segment_count, marker_fields, vertex_count):
        for name, value in (("projection_segment_count", segment_count), ("projection_vertex_count", vertex_count)):
            if isinstance(value, bool) or not 0 < operator.index(value) <= self.marker_capacity:
                raise ValueError(f"{name} exceeds registered assembler capacity")
        if len(segments.shape) != 1 or segments.shape[0] < segment_count:
            raise ValueError("registered segment field has insufficient capacity")
        for field in marker_fields:
            if len(field.shape) != 1 or field.shape[0] < vertex_count:
                raise ValueError("registered marker field has insufficient capacity")
        for index, coordinate in enumerate(coordinates):
            required = self.grid_nodes[index % 3] + (1 if index < 3 else 0)
            if tuple(coordinate.shape) != (required,):
                raise ValueError("registered coordinate field shape does not match MAC grid")

    def certify_active_raw_routes_device(
        self, *, inactive_axis: int, support_available: int, support_anisotropic: int,
        strict_support_radius_xyz_m: tuple[float, float, float], expected_generation: int,
        cell_face_x_m, cell_face_y_m, cell_face_z_m, cell_center_x_m, cell_center_y_m, cell_center_z_m,
        projection_segment_indices, projection_segment_count: int, marker_position_m,
        marker_velocity_mps, marker_normal_m, marker_region_id, marker_role, projection_vertex_count: int,
    ) -> None:
        """Certify every live raw route, with no canonical-ledger writes."""
        if inactive_axis not in (0, 1, 2) or support_available != 1 or support_anisotropic not in (0, 1):
            raise ValueError("registered audit requires declared 2-D geometry and strict source support")
        radius = tuple(float(value) for value in strict_support_radius_xyz_m)
        if len(radius) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in radius):
            raise ValueError("strict_support_radius_xyz_m must contain three finite positive radii")
        if isinstance(expected_generation, bool) or not 0 < operator.index(expected_generation) <= 2147483647:
            raise ValueError("registered audit generation must be a positive signed 32-bit integer")
        coordinates = (cell_face_x_m, cell_face_y_m, cell_face_z_m, cell_center_x_m, cell_center_y_m, cell_center_z_m)
        self._validate_registered_field_shapes(
            coordinates, projection_segment_indices, projection_segment_count,
            (marker_position_m, marker_velocity_mps, marker_normal_m, marker_region_id, marker_role), projection_vertex_count,
        )
        if (self.registered_topology_ready[None] != 1
                or self.registered_topology_vertex_count[None] != projection_vertex_count
                or self.registered_topology_segment_count[None] != projection_segment_count):
            raise ValueError("registered audit topology does not match active geometry")
        self._clear_registered_segment_audit_kernel()
        self._prepare_registered_audit_geometry_kernel(
            inactive_axis, projection_segment_indices, projection_segment_count, marker_position_m,
            marker_velocity_mps, marker_normal_m, marker_region_id, marker_role, projection_vertex_count,
        )
        self._certify_registered_owners_kernel(
            inactive_axis, support_anisotropic, radius, *coordinates, projection_segment_indices,
            projection_segment_count, marker_position_m, marker_velocity_mps, marker_region_id,
        )
        self._certify_active_raw_routes_kernel(
            inactive_axis, support_anisotropic, radius, expected_generation, *coordinates, projection_segment_indices,
            projection_segment_count, marker_position_m, marker_velocity_mps, marker_region_id, projection_vertex_count,
        )
        self._finalize_registered_segment_audit_kernel()

    def audit_rejection_detail(self) -> dict[str, object]:
        """Read one bounded failure record; no successful-path geometry download."""
        def decode(encoded):
            node, axis = divmod(int(encoded), 3)
            plane, k = divmod(node, self.grid_nodes[2])
            i, j = divmod(plane, self.grid_nodes[1])
            return i, j, k, axis

        detail: dict[str, object] = {}
        encoded = int(self.audit_first_rejected_source_key[None])
        if encoded < self.source_axis_record_capacity:
            key = decode(encoded)
            detail.update(source=key, reason=int(self.raw_route_audit_failure[key]),
                          target=tuple(int(value) for value in self.raw_route_target[key]),
                          primitive=tuple(int(value) for value in self.raw_route_primitive[key]))
        encoded = int(self.audit_first_rejected_face_key[None])
        if encoded < self.source_axis_record_capacity:
            key = decode(encoded)
            detail.update(face=key, owner_reason=int(self._audited_owner_failure[key]),
                          owner=tuple(int(value) for value in self.owner_segment[key]),
                          raw_count=int(self.face_raw_count[key[:3]][key[3]]))
        return detail
