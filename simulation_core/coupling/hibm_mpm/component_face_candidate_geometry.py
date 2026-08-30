"""Preselectable MAC-face geometry using the same final-owner certificate.

Requests are geometry scratch, never raw authors. Every actual selected route
still goes through the original full-source audit before canonical publication.
"""

import math

import taichi as ti


class RegisteredCandidateGeometry:
    @ti.kernel
    def _scan_and_certify_candidate_faces_kernel(
        self, inactive_axis: ti.i32, anisotropic: ti.i32, radius: ti.types.vector(3, ti.f64),
        fx: ti.template(), fy: ti.template(), fz: ti.template(),
        cx: ti.template(), cy: ti.template(), cz: ti.template(),
        segments: ti.template(), segment_count: ti.i32,
        positions: ti.template(), velocities: ti.template(), regions: ti.template(), vertex_count: ti.i32,
    ):
        for i, j, k, axis in ti.ndrange(self.grid_nodes[0], self.grid_nodes[1], self.grid_nodes[2], 3):
            key = ti.Vector([i, j, k, axis])
            self.candidate_owner_permission[key] = 0
            self.candidate_owner_failure[key] = 0
            if self.candidate_face_requested[i, j, k][axis] != 0:
                point = ti.Vector([cx[i], cy[j], cz[k]])
                if axis == 0:
                    point.x = fx[i]
                elif axis == 1:
                    point.y = fy[j]
                else:
                    point.z = fz[k]
                self._scan_registered_owner_device(
                    i, j, k, axis, inactive_axis, point.x, point.y, point.z,
                    segments, segment_count, positions, velocities, regions, vertex_count,
                )
                failure = self._audit_owner(
                    key, point.cast(ti.f64), inactive_axis, anisotropic, radius,
                    segments, segment_count, positions, velocities, regions,
                )
                if axis == inactive_axis:
                    failure = 1
                self.candidate_owner_permission[key] = ti.cast(failure == 0, ti.i32)
                self.candidate_owner_failure[key] = failure

    def prepare_candidate_owner_geometry(
        self, *, inactive_axis, support_available, support_anisotropic, strict_support_radius_xyz_m,
        cell_face_x_m, cell_face_y_m, cell_face_z_m, cell_center_x_m, cell_center_y_m, cell_center_z_m,
        projection_segment_indices, projection_segment_count, marker_position_m, marker_velocity_mps,
        marker_normal_m, marker_region_id, marker_role, projection_vertex_count,
    ):
        if inactive_axis not in (0, 1, 2) or support_available != 1 or support_anisotropic not in (0, 1):
            raise ValueError("candidate owner geometry requires declared 2-D support")
        radius = tuple(float(value) for value in strict_support_radius_xyz_m)
        if len(radius) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in radius):
            raise ValueError("candidate owner geometry requires finite positive radii")
        coordinates = (cell_face_x_m, cell_face_y_m, cell_face_z_m,
                       cell_center_x_m, cell_center_y_m, cell_center_z_m)
        self._validate_registered_field_shapes(
            coordinates, projection_segment_indices, projection_segment_count,
            (marker_position_m, marker_velocity_mps, marker_normal_m, marker_region_id, marker_role),
            projection_vertex_count,
        )
        if (self.registered_topology_ready[None] != 1
                or self.registered_topology_vertex_count[None] != projection_vertex_count
                or self.registered_topology_segment_count[None] != projection_segment_count):
            raise ValueError("candidate owner topology does not match active geometry")
        self._prepare_registered_audit_geometry_kernel(
            inactive_axis, projection_segment_indices, projection_segment_count,
            marker_position_m, marker_velocity_mps, marker_normal_m, marker_region_id,
            marker_role, projection_vertex_count,
        )
        self._scan_and_certify_candidate_faces_kernel(
            inactive_axis, support_anisotropic, radius, *coordinates,
            projection_segment_indices, projection_segment_count, marker_position_m,
            marker_velocity_mps, marker_region_id, projection_vertex_count,
        )
