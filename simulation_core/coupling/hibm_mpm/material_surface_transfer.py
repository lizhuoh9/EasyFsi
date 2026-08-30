"""Device-resident material interpolation and its work-conjugate load map.

The fixed reference stencil is shared by position, velocity and force transfer.
Geometry normals follow oriented material edges, not separately advected normals.
IQN trial wall velocity remains a separate algebraic unknown.
"""

import hashlib
import math

import numpy as np
import taichi as ti

from .material_surface_binding import MaterialSurfaceBinding, interpolate_material_surface
from .reports import HibmMpmMpmForceScatterReport, HibmMpmSurfaceUpdateReport


_EPS32 = float(np.finfo(np.float32).eps)
_EPS64 = float(np.finfo(np.float64).eps)
_SOURCE_FORCE = 0
_APPLIED_FORCE = 3
_SOURCE_TORQUE = 6
_APPLIED_TORQUE = 9
_SOURCE_POWER = 12
_APPLIED_POWER = 13
_FORCE_SCALE = 14
_TORQUE_SCALE = 15
_POWER_SCALE = 16
_INVALID_FORCE = 17
_INVALID_GEOMETRY = 18
_MAX_APPLIED = 19


def _oriented_incident_edges(markers, binding, inactive_axis):
    count = binding.marker_count
    edges = np.zeros((count, 2, 2), dtype=np.int32)
    signs = np.zeros((count, 2), dtype=np.float64)
    degree = np.zeros(count, dtype=np.int32)
    normal = markers.n_gamma.to_numpy()[:count].astype(np.float64)
    axis_vector = np.eye(3)[inactive_axis]
    topology = markers.projection_triangle_indices.to_numpy()[:markers.projection_segment_count]
    for first, second, _ in topology:
        first, second = int(first), int(second)
        if first >= count or second >= count:
            continue  # Cap aliases are derived separately from physical rows.
        tangent = binding.reference_marker_positions_m[second] - binding.reference_marker_positions_m[first]
        candidate = np.cross(axis_vector, tangent)
        alignment = float(candidate @ (normal[first] + normal[second]))
        if not np.isfinite(alignment) or alignment == 0.0:
            raise ValueError("material reference edge has no consistent outward orientation")
        for marker in (first, second):
            if degree[marker] >= 2:
                raise ValueError("material surface has a non-manifold physical edge")
            slot = degree[marker]
            edges[marker, slot] = (first, second)
            signs[marker, slot] = math.copysign(1.0, alignment)
            degree[marker] += 1
    if np.any(degree == 0):
        raise ValueError("material surface requires incident physical projection segments")
    return edges, signs, degree


@ti.data_oriented
class MaterialSurfaceTransfer:
    """Fixed stencils and derived scratch, never a second physical state."""

    def __init__(self, markers, binding: MaterialSurfaceBinding, inactive_axis: int):
        if type(inactive_axis) is not int or inactive_axis not in (0, 1, 2):
            raise ValueError("inactive_axis must be 0, 1 or 2")
        if markers.projection_triangle_count != 0 or markers.marker_count != binding.marker_count:
            raise ValueError("material surface transfer requires the matching segment layout")
        self.binding = binding
        self.markers = markers
        self.marker_count = binding.marker_count
        self.particle_count = binding.particle_count
        self.inactive_axis = inactive_axis
        edges, signs, degree = _oriented_incident_edges(markers, binding, inactive_axis)
        self._host_edges = edges.copy()
        self._host_signs = signs.copy()
        self._host_degree = degree.copy()
        self._reference_area = markers.A_gamma_m2.to_numpy()[:self.marker_count].copy()
        self._projection_vertex_count = int(markers.projection_vertex_count)
        self._projection_segment_count = int(markers.projection_segment_count)
        self.indices = ti.field(ti.i32, shape=(self.marker_count, 8))
        self.weights = ti.field(ti.f64, shape=(self.marker_count, 8))
        self.indices.from_numpy(binding.particle_indices)
        self.weights.from_numpy(binding.weights)
        self.edges = ti.field(ti.i32, shape=(self.marker_count, 2, 2))
        self.edge_sign = ti.field(ti.f64, shape=(self.marker_count, 2))
        self.degree = ti.field(ti.i32, shape=self.marker_count)
        self.edges.from_numpy(edges)
        self.edge_sign.from_numpy(signs)
        self.degree.from_numpy(degree)
        normal = markers.n_gamma.to_numpy()[:self.marker_count].astype(np.float64)
        probe = markers.pressure_probe_origin_m.to_numpy()[:self.marker_count].astype(np.float64)
        probe_offsets = np.sum((probe - binding.reference_marker_positions_m) * normal, axis=1)
        self._host_probe_offsets = probe_offsets.copy()
        self.probe_offset = ti.field(ti.f64, shape=self.marker_count)
        self.probe_offset.from_numpy(probe_offsets)

        # Store W.T as deterministic per-particle CSR: no atomic force scatter.
        columns = [[] for _ in range(self.particle_count)]
        for marker in range(self.marker_count):
            for particle, weight in zip(binding.particle_indices[marker], binding.weights[marker]):
                if weight != 0.0:
                    columns[int(particle)].append((marker, float(weight)))
        offsets = np.cumsum([0] + [len(column) for column in columns], dtype=np.int32)
        entries = [entry for column in columns for entry in column]
        self.pair_count = len(entries)
        self.column_offsets = ti.field(ti.i32, shape=self.particle_count + 1)
        self.column_marker = ti.field(ti.i32, shape=max(1, self.pair_count))
        self.column_weight = ti.field(ti.f64, shape=max(1, self.pair_count))
        self.column_offsets.from_numpy(offsets)
        if entries:
            self.column_marker.from_numpy(np.asarray([entry[0] for entry in entries], dtype=np.int32))
            self.column_weight.from_numpy(np.asarray([entry[1] for entry in entries], dtype=np.float64))

        self.cap_binding = getattr(markers, "_open_ribbon_tip_cap_binding", None)
        cap_owner = np.full(self.marker_count, -1, dtype=np.int32)
        cap_factor = np.zeros(self.marker_count, dtype=np.float64)
        if self.cap_binding is not None:
            for previous, tip, cap in ((0, 1, 6), (2, 3, 7)):
                cap_owner[int(self.cap_binding[previous])] = int(self.cap_binding[cap])
                cap_owner[int(self.cap_binding[tip])] = int(self.cap_binding[cap])
                cap_factor[int(self.cap_binding[previous])] = -0.5
                cap_factor[int(self.cap_binding[tip])] = 1.5
        self.cap_owner = ti.field(ti.i32, shape=self.marker_count)
        self.cap_factor = ti.field(ti.f64, shape=self.marker_count)
        self.cap_owner.from_numpy(cap_owner)
        self.cap_factor.from_numpy(cap_factor)
        self.effective_force = ti.Vector.field(3, ti.f64, shape=self.marker_count)
        self.material_velocity = ti.Vector.field(3, ti.f64, shape=self.marker_count)
        self.pending_external_force = ti.Vector.field(3, ti.f32, shape=self.particle_count)
        self.geometry_report = ti.field(ti.f64, shape=5)
        self.audit = ti.field(ti.f64, shape=20)
        digest = hashlib.sha256(b"material-device-transfer-v1")
        digest.update(binding.identity_sha256.encode("ascii"))
        digest.update(bytes([inactive_axis]))
        digest.update(repr(self.cap_binding).encode("ascii"))
        for array in (edges, signs, degree, probe_offsets):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        self.identity_sha256 = digest.hexdigest()

    def require_matching_layout(self):
        if (
            int(self.markers.marker_count) != self.marker_count
            or int(self.markers.projection_vertex_count) != self._projection_vertex_count
            or int(self.markers.projection_segment_count) != self._projection_segment_count
            or int(self.markers.projection_triangle_count) != 0
            or getattr(self.markers, "_open_ribbon_tip_cap_binding", None) != self.cap_binding
        ):
            raise ValueError("material surface bound layout differs from its immutable reference")

    def validate_accepted_state(self, state, particle_positions_m, particle_velocities_mps):
        """Validate a checkpoint before writes; IQN trial velocity is not an accepted state."""
        self.require_matching_layout()
        expected = {}
        arithmetic_roundoff = {}
        for name, particle_values in (("x_gamma_m", particle_positions_m), ("v_gamma_mps", particle_velocities_mps)):
            values = np.asarray(particle_values)
            if values.ndim != 2 or values.shape[0] < self.particle_count or values.shape[1] != 3:
                raise ValueError("material accepted particle state has invalid shape")
            expected[name] = interpolate_material_surface(self.binding, values[:self.particle_count])
            terms = (values[:self.particle_count].astype(np.float64)[self.binding.particle_indices]
                     * self.binding.weights[:, :, None])
            # A dot-product's arithmetic error depends on its absolute terms,
            # not its possibly cancelling sum. This covers host/device FMA
            # differences without introducing a physical speed tolerance.
            arithmetic_roundoff[name] = 32.0 * _EPS64 * np.sum(np.abs(terms), axis=1)
            _validate_rounded_material_field(name, state[name], expected[name], arithmetic_roundoff[name])
        # Geometry is derived from the accepted f32 positions, whose material
        # consistency was just checked, not a second host-rounded realization.
        points = np.asarray(state["x_gamma_m"], dtype=np.float64)
        normals = np.zeros_like(points)
        normal_roundoff = np.zeros_like(points)
        axis = np.eye(3)[self.inactive_axis]
        for marker in range(self.marker_count):
            absolute_normal_sum = 0.0
            for slot in range(int(self._host_degree[marker])):
                first, second = self._host_edges[marker, slot]
                outward = self._host_signs[marker, slot] * np.cross(axis, points[second] - points[first])
                length = float(np.linalg.norm(outward))
                if not np.isfinite(length) or length <= 0.0:
                    raise ValueError("material accepted geometry has a degenerate edge")
                normals[marker] += outward / length
                absolute_normal_sum += float(np.linalg.norm(outward / length, ord=1))
            length = float(np.linalg.norm(normals[marker]))
            if not np.isfinite(length) or length <= 1.0e-12:
                raise ValueError("material accepted geometry has ambiguous orientation")
            normals[marker] /= length
            normal_roundoff[marker] = 64.0 * _EPS64 * absolute_normal_sum / length
        expected["n_gamma"] = normals
        expected["pressure_probe_origin_m"] = points + self._host_probe_offsets[:, None] * normals
        expected["A_gamma_m2"] = self._reference_area
        arithmetic_roundoff["n_gamma"] = normal_roundoff
        arithmetic_roundoff["pressure_probe_origin_m"] = (
            np.abs(self._host_probe_offsets[:, None]) * normal_roundoff
            + 32.0 * _EPS64 * (np.abs(points) + np.abs(self._host_probe_offsets[:, None] * normals))
        )
        arithmetic_roundoff["A_gamma_m2"] = np.zeros_like(self._reference_area)
        for name in ("n_gamma", "pressure_probe_origin_m", "A_gamma_m2"):
            _validate_rounded_material_field(name, state[name], expected[name], arithmetic_roundoff[name])
        _validate_accepted_cap_geometry(state, self.cap_binding, self.inactive_axis)

    @ti.kernel
    def _gather_geometry(self, position: ti.template(), velocity: ti.template()):
        for marker in range(self.marker_count):
            mapped_x = ti.Vector.zero(ti.f64, 3)
            mapped_v = ti.Vector.zero(ti.f64, 3)
            for slot in ti.static(range(8)):
                particle = self.indices[marker, slot]
                weight = self.weights[marker, slot]
                mapped_x += weight * position[particle].cast(ti.f64)
                mapped_v += weight * velocity[particle].cast(ti.f64)
            if self.markers._vector3_fits_f32(mapped_x) != 0 and self.markers._vector3_fits_f32(mapped_v) != 0:
                old_x = self.markers.x_gamma_m[marker]
                self.markers.x_gamma_m[marker] = mapped_x.cast(ti.f32)
                self.markers.v_gamma_mps[marker] = mapped_v.cast(ti.f32)
                self.geometry_report[0] += 1.0
                ti.atomic_max(self.geometry_report[2], (mapped_x - old_x.cast(ti.f64)).norm())
                ti.atomic_max(self.geometry_report[3], mapped_v.norm())
            else:
                self.geometry_report[1] += 1.0

    @ti.kernel
    def _derive_geometry(self):
        for marker in range(self.marker_count):
            normal_sum = ti.Vector.zero(ti.f64, 3)
            valid = 1
            for slot in ti.static(range(2)):
                if slot < self.degree[marker]:
                    first, second = self.edges[marker, slot, 0], self.edges[marker, slot, 1]
                    tangent = self.markers.x_gamma_m[second].cast(ti.f64) - self.markers.x_gamma_m[first].cast(ti.f64)
                    axis = ti.Vector.zero(ti.f64, 3)
                    axis[self.inactive_axis] = 1.0
                    edge_normal = self.edge_sign[marker, slot] * axis.cross(tangent)
                    length = edge_normal.norm()
                    if length > 0.0 and not ti.math.isinf(length) and not ti.math.isnan(length):
                        normal_sum += edge_normal / length
                    else:
                        valid = 0
            length = normal_sum.norm()
            if valid != 0 and length > 1.0e-12 and not ti.math.isnan(length) and not ti.math.isinf(length):
                normal = normal_sum / length
                change = (normal - self.markers.n_gamma[marker].cast(ti.f64)).norm()
                ti.atomic_max(self.geometry_report[4], change)
                self.markers.n_gamma[marker] = normal.cast(ti.f32)
                self.markers.pressure_probe_origin_m[marker] = (
                    self.markers.x_gamma_m[marker].cast(ti.f64) + self.probe_offset[marker] * normal
                ).cast(ti.f32)
            else:
                self.geometry_report[1] += 1.0

    def update_geometry(self, position, velocity):
        self.require_matching_layout()
        self.geometry_report.fill(0)
        self._gather_geometry(position, velocity)
        self._derive_geometry()
        values = self.geometry_report.to_numpy()
        if not np.isfinite(values).all() or int(values[1]) != 0:
            raise RuntimeError("material surface has degenerate or nonfinite oriented geometry")
        self.markers._refresh_open_ribbon_tip_cap_projection_vertices()
        return HibmMpmSurfaceUpdateReport(
            updated_marker_count=int(values[0]), invalid_marker_count=0,
            max_marker_displacement_m=float(values[2]), max_marker_speed_mps=float(values[3]),
            geometry_updated_marker_count=int(values[0]), geometry_invalid_marker_count=0,
            max_marker_normal_change=float(values[4]), max_marker_area_change_m2=0.0,
            candidate_pair_count=self.pair_count,
        )

    @ti.kernel
    def _prepare_load(self, position: ti.template(), velocity: ti.template(), include_cap: ti.i32):
        for marker in range(self.marker_count):
            force = self.markers.F_gamma_n[marker].cast(ti.f64)
            if include_cap != 0 and self.cap_owner[marker] >= 0:
                force += self.cap_factor[marker] * self.markers.F_gamma_n[self.cap_owner[marker]].cast(ti.f64)
            self.effective_force[marker] = force
            mapped_x = ti.Vector.zero(ti.f64, 3)
            mapped_v = ti.Vector.zero(ti.f64, 3)
            coordinate_scale = ti.cast(0.0, ti.f64)
            for slot in ti.static(range(8)):
                particle = self.indices[marker, slot]
                weight = self.weights[marker, slot]
                mapped_x += weight * position[particle].cast(ti.f64)
                mapped_v += weight * velocity[particle].cast(ti.f64)
                coordinate_scale += ti.abs(weight) * position[particle].cast(ti.f64).norm()
            self.material_velocity[marker] = mapped_v
            error = (mapped_x - self.markers.x_gamma_m[marker].cast(ti.f64)).norm()
            if error > 8.0 * _EPS32 * coordinate_scale:
                self.audit[_INVALID_GEOMETRY] += 1.0
            if self.markers._vector3_is_finite(force) == 0 or self.markers._vector3_is_finite(mapped_v) == 0:
                self.audit[_INVALID_FORCE] += 1.0

    @ti.kernel
    def _source_load_audit(self, cap_first: ti.i32, cap_count: ti.i32):
        for source in range(self.marker_count + cap_count):
            marker = source
            material_v = ti.Vector.zero(ti.f64, 3)
            if source < self.marker_count:
                material_v = self.material_velocity[marker]
            else:
                marker = cap_first + source - self.marker_count
                for physical in range(self.marker_count):
                    if self.cap_owner[physical] == marker:
                        material_v += self.cap_factor[physical] * self.material_velocity[physical]
            force = self.markers.F_gamma_n[marker].cast(ti.f64)
            point = self.markers.x_gamma_m[marker].cast(ti.f64)
            torque = point.cross(force)
            if self.markers._vector3_is_finite(force) == 0 or self.markers._vector3_is_finite(point) == 0:
                self.audit[_INVALID_FORCE] += 1.0
            for component in ti.static(range(3)):
                self.audit[_SOURCE_FORCE + component] += force[component]
                self.audit[_SOURCE_TORQUE + component] += torque[component]
            self.audit[_SOURCE_POWER] += force.dot(material_v)
            # Coordinate/cap rounding scales use absolute source loads, not net
            # load, so cancellations cannot erase the arithmetic error bound.
            self.audit[_TORQUE_SCALE] += point.norm() * force.norm()
            self.audit[_POWER_SCALE] += material_v.norm() * force.norm()

    @ti.kernel
    def _stage_load(self, external: ti.template(), position: ti.template(), velocity: ti.template()):
        for particle in range(self.particle_count):
            expected = ti.Vector.zero(ti.f64, 3)
            absolute_sum = ti.cast(0.0, ti.f64)
            for entry in range(self.column_offsets[particle], self.column_offsets[particle + 1]):
                contribution = self.column_weight[entry] * self.effective_force[self.column_marker[entry]]
                expected += contribution
                absolute_sum += contribution.norm()
            before = external[particle].cast(ti.f64)
            self.pending_external_force[particle] = (before + expected).cast(ti.f32)
            actual = self.pending_external_force[particle].cast(ti.f64) - before
            point = position[particle].cast(ti.f64)
            speed = velocity[particle].cast(ti.f64)
            torque = point.cross(actual)
            if self.markers._vector3_is_finite(self.pending_external_force[particle]) == 0:
                self.audit[_INVALID_FORCE] += 1.0
            for component in ti.static(range(3)):
                self.audit[_APPLIED_FORCE + component] += actual[component]
                self.audit[_APPLIED_TORQUE + component] += torque[component]
                ti.atomic_max(self.audit[_MAX_APPLIED], ti.abs(ti.cast(self.pending_external_force[particle][component], ti.f64)))
            self.audit[_APPLIED_POWER] += speed.dot(actual)
            self.audit[_FORCE_SCALE] += before.norm() + absolute_sum
            self.audit[_TORQUE_SCALE] += point.norm() * (before.norm() + absolute_sum)
            self.audit[_POWER_SCALE] += speed.norm() * (before.norm() + absolute_sum)

    @ti.kernel
    def _commit_load(self, external: ti.template()):
        for particle in range(self.particle_count):
            external[particle] = self.pending_external_force[particle]

    def scatter_load(self, external, position, velocity):
        self.require_matching_layout()
        if external.dtype != ti.f32:
            raise ValueError("material particle external-force storage must be f32")
        cap_first, cap_count, _ = self.markers._tip_cap_force_layout()
        self.audit.fill(0)
        self._prepare_load(position, velocity, int(cap_count != 0))
        self._source_load_audit(cap_first, cap_count)
        before_write = self.audit.to_numpy()
        if (
            not np.isfinite(before_write).all() or before_write[_INVALID_FORCE] != 0
            or before_write[_INVALID_GEOMETRY] != 0
        ):
            raise ValueError("material load or geometry is invalid before particle-force write")
        self._stage_load(external, position, velocity)
        values = self.audit.to_numpy()
        source_force = values[_SOURCE_FORCE:_SOURCE_FORCE + 3]
        applied_force = values[_APPLIED_FORCE:_APPLIED_FORCE + 3]
        force_error = float(np.linalg.norm(source_force - applied_force))
        torque_error = float(np.linalg.norm(
            values[_SOURCE_TORQUE:_SOURCE_TORQUE + 3] - values[_APPLIED_TORQUE:_APPLIED_TORQUE + 3]
        ))
        power_error = float(abs(values[_SOURCE_POWER] - values[_APPLIED_POWER]))
        # One f32 store per particle, fixed f64 stencil sums/reductions, plus
        # f32 material-coordinate/cap reconstruction. These are arithmetic
        # bounds in N, N m and W, not adjustable physical convergence tolerances.
        reduction_roundoff = 8.0 * _EPS64 * max(self.pair_count, self.particle_count)
        force_bound = (8.0 * _EPS32 + reduction_roundoff) * float(values[_FORCE_SCALE])
        geometry_gain = max(1.0, self.binding.maximum_row_l1) * (2.0 if cap_count else 1.0)
        torque_bound = (16.0 * _EPS32 * geometry_gain + reduction_roundoff) * float(values[_TORQUE_SCALE])
        power_bound = (8.0 * _EPS32 + reduction_roundoff) * float(values[_POWER_SCALE])
        verified = bool(
            np.isfinite(values).all() and values[_INVALID_FORCE] == 0
            and values[_INVALID_GEOMETRY] == 0 and force_error <= force_bound
            and torque_error <= torque_bound and power_error <= power_bound
        )
        if not verified:
            raise RuntimeError(
                "material adjoint transfer failed rounded-field conservation: "
                f"force={force_error}/{force_bound}, torque={torque_error}/{torque_bound}, "
                f"power={power_error}/{power_bound}, geometry={values[_INVALID_GEOMETRY]}"
            )
        # Both fields are f32, so the commit copies the audited bits exactly.
        # Any arithmetic/geometry failure above leaves the destination intact.
        self._commit_load(external)
        return HibmMpmMpmForceScatterReport(
            active_marker_count=self.marker_count + cap_count, invalid_marker_count=0,
            active_pair_count=self.pair_count, total_marker_force_n=tuple(source_force),
            total_mpm_external_force_n=tuple(applied_force), action_reaction_residual_n=force_error,
            candidate_pair_count=self.pair_count, invalid_external_force_particle_count=0,
            max_abs_external_force_component_n=float(values[_MAX_APPLIED]),
            material_transfer_verified=True, material_binding_identity=self.identity_sha256,
            force_roundoff_bound_n=force_bound, torque_residual_n_m=torque_error,
            torque_roundoff_bound_n_m=torque_bound, material_power_residual_w=power_error,
            material_power_roundoff_bound_w=power_bound,
        )


def _validate_accepted_cap_geometry(state, binding, inactive_axis):
    """Preflight the existing f32 cap derivation before any restore write."""
    if binding is None:
        return
    previous = [int(binding[0]), int(binding[2])]
    tips = [int(binding[1]), int(binding[3])]
    points = np.asarray(state["x_gamma_m"], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        edges = {
            name: np.float32(1.5) * np.asarray(state[name], dtype=np.float32)[tips]
            - np.float32(0.5) * np.asarray(state[name], dtype=np.float32)[previous]
            for name in ("x_gamma_m", "v_gamma_mps", "n_gamma")
        }
        tangent = edges["x_gamma_m"][1] - edges["x_gamma_m"][0]
        tip_direction = (
            np.float32(0.5) * (edges["x_gamma_m"][0] + edges["x_gamma_m"][1])
            - np.float32(0.5) * (points[tips[0]] + points[tips[1]])
        )
        tangent[inactive_axis] = tip_direction[inactive_axis] = 0.0
        cap_normal = np.cross(np.eye(3, dtype=np.float32)[inactive_axis], tangent)
        lengths = np.asarray([
            np.linalg.norm(edges["n_gamma"][0]), np.linalg.norm(edges["n_gamma"][1]),
            np.linalg.norm(tangent), np.linalg.norm(tip_direction), np.linalg.norm(cap_normal),
        ], dtype=np.float32)
        area = np.float32(0.5) * lengths[2] * np.float32(binding[10])
    if (not all(np.isfinite(value).all() for value in edges.values())
            or not np.isfinite(lengths).all() or np.any(lengths <= np.float32(1.0e-12))
            or not np.isfinite(area) or area <= 0.0):
        raise ValueError("material accepted cap geometry is degenerate or nonfinite")


def _validate_rounded_material_field(name, proposed, reference, arithmetic_roundoff):
    proposed = np.asarray(proposed)
    if proposed.shape != reference.shape or not np.isfinite(proposed).all():
        raise ValueError(f"material accepted {name} has invalid shape or values")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = reference.astype(np.float32)
        lower = np.nextafter(stored, np.float32(-np.inf), dtype=np.float32).astype(np.float64)
        upper = np.nextafter(stored, np.float32(np.inf), dtype=np.float32).astype(np.float64)
    if not np.isfinite(stored).all():
        raise ValueError(f"material accepted {name} is not finite in field storage")
    stored64 = stored.astype(np.float64)
    spacing = np.maximum(
        np.where(np.isfinite(lower), np.abs(stored64 - lower), 0.0),
        np.where(np.isfinite(upper), np.abs(upper - stored64), 0.0),
    )
    bound = spacing + arithmetic_roundoff
    if not np.isfinite(bound).all() or np.any(np.abs(proposed.astype(np.float64) - reference) > bound):
        raise ValueError(f"material accepted {name} is inconsistent with particle state")
