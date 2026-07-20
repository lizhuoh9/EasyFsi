"""Transactional marker-space constraints for staggered MAC velocity fields.

The sharp component-face ledger prescribes selected storage faces.  It does
not, by itself, enforce the marker interpolation equation ``J u = U_gamma``.
This module supplies that second, generic operation without weakening the
canonical face ledger: hard/external-exact faces have zero correction mobility,
the linear solve is performed entirely in private fields, and the physical
velocity field is changed once, only after convergence has been established.
"""

from dataclasses import dataclass
import math

import taichi as ti

from .core import (
    HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NO_COMPLETE_MAC_SUPPORT,
    HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NONE,
    HIBM_NO_SLIP_SAMPLE_INVALID_REASON_OUTSIDE_HALF_OPEN_DOMAIN,
)
from .mac_stencil import (
    mac_component_stencil_base_fraction,
    mac_stencil_weight,
)


# The exact dense pressure Schur factor is opt-in and deliberately bounded.
# Ordinary affine-Q users allocate none of these resources.  A larger marker
# system needs a sparse/device factor backend rather than an accidental O(M^2)
# allocation followed by an O(M^3) factorization.
HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS = 512
HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_BYTES = (
    HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS**2 * 8
)
HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class HibmMpmMarkerMacConstraintReport:
    """Immutable result of one marker-MAC constraint transaction."""

    prepared: bool
    converged: bool
    committed: bool
    active_marker_count: int
    constraint_count: int
    iterations: int
    max_residual_mps: float
    sample_identity_generation: int = 0


@dataclass(frozen=True)
class HibmMpmMarkerPressureNullspaceReport:
    """Immutable state of one prepared homogeneous pressure transaction.

    The two legacy ``last_max_*`` names carry solve-wide maxima accumulated
    across every device-only apply in the prepared transaction.  Keeping the
    names avoids breaking existing report consumers while making one final
    host audit sufficient for a complete pressure solve.
    """

    prepared: bool
    active_constraint_count: int
    apply_count: int
    pressure_actuation_generation: int
    min_factor_pivot: float
    last_max_input_constraint: float
    last_max_constraint_residual: float
    resource_bytes: int = 0
    independent_constraint_count: int = 0
    dependent_constraint_count: int = 0
    unactuated_constraint_count: int = 0
    max_dependent_normalized_pivot: float = 0.0
    max_unactuated_input_constraint: float = 0.0


@ti.data_oriented
class HibmMpmMarkerMacConstraintOperator:
    """Matrix-free ``(J_F M_F^-1 J_F.T) lambda = U_gamma - J u`` solver.

    ``F`` denotes the valid MAC support excluding canonical hard-fixed and
    external-exact components.  The same shared stencil functions build both
    ``J`` and ``J.T`` so interpolation and scatter cannot drift apart.
    """

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        marker_capacity: int,
    ) -> None:
        shape = tuple(int(value) for value in grid_nodes)
        if len(shape) != 3 or any(value < 2 for value in shape):
            raise ValueError("grid_nodes must contain three values >= 2")
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")

        self.grid_nodes = shape
        self.marker_capacity = int(marker_capacity)
        self.constraint_capacity = 3 * self.marker_capacity

        self._row_active = ti.field(dtype=ti.i32, shape=self.constraint_capacity)
        self._row_pcg_active = ti.field(
            dtype=ti.i32,
            shape=self.constraint_capacity,
        )
        self._rhs = ti.field(dtype=ti.f32, shape=self.constraint_capacity)
        self._diagonal = ti.field(dtype=ti.f32, shape=self.constraint_capacity)
        self._lambda = ti.field(dtype=ti.f32, shape=self.constraint_capacity)
        self._residual = ti.field(dtype=ti.f32, shape=self.constraint_capacity)
        self._preconditioned = ti.field(
            dtype=ti.f32,
            shape=self.constraint_capacity,
        )
        self._direction = ti.field(dtype=ti.f32, shape=self.constraint_capacity)
        self._matrix_direction = ti.field(
            dtype=ti.f32,
            shape=self.constraint_capacity,
        )
        self._stencil_index = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=(self.constraint_capacity, 8),
        )
        self._stencil_weight = ti.field(
            dtype=ti.f32,
            shape=(self.constraint_capacity, 8),
        )
        self._stencil_free = ti.field(
            dtype=ti.i32,
            shape=(self.constraint_capacity, 8),
        )
        self._stencil_inverse_mass_per_kg = ti.field(
            dtype=ti.f32,
            shape=(self.constraint_capacity, 8),
        )
        self._support_velocity_snapshot_mps = ti.field(
            dtype=ti.f32,
            shape=(self.constraint_capacity, 8),
        )
        self._support_valid_mask_snapshot = ti.field(
            dtype=ti.i32,
            shape=(self.constraint_capacity, 8),
        )
        self._support_hard_mask_snapshot = ti.field(
            dtype=ti.i32,
            shape=(self.constraint_capacity, 8),
        )
        self._support_external_mask_snapshot = ti.field(
            dtype=ti.i32,
            shape=(self.constraint_capacity, 8),
        )
        self._marker_snapshot_active = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._marker_position_snapshot_m = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        self._marker_input_position_snapshot_m = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        self._marker_target_snapshot_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        self._marker_region_snapshot = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._sampling_valid_snapshot = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._sampling_source_snapshot = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._sampling_invalid_reason_snapshot = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._sampling_position_snapshot_m = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        nx, ny, nz = shape
        self._cell_face_x_snapshot_m = ti.field(dtype=ti.f32, shape=nx + 1)
        self._cell_face_y_snapshot_m = ti.field(dtype=ti.f32, shape=ny + 1)
        self._cell_face_z_snapshot_m = ti.field(dtype=ti.f32, shape=nz + 1)
        self._cell_center_x_snapshot_m = ti.field(dtype=ti.f32, shape=nx)
        self._cell_center_y_snapshot_m = ti.field(dtype=ti.f32, shape=ny)
        self._cell_center_z_snapshot_m = ti.field(dtype=ti.f32, shape=nz)
        self._cell_width_x_snapshot_m = ti.field(dtype=ti.f32, shape=nx)
        self._cell_width_y_snapshot_m = ti.field(dtype=ti.f32, shape=ny)
        self._cell_width_z_snapshot_m = ti.field(dtype=ti.f32, shape=nz)
        self._rho_snapshot_kgm3 = ti.field(dtype=ti.f32, shape=())
        self._correction = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self._solved_correction_snapshot = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=shape,
        )
        self._grid_scratch = ti.Vector.field(3, dtype=ti.f32, shape=shape)

        # Pressure increments need the *linear* homogeneous projector
        #
        #   N = I - P J_I.T (J_I P J_I.T)^-1 J_I,
        #
        # where P is the caller-materialized pressure-actuated inverse face
        # mass.  The dense marker matrix is small (3 * marker_capacity) and is
        # where I is a deterministic independent basis for the pressure-
        # actuated marker row space.  The normalized dense marker Gram matrix
        # is factored once in f64.  Reusing that immutable factor makes every
        # pressure matvec see the same linear, self-adjoint operation; a
        # residual-stopped nested Krylov solve would not provide that contract.
        self._pressure_nullspace_resources_allocated = False
        self._pressure_nullspace_resource_bytes = 0
        self._pressure_nullspace_row_active = None
        self._pressure_nullspace_mobility_snapshot = None
        self._pressure_nullspace_inverse_mass_per_kg = None
        self._pressure_nullspace_schur = None
        self._pressure_nullspace_factor = None
        self._pressure_nullspace_row_inverse_norm = None
        self._pressure_nullspace_factor_row_selected = None
        self._pressure_nullspace_factor_order = None
        self._pressure_nullspace_rhs = None
        self._pressure_nullspace_forward = None
        self._pressure_nullspace_lambda = None
        self._pressure_nullspace_correction = None
        self._pressure_nullspace_candidate = None
        self._pressure_nullspace_failure_code = None
        self._pressure_nullspace_active_constraint_count = None
        self._pressure_nullspace_independent_constraint_count = None
        self._pressure_nullspace_dependent_constraint_count = None
        self._pressure_nullspace_unactuated_constraint_count = None
        self._pressure_nullspace_min_factor_pivot = None
        self._pressure_nullspace_max_dependent_normalized_pivot = None
        self._pressure_nullspace_max_input_constraint = None
        self._pressure_nullspace_max_unactuated_input_constraint = None
        self._pressure_nullspace_max_constraint_residual = None

        self._rz_old = ti.field(dtype=ti.f64, shape=())
        self._rz_new = ti.field(dtype=ti.f64, shape=())
        self._p_ap = ti.field(dtype=ti.f64, shape=())
        self._max_residual = ti.field(dtype=ti.f32, shape=())
        self._true_candidate_max_residual = ti.field(dtype=ti.f32, shape=())
        self._failure_code = ti.field(dtype=ti.i32, shape=())
        self._audit_failure_code = ti.field(dtype=ti.i32, shape=())
        self._solved_correction_integrity_failure = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._device_converged = ti.field(dtype=ti.i32, shape=())
        self._device_iterations = ti.field(dtype=ti.i32, shape=())
        self._device_active_marker_count = ti.field(dtype=ti.i32, shape=())
        self._device_constraint_count = ti.field(dtype=ti.i32, shape=())

        self._markers = None
        self._fluid = None
        self._component_face_valid_mask = None
        self._prepared_obstacle_field = None
        self._marker_count = 0
        self._active_marker_count = 0
        self._constraint_count = 0
        self._prepared = False
        self._converged = False
        self._committed = False
        self._iterations = 0
        self._absolute_tolerance_mps = math.nan
        self._max_residual_mps = math.inf
        self._phase = "idle"
        self._prepared_ledger_generation = -1
        self._prepared_primary_region_id = -1
        self._prepared_secondary_region_id = -1
        self.prepared_sampling_identity = None
        self._prepared_sampling_identity_generation = 0
        self._prepared_topology_generation = 0
        self._prepared_component_face_valid_mask_generation = 0
        self._pressure_nullspace_prepared = False
        self._pressure_nullspace_apply_count = 0
        self._pressure_nullspace_fluid = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuated_component_mobility = None
        self._pressure_actuation_generation = 0
        self._pressure_nullspace_topology_generation = 0
        self._pressure_nullspace_component_face_valid_mask_generation = 0
        self._pressure_nullspace_poisoned = False

    @ti.kernel
    def _reset_transaction_kernel(self):
        for row in range(self.constraint_capacity):
            self._row_active[row] = 0
            self._row_pcg_active[row] = 0
            self._rhs[row] = 0.0
            self._diagonal[row] = 0.0
            self._lambda[row] = 0.0
            self._residual[row] = 0.0
            self._preconditioned[row] = 0.0
            self._direction[row] = 0.0
            self._matrix_direction[row] = 0.0
        for i, j, k in self._correction:
            self._correction[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self._solved_correction_snapshot[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
            self._grid_scratch[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        for row, support in self._stencil_weight:
            self._stencil_index[row, support] = ti.Vector([-1, -1, -1])
            self._stencil_weight[row, support] = 0.0
            self._stencil_free[row, support] = 0
            self._stencil_inverse_mass_per_kg[row, support] = 0.0
            self._support_velocity_snapshot_mps[row, support] = 0.0
            self._support_valid_mask_snapshot[row, support] = 0
            self._support_hard_mask_snapshot[row, support] = 0
            self._support_external_mask_snapshot[row, support] = 0
        for marker in range(self.marker_capacity):
            self._marker_snapshot_active[marker] = 0
            self._marker_position_snapshot_m[marker] = ti.Vector([0.0, 0.0, 0.0])
            self._marker_input_position_snapshot_m[marker] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
            self._marker_target_snapshot_mps[marker] = ti.Vector([0.0, 0.0, 0.0])
            self._marker_region_snapshot[marker] = -1
            self._sampling_valid_snapshot[marker] = 0
            self._sampling_source_snapshot[marker] = 0
            self._sampling_position_snapshot_m[marker] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
        self._rz_old[None] = 0.0
        self._rz_new[None] = 0.0
        self._p_ap[None] = 0.0
        self._max_residual[None] = 0.0
        self._true_candidate_max_residual[None] = 0.0
        self._failure_code[None] = 0
        self._audit_failure_code[None] = 0
        self._solved_correction_integrity_failure[None] = 0
        self._device_converged[None] = 0
        self._device_iterations[None] = 0
        self._device_active_marker_count[None] = 0
        self._device_constraint_count[None] = 0

    @ti.kernel
    def _reset_pressure_nullspace_prepare_kernel(self):
        self._pressure_nullspace_failure_code[None] = 0
        self._pressure_nullspace_active_constraint_count[None] = 0
        self._pressure_nullspace_independent_constraint_count[None] = 0
        self._pressure_nullspace_dependent_constraint_count[None] = 0
        self._pressure_nullspace_unactuated_constraint_count[None] = 0
        self._pressure_nullspace_min_factor_pivot[None] = 0.0
        self._pressure_nullspace_max_dependent_normalized_pivot[None] = 0.0
        self._pressure_nullspace_max_input_constraint[None] = 0.0
        self._pressure_nullspace_max_unactuated_input_constraint[None] = 0.0
        self._pressure_nullspace_max_constraint_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            self._pressure_nullspace_row_active[row] = self._row_active[row]
            if self._row_active[row] != 0:
                ti.atomic_add(
                    self._pressure_nullspace_active_constraint_count[None],
                    1,
                )
            self._pressure_nullspace_row_inverse_norm[row] = 0.0
            self._pressure_nullspace_factor_row_selected[row] = 0
            self._pressure_nullspace_factor_order[row] = -1
            self._pressure_nullspace_rhs[row] = 0.0
            self._pressure_nullspace_forward[row] = 0.0
            self._pressure_nullspace_lambda[row] = 0.0
        for row, support in self._pressure_nullspace_mobility_snapshot:
            self._pressure_nullspace_mobility_snapshot[row, support] = 0.0
            self._pressure_nullspace_inverse_mass_per_kg[row, support] = 0.0
        for row, column in self._pressure_nullspace_schur:
            self._pressure_nullspace_schur[row, column] = 0.0
            self._pressure_nullspace_factor[row, column] = 0.0
        for i, j, k in self._pressure_nullspace_correction:
            self._pressure_nullspace_correction[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
            self._pressure_nullspace_candidate[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )

    @ti.kernel
    def _snapshot_pressure_nullspace_mobility_kernel(
        self,
        pressure_actuated_component_mobility: ti.template(),
    ):
        for row, support in self._pressure_nullspace_mobility_snapshot:
            if self._pressure_nullspace_row_active[row] != 0:
                index = self._stencil_index[row, support]
                weight = self._stencil_weight[row, support]
                if index.x >= 0 and weight != 0.0:
                    axis = row % 3
                    actuation_weight = ti.cast(
                        pressure_actuated_component_mobility[
                            index.x, index.y, index.z
                        ][axis],
                        ti.f64,
                    )
                    finite = not ti.math.isnan(actuation_weight)
                    finite = finite and not ti.math.isinf(actuation_weight)
                    if not finite or actuation_weight < 0.0:
                        ti.atomic_max(
                            self._pressure_nullspace_failure_code[None],
                            1,
                        )
                    else:
                        # The caller supplies the complete diagonal pressure
                        # actuation metric A=R^-1.  It already includes
                        # pressure mobility/incidence and inverse dual mass;
                        # multiplying by the ordinary affine-Q metric here
                        # would silently apply inverse mass twice.
                        inverse_mass = actuation_weight
                        inverse_mass_finite = not ti.math.isnan(inverse_mass)
                        inverse_mass_finite = (
                            inverse_mass_finite
                            and not ti.math.isinf(inverse_mass)
                        )
                        if not inverse_mass_finite or inverse_mass < 0.0:
                            ti.atomic_max(
                                self._pressure_nullspace_failure_code[None],
                                1,
                            )
                        else:
                            self._pressure_nullspace_mobility_snapshot[
                                row, support
                            ] = actuation_weight
                            support_is_free = (
                                self._stencil_free[row, support] != 0
                            )
                            hard_owned = (
                                self._support_hard_mask_snapshot[row, support]
                                & (1 << axis)
                            ) != 0
                            external_owned = (
                                self._support_external_mask_snapshot[row, support]
                                & (1 << axis)
                            ) != 0
                            if not support_is_free:
                                if actuation_weight > 0.0 and hard_owned:
                                    ti.atomic_max(
                                        self._pressure_nullspace_failure_code[None],
                                        6,
                                    )
                                if actuation_weight > 0.0 and external_owned:
                                    ti.atomic_max(
                                        self._pressure_nullspace_failure_code[None],
                                        7,
                                    )
                            else:
                                self._pressure_nullspace_inverse_mass_per_kg[
                                    row, support
                                ] = inverse_mass

    @ti.kernel
    def _assemble_pressure_nullspace_schur_kernel(self):
        for first_row, second_row in self._pressure_nullspace_schur:
            value = ti.cast(0.0, ti.f64)
            first_active = self._pressure_nullspace_row_active[first_row] != 0
            second_active = self._pressure_nullspace_row_active[second_row] != 0
            if first_active and second_active:
                first_axis = first_row % 3
                second_axis = second_row % 3
                if first_axis == second_axis:
                    # Ordinary support loops avoid statically expanding this
                    # sizeable body 64 times during Taichi's cold compile.
                    for first_support in range(8):
                        for second_support in range(8):
                            first_index = self._stencil_index[
                                first_row, first_support
                            ]
                            second_index = self._stencil_index[
                                second_row, second_support
                            ]
                            same_index = (
                                self._stencil_free[first_row, first_support] != 0
                            )
                            same_index = same_index and (
                                self._stencil_free[second_row, second_support] != 0
                            )
                            same_index = same_index and first_index.x >= 0
                            same_index = same_index and second_index.x >= 0
                            same_index = (
                                same_index and first_index.x == second_index.x
                            )
                            same_index = (
                                same_index and first_index.y == second_index.y
                            )
                            same_index = (
                                same_index and first_index.z == second_index.z
                            )
                            if same_index:
                                first_inverse_mass = (
                                    self._pressure_nullspace_inverse_mass_per_kg[
                                        first_row, first_support
                                    ]
                                )
                                second_inverse_mass = (
                                    self._pressure_nullspace_inverse_mass_per_kg[
                                        second_row, second_support
                                    ]
                                )
                                if first_inverse_mass != second_inverse_mass:
                                    ti.atomic_max(
                                        self._pressure_nullspace_failure_code[None],
                                        3,
                                    )
                                value += (
                                    ti.cast(
                                        self._stencil_weight[
                                            first_row, first_support
                                        ],
                                        ti.f64,
                                    )
                                    * first_inverse_mass
                                    * ti.cast(
                                        self._stencil_weight[
                                            second_row, second_support
                                        ],
                                        ti.f64,
                                    )
                                )
            self._pressure_nullspace_schur[first_row, second_row] = value

    @ti.kernel
    def _symmetrize_pressure_nullspace_schur_kernel(self):
        # The two independently accumulated triangles differ, at most, by f64
        # addition order.  Publish one exactly symmetric Gram matrix before
        # rank revelation so the selected operator is self-adjoint bit for bit.
        for first_row, second_row in self._pressure_nullspace_schur:
            if first_row < second_row:
                value = 0.5 * (
                    self._pressure_nullspace_schur[first_row, second_row]
                    + self._pressure_nullspace_schur[second_row, first_row]
                )
                self._pressure_nullspace_schur[first_row, second_row] = value
                self._pressure_nullspace_schur[second_row, first_row] = value

    @ti.kernel
    def _factor_pressure_nullspace_schur_kernel(
        self,
        relative_pivot_tolerance: ti.f64,
    ):
        # S=J A J.T is a positive-semidefinite Gram matrix.  Normalize every
        # pressure-actuated row before revealing rank so physical row scale
        # and marker order cannot decide which constraints survive.  Exact
        # zero-energy rows are not algebraic dependencies: pressure simply
        # has no authority on them, so every apply audits their compatibility.
        ti.loop_config(serialize=True)
        for row in range(self.constraint_capacity):
            if self._pressure_nullspace_row_active[row] != 0:
                diagonal = self._pressure_nullspace_schur[row, row]
                finite = not ti.math.isnan(diagonal)
                finite = finite and not ti.math.isinf(diagonal)
                if not finite or diagonal < 0.0:
                    ti.atomic_max(
                        self._pressure_nullspace_failure_code[None],
                        2,
                    )
                elif diagonal == 0.0:
                    ti.atomic_add(
                        self._pressure_nullspace_unactuated_constraint_count[
                            None
                        ],
                        1,
                    )
                else:
                    self._pressure_nullspace_row_inverse_norm[row] = (
                        1.0 / ti.sqrt(diagonal)
                    )

        self._pressure_nullspace_min_factor_pivot[None] = ti.cast(
            1.0e300,
            ti.f64,
        )
        # J never couples different velocity components, so the normalized
        # Gram matrix consists of three independent marker blocks.  Complete
        # diagonal pivoting chooses the largest remaining row-space energy;
        # ties keep the lowest original row because the scan is serialized.
        for axis in ti.static(range(3)):
            factor_rank = 0
            ti.loop_config(serialize=True)
            for selection_marker in range(self.marker_capacity):
                # Once a block has no pivot above tolerance, factor_rank stops
                # advancing and later selection_marker values skip this body.
                if selection_marker == factor_rank:
                    best_row = -1
                    best_residual_diagonal = ti.cast(-1.0e300, ti.f64)
                    minimum_residual_diagonal = ti.cast(1.0e300, ti.f64)
                    remaining_count = 0
                    for candidate_marker in range(self.marker_capacity):
                        candidate = 3 * candidate_marker + axis
                        candidate_available = (
                            self._pressure_nullspace_row_active[candidate] != 0
                        )
                        candidate_available = candidate_available and (
                            self._pressure_nullspace_row_inverse_norm[candidate]
                            > 0.0
                        )
                        candidate_available = candidate_available and (
                            self._pressure_nullspace_factor_row_selected[
                                candidate
                            ]
                            == 0
                        )
                        if candidate_available:
                            residual_diagonal = ti.cast(1.0, ti.f64)
                            for prior_marker in range(self.marker_capacity):
                                if prior_marker < factor_rank:
                                    prior_slot = 3 * prior_marker + axis
                                    factor_value = self._pressure_nullspace_factor[
                                        candidate, prior_slot
                                    ]
                                    residual_diagonal -= factor_value * factor_value
                            remaining_count += 1
                            minimum_residual_diagonal = ti.min(
                                minimum_residual_diagonal,
                                residual_diagonal,
                            )
                            if residual_diagonal > best_residual_diagonal:
                                best_residual_diagonal = residual_diagonal
                                best_row = candidate

                    if best_row >= 0:
                        if (
                            minimum_residual_diagonal
                            < -8.0 * relative_pivot_tolerance
                        ):
                            ti.atomic_max(
                                self._pressure_nullspace_failure_code[None],
                                2,
                            )
                        elif best_residual_diagonal > relative_pivot_tolerance:
                            factor_slot = 3 * factor_rank + axis
                            pivot_sqrt = ti.sqrt(best_residual_diagonal)
                            self._pressure_nullspace_factor_order[
                                factor_slot
                            ] = best_row
                            self._pressure_nullspace_factor_row_selected[
                                best_row
                            ] = 1
                            self._pressure_nullspace_factor[
                                best_row, factor_slot
                            ] = pivot_sqrt
                            ti.atomic_add(
                                self._pressure_nullspace_independent_constraint_count[
                                    None
                                ],
                                1,
                            )
                            self._pressure_nullspace_min_factor_pivot[None] = ti.min(
                                self._pressure_nullspace_min_factor_pivot[None],
                                best_residual_diagonal,
                            )

                            for candidate_marker in range(self.marker_capacity):
                                candidate = 3 * candidate_marker + axis
                                candidate_available = (
                                    self._pressure_nullspace_row_active[candidate]
                                    != 0
                                )
                                candidate_available = candidate_available and (
                                    self._pressure_nullspace_row_inverse_norm[
                                        candidate
                                    ]
                                    > 0.0
                                )
                                candidate_available = candidate_available and (
                                    self._pressure_nullspace_factor_row_selected[
                                        candidate
                                    ]
                                    == 0
                                )
                                if candidate_available:
                                    value = (
                                        self._pressure_nullspace_schur[
                                            candidate, best_row
                                        ]
                                        * self._pressure_nullspace_row_inverse_norm[
                                            candidate
                                        ]
                                        * self._pressure_nullspace_row_inverse_norm[
                                            best_row
                                        ]
                                    )
                                    for prior_marker in range(
                                        self.marker_capacity
                                    ):
                                        if prior_marker < factor_rank:
                                            prior_slot = 3 * prior_marker + axis
                                            value -= (
                                                self._pressure_nullspace_factor[
                                                    candidate, prior_slot
                                                ]
                                                * self._pressure_nullspace_factor[
                                                    best_row, prior_slot
                                                ]
                                            )
                                    self._pressure_nullspace_factor[
                                        candidate, factor_slot
                                    ] = value / pivot_sqrt
                            factor_rank += 1
                        else:
                            ti.atomic_add(
                                self._pressure_nullspace_dependent_constraint_count[
                                    None
                                ],
                                remaining_count,
                            )
                            self._pressure_nullspace_max_dependent_normalized_pivot[
                                None
                            ] = ti.max(
                                self._pressure_nullspace_max_dependent_normalized_pivot[
                                    None
                                ],
                                ti.max(
                                    ti.cast(0.0, ti.f64),
                                    best_residual_diagonal,
                                ),
                            )

    @ti.kernel
    def _reset_pressure_nullspace_audit_kernel(self):
        self._pressure_nullspace_failure_code[None] = 0

    @ti.kernel
    def _audit_pressure_nullspace_support_kernel(
        self,
        pressure_actuated_component_mobility: ti.template(),
        component_face_valid_mask: ti.template(),
        hard_fixed_component_mask: ti.template(),
        external_exact_component_mask: ti.template(),
    ):
        for row, support in self._pressure_nullspace_mobility_snapshot:
            if self._pressure_nullspace_row_active[row] != 0:
                index = self._stencil_index[row, support]
                if index.x >= 0 and self._stencil_weight[row, support] != 0.0:
                    axis = row % 3
                    mobility = ti.cast(
                        pressure_actuated_component_mobility[
                            index.x, index.y, index.z
                        ][axis],
                        ti.f64,
                    )
                    mismatch = mobility != (
                        self._pressure_nullspace_mobility_snapshot[row, support]
                    )
                    if mismatch:
                        ti.atomic_max(
                            self._pressure_nullspace_failure_code[None],
                            4,
                        )
                    support_mismatch = (
                        component_face_valid_mask[index.x, index.y, index.z]
                        != self._support_valid_mask_snapshot[row, support]
                    )
                    support_mismatch = support_mismatch or (
                        hard_fixed_component_mask[index.x, index.y, index.z]
                        != self._support_hard_mask_snapshot[row, support]
                    )
                    support_mismatch = support_mismatch or (
                        external_exact_component_mask[index.x, index.y, index.z]
                        != self._support_external_mask_snapshot[row, support]
                    )
                    if support_mismatch:
                        ti.atomic_max(
                            self._pressure_nullspace_failure_code[None],
                            5,
                        )

    @ti.kernel
    def _gather_pressure_nullspace_rhs_kernel(
        self,
        input_face_correction: ti.template(),
    ):
        for row in range(self.constraint_capacity):
            value = ti.cast(0.0, ti.f64)
            if self._pressure_nullspace_row_active[row] != 0:
                axis = row % 3
                for support in ti.static(range(8)):
                    index = self._stencil_index[row, support]
                    if index.x >= 0:
                        value += ti.cast(
                            self._stencil_weight[row, support],
                            ti.f64,
                        ) * ti.cast(
                            input_face_correction[index.x, index.y, index.z][axis],
                            ti.f64,
                        )
            self._pressure_nullspace_rhs[row] = value
            self._pressure_nullspace_forward[row] = 0.0
            self._pressure_nullspace_lambda[row] = 0.0

    @ti.kernel
    def _solve_pressure_nullspace_factor_kernel(self):
        max_input = ti.cast(0.0, ti.f64)
        max_unactuated_input = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for row in range(self.constraint_capacity):
            max_input = ti.max(
                max_input,
                ti.abs(self._pressure_nullspace_rhs[row]),
            )
            if (
                self._pressure_nullspace_row_active[row] != 0
                and self._pressure_nullspace_row_inverse_norm[row] == 0.0
            ):
                max_unactuated_input = ti.max(
                    max_unactuated_input,
                    ti.abs(self._pressure_nullspace_rhs[row]),
                )
        self._pressure_nullspace_max_input_constraint[None] = ti.max(
            self._pressure_nullspace_max_input_constraint[None],
            max_input,
        )
        self._pressure_nullspace_max_unactuated_input_constraint[None] = ti.max(
            self._pressure_nullspace_max_unactuated_input_constraint[None],
            max_unactuated_input,
        )
        for axis in ti.static(range(3)):
            ti.loop_config(serialize=True)
            for factor_marker in range(self.marker_capacity):
                factor_slot = 3 * factor_marker + axis
                row = self._pressure_nullspace_factor_order[factor_slot]
                if row >= 0:
                    value = (
                        self._pressure_nullspace_rhs[row]
                        * self._pressure_nullspace_row_inverse_norm[row]
                    )
                    for prior_marker in range(self.marker_capacity):
                        if prior_marker < factor_marker:
                            prior_slot = 3 * prior_marker + axis
                            value -= (
                                self._pressure_nullspace_factor[row, prior_slot]
                                * self._pressure_nullspace_forward[prior_slot]
                            )
                    value /= self._pressure_nullspace_factor[row, factor_slot]
                    self._pressure_nullspace_forward[factor_slot] = value
            ti.loop_config(serialize=True)
            for reverse_marker in range(self.marker_capacity):
                factor_marker = self.marker_capacity - 1 - reverse_marker
                factor_slot = 3 * factor_marker + axis
                row = self._pressure_nullspace_factor_order[factor_slot]
                if row >= 0:
                    value = self._pressure_nullspace_forward[factor_slot]
                    for following_marker in range(self.marker_capacity):
                        if following_marker > factor_marker:
                            following_slot = 3 * following_marker + axis
                            following_row = self._pressure_nullspace_factor_order[
                                following_slot
                            ]
                            if following_row >= 0:
                                value -= (
                                    self._pressure_nullspace_factor[
                                        following_row, factor_slot
                                    ]
                                    * self._pressure_nullspace_lambda[
                                        following_slot
                                    ]
                                )
                    value /= self._pressure_nullspace_factor[row, factor_slot]
                    self._pressure_nullspace_lambda[factor_slot] = value

    @ti.kernel
    def _clear_pressure_nullspace_candidate_kernel(self):
        for i, j, k in self._pressure_nullspace_correction:
            self._pressure_nullspace_correction[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
            self._pressure_nullspace_candidate[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )

    @ti.kernel
    def _scatter_pressure_nullspace_correction_kernel(self):
        # Marker supports overlap.  A grid-parallel atomic scatter would be
        # mathematically linear but could change f64 addition order between
        # outer FV-CG matvecs.  The support list is tiny (8 entries per row), so
        # serialize it and make the prepared operator bitwise deterministic.
        ti.loop_config(serialize=True)
        for factor_slot in range(self.constraint_capacity):
            row = self._pressure_nullspace_factor_order[factor_slot]
            if row >= 0:
                coefficient = (
                    self._pressure_nullspace_lambda[factor_slot]
                    * self._pressure_nullspace_row_inverse_norm[row]
                )
                for support in ti.static(range(8)):
                    inverse_mass = self._pressure_nullspace_inverse_mass_per_kg[
                        row, support
                    ]
                    if (
                        self._stencil_free[row, support] != 0
                        and inverse_mass > 0.0
                    ):
                        index = self._stencil_index[row, support]
                        axis = row % 3
                        self._pressure_nullspace_correction[
                            index.x, index.y, index.z
                        ][axis] += (
                            inverse_mass
                            * ti.cast(self._stencil_weight[row, support], ti.f64)
                            * coefficient
                        )

    @ti.kernel
    def _build_pressure_nullspace_candidate_kernel(
        self,
        input_face_correction: ti.template(),
    ):
        for i, j, k in self._pressure_nullspace_candidate:
            input_value = ti.cast(
                input_face_correction[i, j, k],
                ti.f64,
            )
            candidate = (
                input_value - self._pressure_nullspace_correction[i, j, k]
            )
            finite = True
            for axis in ti.static(range(3)):
                finite = finite and not ti.math.isnan(input_value[axis])
                finite = finite and not ti.math.isinf(input_value[axis])
                finite = finite and not ti.math.isnan(candidate[axis])
                finite = finite and not ti.math.isinf(candidate[axis])
            if not finite:
                ti.atomic_max(
                    self._pressure_nullspace_failure_code[None],
                    8,
                )
                candidate = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            self._pressure_nullspace_candidate[i, j, k] = candidate

    @ti.kernel
    def _measure_pressure_nullspace_residual_kernel(self):
        max_residual = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for row in range(self.constraint_capacity):
            if self._pressure_nullspace_row_active[row] != 0:
                axis = row % 3
                residual = ti.cast(0.0, ti.f64)
                for support in ti.static(range(8)):
                    index = self._stencil_index[row, support]
                    if index.x >= 0:
                        residual += ti.cast(
                            self._stencil_weight[row, support],
                            ti.f64,
                        ) * self._pressure_nullspace_candidate[
                            index.x, index.y, index.z
                        ][axis]
                max_residual = ti.max(max_residual, ti.abs(residual))
        self._pressure_nullspace_max_constraint_residual[None] = ti.max(
            self._pressure_nullspace_max_constraint_residual[None],
            max_residual,
        )

    @ti.kernel
    def _commit_pressure_nullspace_candidate_kernel(
        self,
        output_face_correction: ti.template(),
    ):
        for i, j, k in self._pressure_nullspace_candidate:
            output_face_correction[i, j, k] = self._pressure_nullspace_candidate[
                i, j, k
            ]

    @ti.kernel
    def _reset_validation_kernel(self):
        self._failure_code[None] = 0
        self._device_active_marker_count[None] = 0

    @ti.func
    def _coincident_marker_constraint_relation(
        self,
        marker_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        first_marker: ti.i32,
        second_marker: ti.i32,
    ):
        """Return 0 for distinct positions, 1 for conflict, and 2 for identity."""

        same_position = True
        same_target = True
        for axis in ti.static(range(3)):
            same_position = same_position and (
                marker_position_m[first_marker][axis]
                == marker_position_m[second_marker][axis]
            )
            same_target = same_target and (
                marker_velocity_mps[first_marker][axis]
                == marker_velocity_mps[second_marker][axis]
            )
        relation = 0
        if same_position:
            relation = 1
            if same_target:
                relation = 2
        return relation

    @ti.kernel
    def _validate_markers_kernel(
        self,
        marker_position_m: ti.template(),
        marker_input_position_m: ti.template(),
        prepared_sample_valid: ti.template(),
        prepared_sample_invalid_reason_code: ti.template(),
        use_prepared_sampling_identity: ti.i32,
        marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(),
        marker_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
    ):
        nx = ti.static(self.grid_nodes[0])
        ny = ti.static(self.grid_nodes[1])
        nz = ti.static(self.grid_nodes[2])
        for marker in range(marker_count):
            region = marker_region_id[marker]
            active = region == primary_region_id or region == secondary_region_id
            if active:
                ti.atomic_add(self._device_active_marker_count[None], 1)
                if use_prepared_sampling_identity != 0:
                    sample_valid = prepared_sample_valid[marker] != 0
                    invalid_reason = prepared_sample_invalid_reason_code[marker]
                    if sample_valid:
                        if invalid_reason != HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NONE:
                            ti.atomic_max(self._failure_code[None], 7)
                    elif (
                        invalid_reason
                        == HIBM_NO_SLIP_SAMPLE_INVALID_REASON_OUTSIDE_HALF_OPEN_DOMAIN
                    ):
                        ti.atomic_max(self._failure_code[None], 6)
                    elif (
                        invalid_reason
                        == HIBM_NO_SLIP_SAMPLE_INVALID_REASON_NO_COMPLETE_MAC_SUPPORT
                    ):
                        ti.atomic_max(self._failure_code[None], 8)
                    else:
                        ti.atomic_max(self._failure_code[None], 7)
                position = marker_position_m[marker]
                input_position = marker_input_position_m[marker]
                target = marker_velocity_mps[marker]
                finite = True
                for axis in ti.static(range(3)):
                    finite = finite and position[axis] == position[axis]
                    finite = finite and input_position[axis] == input_position[axis]
                    finite = finite and target[axis] == target[axis]
                    finite = finite and ti.abs(position[axis]) < 3.4e38
                    finite = finite and ti.abs(input_position[axis]) < 3.4e38
                    finite = finite and ti.abs(target[axis]) < 3.4e38
                if not finite:
                    ti.atomic_max(self._failure_code[None], 1)
                inside_half_open_domain = (
                    position.x >= cell_face_x_m[0]
                    and position.x < cell_face_x_m[nx]
                    and position.y >= cell_face_y_m[0]
                    and position.y < cell_face_y_m[ny]
                    and position.z >= cell_face_z_m[0]
                    and position.z < cell_face_z_m[nz]
                    and input_position.x >= cell_face_x_m[0]
                    and input_position.x < cell_face_x_m[nx]
                    and input_position.y >= cell_face_y_m[0]
                    and input_position.y < cell_face_y_m[ny]
                    and input_position.z >= cell_face_z_m[0]
                    and input_position.z < cell_face_z_m[nz]
                )
                if not inside_half_open_domain:
                    ti.atomic_max(self._failure_code[None], 6)
                for other in range(marker + 1, marker_count):
                    other_region = marker_region_id[other]
                    other_active = (
                        other_region == primary_region_id
                        or other_region == secondary_region_id
                    )
                    if other_active:
                        relation = self._coincident_marker_constraint_relation(
                            marker_position_m,
                            marker_velocity_mps,
                            marker,
                            other,
                        )
                        if relation == 1:
                            ti.atomic_max(self._failure_code[None], 2)

    @ti.func
    def _component_is_free(
        self,
        component_face_valid_mask: ti.template(),
        hard_fixed_component_mask: ti.template(),
        external_exact_component_mask: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        axis: ti.i32,
    ):
        bit = 1 << axis
        return (
            (component_face_valid_mask[i, j, k] & bit) != 0
            and (hard_fixed_component_mask[i, j, k] & bit) == 0
            and (external_exact_component_mask[i, j, k] & bit) == 0
        )

    @ti.func
    def _component_inverse_mass_per_kg(
        self,
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        density_kgm3: ti.f32,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        axis: ti.i32,
    ):
        index = ti.Vector([i, j, k])
        widths = ti.Vector(
            [cell_width_x_m[i], cell_width_y_m[j], cell_width_z_m[k]]
        )
        normal_width = 0.0
        if axis == 0:
            normal_width = 0.5 * cell_width_x_m[0]
            if i > 0:
                normal_width = 0.5 * (
                    cell_width_x_m[i - 1] + cell_width_x_m[i]
                )
        elif axis == 1:
            normal_width = 0.5 * cell_width_y_m[0]
            if j > 0:
                normal_width = 0.5 * (
                    cell_width_y_m[j - 1] + cell_width_y_m[j]
                )
        else:
            normal_width = 0.5 * cell_width_z_m[0]
            if k > 0:
                normal_width = 0.5 * (
                    cell_width_z_m[k - 1] + cell_width_z_m[k]
                )
        dual_volume = normal_width
        for component in ti.static(range(3)):
            if component != axis:
                dual_volume *= widths[component]
        mass = density_kgm3 * dual_volume
        inverse_mass = 0.0
        if mass > 1.0e-30:
            inverse_mass = 1.0 / mass
        return inverse_mass

    @ti.kernel
    def _prepare_rows_kernel(
        self,
        marker_position_m: ti.template(),
        marker_input_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(),
        marker_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        velocity: ti.template(),
        component_face_valid_mask: ti.template(),
        hard_fixed_component_mask: ti.template(),
        external_exact_component_mask: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        density_kgm3: ti.f32,
    ):
        nx = ti.static(self.grid_nodes[0])
        ny = ti.static(self.grid_nodes[1])
        nz = ti.static(self.grid_nodes[2])
        for marker in range(marker_count):
            region = marker_region_id[marker]
            marker_active = (
                region == primary_region_id or region == secondary_region_id
            )
            self._marker_snapshot_active[marker] = 0
            if marker_active:
                self._marker_snapshot_active[marker] = 1
            self._marker_input_position_snapshot_m[marker] = (
                marker_input_position_m[marker]
            )
            self._marker_target_snapshot_mps[marker] = marker_velocity_mps[marker]
            self._marker_region_snapshot[marker] = region
            if marker_active:
                self._marker_position_snapshot_m[marker] = marker_position_m[marker]
            unique_constraint = marker_active
            if marker_active:
                for prior in range(marker):
                    prior_region = marker_region_id[prior]
                    prior_active = (
                        prior_region == primary_region_id
                        or prior_region == secondary_region_id
                    )
                    if prior_active:
                        relation = self._coincident_marker_constraint_relation(
                            marker_position_m,
                            marker_velocity_mps,
                            marker,
                            prior,
                        )
                        if relation == 2:
                            unique_constraint = False
            for axis in ti.static(range(3)):
                row = 3 * marker + axis
                if unique_constraint:
                    self._row_active[row] = 1
                    ti.atomic_add(self._device_constraint_count[None], 1)
                    position = marker_position_m[marker]
                    base, fraction = mac_component_stencil_base_fraction(
                        position,
                        axis,
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    sampled = 0.0
                    valid_weight = 0.0
                    free_diagonal = 0.0
                    for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                        i = base.x + oi
                        j = base.y + oj
                        k = base.z + ok
                        weight = mac_stencil_weight(fraction, oi, oj, ok)
                        bit = 1 << axis
                        if (component_face_valid_mask[i, j, k] & bit) != 0:
                            sampled += weight * velocity[i, j, k][axis]
                            valid_weight += weight
                    if valid_weight > 1.0e-12:
                        sampled /= valid_weight
                        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                            i = base.x + oi
                            j = base.y + oj
                            k = base.z + ok
                            support = 4 * oi + 2 * oj + ok
                            bit = 1 << axis
                            self._stencil_index[row, support] = ti.Vector([i, j, k])
                            self._support_velocity_snapshot_mps[row, support] = (
                                velocity[i, j, k][axis]
                            )
                            self._support_valid_mask_snapshot[row, support] = (
                                component_face_valid_mask[i, j, k]
                            )
                            self._support_hard_mask_snapshot[row, support] = (
                                hard_fixed_component_mask[i, j, k]
                            )
                            self._support_external_mask_snapshot[row, support] = (
                                external_exact_component_mask[i, j, k]
                            )
                            if (component_face_valid_mask[i, j, k] & bit) != 0:
                                self._stencil_weight[row, support] = (
                                    mac_stencil_weight(fraction, oi, oj, ok)
                                    / valid_weight
                                )
                            if self._component_is_free(
                                component_face_valid_mask,
                                hard_fixed_component_mask,
                                external_exact_component_mask,
                                i,
                                j,
                                k,
                                axis,
                            ):
                                weight = (
                                    mac_stencil_weight(fraction, oi, oj, ok)
                                    / valid_weight
                                )
                                inverse_mass = self._component_inverse_mass_per_kg(
                                    cell_width_x_m,
                                    cell_width_y_m,
                                    cell_width_z_m,
                                    density_kgm3,
                                    i,
                                    j,
                                    k,
                                    axis,
                                )
                                free_diagonal += weight * weight * inverse_mass
                                self._stencil_free[row, support] = 1
                                self._stencil_inverse_mass_per_kg[
                                    row, support
                                ] = inverse_mass
                    if valid_weight <= 1.0e-12:
                        ti.atomic_max(self._failure_code[None], 8)
                    self._rhs[row] = marker_velocity_mps[marker][axis] - sampled
                    self._diagonal[row] = free_diagonal

    @ti.kernel
    def _snapshot_geometry_kernel(
        self,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        density_kgm3: ti.f32,
    ):
        for index in self._cell_face_x_snapshot_m:
            self._cell_face_x_snapshot_m[index] = cell_face_x_m[index]
        for index in self._cell_face_y_snapshot_m:
            self._cell_face_y_snapshot_m[index] = cell_face_y_m[index]
        for index in self._cell_face_z_snapshot_m:
            self._cell_face_z_snapshot_m[index] = cell_face_z_m[index]
        for index in self._cell_center_x_snapshot_m:
            self._cell_center_x_snapshot_m[index] = cell_center_x_m[index]
            self._cell_width_x_snapshot_m[index] = cell_width_x_m[index]
        for index in self._cell_center_y_snapshot_m:
            self._cell_center_y_snapshot_m[index] = cell_center_y_m[index]
            self._cell_width_y_snapshot_m[index] = cell_width_y_m[index]
        for index in self._cell_center_z_snapshot_m:
            self._cell_center_z_snapshot_m[index] = cell_center_z_m[index]
            self._cell_width_z_snapshot_m[index] = cell_width_z_m[index]
        self._rho_snapshot_kgm3[None] = density_kgm3

    @ti.kernel
    def _snapshot_sampling_payload_kernel(
        self,
        sample_valid: ti.template(),
        sample_source_code: ti.template(),
        sample_invalid_reason_code: ti.template(),
        sample_position_m: ti.template(),
        marker_count: ti.i32,
    ):
        for marker in range(marker_count):
            self._sampling_valid_snapshot[marker] = sample_valid[marker]
            self._sampling_source_snapshot[marker] = sample_source_code[marker]
            self._sampling_invalid_reason_snapshot[marker] = (
                sample_invalid_reason_code[marker]
            )
            self._sampling_position_snapshot_m[marker] = sample_position_m[marker]

    @ti.kernel
    def _reset_audit_kernel(self):
        self._audit_failure_code[None] = 0

    @ti.kernel
    def _audit_sampling_payload_kernel(
        self,
        sample_valid: ti.template(),
        sample_source_code: ti.template(),
        sample_invalid_reason_code: ti.template(),
        sample_position_m: ti.template(),
        marker_count: ti.i32,
    ):
        for marker in range(marker_count):
            mismatch = (
                sample_valid[marker] != self._sampling_valid_snapshot[marker]
                or sample_source_code[marker]
                != self._sampling_source_snapshot[marker]
                or sample_invalid_reason_code[marker]
                != self._sampling_invalid_reason_snapshot[marker]
            )
            for axis in ti.static(range(3)):
                mismatch = mismatch or (
                    sample_position_m[marker][axis]
                    != self._sampling_position_snapshot_m[marker][axis]
                )
            if mismatch:
                ti.atomic_max(self._audit_failure_code[None], 4)

    @ti.kernel
    def _audit_transaction_kernel(
        self,
        marker_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(),
        marker_count: ti.i32,
        velocity: ti.template(),
        component_face_valid_mask: ti.template(),
        hard_fixed_component_mask: ti.template(),
        external_exact_component_mask: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        density_kgm3: ti.f32,
    ):
        for marker in range(marker_count):
            region_mismatch = (
                marker_region_id[marker] != self._marker_region_snapshot[marker]
            )
            if region_mismatch:
                ti.atomic_max(self._audit_failure_code[None], 1)
            if self._marker_snapshot_active[marker] != 0:
                mismatch = False
                for axis in ti.static(range(3)):
                    mismatch = mismatch or (
                        marker_position_m[marker][axis]
                        != self._marker_input_position_snapshot_m[marker][axis]
                    )
                    mismatch = mismatch or (
                        marker_velocity_mps[marker][axis]
                        != self._marker_target_snapshot_mps[marker][axis]
                    )
                if mismatch:
                    ti.atomic_max(self._audit_failure_code[None], 1)
        for row, support in self._stencil_weight:
            if self._row_active[row] != 0:
                index = self._stencil_index[row, support]
                axis = row % 3
                if index.x < 0:
                    ti.atomic_max(self._audit_failure_code[None], 2)
                else:
                    mismatch = (
                        velocity[index.x, index.y, index.z][axis]
                        != self._support_velocity_snapshot_mps[row, support]
                    )
                    mismatch = mismatch or (
                        component_face_valid_mask[index.x, index.y, index.z]
                        != self._support_valid_mask_snapshot[row, support]
                    )
                    mismatch = mismatch or (
                        hard_fixed_component_mask[index.x, index.y, index.z]
                        != self._support_hard_mask_snapshot[row, support]
                    )
                    mismatch = mismatch or (
                        external_exact_component_mask[index.x, index.y, index.z]
                        != self._support_external_mask_snapshot[row, support]
                    )
                    if mismatch:
                        ti.atomic_max(self._audit_failure_code[None], 2)
        for index in self._cell_face_x_snapshot_m:
            if self._cell_face_x_snapshot_m[index] != cell_face_x_m[index]:
                ti.atomic_max(self._audit_failure_code[None], 3)
        for index in self._cell_face_y_snapshot_m:
            if self._cell_face_y_snapshot_m[index] != cell_face_y_m[index]:
                ti.atomic_max(self._audit_failure_code[None], 3)
        for index in self._cell_face_z_snapshot_m:
            if self._cell_face_z_snapshot_m[index] != cell_face_z_m[index]:
                ti.atomic_max(self._audit_failure_code[None], 3)
        for index in self._cell_center_x_snapshot_m:
            if (
                self._cell_center_x_snapshot_m[index] != cell_center_x_m[index]
                or self._cell_width_x_snapshot_m[index] != cell_width_x_m[index]
            ):
                ti.atomic_max(self._audit_failure_code[None], 3)
        for index in self._cell_center_y_snapshot_m:
            if (
                self._cell_center_y_snapshot_m[index] != cell_center_y_m[index]
                or self._cell_width_y_snapshot_m[index] != cell_width_y_m[index]
            ):
                ti.atomic_max(self._audit_failure_code[None], 3)
        for index in self._cell_center_z_snapshot_m:
            if (
                self._cell_center_z_snapshot_m[index] != cell_center_z_m[index]
                or self._cell_width_z_snapshot_m[index] != cell_width_z_m[index]
            ):
                ti.atomic_max(self._audit_failure_code[None], 3)
        if self._rho_snapshot_kgm3[None] != density_kgm3:
            self._audit_failure_code[None] = 3

    @ti.kernel
    def _initialize_pcg_kernel(self, tolerance: ti.f32):
        self._max_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            self._row_pcg_active[row] = 0
            if self._row_active[row] != 0:
                residual = self._rhs[row]
                self._residual[row] = residual
                diagonal = self._diagonal[row]
                value = 0.0
                if diagonal > 1.0e-20:
                    self._row_pcg_active[row] = 1
                    value = residual / diagonal
                elif ti.abs(residual) > tolerance:
                    ti.atomic_max(self._failure_code[None], 3)
                self._preconditioned[row] = value
                self._direction[row] = value
                ti.atomic_max(self._max_residual[None], ti.abs(residual))

    @ti.kernel
    def _check_convergence_kernel(self, tolerance: ti.f32):
        if self._failure_code[None] == 0 and self._max_residual[None] <= tolerance:
            self._device_converged[None] = 1

    @ti.kernel
    def _compute_initial_rz_kernel(self):
        self._rz_old[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._row_pcg_active[row] != 0:
                ti.atomic_add(
                    self._rz_old[None],
                    ti.cast(self._residual[row], ti.f64)
                    * ti.cast(self._preconditioned[row], ti.f64),
                )

    @ti.kernel
    def _compute_p_ap_kernel(self):
        self._p_ap[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._row_pcg_active[row] != 0:
                ti.atomic_add(
                    self._p_ap[None],
                    ti.cast(self._direction[row], ti.f64)
                    * ti.cast(self._matrix_direction[row], ti.f64),
                )

    @ti.kernel
    def _reset_iteration_residual_kernel(self):
        if self._device_converged[None] == 0 and self._failure_code[None] == 0:
            self._max_residual[None] = 0.0

    @ti.kernel
    def _pcg_step_device_kernel(self):
        if self._device_converged[None] == 0 and self._failure_code[None] == 0:
            denominator = self._p_ap[None]
            numerator = self._rz_old[None]
            finite = denominator == denominator and numerator == numerator
            finite = finite and ti.abs(denominator) < 1.0e300
            finite = finite and ti.abs(numerator) < 1.0e300
            if not finite or denominator <= 1.0e-30:
                self._failure_code[None] = 4
            else:
                alpha = ti.cast(numerator / denominator, ti.f32)
                ti.atomic_add(self._device_iterations[None], 1)
                for row in range(self.constraint_capacity):
                    if self._row_pcg_active[row] != 0:
                        self._lambda[row] += alpha * self._direction[row]
                        self._residual[row] -= alpha * self._matrix_direction[row]
                        self._preconditioned[row] = (
                            self._residual[row] / self._diagonal[row]
                        )
                        ti.atomic_max(
                            self._max_residual[None],
                            ti.abs(self._residual[row]),
                        )

    @ti.kernel
    def _pcg_update_direction_device_kernel(self):
        if self._device_converged[None] == 0 and self._failure_code[None] == 0:
            self._rz_new[None] = 0.0
            for row in range(self.constraint_capacity):
                if self._row_pcg_active[row] != 0:
                    ti.atomic_add(
                        self._rz_new[None],
                        ti.cast(self._residual[row], ti.f64)
                        * ti.cast(self._preconditioned[row], ti.f64),
                    )

    @ti.kernel
    def _pcg_finish_direction_device_kernel(self):
        if self._device_converged[None] == 0 and self._failure_code[None] == 0:
            old_value = self._rz_old[None]
            new_value = self._rz_new[None]
            finite = old_value == old_value and new_value == new_value
            finite = finite and ti.abs(old_value) < 1.0e300
            finite = finite and ti.abs(new_value) < 1.0e300
            if not finite or old_value <= 0.0:
                self._failure_code[None] = 5
            else:
                beta = ti.cast(new_value / old_value, ti.f32)
                for row in range(self.constraint_capacity):
                    if self._row_pcg_active[row] != 0:
                        self._direction[row] = self._preconditioned[row] + (
                            beta * self._direction[row]
                        )
                self._rz_old[None] = new_value

    @ti.kernel
    def _clear_grid_scratch_kernel(self, force_run: ti.i32):
        for i, j, k in self._grid_scratch:
            if force_run != 0 or (
                self._device_converged[None] == 0
                and self._failure_code[None] == 0
            ):
                self._grid_scratch[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _scatter_rows_to_grid_kernel(
        self,
        row_values: ti.template(),
        force_run: ti.i32,
    ):
        for row, support in self._stencil_weight:
            if force_run != 0 or (
                self._device_converged[None] == 0
                and self._failure_code[None] == 0
            ):
                if (
                    self._row_pcg_active[row] != 0
                    and self._stencil_free[row, support] != 0
                ):
                    index = self._stencil_index[row, support]
                    axis = row % 3
                    ti.atomic_add(
                        self._grid_scratch[index.x, index.y, index.z][axis],
                        self._stencil_weight[row, support]
                        * self._stencil_inverse_mass_per_kg[row, support]
                        * row_values[row],
                    )

    @ti.kernel
    def _gather_grid_to_rows_kernel(
        self,
        output_rows: ti.template(),
        force_run: ti.i32,
    ):
        for row in range(self.constraint_capacity):
            if force_run != 0 or (
                self._device_converged[None] == 0
                and self._failure_code[None] == 0
            ):
                value = 0.0
                if self._row_pcg_active[row] != 0:
                    axis = row % 3
                    for support in ti.static(range(8)):
                        index = self._stencil_index[row, support]
                        if index.x >= 0:
                            value += (
                                self._stencil_weight[row, support]
                                * self._grid_scratch[index.x, index.y, index.z][axis]
                            )
                output_rows[row] = value

    @ti.kernel
    def _copy_grid_scratch_to_correction_kernel(self):
        for i, j, k in self._correction:
            self._correction[i, j, k] = self._grid_scratch[i, j, k]

    @ti.kernel
    def _snapshot_solved_correction_kernel(self):
        for i, j, k in self._correction:
            self._solved_correction_snapshot[i, j, k] = self._correction[i, j, k]

    @ti.kernel
    def _audit_solved_correction_integrity_kernel(self):
        self._solved_correction_integrity_failure[None] = 0
        for i, j, k in self._correction:
            current = self._correction[i, j, k]
            solved = self._solved_correction_snapshot[i, j, k]
            for axis in ti.static(range(3)):
                current_finite = current[axis] == current[axis]
                current_finite = current_finite and ti.abs(current[axis]) < 3.4e38
                solved_finite = solved[axis] == solved[axis]
                solved_finite = solved_finite and ti.abs(solved[axis]) < 3.4e38
                if (
                    not current_finite
                    or not solved_finite
                    or current[axis] != solved[axis]
                ):
                    ti.atomic_max(
                        self._solved_correction_integrity_failure[None],
                        1,
                    )

    @ti.kernel
    def _compute_true_candidate_residual_kernel(self):
        """Measure ``rhs - J correction`` for the exact pending commit."""

        self._true_candidate_max_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._row_active[row] != 0:
                axis = row % 3
                sampled_correction = 0.0
                for support in ti.static(range(8)):
                    if self._stencil_free[row, support] != 0:
                        index = self._stencil_index[row, support]
                        sampled_correction += (
                            self._stencil_weight[row, support]
                            * self._correction[index.x, index.y, index.z][axis]
                        )
                true_residual = ti.abs(self._rhs[row] - sampled_correction)
                finite = true_residual == true_residual
                finite = finite and true_residual < 3.4e38
                if not finite:
                    true_residual = 3.4e38
                ti.atomic_max(
                    self._true_candidate_max_residual[None],
                    true_residual,
                )

    @ti.kernel
    def _commit_kernel(
        self,
        velocity: ti.template(),
        hard_fixed_component_mask: ti.template(),
        external_exact_component_mask: ti.template(),
    ):
        for i, j, k in velocity:
            value = velocity[i, j, k]
            hard = hard_fixed_component_mask[i, j, k]
            external = external_exact_component_mask[i, j, k]
            for axis in ti.static(range(3)):
                if ((hard | external) & (1 << axis)) == 0:
                    value[axis] += self._correction[i, j, k][axis]
            velocity[i, j, k] = value

    def _apply_matrix(
        self,
        input_rows,
        output_rows,
        *,
        force_run: bool,
    ) -> None:
        device_force_run = 1 if force_run else 0
        self._clear_grid_scratch_kernel(device_force_run)
        self._scatter_rows_to_grid_kernel(input_rows, device_force_run)
        self._gather_grid_to_rows_kernel(output_rows, device_force_run)

    def _clear_pressure_nullspace_lifecycle(self) -> None:
        """Invalidate owners/generations while retaining opt-in allocations."""

        self._pressure_nullspace_prepared = False
        self._pressure_nullspace_poisoned = False
        self._pressure_nullspace_apply_count = 0
        self._pressure_nullspace_fluid = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuated_component_mobility = None
        self._pressure_actuation_generation = 0
        self._pressure_nullspace_topology_generation = 0
        self._pressure_nullspace_component_face_valid_mask_generation = 0

    def _poison_pressure_nullspace_transaction(self) -> None:
        """Make a failed pressure transaction impossible to reuse.

        Device diagnostics and ``apply_count`` are intentionally retained for
        a debugger.  A new ordinary affine-Q prepare is required before the
        pressure factor can be published again.
        """

        self._pressure_nullspace_prepared = False
        self._pressure_nullspace_poisoned = True
        self._pressure_nullspace_fluid = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuated_component_mobility = None
        self._pressure_actuation_generation = 0
        self._pressure_nullspace_topology_generation = 0
        self._pressure_nullspace_component_face_valid_mask_generation = 0
        self._phase = "failed"
        self._prepared = False
        self._converged = False
        self._committed = False

    @staticmethod
    def _validate_pressure_nullspace_generation(
        name: str,
        value: int,
    ) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative integer")
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be a non-negative integer"
            ) from exc
        if normalized != value or normalized < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return normalized

    def _validate_pressure_nullspace_python_transaction(
        self,
        *,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> None:
        """Validate immutable Python identities without synchronizing Taichi."""

        if self._pressure_nullspace_poisoned:
            raise RuntimeError("pressure constraint nullspace transaction is poisoned")
        if not self._pressure_nullspace_prepared:
            raise RuntimeError(
                "pressure constraint nullspace transaction is not prepared"
            )
        if fluid is not self._pressure_nullspace_fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        if (
            pressure_actuated_component_mobility
            is not self._pressure_actuated_component_mobility
        ):
            raise RuntimeError("pressure actuation weight owner changed")
        if (
            component_face_valid_mask
            is not self._pressure_nullspace_component_face_valid_mask
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if int(pressure_actuation_generation) != int(
            self._pressure_actuation_generation
        ):
            raise RuntimeError("pressure actuation generation changed")
        if int(topology_generation) != int(
            self._pressure_nullspace_topology_generation
        ):
            raise RuntimeError("pressure nullspace topology generation changed")
        if int(component_face_valid_mask_generation) != int(
            self._pressure_nullspace_component_face_valid_mask_generation
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask generation changed"
            )
        if int(fluid.velocity_dirichlet_component_ledger_generation) != int(
            self._prepared_ledger_generation
        ):
            raise RuntimeError(
                "pressure nullspace velocity ledger generation changed"
            )
        if not self._committed or self._phase != "committed":
            raise RuntimeError(
                "ordinary affine marker Q transaction must be committed before "
                "pressure nullspace apply"
            )

    def _validate_pressure_nullspace_vector_fields(
        self,
        *,
        input_face_correction,
        output_face_correction,
        fluid,
    ) -> None:
        if tuple(input_face_correction.shape) != self.grid_nodes:
            raise ValueError("input_face_correction shape does not match grid_nodes")
        if tuple(output_face_correction.shape) != self.grid_nodes:
            raise ValueError("output_face_correction shape does not match grid_nodes")
        if getattr(input_face_correction, "dtype", None) not in (ti.f32, ti.f64):
            raise ValueError(
                "pressure nullspace input_face_correction must use f32 or f64 storage"
            )
        if output_face_correction is fluid.velocity:
            raise RuntimeError(
                "pressure nullspace transaction must not write fluid.velocity"
            )
        if getattr(output_face_correction, "dtype", None) != ti.f64:
            raise ValueError(
                "pressure nullspace output_face_correction must use f64 storage"
            )

    def _require_pressure_nullspace_python_transaction(
        self,
        *,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> tuple[int, int, int]:
        generations = (
            self._validate_pressure_nullspace_generation(
                "pressure_actuation_generation",
                pressure_actuation_generation,
            ),
            self._validate_pressure_nullspace_generation(
                "topology_generation",
                topology_generation,
            ),
            self._validate_pressure_nullspace_generation(
                "component_face_valid_mask_generation",
                component_face_valid_mask_generation,
            ),
        )
        try:
            self._validate_pressure_nullspace_python_transaction(
                fluid=fluid,
                pressure_actuated_component_mobility=(
                    pressure_actuated_component_mobility
                ),
                component_face_valid_mask=component_face_valid_mask,
                pressure_actuation_generation=generations[0],
                topology_generation=generations[1],
                component_face_valid_mask_generation=generations[2],
            )
        except RuntimeError:
            self._poison_pressure_nullspace_transaction()
            raise
        return generations

    def _enqueue_pressure_nullspace_device_apply(
        self,
        *,
        input_face_correction,
    ) -> None:
        """Launch one projector apply without reading a device scalar."""

        self._gather_pressure_nullspace_rhs_kernel(input_face_correction)
        self._solve_pressure_nullspace_factor_kernel()
        self._clear_pressure_nullspace_candidate_kernel()
        self._scatter_pressure_nullspace_correction_kernel()
        self._build_pressure_nullspace_candidate_kernel(input_face_correction)
        self._measure_pressure_nullspace_residual_kernel()

    def _finalize_pressure_nullspace_device_state(
        self,
        *,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        absolute_tolerance_mps: float,
    ) -> HibmMpmMarkerPressureNullspaceReport:
        """Perform the sole solve-end device audit and host synchronization."""

        self._audit_pressure_nullspace_support_kernel(
            pressure_actuated_component_mobility,
            component_face_valid_mask,
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            fluid.velocity_dirichlet_boundary_external_exact_component_mask,
        )
        failure_code = int(self._pressure_nullspace_failure_code[None])
        max_residual = float(
            self._pressure_nullspace_max_constraint_residual[None]
        )
        max_unactuated_input = float(
            self._pressure_nullspace_max_unactuated_input_constraint[None]
        )
        failure_message = ""
        if failure_code == 4:
            failure_message = (
                "pressure mobility/actuation weight changed after nullspace prepare"
            )
        elif failure_code == 5:
            failure_message = (
                "pressure nullspace cached component-face inputs changed"
            )
        elif failure_code == 8:
            failure_message = (
                "pressure nullspace input and candidate must be finite"
            )
        elif failure_code != 0:
            failure_message = "pressure nullspace device audit failed"
        elif not math.isfinite(max_unactuated_input):
            failure_message = (
                "pressure marker unactuated constraint input must be finite"
            )
        elif max_unactuated_input > absolute_tolerance_mps:
            failure_message = (
                "pressure marker unactuated constraint input is incompatible: "
                f"{max_unactuated_input} > {absolute_tolerance_mps}"
            )
        elif not math.isfinite(max_residual):
            failure_message = "pressure marker nullspace residual must be finite"
        elif max_residual > absolute_tolerance_mps:
            failure_message = (
                "pressure marker nullspace candidate exceeds absolute tolerance: "
                f"{max_residual} > {absolute_tolerance_mps}"
            )
        if failure_message:
            self._poison_pressure_nullspace_transaction()
            raise RuntimeError(failure_message)
        return self.pressure_nullspace_report()

    def _apply_pressure_nullspace_transaction_immediate(
        self,
        *,
        input_face_correction,
        output_face_correction,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
        absolute_tolerance_mps: float,
    ) -> HibmMpmMarkerPressureNullspaceReport:
        """Compatibility apply with atomic output publication."""

        self._require_pressure_nullspace_python_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=pressure_actuation_generation,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=(
                component_face_valid_mask_generation
            ),
        )
        self._validate_pressure_nullspace_vector_fields(
            input_face_correction=input_face_correction,
            output_face_correction=output_face_correction,
            fluid=fluid,
        )
        self._enqueue_pressure_nullspace_device_apply(
            input_face_correction=input_face_correction,
        )
        self._pressure_nullspace_apply_count += 1
        report = self._finalize_pressure_nullspace_device_state(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            absolute_tolerance_mps=absolute_tolerance_mps,
        )
        self._commit_pressure_nullspace_candidate_kernel(output_face_correction)
        return report

    def _ensure_pressure_nullspace_resources(self) -> None:
        """Lazily allocate the bounded dense pressure projector resources."""

        if self._pressure_nullspace_resources_allocated:
            return
        constraints = int(self.constraint_capacity)
        dense_bytes = constraints * constraints * 8
        grid_cells = math.prod(int(value) for value in self.grid_nodes)
        # Two f64 vector grids plus marker support/factor/work storage.  This is
        # an honest upper estimate for fail-fast purposes, not an allocator
        # accounting API.
        estimated_bytes = (
            2 * dense_bytes
            + 2 * grid_cells * 3 * 8
            + 2 * constraints * 8 * 8
            + 5 * constraints * 8
            + 3 * constraints * 4
            + 10 * 8
        )
        if constraints > HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS:
            raise RuntimeError(
                "pressure marker dense nullspace capacity exceeds the exact "
                "backend limit: "
                f"{constraints} > "
                f"{HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS}"
            )
        if dense_bytes > HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_BYTES:
            raise RuntimeError(
                "pressure marker dense nullspace factor exceeds memory budget"
            )
        if estimated_bytes > HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES:
            raise RuntimeError(
                "pressure marker nullspace resources exceed memory budget: "
                f"{estimated_bytes} > "
                f"{HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES} bytes"
            )

        shape = self.grid_nodes
        self._pressure_nullspace_row_active = ti.field(
            dtype=ti.i32,
            shape=constraints,
        )
        self._pressure_nullspace_mobility_snapshot = ti.field(
            dtype=ti.f64,
            shape=(constraints, 8),
        )
        self._pressure_nullspace_inverse_mass_per_kg = ti.field(
            dtype=ti.f64,
            shape=(constraints, 8),
        )
        self._pressure_nullspace_schur = ti.field(
            dtype=ti.f64,
            shape=(constraints, constraints),
        )
        self._pressure_nullspace_factor = ti.field(
            dtype=ti.f64,
            shape=(constraints, constraints),
        )
        self._pressure_nullspace_row_inverse_norm = ti.field(
            dtype=ti.f64,
            shape=constraints,
        )
        self._pressure_nullspace_factor_row_selected = ti.field(
            dtype=ti.i32,
            shape=constraints,
        )
        self._pressure_nullspace_factor_order = ti.field(
            dtype=ti.i32,
            shape=constraints,
        )
        self._pressure_nullspace_rhs = ti.field(dtype=ti.f64, shape=constraints)
        self._pressure_nullspace_forward = ti.field(
            dtype=ti.f64,
            shape=constraints,
        )
        self._pressure_nullspace_lambda = ti.field(
            dtype=ti.f64,
            shape=constraints,
        )
        self._pressure_nullspace_correction = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=shape,
        )
        self._pressure_nullspace_candidate = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=shape,
        )
        self._pressure_nullspace_failure_code = ti.field(dtype=ti.i32, shape=())
        self._pressure_nullspace_active_constraint_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._pressure_nullspace_independent_constraint_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._pressure_nullspace_dependent_constraint_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._pressure_nullspace_unactuated_constraint_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._pressure_nullspace_min_factor_pivot = ti.field(
            dtype=ti.f64,
            shape=(),
        )
        self._pressure_nullspace_max_dependent_normalized_pivot = ti.field(
            dtype=ti.f64,
            shape=(),
        )
        self._pressure_nullspace_max_input_constraint = ti.field(
            dtype=ti.f64,
            shape=(),
        )
        self._pressure_nullspace_max_unactuated_input_constraint = ti.field(
            dtype=ti.f64,
            shape=(),
        )
        self._pressure_nullspace_max_constraint_residual = ti.field(
            dtype=ti.f64,
            shape=(),
        )
        self._pressure_nullspace_resource_bytes = int(estimated_bytes)
        self._pressure_nullspace_resources_allocated = True

    def _audit_pressure_nullspace_transaction_inputs(
        self,
        *,
        fluid,
        pressure_actuation_weight,
        component_face_valid_mask,
    ) -> None:
        if not self._pressure_nullspace_prepared:
            raise RuntimeError(
                "pressure constraint nullspace transaction is not prepared"
            )
        if fluid is not self._pressure_nullspace_fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        if pressure_actuation_weight is not self._pressure_actuated_component_mobility:
            raise RuntimeError("pressure actuation weight owner changed")
        if (
            component_face_valid_mask
            is not self._pressure_nullspace_component_face_valid_mask
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask owner changed"
            )
        if int(fluid.velocity_dirichlet_component_ledger_generation) != int(
            self._prepared_ledger_generation
        ):
            raise RuntimeError(
                "pressure nullspace velocity ledger generation changed"
            )
        self._reset_pressure_nullspace_audit_kernel()
        self._audit_pressure_nullspace_support_kernel(
            pressure_actuation_weight,
            component_face_valid_mask,
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            fluid.velocity_dirichlet_boundary_external_exact_component_mask,
        )
        failure_code = int(self._pressure_nullspace_failure_code[None])
        if failure_code == 4:
            raise RuntimeError(
                "pressure mobility/actuation weight changed after nullspace prepare"
            )
        if failure_code == 5:
            raise RuntimeError(
                "pressure nullspace cached component-face inputs changed"
            )
        if failure_code != 0:
            raise RuntimeError("pressure nullspace cached inputs changed")

    def prepare_pressure_constraint_nullspace(
        self,
        *,
        pressure_actuation_weight,
        component_face_valid_mask,
    ) -> None:
        """Factor the exact homogeneous marker projector once.

        ``pressure_actuation_weight`` is the complete non-negative diagonal
        ``A=R^-1`` in velocity units.  The fluid operator must materialize the
        same pressure mobility/incidence and inverse dual mass that its raw
        pressure-gradient path uses.  This method deliberately does *not*
        derive or multiply the ordinary affine-Q inverse-mass metric.
        """

        if not self._prepared or self._fluid is None:
            raise RuntimeError(
                "ordinary marker transaction must be prepared before pressure nullspace"
            )
        if self._pressure_nullspace_prepared:
            raise RuntimeError(
                "pressure constraint nullspace transaction is already prepared"
            )
        if tuple(pressure_actuation_weight.shape) != self.grid_nodes:
            raise ValueError(
                "pressure_actuation_weight shape does not match grid_nodes"
            )
        if tuple(component_face_valid_mask.shape) != self.grid_nodes:
            raise ValueError(
                "component_face_valid_mask shape does not match grid_nodes"
            )
        if component_face_valid_mask is not self._component_face_valid_mask:
            raise RuntimeError(
                "pressure nullspace must use the ordinary marker transaction valid mask"
            )

        # From this point onward any failure must leave no half-prepared owner
        # or generation token.  Lazy resources, once explicitly requested, may
        # be retained and reused by the next valid affine transaction.
        self._clear_pressure_nullspace_lifecycle()
        self._ensure_pressure_nullspace_resources()
        self._reset_pressure_nullspace_prepare_kernel()
        self._snapshot_pressure_nullspace_mobility_kernel(
            pressure_actuation_weight,
        )
        failure_code = int(self._pressure_nullspace_failure_code[None])
        if failure_code == 1:
            raise RuntimeError(
                "pressure actuation weight must be finite and non-negative"
            )
        if failure_code == 6:
            raise RuntimeError(
                "pressure actuation weight must be zero on hard-fixed marker support"
            )
        if failure_code == 7:
            raise RuntimeError(
                "pressure actuation weight must be zero on external-exact marker support"
            )
        self._assemble_pressure_nullspace_schur_kernel()
        self._symmetrize_pressure_nullspace_schur_kernel()
        failure_code = int(self._pressure_nullspace_failure_code[None])
        if failure_code == 3:
            raise RuntimeError(
                "pressure actuation weight is inconsistent on shared marker support"
            )
        relative_pivot_tolerance = max(
            1.0e-14,
            64.0 * math.ulp(1.0) * float(self.marker_capacity),
        )
        self._factor_pressure_nullspace_schur_kernel(
            relative_pivot_tolerance,
        )
        failure_code = int(self._pressure_nullspace_failure_code[None])
        if failure_code == 2:
            raise RuntimeError(
                "pressure marker Schur complement is not positive semidefinite"
            )
        if failure_code != 0:
            raise RuntimeError("pressure marker Schur factorization failed")
        active_count = int(
            self._pressure_nullspace_active_constraint_count[None]
        )
        independent_count = int(
            self._pressure_nullspace_independent_constraint_count[None]
        )
        dependent_count = int(
            self._pressure_nullspace_dependent_constraint_count[None]
        )
        unactuated_count = int(
            self._pressure_nullspace_unactuated_constraint_count[None]
        )
        if (
            independent_count < 0
            or dependent_count < 0
            or unactuated_count < 0
            or independent_count + dependent_count + unactuated_count
            != active_count
        ):
            raise RuntimeError(
                "pressure marker rank partition is inconsistent: "
                f"active={active_count}, independent={independent_count}, "
                f"dependent={dependent_count}, unactuated={unactuated_count}"
            )

        self._pressure_nullspace_fluid = self._fluid
        self._pressure_nullspace_component_face_valid_mask = (
            component_face_valid_mask
        )
        self._pressure_actuated_component_mobility = pressure_actuation_weight
        self._pressure_actuation_generation += 1
        self._pressure_nullspace_topology_generation = int(
            self._prepared_topology_generation
        )
        self._pressure_nullspace_component_face_valid_mask_generation = int(
            self._prepared_component_face_valid_mask_generation
        )
        self._pressure_nullspace_apply_count = 0
        self._pressure_nullspace_prepared = True

    def project_pressure_actuated_grid_vector_to_marker_nullspace(
        self,
        *,
        input_velocity_mps,
        output_velocity_mps,
        max_iterations: int,
        absolute_tolerance_mps: float,
        component_face_valid_mask,
    ) -> HibmMpmMarkerPressureNullspaceReport:
        """Apply one reusable, direct-factor homogeneous projection.

        ``max_iterations`` remains in the generic API so an iterative backend
        can be substituted without changing callers.  This implementation is
        intentionally a prepared f64 Cholesky solve: it is linear across outer
        FV-CG matvecs and therefore performs no input-dependent inner stopping.
        """

        if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if int(max_iterations) != max_iterations:
            raise ValueError("max_iterations must be a positive integer")
        if isinstance(absolute_tolerance_mps, bool):
            raise ValueError(
                "absolute_tolerance_mps must be finite and positive"
            )
        tolerance = float(absolute_tolerance_mps)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "absolute_tolerance_mps must be finite and positive"
            )
        fluid = self._pressure_nullspace_fluid
        return self._apply_pressure_nullspace_transaction_immediate(
            input_face_correction=input_velocity_mps,
            output_face_correction=output_velocity_mps,
            fluid=fluid,
            pressure_actuated_component_mobility=(
                self._pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=int(
                self._pressure_actuation_generation
            ),
            topology_generation=int(
                self._pressure_nullspace_topology_generation
            ),
            component_face_valid_mask_generation=int(
                self._pressure_nullspace_component_face_valid_mask_generation
            ),
            absolute_tolerance_mps=tolerance,
        )

    def prepare_pressure_nullspace_transaction(
        self,
        *,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> None:
        """Generation-explicit adapter for a fluid-owned pressure protocol.

        Despite the compatibility argument name, the supplied vector is the
        complete actuation weight ``A`` rather than a dimensionless mobility.
        """

        if fluid is not self._fluid:
            raise RuntimeError("pressure nullspace fluid owner changed")
        for name, value in (
            ("pressure_actuation_generation", pressure_actuation_generation),
            ("topology_generation", topology_generation),
            (
                "component_face_valid_mask_generation",
                component_face_valid_mask_generation,
            ),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if int(topology_generation) != int(self._prepared_topology_generation):
            raise RuntimeError("pressure nullspace topology generation changed")
        if int(component_face_valid_mask_generation) != int(
            self._prepared_component_face_valid_mask_generation
        ):
            raise RuntimeError(
                "pressure nullspace component-face valid-mask generation changed"
            )
        self.prepare_pressure_constraint_nullspace(
            pressure_actuation_weight=pressure_actuated_component_mobility,
            component_face_valid_mask=component_face_valid_mask,
        )
        self._pressure_actuation_generation = int(pressure_actuation_generation)

    def apply_pressure_nullspace_transaction_device_only(
        self,
        *,
        input_face_correction,
        output_face_correction,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> None:
        """Queue one prepared projection without a device-to-host scalar read.

        The output is private pressure-solver scratch.  Device failures and
        solve-wide maxima accumulate until
        :meth:`finalize_pressure_nullspace_transaction` is called before any
        projected correction is committed to physical ``fluid.velocity``.
        """

        self._require_pressure_nullspace_python_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=pressure_actuation_generation,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=(
                component_face_valid_mask_generation
            ),
        )
        self._validate_pressure_nullspace_vector_fields(
            input_face_correction=input_face_correction,
            output_face_correction=output_face_correction,
            fluid=fluid,
        )
        self._enqueue_pressure_nullspace_device_apply(
            input_face_correction=input_face_correction,
        )
        self._commit_pressure_nullspace_candidate_kernel(output_face_correction)
        self._pressure_nullspace_apply_count += 1

    def finalize_pressure_nullspace_transaction(
        self,
        *,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
        absolute_tolerance_mps: float,
    ) -> HibmMpmMarkerPressureNullspaceReport:
        """Audit all queued applies once and poison any failed transaction."""

        if isinstance(absolute_tolerance_mps, bool):
            raise ValueError(
                "absolute_tolerance_mps must be finite and positive"
            )
        tolerance = float(absolute_tolerance_mps)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "absolute_tolerance_mps must be finite and positive"
            )
        self._require_pressure_nullspace_python_transaction(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=pressure_actuation_generation,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=(
                component_face_valid_mask_generation
            ),
        )
        return self._finalize_pressure_nullspace_device_state(
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            absolute_tolerance_mps=tolerance,
        )

    def apply_pressure_nullspace_transaction(
        self,
        *,
        input_face_correction,
        output_face_correction,
        fluid,
        pressure_actuated_component_mobility,
        component_face_valid_mask,
        pressure_actuation_generation: int,
        topology_generation: int,
        component_face_valid_mask_generation: int,
    ) -> HibmMpmMarkerPressureNullspaceReport:
        """Compatibility apply with one audit and atomic output publication."""

        return self._apply_pressure_nullspace_transaction_immediate(
            input_face_correction=input_face_correction,
            output_face_correction=output_face_correction,
            fluid=fluid,
            pressure_actuated_component_mobility=(
                pressure_actuated_component_mobility
            ),
            component_face_valid_mask=component_face_valid_mask,
            pressure_actuation_generation=pressure_actuation_generation,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=(
                component_face_valid_mask_generation
            ),
            absolute_tolerance_mps=2.0e-12,
        )

    def pressure_nullspace_report(
        self,
    ) -> HibmMpmMarkerPressureNullspaceReport:
        """Return scalar-only diagnostics without downloading grid fields."""

        if (
            not self._pressure_nullspace_resources_allocated
            or not self._pressure_nullspace_prepared
        ):
            return HibmMpmMarkerPressureNullspaceReport(
                prepared=False,
                active_constraint_count=0,
                apply_count=0,
                pressure_actuation_generation=0,
                min_factor_pivot=0.0,
                last_max_input_constraint=0.0,
                last_max_constraint_residual=0.0,
                resource_bytes=0,
                independent_constraint_count=0,
                dependent_constraint_count=0,
                unactuated_constraint_count=0,
                max_dependent_normalized_pivot=0.0,
                max_unactuated_input_constraint=0.0,
            )
        active_count = int(
            self._pressure_nullspace_active_constraint_count[None]
        )
        independent_count = int(
            self._pressure_nullspace_independent_constraint_count[None]
        )
        min_pivot = (
            float(self._pressure_nullspace_min_factor_pivot[None])
            if independent_count > 0
            else 0.0
        )
        return HibmMpmMarkerPressureNullspaceReport(
            prepared=bool(self._pressure_nullspace_prepared),
            active_constraint_count=active_count,
            apply_count=int(self._pressure_nullspace_apply_count),
            pressure_actuation_generation=int(
                self._pressure_actuation_generation
            ),
            min_factor_pivot=min_pivot,
            last_max_input_constraint=float(
                self._pressure_nullspace_max_input_constraint[None]
            ),
            last_max_constraint_residual=float(
                self._pressure_nullspace_max_constraint_residual[None]
            ),
            resource_bytes=int(self._pressure_nullspace_resource_bytes),
            independent_constraint_count=independent_count,
            dependent_constraint_count=int(
                self._pressure_nullspace_dependent_constraint_count[None]
            ),
            unactuated_constraint_count=int(
                self._pressure_nullspace_unactuated_constraint_count[None]
            ),
            max_dependent_normalized_pivot=float(
                self._pressure_nullspace_max_dependent_normalized_pivot[None]
            ),
            max_unactuated_input_constraint=float(
                self._pressure_nullspace_max_unactuated_input_constraint[None]
            ),
        )

    def _invalidate_stale_transaction(self, reason: str) -> None:
        self._phase = "failed"
        self._prepared = False
        self._converged = False
        raise RuntimeError(f"stale marker MAC constraint transaction: {reason}")

    def _audit_transaction_inputs(
        self,
        fluid,
        *,
        component_face_valid_mask,
        topology_generation: int | None = None,
        component_face_valid_mask_generation: int | None = None,
        obstacle_field=None,
    ) -> None:
        if fluid is not self._fluid:
            self._invalidate_stale_transaction("fluid identity changed")
        if component_face_valid_mask is not self._component_face_valid_mask:
            self._invalidate_stale_transaction(
                "component-face valid-mask owner changed"
            )
        if int(self._markers.marker_count) != self._marker_count:
            self._invalidate_stale_transaction("marker count changed")
        current_generation = int(
            fluid.velocity_dirichlet_component_ledger_generation
        )
        if current_generation != self._prepared_ledger_generation:
            self._invalidate_stale_transaction("ledger generation changed")
        if self.prepared_sampling_identity is not None:
            if topology_generation is None:
                self._invalidate_stale_transaction(
                    "current topology generation is required"
                )
            if component_face_valid_mask_generation is None:
                self._invalidate_stale_transaction(
                    "current component-face valid-mask generation is required"
                )
            current_obstacle_field = obstacle_field
            if current_obstacle_field is None:
                if not hasattr(fluid, "obstacle"):
                    self._invalidate_stale_transaction(
                        "current sampling obstacle owner is required"
                    )
                current_obstacle_field = fluid.obstacle
            try:
                self._markers._audit_prepared_no_slip_sampling_identity(
                    self.prepared_sampling_identity,
                    topology_generation=int(topology_generation),
                    component_face_valid_mask_generation=(
                        int(component_face_valid_mask_generation)
                    ),
                    component_face_valid_mask=component_face_valid_mask,
                    obstacle_field=current_obstacle_field,
                    cell_face_x_m=fluid.cell_face_x_m,
                    cell_face_y_m=fluid.cell_face_y_m,
                    cell_face_z_m=fluid.cell_face_z_m,
                    cell_center_x_m=fluid.cell_center_x_m,
                    cell_center_y_m=fluid.cell_center_y_m,
                    cell_center_z_m=fluid.cell_center_z_m,
                )
            except RuntimeError as exc:
                self._invalidate_stale_transaction(str(exc))
        self._reset_audit_kernel()
        if self.prepared_sampling_identity is not None:
            self._audit_sampling_payload_kernel(
                self.prepared_sampling_identity.sample_valid,
                self.prepared_sampling_identity.sample_source_code,
                self.prepared_sampling_identity.sample_invalid_reason_code,
                self.prepared_sampling_identity.sample_position_m,
                self._marker_count,
            )
        self._audit_transaction_kernel(
            self._markers.x_gamma_m,
            self._markers.v_gamma_mps,
            self._markers.region_id,
            self._marker_count,
            fluid.velocity,
            component_face_valid_mask,
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            fluid.velocity_dirichlet_boundary_external_exact_component_mask,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            float(fluid.rho),
        )
        audit_failure_code = int(self._audit_failure_code[None])
        if audit_failure_code == 4:
            self._invalidate_stale_transaction(
                "sampling identity payload changed"
            )
        if audit_failure_code != 0:
            self._invalidate_stale_transaction("cached inputs changed")

    def prepare(
        self,
        *,
        markers,
        fluid,
        component_face_valid_mask,
        primary_region_id: int,
        secondary_region_id: int,
        prepared_sampling_identity=None,
        topology_generation: int | None = None,
        component_face_valid_mask_generation: int | None = None,
    ) -> None:
        """Build one immutable transaction without changing ``fluid.velocity``."""

        if self._phase in ("prepared", "solved"):
            raise RuntimeError(
                "cannot prepare over a pending uncommitted marker MAC transaction"
            )
        # A new affine J transaction invalidates every pressure factor built
        # from the preceding marker/topology generation, even if validation of
        # the new transaction later fails.
        self._clear_pressure_nullspace_lifecycle()

        marker_count = int(markers.marker_count)
        if marker_count < 0 or marker_count > self.marker_capacity:
            raise ValueError("marker_count exceeds operator marker_capacity")
        if tuple(fluid.velocity.shape) != self.grid_nodes:
            raise ValueError("fluid velocity shape does not match grid_nodes")
        if tuple(component_face_valid_mask.shape) != self.grid_nodes:
            raise ValueError("component_face_valid_mask shape does not match grid_nodes")

        selected_regions = (int(primary_region_id), int(secondary_region_id))
        sample_position_m = markers.x_gamma_m
        sample_valid = markers.region_id
        sample_invalid_reason_code = markers.region_id
        use_prepared_sampling_identity = 0
        prepared_topology_generation = 0
        prepared_valid_mask_generation = 0
        if prepared_sampling_identity is not None:
            if (
                topology_generation is None
                or component_face_valid_mask_generation is None
            ):
                raise RuntimeError(
                    "current topology and component-face-valid-mask generations "
                    "are required for a prepared sampling identity"
                )
            prepared_topology_generation = int(topology_generation)
            prepared_valid_mask_generation = int(
                component_face_valid_mask_generation
            )
            prepared_sampling_identity = (
                markers._audit_prepared_no_slip_sampling_identity(
                    prepared_sampling_identity,
                    topology_generation=prepared_topology_generation,
                    component_face_valid_mask_generation=(
                        prepared_valid_mask_generation
                    ),
                    component_face_valid_mask=component_face_valid_mask,
                    obstacle_field=prepared_sampling_identity._obstacle_field,
                    cell_face_x_m=fluid.cell_face_x_m,
                    cell_face_y_m=fluid.cell_face_y_m,
                    cell_face_z_m=fluid.cell_face_z_m,
                    cell_center_x_m=fluid.cell_center_x_m,
                    cell_center_y_m=fluid.cell_center_y_m,
                    cell_center_z_m=fluid.cell_center_z_m,
                )
            )
            sample_position_m = prepared_sampling_identity.sample_position_m
            sample_valid = prepared_sampling_identity.sample_valid
            sample_invalid_reason_code = (
                prepared_sampling_identity.sample_invalid_reason_code
            )
            use_prepared_sampling_identity = 1
        self._reset_validation_kernel()
        self._validate_markers_kernel(
            sample_position_m,
            markers.x_gamma_m,
            sample_valid,
            sample_invalid_reason_code,
            use_prepared_sampling_identity,
            markers.v_gamma_mps,
            markers.region_id,
            marker_count,
            selected_regions[0],
            selected_regions[1],
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
        )
        failure_code = int(self._failure_code[None])
        if failure_code == 1:
            raise RuntimeError("nonfinite marker constraint input")
        if failure_code == 2:
            raise RuntimeError(
                "conflicting incompatible coincident marker constraints"
            )
        if failure_code == 6:
            raise RuntimeError(
                "active marker is outside the half-open fluid domain upper face"
            )
        if failure_code == 7:
            raise RuntimeError(
                "prepared sampling identity contains an invalid active marker"
            )
        if failure_code == 8:
            raise RuntimeError(
                "active marker constraint has no valid MAC component support"
            )
        active_marker_count = int(self._device_active_marker_count[None])
        self._reset_transaction_kernel()
        if prepared_sampling_identity is not None:
            self._snapshot_sampling_payload_kernel(
                prepared_sampling_identity.sample_valid,
                prepared_sampling_identity.sample_source_code,
                prepared_sampling_identity.sample_invalid_reason_code,
                prepared_sampling_identity.sample_position_m,
                marker_count,
            )
        self._markers = markers
        self._fluid = fluid
        self._component_face_valid_mask = component_face_valid_mask
        self._prepared_obstacle_field = (
            None
            if prepared_sampling_identity is None
            else prepared_sampling_identity._obstacle_field
        )
        self._marker_count = marker_count
        self._active_marker_count = active_marker_count
        self._prepare_rows_kernel(
            sample_position_m,
            markers.x_gamma_m,
            markers.v_gamma_mps,
            markers.region_id,
            marker_count,
            selected_regions[0],
            selected_regions[1],
            fluid.velocity,
            component_face_valid_mask,
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            fluid.velocity_dirichlet_boundary_external_exact_component_mask,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            float(fluid.rho),
        )
        if int(self._failure_code[None]) == 8:
            raise RuntimeError(
                "active marker constraint has no valid MAC component support"
            )
        self._snapshot_geometry_kernel(
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            float(fluid.rho),
        )
        self._constraint_count = int(self._device_constraint_count[None])
        self._prepared = True
        self._converged = False
        self._committed = False
        self._iterations = 0
        self._absolute_tolerance_mps = math.nan
        self._max_residual_mps = math.inf
        self._prepared_ledger_generation = int(
            fluid.velocity_dirichlet_component_ledger_generation
        )
        self._prepared_primary_region_id = selected_regions[0]
        self._prepared_secondary_region_id = selected_regions[1]
        self.prepared_sampling_identity = prepared_sampling_identity
        self._prepared_sampling_identity_generation = (
            0
            if prepared_sampling_identity is None
            else int(prepared_sampling_identity.generation)
        )
        self._prepared_topology_generation = prepared_topology_generation
        self._prepared_component_face_valid_mask_generation = (
            prepared_valid_mask_generation
        )
        self._phase = "prepared"

    def _unsatisfiable_support_provenance(
        self,
        row: int,
        axis: int,
    ) -> tuple[int, int, int, int, float, str]:
        indices = self._stencil_index.to_numpy()
        weights = self._stencil_weight.to_numpy()
        free = self._stencil_free.to_numpy()
        velocities = self._support_velocity_snapshot_mps.to_numpy()
        valid_masks = self._support_valid_mask_snapshot.to_numpy()
        hard_masks = self._support_hard_mask_snapshot.to_numpy()
        external_masks = self._support_external_mask_snapshot.to_numpy()
        bit = 1 << axis
        weighted_count = 0
        free_count = 0
        hard_weighted_count = 0
        external_weighted_count = 0
        valid_weight_sum = 0.0
        support_details: list[str] = []
        for support in range(8):
            index = tuple(int(value) for value in indices[row, support])
            weight = float(weights[row, support])
            is_weighted = abs(weight) > 0.0
            is_free = bool(int(free[row, support]))
            is_valid = bool(int(valid_masks[row, support]) & bit)
            is_hard = bool(int(hard_masks[row, support]) & bit)
            is_external = bool(int(external_masks[row, support]) & bit)
            if is_weighted:
                weighted_count += 1
                valid_weight_sum += weight
                free_count += int(is_free)
                hard_weighted_count += int(is_hard)
                external_weighted_count += int(is_external)
            support_details.append(
                f"slot={support},index={index},weight={weight:.9g},"
                f"velocity_mps={float(velocities[row, support]):.9g},"
                f"valid={int(is_valid)},hard={int(is_hard)},"
                f"external={int(is_external)},free={int(is_free)}"
            )
        return (
            weighted_count,
            free_count,
            hard_weighted_count,
            external_weighted_count,
            valid_weight_sum,
            "; ".join(support_details),
        )

    def _unsatisfiable_constraint_error(
        self,
        message: str,
        tolerance_mps: float,
    ) -> RuntimeError:
        active = self._row_active.to_numpy()
        diagonal = self._diagonal.to_numpy()
        rhs = self._rhs.to_numpy()
        failure_row = next(
            (
                row
                for row in range(self.constraint_capacity)
                if int(active[row]) != 0
                and float(diagonal[row]) <= 1.0e-20
                and abs(float(rhs[row])) > tolerance_mps
            ),
            None,
        )
        if failure_row is None:
            return RuntimeError(
                f"{message}: offending row unavailable, "
                f"tolerance_mps={tolerance_mps:.9g}"
            )
        marker = failure_row // 3
        axis = failure_row % 3
        target = float(self._marker_target_snapshot_mps.to_numpy()[marker][axis])
        row_rhs = float(rhs[failure_row])
        position = tuple(
            float(value)
            for value in self._marker_position_snapshot_m.to_numpy()[marker]
        )
        region = int(self._marker_region_snapshot.to_numpy()[marker])
        weighted, free, hard, external, weight_sum, supports = (
            self._unsatisfiable_support_provenance(failure_row, axis)
        )
        return RuntimeError(
            f"{message}: row={failure_row}, marker={marker}, axis={'xyz'[axis]}, "
            f"region={region}, rhs_mps={row_rhs:.9g}, "
            f"diagonal={float(diagonal[failure_row]):.9g}, "
            f"target_mps={target:.9g}, sampled_mps={target - row_rhs:.9g}, "
            f"free_support_count={free}, weighted_support_count={weighted}, "
            f"hard_weighted_support_count={hard}, "
            f"external_weighted_support_count={external}, "
            f"valid_weight_sum={weight_sum:.9g}, tolerance_mps={tolerance_mps:.9g}, "
            f"marker_position_m={position}, supports=[{supports}]"
        )

    def solve_device(
        self,
        *,
        max_iterations: int,
        absolute_tolerance_mps: float,
        component_face_valid_mask,
        topology_generation: int | None = None,
        component_face_valid_mask_generation: int | None = None,
        obstacle_field=None,
    ) -> None:
        """Solve the marker Schur complement on device-resident fields."""

        if self._phase != "prepared":
            raise RuntimeError(
                "marker MAC constraint transaction state is not prepared or was already solved"
            )
        iterations = int(max_iterations)
        tolerance = float(absolute_tolerance_mps)
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("absolute_tolerance_mps must be finite and positive")

        self._audit_transaction_inputs(
            self._fluid,
            component_face_valid_mask=component_face_valid_mask,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=(
                component_face_valid_mask_generation
            ),
            obstacle_field=obstacle_field,
        )
        self._absolute_tolerance_mps = tolerance
        self._initialize_pcg_kernel(tolerance)
        if int(self._failure_code[None]) == 3:
            self._phase = "failed"
            raise self._unsatisfiable_constraint_error(
                "unsatisfiable marker constraint has no free MAC support",
                tolerance,
            )
        self._check_convergence_kernel(tolerance)
        self._compute_initial_rz_kernel()
        for _ in range(iterations):
            self._apply_matrix(
                self._direction,
                self._matrix_direction,
                force_run=False,
            )
            self._compute_p_ap_kernel()
            self._reset_iteration_residual_kernel()
            self._pcg_step_device_kernel()
            self._check_convergence_kernel(tolerance)
            self._pcg_update_direction_device_kernel()
            self._pcg_finish_direction_device_kernel()

        failure_code = int(self._failure_code[None])
        self._converged = bool(int(self._device_converged[None]))
        self._iterations = int(self._device_iterations[None])
        self._max_residual_mps = float(self._max_residual[None])
        if failure_code in (4, 5):
            self._phase = "failed"
            raise RuntimeError("unsatisfiable marker constraint PCG breakdown")
        if not self._converged:
            self._phase = "failed"
            raise RuntimeError(
                "marker MAC constraint PCG did not converge within max_iterations"
            )
        self._apply_matrix(
            self._lambda,
            self._matrix_direction,
            force_run=True,
        )
        self._copy_grid_scratch_to_correction_kernel()
        self._compute_true_candidate_residual_kernel()
        true_candidate_residual = float(
            self._true_candidate_max_residual[None]
        )
        self._max_residual_mps = true_candidate_residual
        if (
            not math.isfinite(true_candidate_residual)
            or true_candidate_residual > tolerance
        ):
            self._phase = "failed"
            self._converged = False
            raise RuntimeError(
                "true candidate correction residual exceeds the absolute "
                "marker constraint tolerance after solve: "
                f"{true_candidate_residual} > {tolerance}"
            )
        self._snapshot_solved_correction_kernel()
        self._phase = "solved"

    def commit_if_converged(
        self,
        fluid,
        *,
        component_face_valid_mask,
        topology_generation: int | None = None,
        component_face_valid_mask_generation: int | None = None,
        obstacle_field=None,
    ) -> bool:
        """Commit the private correction exactly once after successful solve."""

        if self._phase != "solved" or not self._prepared or not self._converged:
            raise RuntimeError("marker MAC constraint transaction is not converged")
        if fluid is not self._fluid:
            raise RuntimeError("marker MAC constraint commit fluid does not match prepare")
        if self._committed:
            raise RuntimeError("marker MAC constraint transaction is already committed")
        self._audit_transaction_inputs(
            fluid,
            component_face_valid_mask=component_face_valid_mask,
            topology_generation=topology_generation,
            component_face_valid_mask_generation=(
                component_face_valid_mask_generation
            ),
            obstacle_field=obstacle_field,
        )
        self._compute_true_candidate_residual_kernel()
        true_candidate_residual = float(
            self._true_candidate_max_residual[None]
        )
        self._max_residual_mps = true_candidate_residual
        if (
            not math.isfinite(true_candidate_residual)
            or true_candidate_residual > self._absolute_tolerance_mps
        ):
            self._phase = "failed"
            self._converged = False
            raise RuntimeError(
                "true candidate correction residual exceeds the absolute "
                "marker constraint tolerance: "
                f"{true_candidate_residual} > {self._absolute_tolerance_mps}"
            )
        self._audit_solved_correction_integrity_kernel()
        if int(self._solved_correction_integrity_failure[None]) != 0:
            self._phase = "failed"
            self._converged = False
            raise RuntimeError(
                "solved correction integrity changed before commit"
            )
        self._commit_kernel(
            fluid.velocity,
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            fluid.velocity_dirichlet_boundary_external_exact_component_mask,
        )
        self._committed = True
        self._phase = "committed"
        return True

    def report(self) -> HibmMpmMarkerMacConstraintReport:
        return HibmMpmMarkerMacConstraintReport(
            prepared=bool(self._prepared),
            converged=bool(self._converged),
            committed=bool(self._committed),
            active_marker_count=int(self._active_marker_count),
            constraint_count=int(self._constraint_count),
            iterations=int(self._iterations),
            max_residual_mps=float(self._max_residual_mps),
            sample_identity_generation=int(
                self._prepared_sampling_identity_generation
            ),
        )


__all__ = [
    "HibmMpmMarkerMacConstraintOperator",
    "HibmMpmMarkerMacConstraintReport",
]
