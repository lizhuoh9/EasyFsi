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

import numpy as np
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
HIBM_MARKER_CONSTRAINT_HASH_THRESHOLD = 64


def _sparse_normalized_pivoted_cholesky(
    rows: list[dict[tuple[int, int, int, int], float]],
    *,
    relative_pivot_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a sparse row Gram without materialising its M-by-M matrix."""

    count = len(rows)
    norms = np.fromiter(
        (math.sqrt(sum(value * value for value in row.values())) for row in rows),
        dtype=np.float64,
        count=count,
    )
    if not np.all(np.isfinite(norms)):
        raise RuntimeError("sparse marker row norm is non-finite")
    positive = norms > 0.0
    residual = np.where(positive, 1.0, 0.0)
    columns: list[np.ndarray] = []
    selected: list[int] = []

    def normalized_dot(first: int, second: int) -> float:
        if not positive[first] or not positive[second]:
            return 0.0
        left, right = rows[first], rows[second]
        if len(left) > len(right):
            left, right = right, left
        value = sum(weight * right.get(key, 0.0) for key, weight in left.items())
        return value / (norms[first] * norms[second])

    while True:
        pivot = int(np.argmax(residual))
        pivot_value = float(residual[pivot])
        if pivot_value <= relative_pivot_tolerance:
            break
        column = np.fromiter(
            (normalized_dot(row, pivot) for row in range(count)),
            dtype=np.float64,
            count=count,
        )
        if columns:
            # ``column_stack`` creates a transient second M-by-r array.  Fail
            # before that allocation, rather than discovering the pressure
            # resource limit after a large rank-deficient solve has started.
            next_rank = len(columns) + 1
            workspace_bytes = 2 * count * next_rank * 8
            workspace_bytes += next_rank * next_rank * 8
            if workspace_bytes > HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES:
                raise RuntimeError(
                    "sparse marker rank workspace exceeds memory budget: "
                    f"{workspace_bytes} > "
                    f"{HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES} bytes"
                )
            prior = np.column_stack(columns)
            column -= prior @ prior[pivot]
        if not np.all(np.isfinite(column)):
            raise RuntimeError("sparse marker factor column is non-finite")
        pivot_value = float(column[pivot])
        if pivot_value < -8.0 * relative_pivot_tolerance:
            raise RuntimeError("sparse marker Gram is not positive semidefinite")
        if pivot_value <= relative_pivot_tolerance:
            residual[pivot] = 0.0
            continue
        column /= math.sqrt(pivot_value)
        columns.append(column)
        selected.append(pivot)
        next_residual = residual - column * column
        if np.any(next_residual[positive] < -8.0 * relative_pivot_tolerance):
            raise RuntimeError("sparse marker Gram has a negative residual pivot")
        residual = np.where(positive, np.maximum(0.0, next_residual), 0.0)
    factor = (
        np.column_stack(columns)
        if columns
        else np.empty((count, 0), dtype=np.float64)
    )
    return np.asarray(selected, dtype=np.intp), factor, norms


def _sparse_metric_columns(
    rows: list[dict[tuple[int, int, int, int], float]],
    selected: np.ndarray,
) -> np.ndarray:
    """Return only the M-by-r Gram columns needed by a reduced solve."""

    columns = np.empty((len(rows), selected.size), dtype=np.float64)
    for column, pivot in enumerate(selected):
        pivot_row = rows[int(pivot)]
        for row, values in enumerate(rows):
            if len(values) <= len(pivot_row):
                columns[row, column] = sum(
                    value * pivot_row.get(key, 0.0)
                    for key, value in values.items()
                )
            else:
                columns[row, column] = sum(
                    value * values.get(key, 0.0)
                    for key, value in pivot_row.items()
                )
    if not np.all(np.isfinite(columns)):
        raise RuntimeError("sparse marker Gram columns are non-finite")
    return columns



def _sparse_structural_decomposition(
    matrix,
    *,
    column_norm: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a row-space basis and minimum-norm lift without dense M-by-D.

    Twice-reorthogonalized sparse row expansion retains directions down to
    roundoff. The small M-by-r SVD then applies the original singular-value
    cutoff. In particular, mobility never participates in structural rank.
    Only one expanded D-vector and rank-sized workspaces are materialized.
    """

    normalized = matrix.multiply(1.0 / column_norm).tocsr()
    row_count, dof_count = normalized.shape
    basis: list[np.ndarray] = []
    sparse_bytes = sum(
        array.nbytes for array in
        (normalized.data, normalized.indices, normalized.indptr)
    )
    for row in range(row_count):
        candidate = normalized.getrow(row).toarray().ravel()
        initial_norm = float(np.linalg.norm(candidate))
        if not math.isfinite(initial_norm):
            raise RuntimeError("sparse structural row is non-finite")
        if initial_norm == 0.0:
            continue
        for _ in range(2):
            for direction in basis:
                candidate -= np.dot(direction, candidate) * direction
        remaining_norm = float(np.linalg.norm(candidate))
        if remaining_norm <= 8.0 * math.ulp(1.0) * initial_norm:
            continue
        next_rank = len(basis) + 1
        # Include simultaneous QR/SVD/lift and later minimax/weighted work.
        workspace_bytes = sparse_bytes + 8 * (
            6 * (row_count + dof_count) * next_rank
            + 4 * next_rank * next_rank + 2 * dof_count
        )
        if workspace_bytes > HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES:
            raise RuntimeError(
                "sparse structural workspace exceeds memory budget: "
                f"{workspace_bytes} > "
                f"{HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES} bytes"
            )
        basis.append(candidate / remaining_norm)
    if not basis:
        raise RuntimeError("sparse structural rank is zero")
    orthogonal = np.column_stack(basis)
    represented = normalized @ orthogonal
    left, singular, right = np.linalg.svd(represented, full_matrices=False)
    if not np.all(np.isfinite(singular)) or singular[0] <= 0.0:
        raise RuntimeError("sparse structural factor is non-finite or singular")
    rank = int(np.count_nonzero(singular > rcond * singular[0]))
    if rank == 0:
        raise RuntimeError("sparse structural rank is zero")
    row_basis = left[:, :rank]
    lift = (orthogonal @ right[:rank].T) / singular[:rank]
    if not np.all(np.isfinite(lift)):
        raise RuntimeError("sparse structural solution lift is non-finite")
    return row_basis, lift


def _uses_marker_constraint_hash(marker_count: int) -> bool:
    return int(marker_count) > HIBM_MARKER_CONSTRAINT_HASH_THRESHOLD


def _solve_column_normalized_linf(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    failure_context: str,
) -> np.ndarray:
    """Return one Chebyshev witness for a column-normalized dense system."""

    row_count, column_count = matrix.shape
    objective = np.zeros(column_count + 1, dtype=np.float64)
    objective[-1] = 1.0
    upper_matrix = np.vstack(
        (
            np.column_stack(
                (matrix, -np.ones(row_count, dtype=np.float64))
            ),
            np.column_stack(
                (-matrix, -np.ones(row_count, dtype=np.float64))
            ),
        )
    )
    upper_rhs = np.concatenate((rhs, -rhs))
    try:
        from scipy.optimize import linprog

        result = linprog(
            objective,
            A_ub=upper_matrix,
            b_ub=upper_rhs,
            bounds=[(None, None)] * column_count + [(0.0, None)],
            method="highs-ds",
            options={
                "dual_feasibility_tolerance": 1.0e-10,
                "primal_feasibility_tolerance": 1.0e-10,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"{failure_context} minimax solve failed") from exc
    if not result.success:
        raise RuntimeError(
            f"{failure_context} minimax solve failed: {result.message}"
        )
    solution = np.asarray(result.x[:column_count], dtype=np.float64)
    if (
        solution.shape != (column_count,)
        or not np.all(np.isfinite(solution))
        or not math.isfinite(float(result.fun))
    ):
        raise RuntimeError(f"{failure_context} minimax solution is non-finite")
    return solution


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
    backend: str = "pcg"
    rank_revealed: bool = False
    independent_constraint_count: int = 0
    dependent_constraint_count: int = 0
    unactuated_constraint_count: int = 0
    max_structural_residual_mps: float = 0.0
    max_independent_residual_mps: float = 0.0
    max_dependent_residual_mps: float = 0.0
    max_unactuated_residual_mps: float = 0.0


@dataclass(frozen=True)
class _CollectiveFhRepairResult:
    """Accepted device-audited diagnostics for one private F/H candidate."""

    repair_max_residual_mps: float
    global_max_residual_mps: float
    hard_target_dof_count: int
    max_abs_hard_target_delta_mps: float


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
        marker_constraint_hash_capacity = 1
        while marker_constraint_hash_capacity < 2 * self.marker_capacity:
            marker_constraint_hash_capacity *= 2
        self._marker_constraint_hash_capacity = marker_constraint_hash_capacity

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
        self._marker_constraint_owner = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )
        self._marker_constraint_hash_occupied = ti.field(
            dtype=ti.i32,
            shape=self._marker_constraint_hash_capacity,
        )
        self._marker_constraint_hash_position_m = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self._marker_constraint_hash_capacity,
        )
        self._marker_constraint_hash_target_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self._marker_constraint_hash_capacity,
        )
        self._marker_constraint_hash_owner = ti.field(
            dtype=ti.i32,
            shape=self._marker_constraint_hash_capacity,
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

        # Private prospective closure workspace.  This deliberately does not
        # participate in the public Q transaction: the canonical ledger has
        # not committed while target compatibility is being decided.
        self._collective_row_active = ti.field(
            dtype=ti.i32, shape=self.constraint_capacity
        )
        self._collective_row_certificate = ti.field(
            dtype=ti.i32, shape=self.constraint_capacity
        )
        self._collective_row_repair_active = ti.field(
            dtype=ti.i32, shape=self.constraint_capacity
        )
        self._collective_row_immutable_hard = ti.field(
            dtype=ti.i32, shape=self.constraint_capacity
        )
        self._collective_marker_owner = ti.field(
            dtype=ti.i32, shape=self.marker_capacity
        )
        self._collective_hash_occupied = ti.field(
            dtype=ti.i32, shape=self._marker_constraint_hash_capacity
        )
        self._collective_hash_position_m = ti.Vector.field(
            3, dtype=ti.f32, shape=self._marker_constraint_hash_capacity
        )
        self._collective_hash_target_mps = ti.Vector.field(
            3, dtype=ti.f32, shape=self._marker_constraint_hash_capacity
        )
        self._collective_hash_owner = ti.field(
            dtype=ti.i32, shape=self._marker_constraint_hash_capacity
        )
        self._collective_owner_failure = ti.field(dtype=ti.i32, shape=())
        self._collective_rhs = ti.field(dtype=ti.f32, shape=self.constraint_capacity)
        self._collective_index = ti.Vector.field(
            3, dtype=ti.i32, shape=(self.constraint_capacity, 8)
        )
        self._collective_weight = ti.field(
            dtype=ti.f32, shape=(self.constraint_capacity, 8)
        )
        self._collective_free = ti.field(
            dtype=ti.i32, shape=(self.constraint_capacity, 8)
        )
        self._collective_adjustable = ti.field(
            dtype=ti.i32, shape=(self.constraint_capacity, 8)
        )
        self._collective_inverse_mass = ti.field(
            dtype=ti.f32, shape=(self.constraint_capacity, 8)
        )
        self._collective_delta_free = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self._collective_delta_hard = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self._collective_active_count = ti.field(dtype=ti.i32, shape=())
        self._collective_certificate_count = ti.field(dtype=ti.i32, shape=())
        self._collective_immutable_hard_row_count = ti.field(
            dtype=ti.i32, shape=()
        )
        self._collective_max_residual = ti.field(dtype=ti.f32, shape=())
        self._collective_max_repair_residual = ti.field(dtype=ti.f32, shape=())

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
        self._pressure_sparse_factor_capacity = 0
        self._pressure_sparse_dof_capacity = 0
        self._pressure_sparse_dof_count = 0
        self._pressure_sparse_dof_indices = None
        self._pressure_sparse_sqrt_mobility = None
        self._pressure_sparse_factor = None
        self._pressure_sparse_triangular = None
        self._pressure_sparse_actual_rank = 0
        self._pressure_sparse_backend = (
            self.constraint_capacity > HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS
        )
        self._pressure_nullspace_base_resource_bytes = 0

        self._rz_old = ti.field(dtype=ti.f64, shape=())
        self._rz_new = ti.field(dtype=ti.f64, shape=())
        self._p_ap = ti.field(dtype=ti.f64, shape=())
        self._max_residual = ti.field(dtype=ti.f32, shape=())
        self._true_candidate_max_residual = ti.field(dtype=ti.f32, shape=())
        self._final_f32_candidate_max_residual = ti.field(
            dtype=ti.f32,
            shape=(),
        )
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
        self._rank_direct_max_structural_residual = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self._rank_direct_max_independent_residual = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self._rank_direct_max_dependent_residual = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self._rank_direct_max_unactuated_residual = ti.field(
            dtype=ti.f32,
            shape=(),
        )

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
        self._solve_backend = "pcg"
        self._rank_revealed = False
        self._rank_direct_independent_constraint_count = 0
        self._rank_direct_dependent_constraint_count = 0
        self._rank_direct_unactuated_constraint_count = 0

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
        self._final_f32_candidate_max_residual[None] = 0.0
        self._failure_code[None] = 0
        self._audit_failure_code[None] = 0
        self._solved_correction_integrity_failure[None] = 0
        self._device_converged[None] = 0
        self._device_iterations[None] = 0
        self._device_active_marker_count[None] = 0
        self._device_constraint_count[None] = 0
        self._rank_direct_max_structural_residual[None] = 0.0
        self._rank_direct_max_independent_residual[None] = 0.0
        self._rank_direct_max_dependent_residual[None] = 0.0
        self._rank_direct_max_unactuated_residual[None] = 0.0

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
        if ti.static(not self._pressure_sparse_backend):
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
                            same_index = same_index and (
                                self._stencil_weight[first_row, first_support] != 0.0
                            )
                            same_index = same_index and (
                                self._stencil_weight[second_row, second_support] != 0.0
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
    def _project_pressure_nullspace_sparse_kernel(
        self,
        basis: ti.types.ndarray(dtype=ti.f64, ndim=2),
        triangular: ti.types.ndarray(dtype=ti.f64, ndim=2),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=2),
        sqrt_mobility: ti.types.ndarray(dtype=ti.f64, ndim=1),
        rank: ti.i32,
        dof_count: ti.i32,
    ):
        max_input = ti.cast(0.0, ti.f64)
        max_unactuated_input = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for row in range(self.constraint_capacity):
            max_input = ti.max(max_input, ti.abs(self._pressure_nullspace_rhs[row]))
            if (
                self._pressure_nullspace_row_active[row] != 0
                and self._pressure_nullspace_row_inverse_norm[row] == 0.0
            ):
                max_unactuated_input = ti.max(
                    max_unactuated_input, ti.abs(self._pressure_nullspace_rhs[row])
                )
        self._pressure_nullspace_max_input_constraint[None] = ti.max(
            self._pressure_nullspace_max_input_constraint[None], max_input
        )
        self._pressure_nullspace_max_unactuated_input_constraint[None] = ti.max(
            self._pressure_nullspace_max_unactuated_input_constraint[None],
            max_unactuated_input,
        )
        # C.T = Q R for normalized selected rows of J sqrt(A).
        # Use the full Jx, including nonzero input on zero-mobility faces.
        # Solving R.T z = normalized Jx preserves those fixed coordinates
        # while canceling their marker contribution with actuated faces.
        ti.loop_config(serialize=True)
        for column in range(rank):
            row = self._pressure_nullspace_factor_order[column]
            coefficient = (
                self._pressure_nullspace_rhs[row]
                * self._pressure_nullspace_row_inverse_norm[row]
            )
            for previous in range(column):
                coefficient -= (
                    triangular[previous, column]
                    * self._pressure_nullspace_forward[previous]
                )
            self._pressure_nullspace_forward[column] = (
                coefficient / triangular[column, column]
            )
        ti.loop_config(serialize=True)
        for dof in range(dof_count):
            correction = ti.cast(0.0, ti.f64)
            for column in range(rank):
                correction += basis[dof, column] * self._pressure_nullspace_forward[column]
            axis = indices[dof, 0]
            i, j, k = indices[dof, 1], indices[dof, 2], indices[dof, 3]
            self._pressure_nullspace_correction[i, j, k][axis] = (
                sqrt_mobility[dof] * correction
            )

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

    @ti.func
    def _normalized_marker_position_bits(self, value):
        bits = ti.cast(ti.bit_cast(value, ti.i32), ti.i64) & ti.i64(4294967295)
        if (bits & 0x7FFFFFFF) == 0:
            bits = 0
        return bits

    @ti.func
    def _marker_constraint_hash_slot(self, position_m):
        bits_x = self._normalized_marker_position_bits(position_m.x)
        bits_y = self._normalized_marker_position_bits(position_m.y)
        bits_z = self._normalized_marker_position_bits(position_m.z)
        hashed = (
            bits_x * 73856093
            ^ bits_y * 19349663
            ^ bits_z * 83492791
        )
        return ti.cast(
            hashed & (self._marker_constraint_hash_capacity - 1),
            ti.i32,
        )

    @ti.kernel
    def _reset_marker_constraint_hash_kernel(self):
        for slot in range(self._marker_constraint_hash_capacity):
            self._marker_constraint_hash_occupied[slot] = 0

    @ti.kernel
    def _canonicalize_marker_constraints_kernel(
        self,
        marker_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(),
        marker_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
    ):
        # Marker-index order makes the first representative the minimum owner
        # without atomics.  Exact tuple comparisons resolve every hash collision.
        ti.loop_config(serialize=True)
        for marker in range(marker_count):
            region = marker_region_id[marker]
            active = region == primary_region_id or region == secondary_region_id
            self._marker_constraint_owner[marker] = -1
            if active:
                self._marker_constraint_owner[marker] = marker
                position = marker_position_m[marker]
                target = marker_velocity_mps[marker]
                position_has_nan = False
                for axis in ti.static(range(3)):
                    position_has_nan = (
                        position_has_nan or position[axis] != position[axis]
                    )
                if not position_has_nan:
                    slot = self._marker_constraint_hash_slot(position)
                    probe = 0
                    resolved = 0
                    while (
                        probe < self._marker_constraint_hash_capacity
                        and resolved == 0
                    ):
                        if self._marker_constraint_hash_occupied[slot] == 0:
                            self._marker_constraint_hash_occupied[slot] = 1
                            self._marker_constraint_hash_position_m[slot] = position
                            self._marker_constraint_hash_target_mps[slot] = target
                            self._marker_constraint_hash_owner[slot] = marker
                            resolved = 1
                        else:
                            representative = (
                                self._marker_constraint_hash_position_m[slot]
                            )
                            same_position = True
                            for axis in ti.static(range(3)):
                                same_position = same_position and (
                                    position[axis] == representative[axis]
                                )
                            if same_position:
                                representative_target = (
                                    self._marker_constraint_hash_target_mps[slot]
                                )
                                same_target = True
                                for axis in ti.static(range(3)):
                                    same_target = same_target and (
                                        target[axis] == representative_target[axis]
                                    )
                                self._marker_constraint_owner[marker] = (
                                    self._marker_constraint_hash_owner[slot]
                                )
                                if not same_target:
                                    ti.atomic_max(self._failure_code[None], 2)
                                resolved = 1
                            else:
                                slot = (slot + 1) & (
                                    self._marker_constraint_hash_capacity - 1
                                )
                        probe += 1
                    if resolved == 0:
                        ti.atomic_max(self._failure_code[None], 9)

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
            constraint_owner = -1
            if active:
                constraint_owner = marker
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
                if marker_count <= HIBM_MARKER_CONSTRAINT_HASH_THRESHOLD:
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
                            if relation == 1:
                                ti.atomic_max(self._failure_code[None], 2)
                            elif relation == 2:
                                constraint_owner = ti.min(constraint_owner, prior)
            self._marker_constraint_owner[marker] = constraint_owner

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
    def _reset_collective_target_closure_kernel(self):
        self._collective_active_count[None] = 0
        self._collective_certificate_count[None] = 0
        self._collective_immutable_hard_row_count[None] = 0
        self._collective_max_residual[None] = 0.0
        self._collective_max_repair_residual[None] = 0.0
        self._collective_owner_failure[None] = 0
        for row in range(self.constraint_capacity):
            self._collective_row_active[row] = 0
            self._collective_row_certificate[row] = 0
            self._collective_row_repair_active[row] = 0
            self._collective_row_immutable_hard[row] = 0
            self._collective_rhs[row] = 0.0
        for row, support in self._collective_weight:
            self._collective_index[row, support] = ti.Vector([-1, -1, -1])
            self._collective_weight[row, support] = 0.0
            self._collective_free[row, support] = 0
            self._collective_adjustable[row, support] = 0
            self._collective_inverse_mass[row, support] = 0.0
        for i, j, k in self._collective_delta_free:
            self._collective_delta_free[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self._collective_delta_hard[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        for marker in range(self.marker_capacity):
            self._collective_marker_owner[marker] = -1
        for slot in range(self._marker_constraint_hash_capacity):
            self._collective_hash_occupied[slot] = 0

    @ti.kernel
    def _canonicalize_collective_marker_owners_kernel(
        self,
        marker_position_m: ti.template(),
        marker_velocity_mps: ti.template(),
        marker_region_id: ti.template(),
        marker_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
    ):
        # This private map mirrors terminal-Q canonicalization.  Its serial
        # insertion order makes the minimum identical-marker index the owner;
        # a coincident different target remains a fail-closed input error.
        ti.loop_config(serialize=True)
        for marker in range(marker_count):
            region = marker_region_id[marker]
            active = region == primary_region_id or region == secondary_region_id
            if active:
                position = marker_position_m[marker]
                target = marker_velocity_mps[marker]
                finite = True
                for axis in ti.static(range(3)):
                    finite = finite and position[axis] == position[axis]
                    finite = finite and target[axis] == target[axis]
                if not finite:
                    ti.atomic_max(self._collective_owner_failure[None], 1)
                else:
                    self._collective_marker_owner[marker] = marker
                    slot = self._marker_constraint_hash_slot(position)
                    probe = 0
                    resolved = 0
                    while probe < self._marker_constraint_hash_capacity and resolved == 0:
                        if self._collective_hash_occupied[slot] == 0:
                            self._collective_hash_occupied[slot] = 1
                            self._collective_hash_position_m[slot] = position
                            self._collective_hash_target_mps[slot] = target
                            self._collective_hash_owner[slot] = marker
                            resolved = 1
                        else:
                            representative = self._collective_hash_position_m[slot]
                            same_position = True
                            same_target = True
                            for axis in ti.static(range(3)):
                                same_position = same_position and position[axis] == representative[axis]
                                same_target = same_target and target[axis] == self._collective_hash_target_mps[slot][axis]
                            if same_position:
                                self._collective_marker_owner[marker] = self._collective_hash_owner[slot]
                                if not same_target:
                                    ti.atomic_max(self._collective_owner_failure[None], 2)
                                resolved = 1
                            else:
                                slot = (slot + 1) & (self._marker_constraint_hash_capacity - 1)
                        probe += 1
                    if resolved == 0:
                        ti.atomic_max(self._collective_owner_failure[None], 3)

    @ti.kernel
    def _build_collective_target_closure_rows_kernel(
        self, marker_position_m: ti.template(), marker_sample_valid: ti.template(),
        marker_velocity_mps: ti.template(), marker_region_id: ti.template(),
        physical_marker_count: ti.i32, primary_region_id: ti.i32,
        secondary_region_id: ti.i32, prospective_velocity: ti.template(),
        component_face_valid_mask: ti.template(), hard_fixed_component_mask: ti.template(),
        external_exact_component_mask: ti.template(), adjustable_component_mask: ti.template(),
        cell_face_x_m: ti.template(), cell_face_y_m: ti.template(), cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(), cell_center_y_m: ti.template(), cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(), cell_width_y_m: ti.template(), cell_width_z_m: ti.template(),
        density_kgm3: ti.f32,
    ):
        nx = ti.static(self.grid_nodes[0])
        ny = ti.static(self.grid_nodes[1])
        nz = ti.static(self.grid_nodes[2])
        for marker in range(physical_marker_count):
            selected = self._collective_marker_owner[marker] == marker
            selected = selected and marker_sample_valid[marker] != 0
            selected = selected and (marker_region_id[marker] == primary_region_id or marker_region_id[marker] == secondary_region_id)
            for axis in ti.static(range(3)):
                row = 3 * marker + axis
                if selected:
                    base, fraction = mac_component_stencil_base_fraction(
                        marker_position_m[marker], axis, cell_face_x_m, cell_face_y_m,
                        cell_face_z_m, cell_center_x_m, cell_center_y_m, cell_center_z_m,
                        nx, ny, nz,
                    )
                    valid_weight = 0.0
                    sampled = 0.0
                    bit = 1 << axis
                    for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                        i = base.x + oi
                        j = base.y + oj
                        k = base.z + ok
                        weight = mac_stencil_weight(fraction, oi, oj, ok)
                        if (component_face_valid_mask[i, j, k] & bit) != 0:
                            valid_weight += weight
                            sampled += weight * prospective_velocity[i, j, k][axis]
                    if valid_weight > 1.0e-12:
                        self._collective_row_active[row] = 1
                        ti.atomic_add(self._collective_active_count[None], 1)
                        self._collective_rhs[row] = marker_velocity_mps[marker][axis] - sampled / valid_weight
                        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                            i = base.x + oi
                            j = base.y + oj
                            k = base.z + ok
                            support = 4 * oi + 2 * oj + ok
                            weight = mac_stencil_weight(fraction, oi, oj, ok)
                            self._collective_index[row, support] = ti.Vector([i, j, k])
                            if (component_face_valid_mask[i, j, k] & bit) != 0:
                                self._collective_weight[row, support] = weight / valid_weight
                                hard = (hard_fixed_component_mask[i, j, k] & bit) != 0
                                external = (external_exact_component_mask[i, j, k] & bit) != 0
                                self._collective_free[row, support] = 1 if not hard and not external else 0
                                self._collective_adjustable[row, support] = 1 if (hard and not external and (adjustable_component_mask[i, j, k] & bit) != 0) else 0
                                if external or (hard and (adjustable_component_mask[i, j, k] & bit) == 0):
                                    self._collective_row_immutable_hard[row] = 1
                                self._collective_inverse_mass[row, support] = self._component_inverse_mass_per_kg(
                                    cell_width_x_m, cell_width_y_m, cell_width_z_m,
                                    density_kgm3, i, j, k, axis,
                                )
                        if self._collective_row_immutable_hard[row] != 0:
                            ti.atomic_add(self._collective_immutable_hard_row_count[None], 1)

    @ti.func
    def _collective_row_residual(self, row: ti.i32, include_hard: ti.i32):
        axis = row % 3
        residual = self._collective_rhs[row]
        for support in ti.static(range(8)):
            index = self._collective_index[row, support]
            weight = self._collective_weight[row, support]
            if weight != 0.0 and index.x >= 0:
                if self._collective_free[row, support] != 0:
                    residual -= weight * self._collective_delta_free[index.x, index.y, index.z][axis]
                if include_hard != 0 and self._collective_adjustable[row, support] != 0:
                    residual -= weight * self._collective_delta_hard[index.x, index.y, index.z][axis]
        return residual

    @ti.kernel
    def _collective_kaczmarz_sweep_kernel(self, include_hard: ti.i32):
        ti.loop_config(serialize=True)
        for row in range(self.constraint_capacity):
            eligible = self._collective_row_active[row] != 0
            if include_hard != 0:
                eligible = eligible and self._collective_row_certificate[row] != 0
            if eligible:
                axis = row % 3
                denominator = 0.0
                for support in ti.static(range(8)):
                    weight = self._collective_weight[row, support]
                    inv_mass = self._collective_inverse_mass[row, support]
                    if self._collective_free[row, support] != 0:
                        denominator += weight * weight * inv_mass
                    if include_hard != 0 and self._collective_adjustable[row, support] != 0:
                        denominator += weight * weight * inv_mass
                if denominator > 1.0e-24:
                    residual = self._collective_row_residual(row, include_hard)
                    for support in ti.static(range(8)):
                        index = self._collective_index[row, support]
                        weight = self._collective_weight[row, support]
                        inv_mass = self._collective_inverse_mass[row, support]
                        delta = weight * inv_mass * residual / denominator
                        if self._collective_free[row, support] != 0:
                            self._collective_delta_free[index.x, index.y, index.z][axis] += delta
                        if include_hard != 0 and self._collective_adjustable[row, support] != 0:
                            self._collective_delta_hard[index.x, index.y, index.z][axis] += delta

    @ti.kernel
    def _measure_collective_target_closure_kernel(self, include_hard: ti.i32):
        self._collective_max_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._collective_row_active[row] != 0:
                magnitude = ti.abs(
                    self._collective_row_residual(row, include_hard)
                )
                # ``atomic_max`` does not reliably promote NaN.  Saturating
                # every non-finite (or f32-overflow-scale) residual makes the
                # zero-correction fast path fail closed instead.
                if (
                    ti.math.isnan(magnitude)
                    or ti.math.isinf(magnitude)
                ):
                    magnitude = 3.4e38
                ti.atomic_max(self._collective_max_residual[None], magnitude)

    @ti.kernel
    def _measure_collective_fh_repair_residual_kernel(self):
        self._collective_max_residual[None] = 0.0
        self._collective_max_repair_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._collective_row_active[row] != 0:
                magnitude = ti.abs(self._collective_row_residual(row, 1))
                if ti.math.isnan(magnitude) or ti.math.isinf(magnitude):
                    magnitude = 3.4e38
                ti.atomic_max(self._collective_max_residual[None], magnitude)
                if self._collective_row_repair_active[row] != 0:
                    ti.atomic_max(self._collective_max_repair_residual[None], magnitude)

    @ti.kernel
    def _certify_collective_zero_free_rows_kernel(self, tolerance_mps: ti.f64):
        self._collective_certificate_count[None] = 0
        for row in range(self.constraint_capacity):
            if self._collective_row_active[row] != 0:
                rhs = ti.cast(self._collective_rhs[row], ti.f64)
                has_free = 0
                has_owned_hard = 0
                for support in ti.static(range(8)):
                    weight = ti.cast(self._collective_weight[row, support], ti.f64)
                    if self._collective_free[row, support] != 0 and weight != 0.0:
                        has_free = 1
                    mobility = ti.cast(
                        self._collective_inverse_mass[row, support], ti.f64
                    )
                    if (
                        self._collective_adjustable[row, support] != 0
                        and weight != 0.0
                        and not ti.math.isnan(weight)
                        and not ti.math.isinf(weight)
                        and mobility > 0.0
                        and not ti.math.isnan(mobility)
                        and not ti.math.isinf(mobility)
                    ):
                        has_owned_hard = 1
                # J_F[row] is exactly zero, so e_row is a unit left-null
                # witness.  Its defect must meet the stricter owned-H gate.
                if (
                    has_free == 0
                    and has_owned_hard != 0
                    and not ti.math.isnan(rhs)
                    and not ti.math.isinf(rhs)
                    and ti.abs(rhs) > tolerance_mps
                ):
                    self._collective_row_certificate[row] = 1
        # Preserve certificates from other exact families and count each row
        # once when this pass follows the proportional-F witness pass.
        for row in range(self.constraint_capacity):
            if self._collective_row_certificate[row] != 0:
                ti.atomic_add(self._collective_certificate_count[None], 1)

    @ti.kernel
    def _certify_collective_proportional_free_rows_kernel(self, tolerance_mps: ti.f64):
        self._collective_certificate_count[None] = 0
        for row in range(self.constraint_capacity):
            self._collective_row_certificate[row] = 0
        # A finite Kaczmarz timeout is not an infeasibility proof.  This
        # family uses an algebraic proportional-free-row lower-bound witness.
        for first in range(self.constraint_capacity):
            if self._collective_row_active[first] != 0:
                axis = first % 3
                for second in range(first + 1, self.constraint_capacity):
                    if self._collective_row_active[second] != 0 and second % 3 == axis:
                        pivot_a = ti.cast(0.0, ti.f64)
                        pivot_c = ti.cast(0.0, ti.f64)
                        have_pivot = 0
                        first_adjustable = 0
                        second_adjustable = 0
                        for support in ti.static(range(8)):
                            first_adjustable = ti.max(first_adjustable, self._collective_adjustable[first, support])
                            second_adjustable = ti.max(second_adjustable, self._collective_adjustable[second, support])
                            first_weight = ti.cast(self._collective_weight[first, support], ti.f64)
                            if have_pivot == 0 and self._collective_free[first, support] != 0 and first_weight != 0.0:
                                index = self._collective_index[first, support]
                                for candidate in ti.static(range(8)):
                                    other = self._collective_index[second, candidate]
                                    second_weight = ti.cast(self._collective_weight[second, candidate], ti.f64)
                                    if self._collective_free[second, candidate] != 0 and second_weight != 0.0 and index.x == other.x and index.y == other.y and index.z == other.z:
                                        pivot_a = first_weight
                                        pivot_c = second_weight
                                        have_pivot = 1
                        proportional = have_pivot != 0
                        if proportional:
                            for support in ti.static(range(8)):
                                first_weight = ti.cast(self._collective_weight[first, support], ti.f64)
                                if self._collective_free[first, support] != 0 and first_weight != 0.0:
                                    matching_count = 0
                                    matching_cross_product = 0
                                    index = self._collective_index[first, support]
                                    for candidate in ti.static(range(8)):
                                        other = self._collective_index[second, candidate]
                                        second_weight = ti.cast(self._collective_weight[second, candidate], ti.f64)
                                        if self._collective_free[second, candidate] != 0 and second_weight != 0.0 and index.x == other.x and index.y == other.y and index.z == other.z:
                                            matching_count += 1
                                            if second_weight * pivot_a == first_weight * pivot_c:
                                                matching_cross_product = 1
                                    if matching_count != 1 or matching_cross_product == 0:
                                        proportional = False
                                second_weight = ti.cast(self._collective_weight[second, support], ti.f64)
                                if self._collective_free[second, support] != 0 and second_weight != 0.0:
                                    matching_count = 0
                                    matching_cross_product = 0
                                    index = self._collective_index[second, support]
                                    for candidate in ti.static(range(8)):
                                        other = self._collective_index[first, candidate]
                                        first_weight = ti.cast(self._collective_weight[first, candidate], ti.f64)
                                        if self._collective_free[first, candidate] != 0 and first_weight != 0.0 and index.x == other.x and index.y == other.y and index.z == other.z:
                                            matching_count += 1
                                            if second_weight * pivot_a == first_weight * pivot_c:
                                                matching_cross_product = 1
                                    if matching_count != 1 or matching_cross_product == 0:
                                        proportional = False
                        if proportional and first_adjustable != 0 and second_adjustable != 0:
                            first_rhs = ti.cast(self._collective_rhs[first], ti.f64)
                            second_rhs = ti.cast(self._collective_rhs[second], ti.f64)
                            left_product = pivot_a * second_rhs
                            right_product = pivot_c * first_rhs
                            tolerance_product = tolerance_mps * (
                                ti.abs(pivot_a) + ti.abs(pivot_c)
                            )
                            eps64 = ti.cast(2.220446049250313e-16, ti.f64)
                            roundoff_margin = ti.cast(32.0, ti.f64) * eps64 * (
                                ti.abs(left_product)
                                + ti.abs(right_product)
                                + tolerance_product
                            )
                            if ti.abs(left_product - right_product) > tolerance_product + roundoff_margin:
                                self._collective_row_certificate[first] = 1
                                self._collective_row_certificate[second] = 1
        for row in range(self.constraint_capacity):
            if self._collective_row_certificate[row] != 0:
                ti.atomic_add(self._collective_certificate_count[None], 1)

    @ti.kernel
    def _apply_collective_hard_target_delta_kernel(self, claim_target_mps: ti.template(), prospective_velocity: ti.template()):
        for i, j, k in self._collective_delta_hard:
            delta = self._collective_delta_hard[i, j, k]
            if delta.x != 0.0 or delta.y != 0.0 or delta.z != 0.0:
                claim_target_mps[i, j, k] += delta
                prospective_velocity[i, j, k] += delta

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
            unique_constraint = (
                marker_active and self._marker_constraint_owner[marker] == marker
            )
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
    def _compute_final_f32_candidate_residual_kernel(self):
        """Audit the rounded velocity value which commit will actually publish."""

        self._final_f32_candidate_max_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._row_active[row] != 0:
                axis = row % 3
                sampled = ti.cast(0.0, ti.f32)
                for support in ti.static(range(8)):
                    index = self._stencil_index[row, support]
                    if index.x >= 0:
                        value = self._fluid.velocity[index.x, index.y, index.z][axis]
                        hard = self._fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                            index.x, index.y, index.z
                        ]
                        external = self._fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                            index.x, index.y, index.z
                        ]
                        if ((hard | external) & (1 << axis)) == 0:
                            value = ti.cast(
                                value + self._correction[index.x, index.y, index.z][axis],
                                ti.f32,
                            )
                        sampled += self._stencil_weight[row, support] * value
                target = self._marker_target_snapshot_mps[row // 3][axis]
                residual = ti.abs(target - sampled)
                finite = residual == residual and residual < 3.4e38
                if not finite:
                    residual = 3.4e38
                ti.atomic_max(
                    self._final_f32_candidate_max_residual[None],
                    residual,
                )

    def _require_final_f32_candidate_audit(self, tolerance: float) -> None:
        """Fail closed if adding Q to the live f32 grid loses the correction."""

        self._compute_final_f32_candidate_residual_kernel()
        residual = float(self._final_f32_candidate_max_residual[None])
        self._max_residual_mps = max(self._max_residual_mps, residual)
        if not math.isfinite(residual) or residual > tolerance:
            self._phase = "failed"
            self._converged = False
            raise RuntimeError(
                "final f32 candidate velocity residual exceeds the absolute "
                f"marker constraint tolerance: {residual} > {tolerance}"
            )

    @ti.kernel
    def _copy_rank_direct_metric_to_dense_scratch_kernel(self):
        """Borrow the bounded pressure dense storage for one Q direct solve."""

        for row in range(self.constraint_capacity):
            self._pressure_nullspace_row_active[row] = self._row_active[row]
        for row, support in self._pressure_nullspace_inverse_mass_per_kg:
            self._pressure_nullspace_inverse_mass_per_kg[row, support] = (
                ti.cast(self._stencil_inverse_mass_per_kg[row, support], ti.f64)
            )

    @ti.kernel
    def _materialize_rank_direct_correction_and_audit_kernel(self):
        """Round the private f64 direct correction then audit every Q row."""

        self._true_candidate_max_residual[None] = 0.0
        self._rank_direct_max_structural_residual[None] = 0.0
        self._rank_direct_max_independent_residual[None] = 0.0
        self._rank_direct_max_dependent_residual[None] = 0.0
        self._rank_direct_max_unactuated_residual[None] = 0.0
        for i, j, k in self._correction:
            self._correction[i, j, k] = ti.cast(
                self._pressure_nullspace_correction[i, j, k],
                ti.f32,
            )
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
                residual = ti.abs(self._rhs[row] - sampled_correction)
                finite = residual == residual and residual < 3.4e38
                if not finite:
                    residual = 3.4e38
                ti.atomic_max(self._true_candidate_max_residual[None], residual)
                ti.atomic_max(
                    self._rank_direct_max_structural_residual[None],
                    residual,
                )
                if self._pressure_nullspace_factor_row_selected[row] != 0:
                    ti.atomic_max(
                        self._rank_direct_max_independent_residual[None],
                        residual,
                    )
                elif self._pressure_nullspace_row_inverse_norm[row] > 0.0:
                    ti.atomic_max(
                        self._rank_direct_max_dependent_residual[None],
                        residual,
                    )
                else:
                    ti.atomic_max(
                        self._rank_direct_max_unactuated_residual[None],
                        residual,
                    )

    def _solve_rank_revealing_direct(self, tolerance: float) -> None:
        """Solve a compatible rank-deficient Q system without changing PCG.

        The dense resource is intentionally the one owned by the existing
        pressure nullspace implementation.  Q borrows it only while its
        ordinary transaction is prepared; P resets it before establishing its
        separate public lifecycle.  The selected rows provide a correction
        basis, while the least-squares objective includes every positive-energy
        physical row so a representative-only solve cannot hide dependencies.
        Least squares is the fast path; a bounded Chebyshev fallback aligns the
        candidate objective with the authoritative all-row max-residual audit.
        """

        # Publish the attempted backend before every direct-only failure path,
        # but do not let a previous factor partition survive this transaction.
        self._solve_backend = "rank_revealing_direct"
        self._rank_revealed = False
        self._rank_direct_independent_constraint_count = 0
        self._rank_direct_dependent_constraint_count = 0
        self._rank_direct_unactuated_constraint_count = 0
        self._reset_rank_direct_diagnostics_kernel()
        if self.constraint_capacity > HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS:
            try:
                self._solve_rank_revealing_sparse(tolerance)
            except Exception:
                self._phase = "failed"
                self._converged = False
                raise
            return
        self._ensure_pressure_nullspace_resources()
        self._reset_pressure_nullspace_prepare_kernel()
        self._copy_rank_direct_metric_to_dense_scratch_kernel()
        self._assemble_pressure_nullspace_schur_kernel()
        self._symmetrize_pressure_nullspace_schur_kernel()
        relative_pivot_tolerance = max(
            1.0e-14,
            64.0 * math.ulp(1.0) * float(self.marker_capacity),
        )
        self._factor_pressure_nullspace_schur_kernel(relative_pivot_tolerance)
        if int(self._pressure_nullspace_failure_code[None]) != 0:
            self._phase = "failed"
            raise RuntimeError("rank-revealing direct marker factorization failed")

        active = self._pressure_nullspace_row_active.to_numpy().astype(bool)
        inverse_norm = self._pressure_nullspace_row_inverse_norm.to_numpy()
        selected = self._pressure_nullspace_factor_order.to_numpy()
        selected = selected[selected >= 0].astype(np.intp, copy=False)
        active_count = int(np.count_nonzero(active))
        positive = active & (inverse_norm > 0.0)
        selected_mask = np.zeros(self.constraint_capacity, dtype=bool)
        selected_mask[selected] = True
        self._rank_direct_independent_constraint_count = int(selected.size)
        self._rank_direct_dependent_constraint_count = int(
            np.count_nonzero(positive & ~selected_mask)
        )
        self._rank_direct_unactuated_constraint_count = int(
            np.count_nonzero(active & ~positive)
        )
        if (
            self._rank_direct_independent_constraint_count
            + self._rank_direct_dependent_constraint_count
            + self._rank_direct_unactuated_constraint_count
            != active_count
        ):
            self._phase = "failed"
            raise RuntimeError("rank-revealing direct marker partition is inconsistent")
        self._rank_revealed = True

        correction = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
        if selected.size:
            schur = self._pressure_nullspace_schur.to_numpy()
            rhs = self._rhs.to_numpy().astype(np.float64, copy=False)
            positive_rows = np.flatnonzero(positive)
            columns = schur[np.ix_(positive_rows, selected)]
            column_norm = np.linalg.norm(columns, axis=0)
            if (
                not np.all(np.isfinite(column_norm))
                or np.any(column_norm <= 0.0)
            ):
                self._phase = "failed"
                raise RuntimeError("rank-revealing direct marker basis is singular")
            normalized_columns = columns / column_norm
            normal_matrix = normalized_columns.T @ normalized_columns
            normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
            normal_rhs = normalized_columns.T @ rhs[positive_rows]
            try:
                cholesky = np.linalg.cholesky(normal_matrix)
                forward = np.linalg.solve(cholesky, normal_rhs)
                coefficients = np.linalg.solve(cholesky.T, forward) / column_norm
            except np.linalg.LinAlgError as exc:
                self._phase = "failed"
                raise RuntimeError(
                    "rank-revealing direct marker normal factorization failed"
                ) from exc
            indices = self._stencil_index.to_numpy()
            weights = self._stencil_weight.to_numpy()
            free = self._stencil_free.to_numpy()
            inverse_mass = self._stencil_inverse_mass_per_kg.to_numpy()

            def assemble_selected_correction(
                basis_coefficients: np.ndarray,
            ) -> np.ndarray:
                candidate = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
                for coefficient, row in zip(
                    basis_coefficients, selected, strict=True
                ):
                    axis = int(row % 3)
                    for support in range(8):
                        if int(free[row, support]) == 0:
                            continue
                        index = tuple(int(value) for value in indices[row, support])
                        candidate[index][axis] += (
                            float(inverse_mass[row, support])
                            * float(weights[row, support])
                            * float(coefficient)
                        )
                return candidate

            correction = assemble_selected_correction(coefficients)
        self._pressure_nullspace_correction.from_numpy(correction)
        self._materialize_rank_direct_correction_and_audit_kernel()
        self._max_residual_mps = float(self._true_candidate_max_residual[None])
        self._converged = self._max_residual_mps <= tolerance
        self._iterations = 0
        if not self._converged and selected.size:
            try:
                normalized_solution = _solve_column_normalized_linf(
                    normalized_columns,
                    rhs[positive_rows],
                    failure_context="rank-revealing direct marker",
                )
            except Exception:
                self._phase = "failed"
                raise
            correction = assemble_selected_correction(
                normalized_solution / column_norm
            )
            self._pressure_nullspace_correction.from_numpy(correction)
            self._materialize_rank_direct_correction_and_audit_kernel()
            self._max_residual_mps = float(
                self._true_candidate_max_residual[None]
            )
            self._converged = self._max_residual_mps <= tolerance
        if not self._converged:
            self._phase = "failed"
            raise RuntimeError(
                "rank-revealing direct marker correction residual exceeds the "
                "absolute marker constraint tolerance: "
                f"{self._max_residual_mps} > {tolerance}"
            )
        self._require_final_f32_candidate_audit(tolerance)
        self._snapshot_solved_correction_kernel()
        self._phase = "solved"

    def _sparse_metric_rows(
        self, *, row_active, inverse_mobility, allow_zero_mobility: bool = False
    ) -> tuple[list[dict[tuple[int, int, int, int], float]], dict]:
        """Build sparse J sqrt(D) rows using the supplied Q or P metric."""

        active = row_active.to_numpy().astype(bool)
        indices = self._stencil_index.to_numpy()
        weights = self._stencil_weight.to_numpy().astype(np.float64, copy=False)
        free = self._stencil_free.to_numpy().astype(bool)
        inverse_mass = inverse_mobility.to_numpy().astype(
            np.float64,
            copy=False,
        )
        rows: list[dict[tuple[int, int, int, int], float]] = []
        mobility_by_dof: dict[tuple[int, int, int, int], float] = {}
        for row in range(self.constraint_capacity):
            values: dict[tuple[int, int, int, int], float] = {}
            if active[row]:
                axis = row % 3
                for support in range(8):
                    if not free[row, support]:
                        continue
                    weight = float(weights[row, support])
                    mobility = float(inverse_mass[row, support])
                    if weight == 0.0:
                        continue
                    if (
                        not math.isfinite(weight)
                        or not math.isfinite(mobility)
                        or mobility < 0.0
                        or (mobility == 0.0 and not allow_zero_mobility)
                    ):
                        raise RuntimeError("rank-revealing sparse marker support is invalid")
                    index = tuple(int(value) for value in indices[row, support])
                    if any(
                        value < 0 or value >= self.grid_nodes[dimension]
                        for dimension, value in enumerate(index)
                    ):
                        raise RuntimeError("rank-revealing sparse marker support is out of bounds")
                    key = (axis, *index)
                    previous = mobility_by_dof.setdefault(key, mobility)
                    if previous != mobility:
                        raise RuntimeError(
                            "rank-revealing sparse marker shared support has inconsistent mobility"
                        )
                    if mobility > 0.0:
                        values[key] = values.get(key, 0.0) + weight * math.sqrt(mobility)
            rows.append(values)
        return rows, mobility_by_dof

    @ti.kernel
    def _audit_sparse_q_partition_kernel(
        self, partition: ti.types.ndarray(dtype=ti.i32, ndim=1)
    ):
        self._true_candidate_max_residual[None] = 0.0
        self._rank_direct_max_structural_residual[None] = 0.0
        self._rank_direct_max_independent_residual[None] = 0.0
        self._rank_direct_max_dependent_residual[None] = 0.0
        self._rank_direct_max_unactuated_residual[None] = 0.0
        for row in range(self.constraint_capacity):
            if self._row_active[row] != 0:
                axis = row % 3
                sampled = ti.cast(0.0, ti.f32)
                for support in ti.static(range(8)):
                    if self._stencil_free[row, support] != 0:
                        index = self._stencil_index[row, support]
                        sampled += (
                            self._stencil_weight[row, support]
                            * self._correction[index.x, index.y, index.z][axis]
                        )
                residual = ti.abs(self._rhs[row] - sampled)
                if not (residual == residual and residual < 3.4e38):
                    residual = 3.4e38
                ti.atomic_max(self._true_candidate_max_residual[None], residual)
                ti.atomic_max(
                    self._rank_direct_max_structural_residual[None], residual
                )
                if partition[row] == 1:
                    ti.atomic_max(
                        self._rank_direct_max_independent_residual[None], residual
                    )
                elif partition[row] == 2:
                    ti.atomic_max(
                        self._rank_direct_max_dependent_residual[None], residual
                    )
                elif partition[row] == 3:
                    ti.atomic_max(
                        self._rank_direct_max_unactuated_residual[None], residual
                    )

    def _solve_rank_revealing_sparse(self, tolerance: float) -> None:
        """Scalable Q solve retaining sparse supports and M-by-r work only."""

        rows, _mobility = self._sparse_metric_rows(
            row_active=self._row_active,
            inverse_mobility=self._stencil_inverse_mass_per_kg,
        )
        relative_pivot_tolerance = max(
            1.0e-14,
            64.0 * math.ulp(1.0) * float(self.marker_capacity),
        )
        selected, _factor, norms = _sparse_normalized_pivoted_cholesky(
            rows,
            relative_pivot_tolerance=relative_pivot_tolerance,
        )
        active = self._row_active.to_numpy().astype(bool)
        rhs = self._rhs.to_numpy().astype(np.float64, copy=False)
        positive = active & (norms > 0.0)
        selected_mask = np.zeros(self.constraint_capacity, dtype=bool)
        selected_mask[selected] = True
        partition = np.zeros(self.constraint_capacity, dtype=np.int32)
        partition[active & ~positive] = 3
        partition[positive] = 2
        partition[selected] = 1
        self._rank_direct_independent_constraint_count = int(selected.size)
        self._rank_direct_dependent_constraint_count = int(
            np.count_nonzero(positive & ~selected_mask)
        )
        self._rank_direct_unactuated_constraint_count = int(
            np.count_nonzero(active & ~positive)
        )
        self._rank_revealed = True
        if not np.all(np.isfinite(rhs[active])):
            self._phase = "failed"
            raise RuntimeError("rank-revealing sparse Q rhs is non-finite")
        columns = _sparse_metric_columns(rows, selected)
        correction = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
        if selected.size:
            column_norm = np.linalg.norm(columns[positive], axis=0)
            if (
                not np.all(np.isfinite(column_norm))
                or np.any(column_norm <= 0.0)
            ):
                self._phase = "failed"
                raise RuntimeError("rank-revealing sparse Q basis is singular")
            normalized_columns = columns[positive] / column_norm
            try:
                coefficients, _, _, _ = np.linalg.lstsq(
                    normalized_columns,
                    rhs[positive],
                    rcond=relative_pivot_tolerance,
                )
            except np.linalg.LinAlgError as exc:
                self._phase = "failed"
                raise RuntimeError(
                    "rank-revealing sparse Q least-squares failed"
                ) from exc
            indices = self._stencil_index.to_numpy()
            weights = self._stencil_weight.to_numpy().astype(np.float64, copy=False)
            free = self._stencil_free.to_numpy().astype(bool)
            inverse_mass = self._stencil_inverse_mass_per_kg.to_numpy().astype(
                np.float64,
                copy=False,
            )

            def assemble(solution: np.ndarray) -> np.ndarray:
                candidate = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
                for coefficient, row in zip(
                    solution / column_norm,
                    selected,
                    strict=True,
                ):
                    axis = int(row % 3)
                    for support in range(8):
                        if not free[row, support]:
                            continue
                        weight = float(weights[row, support])
                        if weight != 0.0:
                            index = tuple(
                                int(value) for value in indices[row, support]
                            )
                            candidate[index][axis] += (
                                float(inverse_mass[row, support])
                                * weight
                                * float(coefficient)
                            )
                return candidate

            correction = assemble(coefficients)
        correction_f32 = correction.astype(np.float32)
        if not np.all(np.isfinite(correction_f32)):
            self._phase = "failed"
            raise RuntimeError("rank-revealing sparse Q correction is non-finite")
        self._correction.from_numpy(correction_f32)
        self._audit_sparse_q_partition_kernel(partition)
        self._max_residual_mps = float(self._true_candidate_max_residual[None])
        if self._max_residual_mps > tolerance and selected.size:
            minimax = _solve_column_normalized_linf(
                normalized_columns,
                rhs[positive],
                failure_context="rank-revealing sparse Q",
            )
            correction_f32 = assemble(minimax).astype(np.float32)
            if not np.all(np.isfinite(correction_f32)):
                self._phase = "failed"
                raise RuntimeError(
                    "rank-revealing sparse Q minimax correction is non-finite"
                )
            self._correction.from_numpy(correction_f32)
            self._audit_sparse_q_partition_kernel(partition)
            self._max_residual_mps = float(self._true_candidate_max_residual[None])
        self._converged = (
            math.isfinite(self._max_residual_mps)
            and self._max_residual_mps <= tolerance
        )
        self._iterations = 0
        if not self._converged:
            self._phase = "failed"
            raise RuntimeError(
                "rank-revealing sparse Q correction residual exceeds the absolute "
                f"marker constraint tolerance: {self._max_residual_mps} > {tolerance}"
            )
        self._rank_direct_max_structural_residual[None] = self._max_residual_mps
        self._require_final_f32_candidate_audit(tolerance)
        self._snapshot_solved_correction_kernel()
        self._phase = "solved"

    def _collective_isolated_f_only_feasible(self, tolerance: float) -> bool:
        """Seek private F feasibility without altering terminal-Q transaction state.

        A least-squares candidate is the fast path.  If its max residual misses,
        a bounded Chebyshev solve seeks the complete F-only max-norm witness
        before any H authorization is considered.  Neither path alters terminal
        Q's factorization.  Each f64 candidate is cast into f32 and accepted only
        through the existing device all-row audit.
        """

        sparse_backend = (
            self.constraint_capacity
            > HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS
        )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("collective isolated F-only tolerance must be positive")

        # This is the only scratch the isolated witness may mutate.  It is
        # cleared on every exit so an unsuccessful witness cannot leak into
        # the later certificate/H path or a subsequent close attempt.
        self._collective_delta_free.fill(0.0)
        try:
            active = self._collective_row_active.to_numpy().astype(bool)
            rhs = self._collective_rhs.to_numpy().astype(np.float64, copy=False)
            indices = self._collective_index.to_numpy()
            weights = self._collective_weight.to_numpy().astype(
                np.float64, copy=False
            )
            free = self._collective_free.to_numpy().astype(bool)
            inverse_mass = self._collective_inverse_mass.to_numpy().astype(
                np.float64, copy=False
            )
            rows = np.flatnonzero(active)
            if not np.all(np.isfinite(rhs[rows])):
                raise RuntimeError("collective isolated F-only rhs is non-finite")

            correction = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
            # Normalize each correction-space column before SVD.  Mobility
            # determines whether a support is free, but its numerical scale is
            # not part of this yes/no F-space feasibility witness.
            rcond = max(
                1.0e-14,
                64.0 * math.ulp(1.0) * float(self.marker_capacity),
            )
            axis_systems = []
            for axis in range(3):
                axis_rows = rows[rows % 3 == axis]
                dof_column: dict[tuple[int, int, int], int] = {}
                mobility_by_dof: dict[tuple[int, int, int], float] = {}
                for row in axis_rows:
                    for support in range(8):
                        weight = float(weights[row, support])
                        if not math.isfinite(weight):
                            raise RuntimeError(
                                "collective isolated F-only weight is non-finite"
                            )
                        if weight == 0.0:
                            continue
                        index = tuple(
                            int(value) for value in indices[row, support]
                        )
                        if any(
                            value < 0 or value >= self.grid_nodes[dimension]
                            for dimension, value in enumerate(index)
                        ):
                            raise RuntimeError(
                                "collective isolated F-only support is out of bounds"
                            )
                        if not free[row, support]:
                            continue
                        mobility = float(inverse_mass[row, support])
                        if not math.isfinite(mobility) or mobility <= 0.0:
                            raise RuntimeError(
                                "collective isolated F-only free mobility is invalid"
                            )
                        prior_mobility = mobility_by_dof.get(index)
                        if prior_mobility is None:
                            dof_column[index] = len(dof_column)
                            mobility_by_dof[index] = mobility
                        elif mobility != prior_mobility:
                            raise RuntimeError(
                                "collective isolated F-only repeated free support "
                                "has inconsistent mobility"
                            )

                if sparse_backend:
                    from scipy.sparse import lil_matrix

                    matrix = lil_matrix(
                        (axis_rows.size, len(dof_column)), dtype=np.float64
                    )
                else:
                    matrix = np.zeros(
                        (axis_rows.size, len(dof_column)), dtype=np.float64
                    )
                for local_row, row in enumerate(axis_rows):
                    for support in range(8):
                        if not free[row, support]:
                            continue
                        weight = float(weights[row, support])
                        if weight == 0.0:
                            continue
                        index = tuple(
                            int(value) for value in indices[row, support]
                        )
                        matrix[local_row, dof_column[index]] += weight
                if not dof_column:
                    continue
                if sparse_backend:
                    matrix = matrix.tocsr()
                    column_norm = np.sqrt(
                        np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
                    )
                else:
                    column_norm = np.linalg.norm(matrix, axis=0)
                if (
                    not np.all(np.isfinite(column_norm))
                    or np.any(column_norm <= 0.0)
                ):
                    raise RuntimeError(
                        "collective isolated F-only correction basis is singular"
                    )
                solution_lift = None
                if sparse_backend:
                    normalized_matrix, solution_lift = _sparse_structural_decomposition(
                        matrix, column_norm=column_norm, rcond=rcond
                    )
                else:
                    normalized_matrix = matrix / column_norm
                axis_rhs = rhs[axis_rows]
                axis_systems.append(
                    (
                        axis,
                        dof_column,
                        column_norm,
                        normalized_matrix,
                        axis_rhs,
                        solution_lift,
                    )
                )
                try:
                    solution, _, _, _ = np.linalg.lstsq(
                        normalized_matrix,
                        axis_rhs,
                        rcond=rcond,
                    )
                except np.linalg.LinAlgError as exc:
                    raise RuntimeError(
                        "collective isolated F-only least-squares failed"
                    ) from exc
                if solution_lift is not None:
                    solution = solution_lift @ solution
                if not np.all(np.isfinite(solution)):
                    raise RuntimeError(
                        "collective isolated F-only solution is non-finite"
                    )
                for index, column in dof_column.items():
                    correction[index][axis] = solution[column] / column_norm[column]

            correction_f32 = correction.astype(np.float32)
            if not np.all(np.isfinite(correction_f32)):
                raise RuntimeError(
                    "collective isolated F-only correction is non-finite"
                )
            self._collective_delta_free.from_numpy(correction_f32)
            self._measure_collective_target_closure_kernel(0)
            residual = float(self._collective_max_residual[None])
            if not math.isfinite(residual):
                raise RuntimeError(
                    "collective isolated F-only residual is non-finite"
                )
            if residual <= tolerance:
                return True

            correction.fill(0.0)
            for (
                axis,
                dof_column,
                column_norm,
                normalized_matrix,
                axis_rhs,
                solution_lift,
            ) in axis_systems:
                solution = _solve_column_normalized_linf(
                    normalized_matrix,
                    axis_rhs,
                    failure_context="collective isolated F-only",
                )
                if solution_lift is not None:
                    solution = solution_lift @ solution
                for index, column in dof_column.items():
                    correction[index][axis] = solution[column] / column_norm[column]
            correction_f32 = correction.astype(np.float32)
            if not np.all(np.isfinite(correction_f32)):
                raise RuntimeError(
                    "collective isolated F-only minimax correction is non-finite"
                )
            self._collective_delta_free.from_numpy(correction_f32)
            self._measure_collective_target_closure_kernel(0)
            residual = float(self._collective_max_residual[None])
            if not math.isfinite(residual):
                raise RuntimeError(
                    "collective isolated F-only minimax residual is non-finite"
                )
            return residual <= tolerance
        finally:
            self._collective_delta_free.fill(0.0)

    def _collective_isolated_fh_repair(
        self,
        *,
        closure_tolerance: float,
        absolute_tolerance: float,
    ) -> _CollectiveFhRepairResult | None:
        """Build one bounded, certificate-authorized all-row F/H repair.

        F and H use distinct correction-space columns even at one MAC face.
        A certificate row only authorizes H in its row/variable connected
        component; every active row still participates in the per-axis solve.
        The device audit then requires closure accuracy on authorized rows and
        the established absolute accuracy on every active row.
        """

        sparse_backend = (
            self.constraint_capacity
            > HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS
        )
        if (
            not math.isfinite(closure_tolerance)
            or closure_tolerance <= 0.0
            or not math.isfinite(absolute_tolerance)
            or absolute_tolerance <= 0.0
            or closure_tolerance > absolute_tolerance
        ):
            raise ValueError(
                "collective isolated F/H repair requires "
                "0 < closure_tolerance <= absolute_tolerance"
            )

        # The isolated witness is private until the caller accepts its device
        # audit.  A failed or exceptional attempt must leave neither F nor H
        # scratch available to the later apply path.
        self._collective_delta_free.fill(0.0)
        self._collective_delta_hard.fill(0.0)
        self._collective_row_repair_active.fill(0)
        succeeded = False
        try:
            active = self._collective_row_active.to_numpy().astype(bool)
            certificate = self._collective_row_certificate.to_numpy().astype(bool)
            rhs = self._collective_rhs.to_numpy().astype(np.float64, copy=False)
            indices = self._collective_index.to_numpy()
            weights = self._collective_weight.to_numpy().astype(
                np.float64, copy=False
            )
            free = self._collective_free.to_numpy().astype(bool)
            adjustable = self._collective_adjustable.to_numpy().astype(bool)
            inverse_mass = self._collective_inverse_mass.to_numpy().astype(
                np.float64, copy=False
            )
            rows = np.flatnonzero(active)
            if not np.all(np.isfinite(rhs[rows])):
                raise RuntimeError("collective isolated F/H rhs is non-finite")

            correction_free = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
            correction_hard = np.zeros((*self.grid_nodes, 3), dtype=np.float64)
            repair_active = np.zeros(self.constraint_capacity, dtype=np.int32)
            # Keep this rank decision identical to the F-only witness.  The
            # mobility fields admit an F or H column but do not scale the
            # feasibility column space.
            rcond = max(
                1.0e-14,
                64.0 * math.ulp(1.0) * float(self.marker_capacity),
            )
            for axis in range(3):
                axis_rows = rows[rows % 3 == axis]
                parent: dict[tuple[object, ...], tuple[object, ...]] = {}

                def find(node: tuple[object, ...]) -> tuple[object, ...]:
                    root = parent.setdefault(node, node)
                    while root != parent[root]:
                        root = parent[root]
                    while node != root:
                        previous = parent[node]
                        parent[node] = root
                        node = previous
                    return root

                def union(
                    first: tuple[object, ...], second: tuple[object, ...]
                ) -> None:
                    first_root = find(first)
                    second_root = find(second)
                    if first_root != second_root:
                        parent[second_root] = first_root

                for row in axis_rows:
                    row_key = ("row", int(row))
                    find(row_key)
                    for support in range(8):
                        weight = float(weights[row, support])
                        if not math.isfinite(weight):
                            raise RuntimeError(
                                "collective isolated F/H weight is non-finite"
                            )
                        if weight == 0.0:
                            continue
                        index = tuple(
                            int(value) for value in indices[row, support]
                        )
                        if any(
                            value < 0 or value >= self.grid_nodes[dimension]
                            for dimension, value in enumerate(index)
                        ):
                            raise RuntimeError(
                                "collective isolated F/H support is out of bounds"
                            )
                        if free[row, support]:
                            union(row_key, ("free", index))
                        if adjustable[row, support]:
                            union(row_key, ("hard", index))

                authorized_roots = {
                    find(("row", int(row)))
                    for row in axis_rows
                    if certificate[row]
                }
                for row in axis_rows:
                    if find(("row", int(row))) in authorized_roots:
                        repair_active[row] = 1

                dof_column: dict[tuple[str, tuple[int, int, int]], int] = {}
                mobility_by_dof: dict[tuple[str, tuple[int, int, int]], float] = {}
                for row in axis_rows:
                    for support in range(8):
                        weight = float(weights[row, support])
                        if not math.isfinite(weight):
                            raise RuntimeError(
                                "collective isolated F/H weight is non-finite"
                            )
                        if weight == 0.0:
                            continue
                        index = tuple(
                            int(value) for value in indices[row, support]
                        )
                        if any(
                            value < 0 or value >= self.grid_nodes[dimension]
                            for dimension, value in enumerate(index)
                        ):
                            raise RuntimeError(
                                "collective isolated F/H support is out of bounds"
                            )
                        for kind, enabled in (
                            ("free", bool(free[row, support])),
                            (
                                "hard",
                                bool(
                                    adjustable[row, support]
                                    and repair_active[row] != 0
                                ),
                            ),
                        ):
                            if not enabled:
                                continue
                            mobility = float(inverse_mass[row, support])
                            if not math.isfinite(mobility) or mobility <= 0.0:
                                raise RuntimeError(
                                    "collective isolated F/H active mobility is invalid"
                                )
                            key = (kind, index)
                            prior_mobility = mobility_by_dof.get(key)
                            if prior_mobility is None:
                                dof_column[key] = len(dof_column)
                                mobility_by_dof[key] = mobility
                            elif mobility != prior_mobility:
                                raise RuntimeError(
                                    "collective isolated F/H repeated active support "
                                    "has inconsistent mobility"
                                )

                if sparse_backend:
                    from scipy.sparse import lil_matrix

                    matrix = lil_matrix(
                        (axis_rows.size, len(dof_column)), dtype=np.float64
                    )
                else:
                    matrix = np.zeros(
                        (axis_rows.size, len(dof_column)), dtype=np.float64
                    )
                for local_row, row in enumerate(axis_rows):
                    for support in range(8):
                        weight = float(weights[row, support])
                        if weight == 0.0:
                            continue
                        index = tuple(
                            int(value) for value in indices[row, support]
                        )
                        if free[row, support]:
                            matrix[local_row, dof_column[("free", index)]] += weight
                        if (
                            adjustable[row, support]
                            and repair_active[row] != 0
                        ):
                            matrix[local_row, dof_column[("hard", index)]] += weight
                if not dof_column:
                    continue
                if sparse_backend:
                    matrix = matrix.tocsr()
                    column_norm = np.sqrt(
                        np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
                    )
                else:
                    column_norm = np.linalg.norm(matrix, axis=0)
                if (
                    not np.all(np.isfinite(column_norm))
                    or np.any(column_norm <= 0.0)
                ):
                    raise RuntimeError(
                        "collective isolated F/H correction basis is singular"
                    )
                try:
                    if sparse_backend:
                        row_basis, _solution_lift = _sparse_structural_decomposition(
                            matrix, column_norm=column_norm, rcond=rcond
                        )
                        reduced_matrix = np.asarray(matrix.T @ row_basis).T
                    else:
                        structural_left, structural_singular, _ = np.linalg.svd(
                            matrix / column_norm,
                            full_matrices=False,
                        )
                        if (
                            not np.all(np.isfinite(structural_singular))
                            or structural_singular.size == 0
                            or structural_singular[0] <= 0.0
                        ):
                            raise RuntimeError(
                                "collective isolated F/H structural basis is singular"
                            )
                        structural_rank = int(
                            np.count_nonzero(
                                structural_singular
                                > rcond * structural_singular[0]
                            )
                        )
                        if structural_rank == 0:
                            raise RuntimeError(
                                "collective isolated F/H structural rank is zero"
                            )

                        row_basis = structural_left[:, :structural_rank]
                        reduced_matrix = row_basis.T @ matrix
                    # Structural rank is independent of mobility. Solve the
                    # same minimum-energy problem in delta=sqrt(D)*y below.
                    reduced_rhs = row_basis.T @ rhs[axis_rows]
                    mobility = np.empty(len(dof_column), dtype=np.float64)
                    for key, column in dof_column.items():
                        mobility[column] = mobility_by_dof[key]
                    mobility /= float(np.max(mobility))
                    square_root_mobility = np.sqrt(mobility)
                    if (
                        not np.all(np.isfinite(square_root_mobility))
                        or np.any(square_root_mobility <= 0.0)
                    ):
                        raise RuntimeError(
                            "collective isolated F/H mobility scale is invalid"
                        )
                    weighted_matrix = (
                        reduced_matrix
                        * square_root_mobility[np.newaxis, :]
                    )
                    row_norm = np.linalg.norm(weighted_matrix, axis=1)
                    if (
                        not np.all(np.isfinite(row_norm))
                        or np.any(row_norm <= 0.0)
                    ):
                        raise RuntimeError(
                            "collective isolated F/H weighted row basis is singular"
                        )
                    scaled_matrix = weighted_matrix / row_norm[:, np.newaxis]
                    scaled_rhs = reduced_rhs / row_norm
                    orthogonal, triangular = np.linalg.qr(
                        scaled_matrix.T,
                        mode="reduced",
                    )
                    diagonal = np.diag(triangular)
                    if (
                        not np.all(np.isfinite(orthogonal))
                        or not np.all(np.isfinite(triangular))
                        or np.any(diagonal == 0.0)
                    ):
                        raise RuntimeError(
                            "collective isolated F/H weighted factor is singular"
                        )
                    minimum_norm_coordinates = orthogonal @ np.linalg.solve(
                        triangular.T,
                        scaled_rhs,
                    )
                    solution = (
                        square_root_mobility * minimum_norm_coordinates
                    )
                except np.linalg.LinAlgError as exc:
                    raise RuntimeError(
                        "collective isolated F/H weighted solve failed"
                    ) from exc
                if not np.all(np.isfinite(solution)):
                    raise RuntimeError(
                        "collective isolated F/H solution is non-finite"
                    )
                for (kind, index), column in dof_column.items():
                    correction = solution[column]
                    if kind == "free":
                        correction_free[index][axis] = correction
                    else:
                        correction_hard[index][axis] = correction

            correction_free_f32 = correction_free.astype(np.float32)
            correction_hard_f32 = correction_hard.astype(np.float32)
            if (
                not np.all(np.isfinite(correction_free_f32))
                or not np.all(np.isfinite(correction_hard_f32))
            ):
                raise RuntimeError("collective isolated F/H correction is non-finite")
            self._collective_delta_free.from_numpy(correction_free_f32)
            self._collective_delta_hard.from_numpy(correction_hard_f32)
            self._collective_row_repair_active.from_numpy(repair_active)
            self._measure_collective_fh_repair_residual_kernel()
            repair_residual = float(self._collective_max_repair_residual[None])
            residual = float(self._collective_max_residual[None])
            if (
                not math.isfinite(repair_residual)
                or not math.isfinite(residual)
            ):
                raise RuntimeError("collective isolated F/H residual is non-finite")
            succeeded = (
                repair_residual <= closure_tolerance
                and residual <= absolute_tolerance
            )
            if not succeeded:
                return None
            return _CollectiveFhRepairResult(
                repair_max_residual_mps=repair_residual,
                global_max_residual_mps=residual,
                hard_target_dof_count=int(
                    np.count_nonzero(correction_hard_f32)
                ),
                max_abs_hard_target_delta_mps=float(
                    np.max(np.abs(correction_hard_f32), initial=0.0)
                ),
            )
        finally:
            if not succeeded:
                self._collective_delta_free.fill(0.0)
                self._collective_delta_hard.fill(0.0)
                self._collective_row_repair_active.fill(0)

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

        self._pressure_sparse_actual_rank = 0
        self._pressure_nullspace_prepared = False
        self._pressure_nullspace_poisoned = False
        self._pressure_nullspace_apply_count = 0
        self._pressure_nullspace_fluid = None
        self._pressure_nullspace_component_face_valid_mask = None
        self._pressure_actuated_component_mobility = None
        self._pressure_actuation_generation = 0
        self._pressure_nullspace_topology_generation = 0
        self._pressure_nullspace_component_face_valid_mask_generation = 0

    @ti.kernel
    def _reset_rank_direct_diagnostics_kernel(self):
        self._rank_direct_max_structural_residual[None] = 0.0
        self._rank_direct_max_independent_residual[None] = 0.0
        self._rank_direct_max_dependent_residual[None] = 0.0
        self._rank_direct_max_unactuated_residual[None] = 0.0

    def _reset_rank_direct_diagnostics(self) -> None:
        """Retire every externally visible direct-backend diagnostic."""

        self._solve_backend = "pcg"
        self._rank_revealed = False
        self._rank_direct_independent_constraint_count = 0
        self._rank_direct_dependent_constraint_count = 0
        self._rank_direct_unactuated_constraint_count = 0
        self._reset_rank_direct_diagnostics_kernel()

    def _retire_transaction_lifecycle(self) -> None:
        """Fail closed before an attempted transaction can touch device rows."""

        self._phase = "failed"
        self._prepared = False
        self._converged = False
        self._committed = False
        self._markers = None
        self._fluid = None
        self._component_face_valid_mask = None
        self._prepared_obstacle_field = None
        self._marker_count = 0
        self._active_marker_count = 0
        self._constraint_count = 0
        self._iterations = 0
        self._absolute_tolerance_mps = math.nan
        self._max_residual_mps = math.inf
        self._prepared_ledger_generation = -1
        self._prepared_primary_region_id = -1
        self._prepared_secondary_region_id = -1
        self.prepared_sampling_identity = None
        self._prepared_sampling_identity_generation = 0
        self._prepared_topology_generation = 0
        self._prepared_component_face_valid_mask_generation = 0
        self._clear_pressure_nullspace_lifecycle()
        self._reset_rank_direct_diagnostics()

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
        self._clear_pressure_nullspace_candidate_kernel()
        if self._pressure_sparse_backend:
            self._project_pressure_nullspace_sparse_kernel(
                self._pressure_sparse_factor,
                self._pressure_sparse_triangular,
                self._pressure_sparse_dof_indices,
                self._pressure_sparse_sqrt_mobility,
                self._pressure_sparse_actual_rank,
                self._pressure_sparse_dof_count,
            )
        else:
            self._solve_pressure_nullspace_factor_kernel()
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
        """Allocate common linear storage and the small-system dense backend once."""

        if self._pressure_nullspace_resources_allocated:
            return
        constraints = int(self.constraint_capacity)
        dense_bytes = (
            0 if self._pressure_sparse_backend else constraints * constraints * 8
        )
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
        if not self._pressure_sparse_backend:
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
        self._pressure_nullspace_base_resource_bytes = int(estimated_bytes)
        self._pressure_nullspace_resource_bytes = int(estimated_bytes)
        self._pressure_nullspace_resources_allocated = True

    def _prepare_sparse_pressure_factor(self, relative_pivot_tolerance: float) -> None:
        rows, _mobility = self._sparse_metric_rows(
            row_active=self._pressure_nullspace_row_active,
            inverse_mobility=self._pressure_nullspace_inverse_mass_per_kg,
            allow_zero_mobility=True,
        )
        selected, rank_factor, norms = _sparse_normalized_pivoted_cholesky(
            rows, relative_pivot_tolerance=relative_pivot_tolerance
        )
        rank = int(selected.size)
        capacity = max(1, self._pressure_sparse_factor_capacity)
        while capacity < rank:
            capacity *= 2
        dofs = sorted({key for row in selected for key in rows[int(row)]})
        dof_count = len(dofs)
        dof_capacity = max(1, self._pressure_sparse_dof_capacity)
        while dof_capacity < dof_count:
            dof_capacity *= 2
        base_bytes = self._pressure_nullspace_base_resource_bytes
        new_buffer_bytes = (
            dof_capacity * (capacity * 8 + 4 * 4 + 8) + capacity * capacity * 8
        )
        old_buffer_bytes = (
            self._pressure_sparse_dof_capacity
            * (self._pressure_sparse_factor_capacity * 8 + 4 * 4 + 8)
            + self._pressure_sparse_factor_capacity ** 2 * 8
        )
        workspace_bytes = (
            base_bytes + old_buffer_bytes + 2 * new_buffer_bytes
            + 3 * self.constraint_capacity * rank * 8
            + 3 * dof_count * rank * 8 + rank * rank * 8
        )
        if workspace_bytes > HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES:
            raise RuntimeError(
                "sparse pressure factor workspace exceeds memory budget: "
                f"{workspace_bytes} bytes"
            )
        factor = np.zeros((dof_capacity, capacity), dtype=np.float64)
        triangular_buffer = np.zeros((capacity, capacity), dtype=np.float64)
        indices = np.zeros((dof_capacity, 4), dtype=np.int32)
        sqrt_mobility = np.ones(dof_capacity, dtype=np.float64)
        triangular_diagonal = np.empty(0, dtype=np.float64)
        if rank:
            dof_index = {key: index for index, key in enumerate(dofs)}
            selected_rows = np.zeros((dof_count, rank), dtype=np.float64)
            for column, row in enumerate(selected):
                for key, weight in rows[int(row)].items():
                    selected_rows[dof_index[key], column] = weight / norms[row]
            try:
                orthogonal, triangular = np.linalg.qr(selected_rows, mode="reduced")
            except np.linalg.LinAlgError as error:
                raise RuntimeError("sparse pressure reduced QR failed") from error
            triangular_diagonal = np.diag(triangular)
            if (
                not np.all(np.isfinite(orthogonal))
                or not np.all(np.isfinite(triangular))
                or np.any(triangular_diagonal == 0.0)
            ):
                raise RuntimeError("sparse pressure reduced QR is singular")
            factor[:dof_count, :rank] = orthogonal
            triangular_buffer[:rank, :rank] = triangular
            indices[:dof_count] = np.asarray(dofs, dtype=np.int32)
            sqrt_mobility[:dof_count] = np.sqrt([_mobility[key] for key in dofs])
        if (
            capacity != self._pressure_sparse_factor_capacity
            or dof_capacity != self._pressure_sparse_dof_capacity
        ):
            self._pressure_sparse_factor = ti.ndarray(
                dtype=ti.f64, shape=(dof_capacity, capacity)
            )
            self._pressure_sparse_triangular = ti.ndarray(
                dtype=ti.f64, shape=(capacity, capacity)
            )
            self._pressure_sparse_dof_indices = ti.ndarray(
                dtype=ti.i32, shape=(dof_capacity, 4)
            )
            self._pressure_sparse_sqrt_mobility = ti.ndarray(
                dtype=ti.f64, shape=dof_capacity
            )
            self._pressure_sparse_factor_capacity = capacity
            self._pressure_sparse_dof_capacity = dof_capacity
        self._pressure_sparse_factor.from_numpy(factor)
        self._pressure_sparse_triangular.from_numpy(triangular_buffer)
        self._pressure_sparse_dof_indices.from_numpy(indices)
        self._pressure_sparse_sqrt_mobility.from_numpy(sqrt_mobility)
        order = np.full(self.constraint_capacity, -1, dtype=np.int32)
        order[:rank] = selected
        selected_mask = np.zeros(self.constraint_capacity, dtype=np.int32)
        selected_mask[selected] = 1
        inverse_norm = np.zeros(self.constraint_capacity, dtype=np.float64)
        positive = norms > 0.0
        inverse_norm[positive] = 1.0 / norms[positive]
        active = self._pressure_nullspace_row_active.to_numpy().astype(bool)
        dependent = active & positive & (selected_mask == 0)
        residual_diagonal = 1.0 - np.sum(rank_factor * rank_factor, axis=1)
        self._pressure_nullspace_factor_order.from_numpy(order)
        self._pressure_nullspace_factor_row_selected.from_numpy(selected_mask)
        self._pressure_nullspace_row_inverse_norm.from_numpy(inverse_norm)
        self._pressure_nullspace_independent_constraint_count[None] = rank
        self._pressure_nullspace_dependent_constraint_count[None] = int(np.count_nonzero(dependent))
        self._pressure_nullspace_unactuated_constraint_count[None] = int(np.count_nonzero(active & ~positive))
        self._pressure_nullspace_min_factor_pivot[None] = (
            float(np.min(triangular_diagonal ** 2)) if rank else 0.0
        )
        self._pressure_nullspace_max_dependent_normalized_pivot[None] = (
            max(0.0, float(np.max(residual_diagonal[dependent])))
            if np.any(dependent) else 0.0
        )
        self._pressure_nullspace_resource_bytes = base_bytes + new_buffer_bytes
        self._pressure_sparse_actual_rank = rank
        self._pressure_sparse_dof_count = dof_count

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
        relative_pivot_tolerance = max(
            1.0e-14,
            64.0 * math.ulp(1.0) * float(self.marker_capacity),
        )
        if self._pressure_sparse_backend:
            self._prepare_sparse_pressure_factor(relative_pivot_tolerance)
        else:
            self._assemble_pressure_nullspace_schur_kernel()
            self._symmetrize_pressure_nullspace_schur_kernel()
            failure_code = int(self._pressure_nullspace_failure_code[None])
            if failure_code == 3:
                raise RuntimeError(
                    "pressure actuation weight is inconsistent on shared marker support"
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
        a prepared f64 direct projection (Cholesky for small systems, reduced
        orthogonal basis for large systems). It is linear across outer FV-CG
        matvecs and performs no input-dependent inner stopping.
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
        self._retire_transaction_lifecycle()
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

    @staticmethod
    def _collective_closure_result(
        *,
        attempted: bool,
        closed: bool,
        constraint_count: int,
        f_only_converged: bool,
        certificate_count: int,
        immutable_hard_row_count: int,
        repair_applied: bool = False,
        repair_backend: str = "none",
        repair_max_residual_mps: float = 0.0,
        global_max_residual_mps: float = 0.0,
        hard_target_dof_count: int = 0,
        max_abs_hard_target_delta_mps: float = 0.0,
    ) -> dict[str, object]:
        """Return one complete, non-stale collective closure diagnostic."""

        return {
            "attempted": attempted,
            "closed": closed,
            "constraint_count": constraint_count,
            "f_only_converged": f_only_converged,
            "certificate_count": certificate_count,
            "immutable_hard_row_count": immutable_hard_row_count,
            "repair_applied": repair_applied,
            "repair_backend": repair_backend,
            "repair_max_residual_mps": repair_max_residual_mps,
            "global_max_residual_mps": global_max_residual_mps,
            "hard_target_dof_count": hard_target_dof_count,
            "max_abs_hard_target_delta_mps": (
                max_abs_hard_target_delta_mps
            ),
        }

    def close_prospective_owned_hard_targets_collectively(
        self,
        *,
        marker_position_m,
        marker_sample_valid,
        marker_velocity_mps,
        marker_region_id,
        physical_marker_count: int,
        primary_region_id: int,
        secondary_region_id: int,
        prospective_velocity,
        component_face_valid_mask,
        hard_fixed_component_mask,
        external_exact_component_mask,
        adjustable_component_mask,
        claim_target_mps,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        cell_width_x_m,
        cell_width_y_m,
        cell_width_z_m,
        density_kgm3: float,
        sweeps_per_batch: int,
        closure_tolerance_mps: float,
        absolute_tolerance_mps: float,
    ) -> dict[str, object]:
        """Close a certified collective F/H defect before ledger publication."""

        if self._phase in {"prepared", "solved"}:
            raise RuntimeError("collective target closure cannot reuse a pending Q transaction")
        count = int(physical_marker_count)
        sweeps = int(sweeps_per_batch)
        closure_tolerance = float(closure_tolerance_mps)
        absolute_tolerance = float(absolute_tolerance_mps)
        density = float(density_kgm3)
        if count < 0 or count > self.marker_capacity:
            raise ValueError("physical_marker_count exceeds marker_capacity")
        if (
            sweeps <= 0
            or not math.isfinite(closure_tolerance)
            or closure_tolerance <= 0.0
            or not math.isfinite(absolute_tolerance)
            or absolute_tolerance <= 0.0
            or closure_tolerance > absolute_tolerance
        ):
            raise ValueError(
                "collective closure requires 0 < closure_tolerance_mps <= "
                "absolute_tolerance_mps"
            )
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("collective closure density must be positive")

        self._reset_collective_target_closure_kernel()
        self._canonicalize_collective_marker_owners_kernel(
            marker_position_m,
            marker_velocity_mps,
            marker_region_id,
            count,
            int(primary_region_id),
            int(secondary_region_id),
        )
        owner_failure = int(self._collective_owner_failure[None])
        if owner_failure != 0:
            self._reset_collective_target_closure_kernel()
            raise RuntimeError(
                "collective target closure marker canonicalization failed before "
                f"canonical commit: failure_code={owner_failure}"
            )
        self._build_collective_target_closure_rows_kernel(
            marker_position_m, marker_sample_valid, marker_velocity_mps,
            marker_region_id, count, int(primary_region_id), int(secondary_region_id),
            prospective_velocity, component_face_valid_mask,
            hard_fixed_component_mask, external_exact_component_mask,
            adjustable_component_mask,
            cell_face_x_m, cell_face_y_m, cell_face_z_m,
            cell_center_x_m, cell_center_y_m, cell_center_z_m,
            cell_width_x_m, cell_width_y_m, cell_width_z_m, density,
        )
        active_count = int(self._collective_active_count[None])
        immutable_hard_row_count = int(
            self._collective_immutable_hard_row_count[None]
        )
        if active_count == 0:
            self._reset_collective_target_closure_kernel()
            return self._collective_closure_result(
                attempted=False,
                closed=False,
                constraint_count=0,
                f_only_converged=False,
                certificate_count=0,
                immutable_hard_row_count=0,
            )

        # Every post-build exit, including a witness invariant failure, retires
        # private collective rows and counters.  H is applied before a normal
        # return; the finally block clears scratch only and never touches it.
        try:
            # Zero-F owned-H defects must satisfy closure_tolerance even
            # when the private F-only witness accepts absolute_tolerance.
            self._certify_collective_zero_free_rows_kernel(closure_tolerance)
            zero_free_certificate_count = int(
                self._collective_certificate_count[None]
            )
            # Terminal Q measures the zero correction before attempting a
            # solve.  Do the same: a cyclic sweep may move an already
            # acceptable system away from its physical absolute tolerance.
            self._measure_collective_target_closure_kernel(0)
            if (
                zero_free_certificate_count == 0
                and float(self._collective_max_residual[None]) <= absolute_tolerance
            ):
                return self._collective_closure_result(
                    attempted=True,
                    closed=False,
                    constraint_count=active_count,
                    f_only_converged=True,
                    certificate_count=0,
                    immutable_hard_row_count=immutable_hard_row_count,
                )

            # F-only is private scratch.  A bounded isolated witness tries the
            # LS fast path, then a minimax fallback; terminal Q still makes the
            # authoritative decision after ledger publication.
            if (
                zero_free_certificate_count == 0
                and self._collective_isolated_f_only_feasible(absolute_tolerance)
            ):
                return self._collective_closure_result(
                    attempted=True,
                    closed=False,
                    constraint_count=active_count,
                    f_only_converged=True,
                    certificate_count=0,
                    immutable_hard_row_count=immutable_hard_row_count,
                )

            self._certify_collective_proportional_free_rows_kernel(
                absolute_tolerance
            )
            self._certify_collective_zero_free_rows_kernel(closure_tolerance)
            certificate_count = int(self._collective_certificate_count[None])
            if certificate_count == 0:
                return self._collective_closure_result(
                    attempted=True,
                    closed=False,
                    constraint_count=active_count,
                    f_only_converged=False,
                    certificate_count=0,
                    immutable_hard_row_count=immutable_hard_row_count,
                )
            repair = self._collective_isolated_fh_repair(
                closure_tolerance=closure_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if repair is not None:
                self._apply_collective_hard_target_delta_kernel(
                    claim_target_mps, prospective_velocity
                )
                return self._collective_closure_result(
                    attempted=True,
                    closed=True,
                    constraint_count=active_count,
                    f_only_converged=False,
                    certificate_count=certificate_count,
                    immutable_hard_row_count=immutable_hard_row_count,
                    repair_applied=True,
                    repair_backend=(
                        "certificate_authorized_inverse_mass_weighted_lstsq"
                    ),
                    repair_max_residual_mps=(
                        repair.repair_max_residual_mps
                    ),
                    global_max_residual_mps=(
                        repair.global_max_residual_mps
                    ),
                    hard_target_dof_count=repair.hard_target_dof_count,
                    max_abs_hard_target_delta_mps=(
                        repair.max_abs_hard_target_delta_mps
                    ),
                )
            return self._collective_closure_result(
                attempted=True,
                closed=False,
                constraint_count=active_count,
                f_only_converged=False,
                certificate_count=certificate_count,
                immutable_hard_row_count=immutable_hard_row_count,
            )
        finally:
            self._reset_collective_target_closure_kernel()

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
        self._retire_transaction_lifecycle()
        # A new affine J transaction invalidates every pressure factor built
        # from the preceding marker/topology generation, even if validation of
        # the new transaction later fails.

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
        if _uses_marker_constraint_hash(marker_count):
            self._reset_marker_constraint_hash_kernel()
            self._canonicalize_marker_constraints_kernel(
                sample_position_m,
                markers.v_gamma_mps,
                markers.region_id,
                marker_count,
                selected_regions[0],
                selected_regions[1],
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
        if failure_code == 9:
            raise RuntimeError("marker constraint canonical hash table is full")
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
        self._solve_backend = "pcg"
        self._rank_revealed = False
        self._rank_direct_independent_constraint_count = 0
        self._rank_direct_dependent_constraint_count = 0
        self._rank_direct_unactuated_constraint_count = 0
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
        rank_revealing_direct: bool = False,
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
        if not isinstance(rank_revealing_direct, bool):
            raise TypeError("rank_revealing_direct must be a bool")

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
        if (
            rank_revealing_direct
            and int(self._device_converged[None]) == 0
        ):
            self._solve_rank_revealing_direct(tolerance)
            return
        self._compute_initial_rz_kernel()
        iteration_budget = (
            0 if int(self._device_converged[None]) != 0 else iterations
        )
        poll_interval = 8
        for iteration_index in range(iteration_budget):
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
            completed_iterations = iteration_index + 1
            if (
                completed_iterations % poll_interval == 0
                or completed_iterations == iteration_budget
            ) and (
                int(self._device_converged[None]) != 0
                or int(self._failure_code[None]) != 0
            ):
                break

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
        self._require_final_f32_candidate_audit(tolerance)
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
        self._require_final_f32_candidate_audit(
            self._absolute_tolerance_mps
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
            backend=self._solve_backend,
            rank_revealed=bool(self._rank_revealed),
            independent_constraint_count=int(
                self._rank_direct_independent_constraint_count
            ),
            dependent_constraint_count=int(
                self._rank_direct_dependent_constraint_count
            ),
            unactuated_constraint_count=int(
                self._rank_direct_unactuated_constraint_count
            ),
            max_structural_residual_mps=float(
                self._rank_direct_max_structural_residual[None]
            ),
            max_independent_residual_mps=float(
                self._rank_direct_max_independent_residual[None]
            ),
            max_dependent_residual_mps=float(
                self._rank_direct_max_dependent_residual[None]
            ),
            max_unactuated_residual_mps=float(
                self._rank_direct_max_unactuated_residual[None]
            ),
        )


__all__ = [
    "HibmMpmMarkerMacConstraintOperator",
    "HibmMpmMarkerMacConstraintReport",
]
