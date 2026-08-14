"""Weighted minimum-norm solve for HIBM-owned hard marker targets."""

from dataclasses import dataclass
import math

import numpy as np


class MarkerTargetClosureIncompatibleError(RuntimeError):
    """Raised when the adjustable hard targets cannot satisfy all marker rows."""


@dataclass(frozen=True)
class MarkerTargetClosureSolution:
    correction_mps: np.ndarray
    rank: int
    constraint_count: int
    adjustable_dof_count: int
    max_residual_mps: float
    l2_residual_mps: float
    max_abs_correction_mps: float


def solve_weighted_marker_target_closure(
    matrix: np.ndarray,
    residual_mps: np.ndarray,
    inverse_mass_per_kg: np.ndarray,
    *,
    absolute_tolerance_mps: float,
) -> MarkerTargetClosureSolution:
    """Solve ``A delta = residual`` in the inverse-mass minimum norm."""

    coefficients = np.asarray(matrix, dtype=np.float64)
    residual = np.asarray(residual_mps, dtype=np.float64)
    inverse_mass = np.asarray(inverse_mass_per_kg, dtype=np.float64)
    tolerance = float(absolute_tolerance_mps)
    if coefficients.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    row_count, column_count = coefficients.shape
    if residual.shape != (row_count,) or inverse_mass.shape != (column_count,):
        raise ValueError("closure matrix, residual, and inverse-mass shapes disagree")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("absolute_tolerance_mps must be finite and positive")
    if (
        not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(inverse_mass))
        or np.any(inverse_mass <= 0.0)
    ):
        raise ValueError("closure system must contain finite positive mobility")

    mobility_scale = inverse_mass / float(np.max(inverse_mass))
    square_root_mobility = np.sqrt(mobility_scale)
    weighted_matrix = coefficients * square_root_mobility[np.newaxis, :]
    rank_rcond = np.finfo(np.float32).eps * max(weighted_matrix.shape)
    scaled_correction, _sum_squares, rank, _singular_values = np.linalg.lstsq(
        weighted_matrix,
        residual,
        # The operator was assembled in f32 Taichi fields.  Do not invent
        # independent directions below the precision of that source data.
        rcond=rank_rcond,
    )
    correction = square_root_mobility * scaled_correction
    remaining = residual - coefficients @ correction
    max_residual = float(np.max(np.abs(remaining), initial=0.0))
    l2_residual = float(np.linalg.norm(remaining))
    if not math.isfinite(max_residual) or max_residual > tolerance:
        raise MarkerTargetClosureIncompatibleError(
            "weighted marker-target closure is incompatible: "
            f"least_squares_max_residual_mps={max_residual:.9g}, "
            f"tolerance_mps={tolerance:.9g}, rank={int(rank)}, "
            f"constraints={row_count}, adjustable_dofs={column_count}"
        )
    return MarkerTargetClosureSolution(
        correction_mps=correction,
        rank=int(rank),
        constraint_count=row_count,
        adjustable_dof_count=column_count,
        max_residual_mps=max_residual,
        l2_residual_mps=l2_residual,
        max_abs_correction_mps=float(np.max(np.abs(correction), initial=0.0)),
    )


__all__ = [
    "MarkerTargetClosureIncompatibleError",
    "MarkerTargetClosureSolution",
    "solve_weighted_marker_target_closure",
]
