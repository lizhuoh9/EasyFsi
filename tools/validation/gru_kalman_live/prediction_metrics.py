"""Frozen diagnostic prediction metrics for the R25B live probe."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

TIP_REGION_HEIGHT_FRACTION = 0.10
TIP_REGION_POLICY = "reference_y_top_10_percent"


class PredictionMetricError(ValueError):
    """A prediction metric input violates the fixed marker contract."""


def _velocity(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.dtype != np.float64
        or array.ndim != 2
        or array.shape[1] != 3
        or array.shape[0] < 1
    ):
        raise PredictionMetricError(f"{name} must be float64 (markers, 3)")
    if not np.all(np.isfinite(array)) or not np.all(array[:, 0] == 0.0):
        raise PredictionMetricError(
            f"{name} must be finite with exact-zero x velocity"
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _area(values: Any, *, marker_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (marker_count,) or not np.issubdtype(
        raw.dtype, np.floating
    ):
        raise PredictionMetricError("marker areas must be a floating marker vector")
    area = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise PredictionMetricError("marker areas must be finite and positive")
    return area


def _reference(values: Any, *, marker_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (marker_count, 3) or not np.issubdtype(
        raw.dtype, np.floating
    ):
        raise PredictionMetricError(
            "marker reference positions must be a floating (markers, 3) array"
        )
    reference = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(reference)):
        raise PredictionMetricError("marker reference positions must be finite")
    return reference


def _tip_mask(reference: np.ndarray) -> tuple[np.ndarray, float]:
    reference_y = reference[:, 1]
    minimum = float(np.min(reference_y))
    maximum = float(np.max(reference_y))
    height = maximum - minimum
    if not np.isfinite(height) or height <= 0.0:
        raise PredictionMetricError("marker reference height must be positive")
    threshold = maximum - TIP_REGION_HEIGHT_FRACTION * height
    mask = np.asarray(reference_y >= threshold, dtype=np.bool_)
    if not np.any(mask):
        raise PredictionMetricError("tip-region policy selected no markers")
    return mask, threshold


def compute_live_prediction_metrics(
    candidate: Any,
    *,
    truth: Any,
    carry: Any,
    marker_area_m2: Any,
    marker_reference_positions_m: Any,
) -> dict[str, float | int | str]:
    """Compare one initial guess with the accepted target and carry baseline.

    RMSE uses active y/z components. Area RMSE normalizes fixed reference areas
    over markers. The diagnostic tip region is frozen to the highest 10% of
    reference y. Direction metrics use the carry-to-accepted vector as oracle.
    """

    predicted = _velocity(candidate, name="candidate")
    accepted = _velocity(truth, name="truth")
    baseline = _velocity(carry, name="carry")
    if accepted.shape != predicted.shape or baseline.shape != predicted.shape:
        raise PredictionMetricError("candidate, truth, and carry shapes differ")
    area = _area(marker_area_m2, marker_count=predicted.shape[0])
    reference = _reference(
        marker_reference_positions_m,
        marker_count=predicted.shape[0],
    )
    error = predicted[:, 1:] - accepted[:, 1:]
    per_marker_mse = np.mean(np.square(error), axis=1)
    tip_mask, tip_threshold = _tip_mask(reference)
    oracle_delta = (accepted[:, 1:] - baseline[:, 1:]).reshape(-1)
    model_delta = (predicted[:, 1:] - baseline[:, 1:]).reshape(-1)
    oracle_energy = float(np.dot(oracle_delta, oracle_delta))
    if not np.isfinite(oracle_energy) or oracle_energy <= 0.0:
        raise PredictionMetricError(
            "carry-to-accepted oracle direction has zero energy"
        )
    alpha_parallel = float(np.dot(model_delta, oracle_delta) / oracle_energy)
    orthogonal = model_delta - alpha_parallel * oracle_delta
    r_perp = float(np.sqrt(np.dot(orthogonal, orthogonal) / oracle_energy))
    tip_digest = hashlib.sha256(
        np.ascontiguousarray(tip_mask).tobytes(order="C")
    ).hexdigest()
    return {
        "rmse_active_yz_mps": float(np.sqrt(np.mean(np.square(error)))),
        "area_weighted_rmse_active_yz_mps": float(
            np.sqrt(np.sum(area * per_marker_mse) / np.sum(area))
        ),
        "tip_region_rmse_active_yz_mps": float(
            np.sqrt(np.mean(np.square(error[tip_mask])))
        ),
        "max_marker_error_mps": float(
            np.max(np.linalg.norm(error, axis=1))
        ),
        "alpha_parallel": alpha_parallel,
        "r_perp": r_perp,
        "tip_region_policy": TIP_REGION_POLICY,
        "tip_region_height_fraction": TIP_REGION_HEIGHT_FRACTION,
        "tip_region_reference_y_threshold_m": tip_threshold,
        "tip_region_marker_count": int(np.count_nonzero(tip_mask)),
        "tip_region_mask_sha256": tip_digest,
        "marker_area_sum_m2": float(np.sum(area)),
    }


__all__ = [
    "PredictionMetricError",
    "TIP_REGION_HEIGHT_FRACTION",
    "TIP_REGION_POLICY",
    "compute_live_prediction_metrics",
]
