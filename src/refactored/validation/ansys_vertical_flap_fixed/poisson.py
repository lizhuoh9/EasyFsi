from __future__ import annotations

import numpy as np


def solve_pressure_poisson(
    rhs: np.ndarray,
    p_initial: np.ndarray,
    fluid_mask: np.ndarray,
    solid_mask: np.ndarray,
    outlet_mask: np.ndarray,
    pressure_reference_value: float,
    ds: float,
    dy: float,
    max_iters: int,
    tolerance: float,
    omega: float,
) -> tuple[np.ndarray, dict]:
    del solid_mask
    fluid = fluid_mask.astype(bool)
    outlet = outlet_mask.astype(bool)
    active = fluid & ~outlet
    p = np.array(p_initial, dtype=np.float64, copy=True)
    rhs = np.asarray(rhs, dtype=np.float64)
    p[~fluid] = 0.0
    p[outlet] = float(pressure_reference_value)

    inv_ds2 = 1.0 / (ds * ds)
    inv_dy2 = 1.0 / (dy * dy)
    denominator = 2.0 * (inv_ds2 + inv_dy2)
    relaxation = min(max(float(omega), 0.1), 1.0)
    residual_linf = float("inf")
    residual_l2 = float("inf")
    iterations = 0

    for iteration in range(1, int(max_iters) + 1):
        p_e = _pressure_neighbor(p, fluid, axis=1, direction=1)
        p_w = _pressure_neighbor(p, fluid, axis=1, direction=-1)
        p_n = _pressure_neighbor(p, fluid, axis=0, direction=1)
        p_s = _pressure_neighbor(p, fluid, axis=0, direction=-1)
        p_next = ((p_e + p_w) * inv_ds2 + (p_n + p_s) * inv_dy2 - rhs) / denominator
        p[active] = (1.0 - relaxation) * p[active] + relaxation * p_next[active]
        p[outlet] = float(pressure_reference_value)
        p[~fluid] = 0.0

        if iteration == int(max_iters) or iteration % 10 == 0:
            residual = _poisson_residual(p, rhs, fluid, ds, dy)
            active_residual = residual[active]
            residual_linf = float(np.max(np.abs(active_residual))) if active_residual.size else 0.0
            residual_l2 = float(np.sqrt(np.mean(active_residual * active_residual))) if active_residual.size else 0.0
            iterations = iteration
            if residual_linf <= float(tolerance):
                break

    return p, {
        "poisson_iters": iterations,
        "poisson_residual_linf": residual_linf,
        "poisson_residual_l2": residual_l2,
    }


def _poisson_residual(
    p: np.ndarray, rhs: np.ndarray, fluid: np.ndarray, ds: float, dy: float
) -> np.ndarray:
    p_e = _pressure_neighbor(p, fluid, axis=1, direction=1)
    p_w = _pressure_neighbor(p, fluid, axis=1, direction=-1)
    p_n = _pressure_neighbor(p, fluid, axis=0, direction=1)
    p_s = _pressure_neighbor(p, fluid, axis=0, direction=-1)
    laplace = (p_e - 2.0 * p + p_w) / (ds * ds)
    laplace += (p_n - 2.0 * p + p_s) / (dy * dy)
    residual = laplace - rhs
    residual[~fluid] = 0.0
    return residual


def _pressure_neighbor(
    p: np.ndarray, fluid: np.ndarray, *, axis: int, direction: int
) -> np.ndarray:
    shifted_p = _shift_with_edge(p, axis=axis, direction=direction)
    shifted_fluid = _shift_with_edge(fluid, axis=axis, direction=direction)
    return np.where(shifted_fluid, shifted_p, p)


def _shift_with_edge(values: np.ndarray, *, axis: int, direction: int) -> np.ndarray:
    shifted = np.empty_like(values)
    if axis == 1 and direction == 1:
        shifted[:, :-1] = values[:, 1:]
        shifted[:, -1] = values[:, -1]
    elif axis == 1 and direction == -1:
        shifted[:, 1:] = values[:, :-1]
        shifted[:, 0] = values[:, 0]
    elif axis == 0 and direction == 1:
        shifted[:-1, :] = values[1:, :]
        shifted[-1, :] = values[-1, :]
    elif axis == 0 and direction == -1:
        shifted[1:, :] = values[:-1, :]
        shifted[0, :] = values[0, :]
    else:
        raise ValueError(f"Unsupported shift axis={axis}, direction={direction}")
    return shifted
