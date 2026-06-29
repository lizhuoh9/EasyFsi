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


def solve_pressure_poisson_sor(
    rhs: np.ndarray,
    p_initial: np.ndarray,
    fluid_mask: np.ndarray,
    solid_mask: np.ndarray,
    outlet_mask: np.ndarray,
    pressure_reference_value: float,
    ds: float,
    dy: float,
    max_iters: int,
    tolerance_abs: float,
    tolerance_rel: float,
    omega: float,
    compatibility_correction: bool = True,
    check_interval: int = 25,
    cg_polish: bool = False,
    cg_max_iters: int = 300,
) -> tuple[np.ndarray, dict]:
    """Solve the masked pressure Poisson equation with red-black SOR updates."""
    del solid_mask
    fluid = fluid_mask.astype(bool)
    outlet = outlet_mask.astype(bool) & fluid
    active = fluid & ~outlet
    p = np.array(p_initial, dtype=np.float64, copy=True)
    rhs_work = np.asarray(rhs, dtype=np.float64).copy()
    rhs_work[~active] = 0.0

    correction_applied = False
    if compatibility_correction and active.any():
        rhs_work[active] -= float(np.mean(rhs_work[active]))
        correction_applied = True

    p[~fluid] = 0.0
    p[outlet] = float(pressure_reference_value)
    relaxation = min(max(float(omega), 0.1), 1.95)
    check_every = max(1, int(check_interval))
    color = np.indices(p.shape).sum(axis=0) & 1
    red = active & (color == 0)
    black = active & (color == 1)

    residual = _masked_poisson_residual(p, rhs_work, fluid, outlet, ds, dy)
    initial_values = residual[active]
    initial_linf = (
        float(np.max(np.abs(initial_values))) if initial_values.size else 0.0
    )
    initial_l2 = (
        float(np.sqrt(np.mean(initial_values * initial_values)))
        if initial_values.size
        else 0.0
    )
    rhs_values = rhs_work[active]
    rhs_linf = float(np.max(np.abs(rhs_values))) if rhs_values.size else 0.0
    rhs_l2 = (
        float(np.sqrt(np.mean(rhs_values * rhs_values))) if rhs_values.size else 0.0
    )
    residual_linf = initial_linf
    residual_l2 = initial_l2
    relative = 0.0 if initial_linf == 0.0 else 1.0
    iterations = 0
    converged = initial_linf <= float(tolerance_abs)

    for iteration in range(1, int(max_iters) + 1):
        _sor_color_update(
            p,
            rhs_work,
            fluid,
            outlet,
            red,
            ds,
            dy,
            relaxation,
            float(pressure_reference_value),
        )
        _sor_color_update(
            p,
            rhs_work,
            fluid,
            outlet,
            black,
            ds,
            dy,
            relaxation,
            float(pressure_reference_value),
        )
        iterations = iteration
        if iteration % check_every == 0 or iteration == int(max_iters):
            residual = _masked_poisson_residual(p, rhs_work, fluid, outlet, ds, dy)
            values = residual[active]
            residual_linf = (
                float(np.max(np.abs(values))) if values.size else 0.0
            )
            residual_l2 = (
                float(np.sqrt(np.mean(values * values))) if values.size else 0.0
            )
            relative = residual_linf / max(initial_linf, 1.0e-30)
            if (
                residual_linf <= float(tolerance_abs)
                or relative <= float(tolerance_rel)
            ):
                converged = True
                break

    if not converged and cg_polish:
        p, cg_info = _cg_polish_pressure(
            p,
            rhs_work,
            fluid,
            outlet,
            ds,
            dy,
            float(tolerance_abs),
            float(tolerance_rel),
            int(cg_max_iters),
            initial_linf,
        )
        residual_linf = cg_info["poisson_residual_linf"]
        residual_l2 = cg_info["poisson_residual_l2"]
        relative = cg_info["poisson_residual_linf_relative"]
        iterations += cg_info["cg_iters"]
        converged = cg_info["converged"]

    p[~fluid] = 0.0
    p[outlet] = float(pressure_reference_value)
    return p, {
        "method": "masked_sor",
        "poisson_iters": int(iterations),
        "poisson_residual_linf": float(residual_linf),
        "poisson_residual_l2": float(residual_l2),
        "poisson_residual_linf_initial": float(initial_linf),
        "poisson_residual_l2_initial": float(initial_l2),
        "poisson_residual_linf_relative": float(relative),
        "rhs_linf": float(rhs_linf),
        "rhs_l2": float(rhs_l2),
        "compatibility_correction_applied": bool(correction_applied),
        "converged": bool(converged),
        "cg_polish": bool(cg_polish),
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


def _sor_color_update(
    p: np.ndarray,
    rhs: np.ndarray,
    fluid: np.ndarray,
    outlet: np.ndarray,
    color_mask: np.ndarray,
    ds: float,
    dy: float,
    omega: float,
    pressure_reference_value: float,
) -> None:
    inv_ds2 = 1.0 / (ds * ds)
    inv_dy2 = 1.0 / (dy * dy)
    p_e, w_e = _masked_neighbor(p, fluid, outlet, axis=1, direction=1)
    p_w, w_w = _masked_neighbor(p, fluid, outlet, axis=1, direction=-1)
    p_n, w_n = _masked_neighbor(p, fluid, outlet, axis=0, direction=1)
    p_s, w_s = _masked_neighbor(p, fluid, outlet, axis=0, direction=-1)
    denominator = (w_e + w_w) * inv_ds2 + (w_n + w_s) * inv_dy2
    numerator = (w_e * p_e + w_w * p_w) * inv_ds2
    numerator += (w_n * p_n + w_s * p_s) * inv_dy2
    numerator -= rhs
    safe = color_mask & (denominator > 0.0)
    next_p = np.zeros_like(p, dtype=np.float64)
    next_p[safe] = numerator[safe] / denominator[safe]
    p[safe] = (1.0 - omega) * p[safe] + omega * next_p[safe]
    p[outlet] = pressure_reference_value
    p[~fluid] = 0.0


def _masked_poisson_residual(
    p: np.ndarray,
    rhs: np.ndarray,
    fluid: np.ndarray,
    outlet: np.ndarray,
    ds: float,
    dy: float,
) -> np.ndarray:
    inv_ds2 = 1.0 / (ds * ds)
    inv_dy2 = 1.0 / (dy * dy)
    p_e, w_e = _masked_neighbor(p, fluid, outlet, axis=1, direction=1)
    p_w, w_w = _masked_neighbor(p, fluid, outlet, axis=1, direction=-1)
    p_n, w_n = _masked_neighbor(p, fluid, outlet, axis=0, direction=1)
    p_s, w_s = _masked_neighbor(p, fluid, outlet, axis=0, direction=-1)
    laplace = (w_e * (p_e - p) + w_w * (p_w - p)) * inv_ds2
    laplace += (w_n * (p_n - p) + w_s * (p_s - p)) * inv_dy2
    residual = laplace - rhs
    residual[~fluid] = 0.0
    residual[outlet] = 0.0
    return residual


def _cg_polish_pressure(
    p: np.ndarray,
    rhs: np.ndarray,
    fluid: np.ndarray,
    outlet: np.ndarray,
    ds: float,
    dy: float,
    tolerance_abs: float,
    tolerance_rel: float,
    max_iters: int,
    initial_linf: float,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    active = fluid & ~outlet
    x = np.array(p, dtype=np.float64, copy=True)
    b = np.zeros_like(x)
    b[active] = -rhs[active]
    r = b - _negative_laplace(x, fluid, outlet, ds, dy)
    r[~active] = 0.0
    z = _jacobi_precondition(r, fluid, outlet, ds, dy)
    direction = z.copy()
    rz_old = float(np.sum(r[active] * z[active]))
    converged = False
    residual_linf = float(np.max(np.abs(r[active]))) if active.any() else 0.0
    residual_l2 = (
        float(np.sqrt(np.mean(r[active] * r[active]))) if active.any() else 0.0
    )
    relative = residual_linf / max(initial_linf, 1.0e-30)
    iterations = 0
    for iteration in range(1, max(1, int(max_iters)) + 1):
        operator_direction = _negative_laplace(direction, fluid, outlet, ds, dy)
        denom = float(np.sum(direction[active] * operator_direction[active]))
        if abs(denom) <= 1.0e-30:
            break
        alpha = rz_old / denom
        x[active] += alpha * direction[active]
        r[active] -= alpha * operator_direction[active]
        residual_linf = float(np.max(np.abs(r[active]))) if active.any() else 0.0
        residual_l2 = (
            float(np.sqrt(np.mean(r[active] * r[active]))) if active.any() else 0.0
        )
        relative = residual_linf / max(initial_linf, 1.0e-30)
        iterations = iteration
        if residual_linf <= tolerance_abs or relative <= tolerance_rel:
            converged = True
            break
        z = _jacobi_precondition(r, fluid, outlet, ds, dy)
        rz_new = float(np.sum(r[active] * z[active]))
        if abs(rz_old) <= 1.0e-30:
            break
        beta = rz_new / rz_old
        direction[active] = z[active] + beta * direction[active]
        direction[~active] = 0.0
        rz_old = rz_new
    x[~fluid] = 0.0
    x[outlet] = 0.0
    return x, {
        "cg_iters": int(iterations),
        "poisson_residual_linf": float(residual_linf),
        "poisson_residual_l2": float(residual_l2),
        "poisson_residual_linf_relative": float(relative),
        "converged": bool(converged),
    }


def _negative_laplace(
    p: np.ndarray,
    fluid: np.ndarray,
    outlet: np.ndarray,
    ds: float,
    dy: float,
) -> np.ndarray:
    return -_masked_poisson_residual(p, np.zeros_like(p), fluid, outlet, ds, dy)


def _jacobi_precondition(
    r: np.ndarray,
    fluid: np.ndarray,
    outlet: np.ndarray,
    ds: float,
    dy: float,
) -> np.ndarray:
    inv_ds2 = 1.0 / (ds * ds)
    inv_dy2 = 1.0 / (dy * dy)
    _, w_e = _masked_neighbor(r, fluid, outlet, axis=1, direction=1)
    _, w_w = _masked_neighbor(r, fluid, outlet, axis=1, direction=-1)
    _, w_n = _masked_neighbor(r, fluid, outlet, axis=0, direction=1)
    _, w_s = _masked_neighbor(r, fluid, outlet, axis=0, direction=-1)
    diagonal = (w_e + w_w) * inv_ds2 + (w_n + w_s) * inv_dy2
    active = fluid & ~outlet & (diagonal > 0.0)
    z = np.zeros_like(r, dtype=np.float64)
    z[active] = r[active] / diagonal[active]
    return z


def _masked_neighbor(
    p: np.ndarray,
    fluid: np.ndarray,
    outlet: np.ndarray,
    *,
    axis: int,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_p = _shift_with_edge(p, axis=axis, direction=direction)
    shifted_fluid = _shift_with_edge(fluid, axis=axis, direction=direction)
    shifted_outlet = _shift_with_edge(outlet, axis=axis, direction=direction)
    valid = np.ones_like(fluid, dtype=bool)
    if axis == 1 and direction == 1:
        valid[:, -1] = False
    elif axis == 1 and direction == -1:
        valid[:, 0] = False
    elif axis == 0 and direction == 1:
        valid[-1, :] = False
    elif axis == 0 and direction == -1:
        valid[0, :] = False
    neighbor_active = valid & (shifted_fluid | shifted_outlet)
    weights = neighbor_active.astype(np.float64)
    return np.where(neighbor_active, shifted_p, p), weights


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
