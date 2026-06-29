from __future__ import annotations

import numpy as np


def infer_spacing(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if s.ndim != 1 or y.ndim != 1 or s.size < 2 or y.size < 2:
        raise ValueError("s and y must be one-dimensional arrays with at least 2 values")
    ds_values = np.diff(s)
    dy_values = np.diff(y)
    if np.any(ds_values <= 0.0) or np.any(dy_values <= 0.0):
        raise ValueError("s and y coordinates must be strictly increasing")
    return float(ds_values.mean()), float(dy_values.mean())


def compute_divergence(
    u: np.ndarray, v: np.ndarray, fluid_mask: np.ndarray, ds: float, dy: float
) -> np.ndarray:
    fluid = fluid_mask.astype(bool)
    du_ds = _central_derivative(u, fluid, ds, axis=1, solid_value=0.0)
    dv_dy = _central_derivative(v, fluid, dy, axis=0, solid_value=0.0)
    divergence = du_ds + dv_dy
    divergence[~fluid] = 0.0
    return divergence


def laplacian(
    phi: np.ndarray,
    fluid_mask: np.ndarray,
    solid_mask: np.ndarray,
    ds: float,
    dy: float,
) -> np.ndarray:
    fluid = fluid_mask.astype(bool)
    solid = solid_mask.astype(bool)
    center = np.asarray(phi, dtype=np.float64)
    east = _neighbor_no_slip(center, fluid, solid, axis=1, direction=1)
    west = _neighbor_no_slip(center, fluid, solid, axis=1, direction=-1)
    north = _neighbor_no_slip(center, fluid, solid, axis=0, direction=1)
    south = _neighbor_no_slip(center, fluid, solid, axis=0, direction=-1)
    result = (east - 2.0 * center + west) / (ds * ds)
    result += (north - 2.0 * center + south) / (dy * dy)
    result[~fluid] = 0.0
    return result


def upwind_advection_u(
    u: np.ndarray, v: np.ndarray, fluid_mask: np.ndarray, ds: float, dy: float
) -> np.ndarray:
    return _upwind_advection(u, u, v, fluid_mask, ds, dy)


def upwind_advection_v(
    u: np.ndarray, v: np.ndarray, fluid_mask: np.ndarray, ds: float, dy: float
) -> np.ndarray:
    return _upwind_advection(v, u, v, fluid_mask, ds, dy)


def compute_pressure_gradient(
    p: np.ndarray, fluid_mask: np.ndarray, ds: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    fluid = fluid_mask.astype(bool)
    dp_ds = _central_derivative(p, fluid, ds, axis=1, solid_value=None)
    dp_dy = _central_derivative(p, fluid, dy, axis=0, solid_value=None)
    dp_ds[~fluid] = 0.0
    dp_dy[~fluid] = 0.0
    return dp_ds, dp_dy


def apply_velocity_bc(
    u: np.ndarray,
    v: np.ndarray,
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    u_out = np.array(u, dtype=np.float64, copy=True)
    v_out = np.array(v, dtype=np.float64, copy=True)
    fluid = masks["fluid_mask"].astype(bool)
    solid = masks["solid_mask"].astype(bool)
    inlet = masks["inlet_mask"].astype(bool)
    outlet = masks["outlet_mask"].astype(bool)

    u_out[solid] = 0.0
    v_out[solid] = 0.0
    u_out[inlet] = float(bc_values["inlet_u"])
    v_out[inlet] = float(bc_values["inlet_v"])

    outlet_rows = np.where(outlet[:, -1])[0]
    if outlet_rows.size:
        u_out[outlet_rows, -1] = u_out[outlet_rows, -2]
        v_out[outlet_rows, -1] = v_out[outlet_rows, -2]

    u_out[~fluid] = 0.0
    v_out[~fluid] = 0.0
    return u_out, v_out


def _upwind_advection(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fluid_mask: np.ndarray,
    ds: float,
    dy: float,
) -> np.ndarray:
    fluid = fluid_mask.astype(bool)
    phi_e = _neighbor_value(phi, fluid, axis=1, direction=1, solid_value=0.0)
    phi_w = _neighbor_value(phi, fluid, axis=1, direction=-1, solid_value=0.0)
    phi_n = _neighbor_value(phi, fluid, axis=0, direction=1, solid_value=0.0)
    phi_s = _neighbor_value(phi, fluid, axis=0, direction=-1, solid_value=0.0)

    dphi_ds_backward = (phi - phi_w) / ds
    dphi_ds_forward = (phi_e - phi) / ds
    dphi_dy_backward = (phi - phi_s) / dy
    dphi_dy_forward = (phi_n - phi) / dy

    dphi_ds = np.where(u >= 0.0, dphi_ds_backward, dphi_ds_forward)
    dphi_dy = np.where(v >= 0.0, dphi_dy_backward, dphi_dy_forward)
    advection = u * dphi_ds + v * dphi_dy
    advection[~fluid] = 0.0
    return advection


def _central_derivative(
    phi: np.ndarray,
    fluid: np.ndarray,
    spacing: float,
    *,
    axis: int,
    solid_value: float | None,
) -> np.ndarray:
    forward = _neighbor_value(phi, fluid, axis=axis, direction=1, solid_value=solid_value)
    backward = _neighbor_value(
        phi, fluid, axis=axis, direction=-1, solid_value=solid_value
    )
    derivative = (forward - backward) / (2.0 * spacing)
    derivative[~fluid] = 0.0
    return derivative


def _neighbor_no_slip(
    values: np.ndarray,
    fluid: np.ndarray,
    solid: np.ndarray,
    *,
    axis: int,
    direction: int,
) -> np.ndarray:
    neighbor = _shift_with_edge(values, axis=axis, direction=direction)
    neighbor_fluid = _shift_with_edge(fluid, axis=axis, direction=direction)
    neighbor_solid = _shift_with_edge(solid, axis=axis, direction=direction)
    center = np.asarray(values, dtype=np.float64)
    return np.where(neighbor_fluid, neighbor, np.where(neighbor_solid, 0.0, center))


def _neighbor_value(
    values: np.ndarray,
    fluid: np.ndarray,
    *,
    axis: int,
    direction: int,
    solid_value: float | None,
) -> np.ndarray:
    neighbor = _shift_with_edge(values, axis=axis, direction=direction)
    neighbor_fluid = _shift_with_edge(fluid, axis=axis, direction=direction)
    center = np.asarray(values, dtype=np.float64)
    fallback = center if solid_value is None else float(solid_value)
    return np.where(neighbor_fluid, neighbor, fallback)


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
