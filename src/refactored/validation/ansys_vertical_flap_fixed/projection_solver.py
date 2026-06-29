from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .operators import (
    apply_velocity_bc,
    compute_divergence,
    compute_pressure_gradient,
    infer_spacing,
    laplacian,
    upwind_advection_u,
    upwind_advection_v,
)
from .poisson import solve_pressure_poisson
from .solver_diagnostics import (
    build_step2_manifest,
    compute_centerline_profile,
    compute_mass_balance,
    compute_velocity_stats,
    metadata_json,
    write_mass_balance_csv,
    write_solver_history_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_SOLVER_CONFIG = {
    "max_steps": 1200,
    "cfl": 0.35,
    "steady_tolerance": 1.0e-5,
    "divergence_tolerance": 1.0e-3,
    "poisson_max_iters": 400,
    "poisson_tolerance": 1.0e-5,
    "poisson_omega": 1.5,
    "history_interval": 10,
    "write_checkpoints": False,
}


def run_projection_solver(
    geometry_path: str | Path,
    bc_path: str | Path,
    output_root: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    geometry_path = Path(geometry_path)
    bc_path = Path(bc_path)
    output_root = _resolve_output_root(output_root)
    solver_config = _merge_solver_config(config)

    geometry = dict(np.load(geometry_path))
    bc_map = dict(np.load(bc_path))
    masks = _build_masks(geometry, bc_map)
    ds, dy = infer_spacing(geometry["s"], geometry["y"])

    rho = 1000.0
    mu = 0.001
    nu = mu / rho
    bc_values = _bc_values(bc_map, masks)
    u, v = _initial_velocity(geometry, masks, bc_values)
    p = np.zeros_like(u, dtype=np.float64)

    history_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    max_steps = int(solver_config["max_steps"])
    interval = max(1, int(solver_config["history_interval"]))
    zero_poisson_info = {
        "poisson_iters": 0,
        "poisson_residual_linf": 0.0,
        "poisson_residual_l2": 0.0,
    }
    _append_history(
        history_rows,
        mass_rows,
        0,
        _stable_dt(u, v, masks["fluid_mask"], ds, dy, nu, solver_config),
        u,
        v,
        masks,
        ds,
        dy,
        0.0,
        zero_poisson_info,
    )

    for step in range(1, max_steps + 1):
        previous_u = u.copy()
        previous_v = v.copy()
        dt = _stable_dt(u, v, masks["fluid_mask"], ds, dy, nu, solver_config)
        u, v, p, poisson_info = _advance_one_step(
            u, v, p, masks, bc_values, rho, nu, ds, dy, dt, solver_config
        )
        velocity_change = _velocity_change(u, v, previous_u, previous_v, masks["fluid_mask"])

        if step % interval == 0 or step == max_steps:
            _append_history(
                history_rows,
                mass_rows,
                step,
                dt,
                u,
                v,
                masks,
                ds,
                dy,
                velocity_change,
                poisson_info,
            )

        if velocity_change <= float(solver_config["steady_tolerance"]):
            if history_rows[-1]["step"] != step:
                _append_history(
                    history_rows,
                    mass_rows,
                    step,
                    dt,
                    u,
                    v,
                    masks,
                    ds,
                    dy,
                    velocity_change,
                    poisson_info,
                )
            break

    final_fields_path = output_root / "fields" / "final_fields.npz"
    solver_history_path = output_root / "logs" / "solver_history.csv"
    mass_balance_path = output_root / "logs" / "mass_balance.csv"
    manifest_path = output_root / "case_manifest_step2.json"

    final_summary = _final_summary(
        u, v, p, masks, geometry["y"], geometry["s"], ds, dy, history_rows
    )
    _write_final_fields(
        final_fields_path,
        geometry,
        masks,
        u,
        v,
        p,
        geometry_path,
        bc_path,
        solver_config,
        final_summary,
    )
    write_solver_history_csv(solver_history_path, history_rows)
    write_mass_balance_csv(mass_balance_path, mass_rows)
    manifest = build_step2_manifest(
        output_root=output_root,
        geometry_path=geometry_path,
        bc_path=bc_path,
        final_fields_path=final_fields_path,
        solver_history_path=solver_history_path,
        mass_balance_path=mass_balance_path,
        manifest_path=manifest_path,
        solver_config=solver_config,
        final_summary=final_summary,
        project_root=PROJECT_ROOT,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "output_root": str(output_root),
        "final_fields": str(final_fields_path),
        "solver_history": str(solver_history_path),
        "mass_balance": str(mass_balance_path),
        "manifest": str(manifest_path),
        "final_summary": final_summary,
        "claims": manifest["claims"],
    }


def _advance_one_step(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
    rho: float,
    nu: float,
    ds: float,
    dy: float,
    dt: float,
    solver_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    u, v = apply_velocity_bc(u, v, masks, bc_values)
    fluid = masks["fluid_mask"]
    solid = masks["solid_mask"]

    adv_u = upwind_advection_u(u, v, fluid, ds, dy)
    adv_v = upwind_advection_v(u, v, fluid, ds, dy)
    diff_u = laplacian(u, fluid, solid, ds, dy)
    diff_v = laplacian(v, fluid, solid, ds, dy)
    u_star = u + dt * (-adv_u + nu * diff_u)
    v_star = v + dt * (-adv_v + nu * diff_v)
    u_star, v_star = apply_velocity_bc(u_star, v_star, masks, bc_values)

    rhs = rho / max(dt, 1.0e-12) * compute_divergence(u_star, v_star, fluid, ds, dy)
    p, poisson_info = solve_pressure_poisson(
        rhs,
        p,
        fluid,
        solid,
        masks["outlet_mask"],
        bc_values["outlet_pressure"],
        ds,
        dy,
        int(solver_config["poisson_max_iters"]),
        float(solver_config["poisson_tolerance"]),
        float(solver_config["poisson_omega"]),
    )
    dp_ds, dp_dy = compute_pressure_gradient(p, fluid, ds, dy)
    u_next = u_star - dt / rho * dp_ds
    v_next = v_star - dt / rho * dp_dy
    u_next, v_next = apply_velocity_bc(u_next, v_next, masks, bc_values)
    return (
        np.nan_to_num(u_next, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(v_next, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0),
        poisson_info,
    )


def _initial_velocity(
    geometry: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    fluid = masks["fluid_mask"].astype(bool)
    s = geometry["s"].astype(np.float64)
    y = geometry["y"].astype(np.float64)
    S = geometry["S"].astype(np.float64)
    Y = geometry["Y"].astype(np.float64)
    gap = geometry["gap_mask"].astype(bool)
    inlet_u = float(bc_values["inlet_u"])

    fluid_counts = np.maximum(fluid.sum(axis=0).astype(np.float64), 1.0)
    inlet_count = max(float(fluid[:, 0].sum()), 1.0)
    area_profile = inlet_u * inlet_count / fluid_counts
    area_profile = _smooth_profile(area_profile, passes=3)

    if gap.any():
        gap_columns = np.where(gap.any(axis=0))[0]
        throat_column = int(gap_columns[len(gap_columns) // 2])
    else:
        throat_column = int(np.argmin(fluid_counts))
    throat_s = float(s[throat_column])
    throat_u = float(max(area_profile[throat_column], inlet_u))
    duct_height = float(y[-1] - y[0])
    gap_height = max(float(np.count_nonzero(gap[:, throat_column])) * (y[1] - y[0]), 1.0e-6)
    downstream_distance = np.maximum(S - throat_s, 0.0)
    decay_length = max(0.25 * float(s[-1] - s[0]), 4.0 * gap_height)
    jet_center_speed = inlet_u + (throat_u - inlet_u) * np.exp(
        -downstream_distance / decay_length
    )
    jet_width = 0.5 * gap_height + 0.20 * downstream_distance + 0.04 * duct_height
    jet_shape = np.exp(-((Y / np.maximum(jet_width, 1.0e-9)) ** 2))
    downstream_jet = inlet_u + (jet_center_speed - inlet_u) * jet_shape

    u = np.maximum(area_profile[np.newaxis, :], downstream_jet)
    u[~fluid] = 0.0
    u = _balance_column_flux(u, fluid, area_profile)
    du_ds = np.gradient(u, s, axis=1, edge_order=1)
    v = _continuity_transverse_velocity(du_ds, y, fluid)
    v = np.clip(v, -0.75 * inlet_u, 0.75 * inlet_u)
    v[~fluid] = 0.0
    return apply_velocity_bc(u, v, masks, bc_values)


def _history_row(
    step: int,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    divergence: np.ndarray,
    masks: dict[str, np.ndarray],
    ds: float,
    dy: float,
    mass: dict[str, float],
    velocity_change: float,
    poisson_info: dict[str, float],
) -> dict[str, Any]:
    fluid = masks["fluid_mask"].astype(bool)
    stats = compute_velocity_stats(u, v, fluid, masks["near_solid_mask"])
    divergence_values = divergence[fluid]
    divergence_linf = float(np.max(np.abs(divergence_values))) if divergence_values.size else 0.0
    divergence_l2 = float(np.sqrt(np.mean(divergence_values * divergence_values))) if divergence_values.size else 0.0
    return {
        "step": int(step),
        "dt": float(dt),
        **stats,
        "divergence_linf": divergence_linf,
        "divergence_l2": divergence_l2,
        "inlet_flux": mass["inlet_flux"],
        "outlet_flux": mass["outlet_flux"],
        "mass_imbalance_rel": mass["mass_imbalance_rel"],
        "velocity_change_l2_rel": float(velocity_change),
        "poisson_iters": int(poisson_info["poisson_iters"]),
        "poisson_residual_linf": float(poisson_info["poisson_residual_linf"]),
        "ds": float(ds),
        "dy": float(dy),
    }


def _append_history(
    history_rows: list[dict[str, Any]],
    mass_rows: list[dict[str, Any]],
    step: int,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    masks: dict[str, np.ndarray],
    ds: float,
    dy: float,
    velocity_change: float,
    poisson_info: dict[str, float],
) -> None:
    divergence = compute_divergence(u, v, masks["fluid_mask"], ds, dy)
    mass = compute_mass_balance(
        u, v, masks["inlet_mask"], masks["outlet_mask"], ds, dy
    )
    history_rows.append(
        _history_row(
            step,
            dt,
            u,
            v,
            divergence,
            masks,
            ds,
            dy,
            mass,
            velocity_change,
            poisson_info,
        )
    )
    mass_rows.append({"step": step, **mass})


def _final_summary(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    masks: dict[str, np.ndarray],
    y: np.ndarray,
    s: np.ndarray,
    ds: float,
    dy: float,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del p
    mass = compute_mass_balance(u, v, masks["inlet_mask"], masks["outlet_mask"], ds, dy)
    stats = compute_velocity_stats(u, v, masks["fluid_mask"], masks["near_solid_mask"])
    centerline = compute_centerline_profile(u, y, s, masks["fluid_mask"])
    return {
        **stats,
        **mass,
        **centerline,
        "history_rows": len(history_rows),
        "completed_steps": int(history_rows[-1]["step"]) if history_rows else 0,
    }


def _write_final_fields(
    path: Path,
    geometry: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    geometry_path: Path,
    bc_path: Path,
    solver_config: dict[str, Any],
    final_summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    speed = np.sqrt(u * u + v * v)
    speed[~masks["fluid_mask"].astype(bool)] = 0.0
    metadata = {
        "sources": {
            "geometry_mask": _manifest_path(geometry_path),
            "bc_map": _manifest_path(bc_path),
        },
        "claims": {
            "fluent_parity": "not_claimed",
            "fsi": "not_claimed",
        },
        "solver_config": solver_config,
        "final_summary": final_summary,
    }
    np.savez_compressed(
        path,
        s=geometry["s"],
        y=geometry["y"],
        S=geometry["S"],
        Y=geometry["Y"],
        Z=geometry["Z"],
        u=u,
        v=v,
        Uz=-u,
        Uy=v,
        p=p,
        speed=speed,
        streamwise_minus_Uz=u,
        fluid_mask=masks["fluid_mask"],
        solid_mask=masks["solid_mask"],
        inlet_mask=masks["inlet_mask"],
        outlet_mask=masks["outlet_mask"],
        wall_noslip_mask=masks["wall_noslip_mask"],
        flap_noslip_mask=masks["flap_noslip_mask"],
        near_solid_mask=masks["near_solid_mask"],
        metadata_json=metadata_json(metadata),
    )


def _build_masks(
    geometry: dict[str, np.ndarray], bc_map: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    fluid = geometry["fluid_mask"].astype(bool)
    solid = geometry["solid_mask"].astype(bool)
    near_solid = _near_solid_mask(fluid, solid)
    return {
        "fluid_mask": fluid,
        "solid_mask": solid,
        "inlet_mask": bc_map["inlet_mask"].astype(bool),
        "outlet_mask": bc_map["outlet_mask"].astype(bool),
        "wall_noslip_mask": bc_map["wall_noslip_mask"].astype(bool),
        "flap_noslip_mask": bc_map["flap_noslip_mask"].astype(bool),
        "near_solid_mask": near_solid,
    }


def _near_solid_mask(fluid: np.ndarray, solid: np.ndarray) -> np.ndarray:
    near = np.zeros_like(fluid, dtype=bool)
    near[:, 1:] |= solid[:, :-1]
    near[:, :-1] |= solid[:, 1:]
    near[1:, :] |= solid[:-1, :]
    near[:-1, :] |= solid[1:, :]
    return near & fluid


def _bc_values(
    bc_map: dict[str, np.ndarray], masks: dict[str, np.ndarray]
) -> dict[str, float]:
    inlet = masks["inlet_mask"].astype(bool)
    outlet = masks["outlet_mask"].astype(bool)
    inlet_u = float(np.mean(-bc_map["inlet_Uz"][inlet])) if inlet.any() else 0.0
    inlet_v = float(np.mean(bc_map["inlet_Uy"][inlet])) if inlet.any() else 0.0
    outlet_pressure = (
        float(np.mean(bc_map["outlet_pressure"][outlet])) if outlet.any() else 0.0
    )
    return {
        "inlet_u": inlet_u,
        "inlet_v": inlet_v,
        "outlet_pressure": outlet_pressure,
    }


def _stable_dt(
    u: np.ndarray,
    v: np.ndarray,
    fluid_mask: np.ndarray,
    ds: float,
    dy: float,
    nu: float,
    solver_config: dict[str, Any],
) -> float:
    fluid = fluid_mask.astype(bool)
    max_velocity = float(
        max(np.max(np.abs(u[fluid])), np.max(np.abs(v[fluid])), 1.0e-12)
    )
    dt_adv = float(solver_config["cfl"]) * min(ds, dy) / max_velocity
    dt_diff = 0.25 * min(ds, dy) ** 2 / max(nu, 1.0e-12)
    return float(min(dt_adv, dt_diff))


def _velocity_change(
    u: np.ndarray,
    v: np.ndarray,
    previous_u: np.ndarray,
    previous_v: np.ndarray,
    fluid_mask: np.ndarray,
) -> float:
    fluid = fluid_mask.astype(bool)
    delta = (u - previous_u)[fluid] ** 2 + (v - previous_v)[fluid] ** 2
    base = previous_u[fluid] ** 2 + previous_v[fluid] ** 2
    return float(np.sqrt(np.sum(delta)) / max(np.sqrt(np.sum(base)), 1.0e-12))


def _merge_solver_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_SOLVER_CONFIG)
    if config:
        solver = config.get("solver", config)
        merged.update(solver)
    return merged


def _resolve_output_root(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _smooth_profile(values: np.ndarray, passes: int) -> np.ndarray:
    smoothed = values.astype(np.float64).copy()
    for _ in range(int(passes)):
        smoothed[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
    return smoothed


def _continuity_transverse_velocity(
    du_ds: np.ndarray, y: np.ndarray, fluid: np.ndarray
) -> np.ndarray:
    dy = float(np.diff(y).mean())
    v = np.zeros_like(du_ds, dtype=np.float64)
    for row in range(1, du_ds.shape[0]):
        v[row, :] = v[row - 1, :] - du_ds[row - 1, :] * dy
    height = max(float(y[-1] - y[0]), 1.0e-12)
    eta = ((y - y[0]) / height)[:, np.newaxis]
    v -= eta * v[-1:, :]
    v[~fluid] = 0.0
    return v


def _balance_column_flux(
    u: np.ndarray, fluid: np.ndarray, area_profile: np.ndarray
) -> np.ndarray:
    balanced = np.array(u, dtype=np.float64, copy=True)
    for column in range(balanced.shape[1]):
        mask = fluid[:, column]
        if not mask.any():
            balanced[:, column] = 0.0
            continue
        target_sum = float(area_profile[column]) * float(mask.sum())
        current_sum = float(np.sum(balanced[mask, column]))
        balanced[mask, column] -= (current_sum - target_sum) / float(mask.sum())
    balanced[~fluid] = 0.0
    return balanced


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)
