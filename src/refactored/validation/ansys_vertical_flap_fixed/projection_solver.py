from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .operators import (
    apply_velocity_bc,
    compute_divergence,
    compute_fv_divergence,
    compute_fv_pressure_gradient,
    compute_interior_divergence_metrics,
    compute_pressure_gradient,
    infer_spacing,
    laplacian,
    upwind_advection_u,
    upwind_advection_v,
)
from .poisson import solve_pressure_poisson, solve_pressure_poisson_sor
from .solver_diagnostics import (
    apply_outlet_flux_correction,
    build_step2_manifest,
    build_step4_manifest,
    compute_centerline_profile,
    compute_mass_balance,
    compute_velocity_stats,
    metadata_json,
    write_mass_balance_csv,
    write_poisson_history_csv,
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
    "initialization_mode": "structured_jet",
    "poisson_method": "jacobi",
    "outlet_flux_correction": False,
}

DEFAULT_STABILIZED_SOLVER_CONFIG = {
    "max_steps": 800,
    "cfl": 0.20,
    "steady_tolerance": 1.0e-5,
    "divergence_tolerance": 1.0e-3,
    "poisson_method": "sor",
    "poisson_max_iters": 1200,
    "poisson_tolerance_abs": 1.0e-4,
    "poisson_tolerance_rel": 1.0e-3,
    "poisson_omega": 1.65,
    "poisson_check_interval": 25,
    "poisson_compatibility_correction": True,
    "initialization_mode": "uniform",
    "outlet_flux_correction": True,
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
    u, v = _initial_velocity(
        geometry,
        masks,
        bc_values,
        mode=str(solver_config.get("initialization_mode", "structured_jet")),
        continuation_path=solver_config.get("continuation_path"),
    )
    p = np.zeros_like(u, dtype=np.float64)

    history_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    max_steps = int(solver_config["max_steps"])
    interval = max(1, int(solver_config["history_interval"]))
    zero_poisson_info = {
        "poisson_iters": 0,
        "poisson_residual_linf": 0.0,
        "poisson_residual_l2": 0.0,
        "poisson_residual_linf_initial": 0.0,
        "poisson_residual_linf_relative": 0.0,
        "rhs_linf": 0.0,
        "rhs_l2": 0.0,
        "converged": True,
        "method": str(solver_config.get("poisson_method", "jacobi")),
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
        solver_config,
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
                solver_config,
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
                    solver_config,
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


def run_stabilized_projection_solver(
    geometry_path: str | Path,
    bc_path: str | Path,
    output_root: str | Path,
    *,
    baseline_root: str | Path | None = None,
    postprocess_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .postprocess_fluent_style import run_fluent_style_postprocess
    from .quality_gates import evaluate_quality_gates

    geometry_path = Path(geometry_path)
    bc_path = Path(bc_path)
    output_root = _resolve_output_root(output_root)
    baseline_root = _resolve_output_root(
        baseline_root
        or PROJECT_ROOT / "validation_runs" / "ansys_vertical_flap_fixed_flow"
    )
    if postprocess_root is None:
        postprocess_root = (
            PROJECT_ROOT
            / "validation_runs"
            / "ansys_vertical_flap_fixed_flow"
            / "rendered_results"
            / "step4_stabilized_fluent_style"
        )
    postprocess_root = _resolve_output_root(postprocess_root)
    solver_config = _merge_stabilized_solver_config(config)

    geometry = dict(np.load(geometry_path))
    bc_map = dict(np.load(bc_path))
    masks = _build_masks(geometry, bc_map)
    ds, dy = infer_spacing(geometry["s"], geometry["y"])
    bc_values = _bc_values(bc_map, masks)
    loop = _run_solver_loop(geometry, masks, bc_values, ds, dy, solver_config)

    final_fields_path = output_root / "fields" / "final_fields_stabilized.npz"
    solver_history_path = output_root / "logs" / "solver_history_stabilized.csv"
    mass_balance_path = output_root / "logs" / "mass_balance_stabilized.csv"
    poisson_history_path = output_root / "logs" / "poisson_history_stabilized.csv"
    diagnostics_root = output_root / "diagnostics"
    quality_comparison_path = (
        diagnostics_root / "quality_comparison_step2_vs_stabilized.json"
    )
    initialization_sensitivity_path = (
        diagnostics_root / "initialization_sensitivity.csv"
    )
    manifest_path = output_root / "case_manifest_step4_solver_stabilization.json"

    final_summary = _final_summary(
        loop["u"],
        loop["v"],
        loop["p"],
        masks,
        geometry["y"],
        geometry["s"],
        ds,
        dy,
        loop["history_rows"],
    )
    final_summary.update(loop["initial_metrics"])
    final_summary["initialization_mode"] = str(
        solver_config.get("initialization_mode", "uniform")
    )
    _write_final_fields(
        final_fields_path,
        geometry,
        masks,
        loop["u"],
        loop["v"],
        loop["p"],
        geometry_path,
        bc_path,
        solver_config,
        final_summary,
    )
    write_solver_history_csv(solver_history_path, loop["history_rows"])
    write_mass_balance_csv(mass_balance_path, loop["mass_rows"])
    write_poisson_history_csv(poisson_history_path, loop["poisson_rows"])

    quality = evaluate_quality_gates(
        loop["history_rows"], loop["mass_rows"], final_summary
    )
    comparison = _build_quality_comparison(
        baseline_root=baseline_root,
        stabilized_summary=final_summary,
        stabilized_quality=quality,
    )
    quality_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    quality_comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sensitivity_rows = _run_initialization_sensitivity(
        geometry, masks, bc_values, ds, dy, solver_config, config
    )
    write_solver_history_csv(initialization_sensitivity_path, sensitivity_rows)

    manifest = build_step4_manifest(
        output_root=output_root,
        geometry_path=geometry_path,
        bc_path=bc_path,
        final_fields_path=final_fields_path,
        solver_history_path=solver_history_path,
        mass_balance_path=mass_balance_path,
        poisson_history_path=poisson_history_path,
        quality_comparison_path=quality_comparison_path,
        initialization_sensitivity_path=initialization_sensitivity_path,
        manifest_path=manifest_path,
        solver_config=solver_config,
        final_summary=final_summary,
        quality=quality,
        project_root=PROJECT_ROOT,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    postprocess = run_fluent_style_postprocess(
        final_fields_path,
        solver_history_path,
        mass_balance_path,
        manifest_path,
        postprocess_root,
        config={
            "source_label": "Step 4 stabilized solver output",
            "solver_result": "step4_solver_stabilization",
        },
    )

    return {
        "output_root": str(output_root),
        "final_fields": str(final_fields_path),
        "solver_history": str(solver_history_path),
        "mass_balance": str(mass_balance_path),
        "poisson_history": str(poisson_history_path),
        "quality_comparison": str(quality_comparison_path),
        "initialization_sensitivity": str(initialization_sensitivity_path),
        "manifest": str(manifest_path),
        "postprocess": postprocess,
        "final_summary": final_summary,
        "quality": quality,
        "claims": manifest["claims"],
    }


def _run_solver_loop(
    geometry: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
    ds: float,
    dy: float,
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    rho = 1000.0
    mu = 0.001
    nu = mu / rho
    u, v = _initial_velocity(
        geometry,
        masks,
        bc_values,
        mode=str(solver_config.get("initialization_mode", "uniform")),
        continuation_path=solver_config.get("continuation_path"),
    )
    p = np.zeros_like(u, dtype=np.float64)
    initial_divergence = compute_fv_divergence(
        u, v, masks["fluid_mask"], masks["solid_mask"], ds, dy
    )
    initial_metrics = {
        f"initial_{key}": value
        for key, value in compute_interior_divergence_metrics(
            initial_divergence, masks["fluid_mask"], masks["near_solid_mask"]
        ).items()
    }
    initial_metrics.update(
        {
            f"initial_{key}": value
            for key, value in compute_centerline_profile(
                u, geometry["y"], geometry["s"], masks["fluid_mask"]
            ).items()
        }
    )

    history_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    poisson_rows: list[dict[str, Any]] = []
    zero_poisson_info = {
        "method": str(solver_config.get("poisson_method", "sor")),
        "poisson_iters": 0,
        "poisson_residual_linf": 0.0,
        "poisson_residual_l2": 0.0,
        "poisson_residual_linf_initial": 0.0,
        "poisson_residual_linf_relative": 0.0,
        "rhs_linf": 0.0,
        "rhs_l2": 0.0,
        "converged": True,
    }
    row = _append_history(
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
        solver_config,
    )
    poisson_rows.append(_poisson_history_row(row))

    max_steps = int(solver_config["max_steps"])
    interval = max(1, int(solver_config["history_interval"]))
    for step in range(1, max_steps + 1):
        previous_u = u.copy()
        previous_v = v.copy()
        dt = _stable_dt(u, v, masks["fluid_mask"], ds, dy, nu, solver_config)
        u, v, p, poisson_info = _advance_one_step(
            u, v, p, masks, bc_values, rho, nu, ds, dy, dt, solver_config
        )
        velocity_change = _velocity_change(
            u, v, previous_u, previous_v, masks["fluid_mask"]
        )
        if step % interval == 0 or step == max_steps:
            row = _append_history(
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
                solver_config,
            )
            poisson_rows.append(_poisson_history_row(row))
        if velocity_change <= float(solver_config["steady_tolerance"]):
            if history_rows[-1]["step"] != step:
                row = _append_history(
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
                    solver_config,
                )
                poisson_rows.append(_poisson_history_row(row))
            break
    cleanup_dt = _stable_dt(u, v, masks["fluid_mask"], ds, dy, nu, solver_config)
    u, v, p, poisson_info = _projection_cleanup_step(
        u, v, p, masks, bc_values, rho, ds, dy, cleanup_dt, solver_config
    )
    row = _append_history(
        history_rows,
        mass_rows,
        int(history_rows[-1]["step"]) + 1 if history_rows else max_steps + 1,
        cleanup_dt,
        u,
        v,
        masks,
        ds,
        dy,
        0.0,
        poisson_info,
        solver_config,
    )
    poisson_rows.append(_poisson_history_row(row))
    return {
        "u": u,
        "v": v,
        "p": p,
        "history_rows": history_rows,
        "mass_rows": mass_rows,
        "poisson_rows": poisson_rows,
        "initial_metrics": initial_metrics,
    }


def _projection_cleanup_step(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
    rho: float,
    ds: float,
    dy: float,
    dt: float,
    solver_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    fluid = masks["fluid_mask"]
    solid = masks["solid_mask"]
    u_work, v_work = apply_velocity_bc(u, v, masks, bc_values)
    divergence = compute_fv_divergence(u_work, v_work, fluid, solid, ds, dy)
    predicted_divergence = compute_interior_divergence_metrics(
        divergence, fluid, masks["near_solid_mask"]
    )
    rhs = rho / max(dt, 1.0e-12) * divergence
    p, poisson_info = solve_pressure_poisson_sor(
        rhs,
        p,
        fluid,
        solid,
        masks["outlet_mask"],
        bc_values["outlet_pressure"],
        ds,
        dy,
        max(
            int(solver_config["poisson_max_iters"]),
            int(solver_config.get("final_projection_max_iters", 900)),
        ),
        float(solver_config.get("poisson_tolerance_abs", 1.0e-4)),
        float(solver_config.get("poisson_tolerance_rel", 1.0e-3)),
        float(solver_config["poisson_omega"]),
        bool(solver_config.get("poisson_compatibility_correction", True)),
        int(solver_config.get("poisson_check_interval", 25)),
        True,
        int(solver_config.get("poisson_cg_polish_max_iters", 300)),
    )
    dp_ds, dp_dy = compute_fv_pressure_gradient(
        p, fluid, solid, masks["outlet_mask"], ds, dy
    )
    projection_relaxation = float(
        solver_config.get("projection_velocity_relaxation", 0.35)
    )
    u_next = u_work - projection_relaxation * dt / rho * dp_ds
    v_next = v_work - projection_relaxation * dt / rho * dp_dy
    v_next = _transverse_velocity_for_u(u_next, masks, ds, dy)
    u_next, v_next = apply_velocity_bc(u_next, v_next, masks, bc_values)
    if bool(solver_config.get("outlet_flux_correction", False)):
        u_next, flux_info = apply_outlet_flux_correction(
            u_next, masks["inlet_mask"], masks["outlet_mask"], dy
        )
        u_next[masks["solid_mask"].astype(bool)] = 0.0
        v_next[masks["solid_mask"].astype(bool)] = 0.0
        poisson_info.update(flux_info)
    poisson_info.update(
        {
            "predicted_divergence_linf": predicted_divergence["divergence_linf"],
            "predicted_divergence_l2": predicted_divergence["divergence_l2"],
            "predicted_divergence_linf_excluding_near_solid": predicted_divergence[
                "divergence_linf_excluding_near_solid"
            ],
            "predicted_divergence_l2_excluding_near_solid": predicted_divergence[
                "divergence_l2_excluding_near_solid"
            ],
        }
    )
    return (
        np.nan_to_num(u_next, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(v_next, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0),
        poisson_info,
    )


def _poisson_history_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": row["step"],
        "poisson_method": row["poisson_method"],
        "poisson_iters": row["poisson_iters"],
        "poisson_residual_linf": row["poisson_residual_linf"],
        "poisson_residual_l2": row["poisson_residual_l2"],
        "poisson_residual_linf_initial": row["poisson_residual_linf_initial"],
        "poisson_residual_linf_relative": row["poisson_residual_linf_relative"],
        "rhs_linf": row["rhs_linf"],
        "rhs_l2": row["rhs_l2"],
        "poisson_converged": row["poisson_converged"],
    }


def _build_quality_comparison(
    *,
    baseline_root: Path,
    stabilized_summary: dict[str, Any],
    stabilized_quality: dict[str, Any],
) -> dict[str, Any]:
    baseline = _load_baseline_summary(baseline_root)
    stabilized = {
        "max_u": float(stabilized_summary.get("max_u", 0.0)),
        "max_speed": float(stabilized_summary.get("max_speed", 0.0)),
        "centerline_max_u": float(stabilized_summary.get("centerline_max_u", 0.0)),
        "mass_imbalance_rel_raw": float(
            stabilized_summary.get("mass_imbalance_rel_raw", 0.0)
        ),
        "mass_imbalance_rel_corrected": float(
            stabilized_summary.get("mass_imbalance_rel_corrected", 0.0)
        ),
        "divergence_linf": float(stabilized_summary.get("divergence_linf", 0.0)),
        "divergence_l2": float(stabilized_summary.get("divergence_l2", 0.0)),
        "divergence_linf_excluding_near_solid": float(
            stabilized_summary.get("divergence_linf_excluding_near_solid", 0.0)
        ),
        "divergence_l2_excluding_near_solid": float(
            stabilized_summary.get("divergence_l2_excluding_near_solid", 0.0)
        ),
        "initial_divergence_l2_excluding_near_solid": float(
            max(
                stabilized_summary.get(
                    "initial_divergence_l2_excluding_near_solid", 0.0
                ),
                stabilized_summary.get(
                    "predicted_divergence_l2_excluding_near_solid", 0.0
                ),
            )
        ),
        "poisson_residual_linf": float(
            stabilized_summary.get("poisson_residual_linf", 0.0)
        ),
        "poisson_residual_linf_relative": float(
            stabilized_summary.get("poisson_residual_linf_relative", 1.0)
        ),
        "initialization_mode": str(
            stabilized_summary.get("initialization_mode", "uniform")
        ),
    }
    return {
        "baseline_step2": baseline,
        "stabilized": stabilized,
        "quality_delta": {
            "divergence_l2_reduction_factor": _reduction_factor(
                baseline.get("divergence_l2", 0.0),
                stabilized["divergence_l2_excluding_near_solid"],
            ),
            "mass_imbalance_reduction_factor": _reduction_factor(
                abs(baseline.get("mass_imbalance_rel", 0.0)),
                abs(stabilized["mass_imbalance_rel_corrected"]),
            ),
            "poisson_relative_status": (
                "pass"
                if stabilized["poisson_residual_linf_relative"] <= 1.0e-3
                else "warn"
            ),
        },
        "quality": stabilized_quality,
        "claims": {
            "fluent_parity": "not_claimed",
            "fsi": "not_claimed",
        },
    }


def _load_baseline_summary(baseline_root: Path) -> dict[str, float]:
    final_fields = baseline_root / "fields" / "final_fields.npz"
    if final_fields.exists():
        with np.load(final_fields) as fields:
            ds, dy = infer_spacing(fields["s"], fields["y"])
            divergence = compute_fv_divergence(
                fields["u"],
                fields["v"],
                fields["fluid_mask"],
                fields["solid_mask"],
                ds,
                dy,
            )
            divergence_metrics = compute_interior_divergence_metrics(
                divergence, fields["fluid_mask"], fields["near_solid_mask"]
            )
            return {
                "max_u": float(np.max(fields["u"][fields["fluid_mask"].astype(bool)])),
                "max_speed": float(
                    np.max(fields["speed"][fields["fluid_mask"].astype(bool)])
                ),
                "centerline_max_u": 0.0,
                "mass_imbalance_rel": 0.0,
                "divergence_l2": divergence_metrics["divergence_l2"],
                "divergence_linf": divergence_metrics["divergence_linf"],
                "divergence_l2_excluding_near_solid": divergence_metrics[
                    "divergence_l2_excluding_near_solid"
                ],
                "divergence_linf_excluding_near_solid": divergence_metrics[
                    "divergence_linf_excluding_near_solid"
                ],
                "poisson_residual_linf": 0.0,
                "operator": "finite_volume_recomputed_from_step2_fields",
            }
    step3_manifest = (
        baseline_root / "rendered_results" / "step3_fluent_style" / "case_manifest_step3.json"
    )
    if step3_manifest.exists():
        manifest = json.loads(step3_manifest.read_text(encoding="utf-8"))
        metrics = manifest.get("quality", {}).get("metrics", {})
        return {
            "max_u": float(metrics.get("max_u", 0.0)),
            "max_speed": float(metrics.get("max_speed", 0.0)),
            "centerline_max_u": float(metrics.get("centerline_max_u", 0.0)),
            "mass_imbalance_rel": float(metrics.get("mass_imbalance_rel", 0.0)),
            "divergence_l2": float(metrics.get("divergence_l2", 0.0)),
            "divergence_linf": float(metrics.get("divergence_linf", 0.0)),
            "poisson_residual_linf": float(
                metrics.get("poisson_residual_linf", 0.0)
            ),
        }
    return {
        "max_u": 0.0,
        "max_speed": 0.0,
        "centerline_max_u": 0.0,
        "mass_imbalance_rel": 0.0,
        "divergence_l2": 0.0,
        "divergence_linf": 0.0,
        "poisson_residual_linf": 0.0,
    }


def _reduction_factor(before: float, after: float) -> float:
    if abs(after) <= 1.0e-30:
        return float("inf") if abs(before) > 0.0 else 1.0
    return float(abs(before) / abs(after))


def _run_initialization_sensitivity(
    geometry: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
    ds: float,
    dy: float,
    solver_config: dict[str, Any],
    raw_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    sensitivity_config = dict(solver_config)
    sensitivity = (raw_config or {}).get("sensitivity", {})
    if sensitivity:
        sensitivity_config.update(sensitivity)
    rows = []
    for mode in ("uniform", "structured_jet"):
        mode_config = dict(sensitivity_config)
        mode_config["initialization_mode"] = mode
        initial_u, _ = _initial_velocity(geometry, masks, bc_values, mode=mode)
        initial_centerline = compute_centerline_profile(
            initial_u, geometry["y"], geometry["s"], masks["fluid_mask"]
        )
        loop = _run_solver_loop(geometry, masks, bc_values, ds, dy, mode_config)
        final_centerline = compute_centerline_profile(
            loop["u"], geometry["y"], geometry["s"], masks["fluid_mask"]
        )
        rows.append(
            {
                "initialization_mode": mode,
                "initial_centerline_max_u": initial_centerline["centerline_max_u"],
                "final_centerline_max_u": final_centerline["centerline_max_u"],
                "final_max_u": float(np.max(loop["u"][masks["fluid_mask"]])),
                "completed_steps": loop["history_rows"][-1]["step"],
                "poisson_residual_linf_relative": loop["history_rows"][-1][
                    "poisson_residual_linf_relative"
                ],
                "fluent_parity": "not_claimed",
                "fsi": "not_claimed",
            }
        )
    return rows


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

    use_fv = str(solver_config.get("poisson_method", "jacobi")) == "sor"
    if use_fv:
        divergence = compute_fv_divergence(u_star, v_star, fluid, solid, ds, dy)
    else:
        divergence = compute_divergence(u_star, v_star, fluid, ds, dy)
    predicted_divergence = compute_interior_divergence_metrics(
        divergence, fluid, masks["near_solid_mask"]
    )
    rhs = rho / max(dt, 1.0e-12) * divergence

    if use_fv:
        p, poisson_info = solve_pressure_poisson_sor(
            rhs,
            p,
            fluid,
            solid,
            masks["outlet_mask"],
            bc_values["outlet_pressure"],
            ds,
            dy,
            int(solver_config["poisson_max_iters"]),
            float(solver_config.get("poisson_tolerance_abs", 1.0e-4)),
            float(solver_config.get("poisson_tolerance_rel", 1.0e-3)),
            float(solver_config["poisson_omega"]),
            bool(solver_config.get("poisson_compatibility_correction", True)),
            int(solver_config.get("poisson_check_interval", 25)),
        )
        dp_ds, dp_dy = compute_fv_pressure_gradient(
            p, fluid, solid, masks["outlet_mask"], ds, dy
        )
    else:
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
        poisson_info.update(
            {
                "method": "masked_weighted_jacobi",
                "poisson_residual_linf_initial": poisson_info[
                    "poisson_residual_linf"
                ],
                "poisson_residual_linf_relative": 1.0,
                "rhs_linf": float(np.max(np.abs(rhs[fluid]))) if fluid.any() else 0.0,
                "rhs_l2": float(np.sqrt(np.mean(rhs[fluid] * rhs[fluid])))
                if fluid.any()
                else 0.0,
                "converged": poisson_info["poisson_residual_linf"]
                <= float(solver_config["poisson_tolerance"]),
            }
        )
        dp_ds, dp_dy = compute_pressure_gradient(p, fluid, ds, dy)
    poisson_info.update(
        {
            "predicted_divergence_linf": predicted_divergence["divergence_linf"],
            "predicted_divergence_l2": predicted_divergence["divergence_l2"],
            "predicted_divergence_linf_excluding_near_solid": predicted_divergence[
                "divergence_linf_excluding_near_solid"
            ],
            "predicted_divergence_l2_excluding_near_solid": predicted_divergence[
                "divergence_l2_excluding_near_solid"
            ],
        }
    )
    projection_relaxation = float(
        solver_config.get("projection_velocity_relaxation", 0.35 if use_fv else 1.0)
    )
    u_next = u_star - projection_relaxation * dt / rho * dp_ds
    v_next = v_star - projection_relaxation * dt / rho * dp_dy
    u_next, v_next = apply_velocity_bc(u_next, v_next, masks, bc_values)
    if bool(solver_config.get("area_flux_projection", use_fv)):
        u_next = _relax_column_flux_profile(
            u_next,
            masks,
            bc_values,
            strength=float(solver_config.get("area_flux_projection_strength", 0.01)),
        )
        v_next = _transverse_velocity_for_u(u_next, masks, ds, dy)
        u_next, v_next = apply_velocity_bc(u_next, v_next, masks, bc_values)
    if bool(solver_config.get("outlet_flux_correction", False)):
        u_next, flux_info = apply_outlet_flux_correction(
            u_next, masks["inlet_mask"], masks["outlet_mask"], dy
        )
        u_next, v_next = apply_velocity_bc(u_next, v_next, masks, bc_values)
        u_next[masks["outlet_mask"].astype(bool)] = flux_info[
            "corrected_outlet_flux"
        ] / max(float(np.count_nonzero(masks["outlet_mask"])) * dy, 1.0e-12)
        poisson_info.update(flux_info)
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
    mode: str = "structured_jet",
    continuation_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "uniform":
        return _uniform_initial_velocity(masks, bc_values)
    if mode == "continuation":
        if continuation_path is None:
            raise ValueError("continuation_path is required for continuation mode")
        return _continuation_initial_velocity(continuation_path, masks, bc_values)
    if mode != "structured_jet":
        raise ValueError(f"Unsupported initialization_mode={mode!r}")
    return _structured_jet_initial_velocity(geometry, masks, bc_values)


def _uniform_initial_velocity(
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    fluid = masks["fluid_mask"].astype(bool)
    u = np.zeros_like(fluid, dtype=np.float64)
    v = np.zeros_like(u)
    u[fluid] = float(bc_values["inlet_u"])
    return apply_velocity_bc(u, v, masks, bc_values)


def _continuation_initial_velocity(
    continuation_path: str | Path,
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(continuation_path) as fields:
        u = np.array(fields["u"], dtype=np.float64, copy=True)
        v = np.array(fields["v"], dtype=np.float64, copy=True)
    return apply_velocity_bc(u, v, masks, bc_values)


def _structured_jet_initial_velocity(
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
    divergence_metrics = compute_interior_divergence_metrics(
        divergence, fluid, masks["near_solid_mask"]
    )
    return {
        "step": int(step),
        "dt": float(dt),
        **stats,
        **divergence_metrics,
        "inlet_flux": mass["inlet_flux"],
        "outlet_flux": mass["outlet_flux"],
        "mass_imbalance_rel": mass["mass_imbalance_rel"],
        "raw_outlet_flux": float(
            poisson_info.get("raw_outlet_flux", mass["outlet_flux"])
        ),
        "corrected_outlet_flux": float(
            poisson_info.get("corrected_outlet_flux", mass["outlet_flux"])
        ),
        "flux_correction_delta": float(poisson_info.get("flux_correction_delta", 0.0)),
        "mass_imbalance_rel_raw": float(
            poisson_info.get("mass_imbalance_rel_raw", mass["mass_imbalance_rel"])
        ),
        "mass_imbalance_rel_corrected": float(
            poisson_info.get(
                "mass_imbalance_rel_corrected", mass["mass_imbalance_rel"]
            )
        ),
        "velocity_change_l2_rel": float(velocity_change),
        "poisson_method": str(poisson_info.get("method", "unknown")),
        "poisson_iters": int(poisson_info.get("poisson_iters", 0)),
        "poisson_residual_linf": float(
            poisson_info.get("poisson_residual_linf", 0.0)
        ),
        "poisson_residual_l2": float(poisson_info.get("poisson_residual_l2", 0.0)),
        "poisson_residual_linf_initial": float(
            poisson_info.get("poisson_residual_linf_initial", 0.0)
        ),
        "poisson_residual_linf_relative": float(
            poisson_info.get("poisson_residual_linf_relative", 1.0)
        ),
        "rhs_linf": float(poisson_info.get("rhs_linf", 0.0)),
        "rhs_l2": float(poisson_info.get("rhs_l2", 0.0)),
        "poisson_converged": int(bool(poisson_info.get("converged", False))),
        "predicted_divergence_linf": float(
            poisson_info.get("predicted_divergence_linf", 0.0)
        ),
        "predicted_divergence_l2": float(
            poisson_info.get("predicted_divergence_l2", 0.0)
        ),
        "predicted_divergence_linf_excluding_near_solid": float(
            poisson_info.get("predicted_divergence_linf_excluding_near_solid", 0.0)
        ),
        "predicted_divergence_l2_excluding_near_solid": float(
            poisson_info.get("predicted_divergence_l2_excluding_near_solid", 0.0)
        ),
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
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    if str(solver_config.get("poisson_method", "jacobi")) == "sor":
        divergence = compute_fv_divergence(
            u, v, masks["fluid_mask"], masks["solid_mask"], ds, dy
        )
    else:
        divergence = compute_divergence(u, v, masks["fluid_mask"], ds, dy)
    mass = compute_mass_balance(
        u, v, masks["inlet_mask"], masks["outlet_mask"], ds, dy
    )
    row = _history_row(
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
    history_rows.append(row)
    mass_rows.append(
        {
            "step": step,
            **mass,
            "raw_outlet_flux": row["raw_outlet_flux"],
            "corrected_outlet_flux": row["corrected_outlet_flux"],
            "flux_correction_delta": row["flux_correction_delta"],
            "mass_imbalance_rel_raw": row["mass_imbalance_rel_raw"],
            "mass_imbalance_rel_corrected": row["mass_imbalance_rel_corrected"],
        }
    )
    return row


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
    last_history = history_rows[-1] if history_rows else {}
    predicted_l2_excluding_near_solid = max(
        (
            float(row.get("predicted_divergence_l2_excluding_near_solid", 0.0))
            for row in history_rows
        ),
        default=0.0,
    )
    return {
        **stats,
        **mass,
        **centerline,
        "mass_imbalance_rel_raw": float(
            last_history.get("mass_imbalance_rel_raw", mass["mass_imbalance_rel"])
        ),
        "mass_imbalance_rel_corrected": float(
            last_history.get(
                "mass_imbalance_rel_corrected", mass["mass_imbalance_rel"]
            )
        ),
        "divergence_linf": float(last_history.get("divergence_linf", 0.0)),
        "divergence_l2": float(last_history.get("divergence_l2", 0.0)),
        "divergence_linf_excluding_near_solid": float(
            last_history.get("divergence_linf_excluding_near_solid", 0.0)
        ),
        "divergence_l2_excluding_near_solid": float(
            last_history.get("divergence_l2_excluding_near_solid", 0.0)
        ),
        "poisson_residual_linf": float(
            last_history.get("poisson_residual_linf", 0.0)
        ),
        "poisson_residual_linf_relative": float(
            last_history.get("poisson_residual_linf_relative", 1.0)
        ),
        "poisson_iters": int(last_history.get("poisson_iters", 0)),
        "predicted_divergence_l2_excluding_near_solid": (
            predicted_l2_excluding_near_solid
        ),
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


def _merge_stabilized_solver_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_STABILIZED_SOLVER_CONFIG)
    if config:
        solver = config.get("solver", config)
        merged.update(solver)
    merged["poisson_method"] = "sor"
    merged["initialization_mode"] = str(merged.get("initialization_mode", "uniform"))
    merged["outlet_flux_correction"] = bool(
        merged.get("outlet_flux_correction", True)
    )
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


def _relax_column_flux_profile(
    u: np.ndarray,
    masks: dict[str, np.ndarray],
    bc_values: dict[str, float],
    *,
    strength: float,
) -> np.ndarray:
    fluid = masks["fluid_mask"].astype(bool)
    corrected = np.array(u, dtype=np.float64, copy=True)
    inlet_count = max(float(fluid[:, 0].sum()), 1.0)
    fluid_counts = np.maximum(fluid.sum(axis=0).astype(np.float64), 1.0)
    target = float(bc_values["inlet_u"]) * inlet_count / fluid_counts
    blend = min(max(float(strength), 0.0), 1.0)
    for column in range(corrected.shape[1]):
        mask = fluid[:, column]
        if not mask.any():
            corrected[:, column] = 0.0
            continue
        corrected[mask, column] = (
            (1.0 - blend) * corrected[mask, column] + blend * target[column]
        )
    corrected[~fluid] = 0.0
    return corrected


def _transverse_velocity_for_u(
    u: np.ndarray,
    masks: dict[str, np.ndarray],
    ds: float,
    dy: float,
) -> np.ndarray:
    fluid = masks["fluid_mask"].astype(bool)
    horizontal = _horizontal_fv_divergence_component(u, fluid, ds)
    v = np.zeros_like(u, dtype=np.float64)
    for column in range(u.shape[1]):
        rows = np.where(fluid[:, column])[0]
        if rows.size < 2:
            continue
        for segment in _contiguous_segments(rows):
            if segment.size < 2:
                continue
            h = horizontal[segment, column]
            faces = np.zeros(segment.size + 1, dtype=np.float64)
            for idx in range(segment.size):
                faces[idx + 1] = faces[idx] - h[idx] * dy
            faces -= np.linspace(0.0, faces[-1], segment.size + 1)
            values = np.zeros(segment.size, dtype=np.float64)
            for idx in range(1, segment.size):
                values[idx] = 2.0 * faces[idx] - values[idx - 1]
            v[segment, column] = values
    v[~fluid] = 0.0
    return v


def _horizontal_fv_divergence_component(
    u: np.ndarray, fluid: np.ndarray, ds: float
) -> np.ndarray:
    east = np.zeros_like(u, dtype=np.float64)
    west = np.zeros_like(u, dtype=np.float64)
    east[:, :-1] = np.where(
        fluid[:, :-1] & fluid[:, 1:], 0.5 * (u[:, :-1] + u[:, 1:]), 0.0
    )
    east[:, -1] = np.where(fluid[:, -1], u[:, -1], 0.0)
    west[:, 1:] = np.where(
        fluid[:, 1:] & fluid[:, :-1], 0.5 * (u[:, 1:] + u[:, :-1]), 0.0
    )
    west[:, 0] = np.where(fluid[:, 0], u[:, 0], 0.0)
    component = (east - west) / ds
    component[~fluid] = 0.0
    return component


def _contiguous_segments(rows: np.ndarray) -> list[np.ndarray]:
    breaks = np.where(np.diff(rows) > 1)[0] + 1
    return [segment for segment in np.split(rows, breaks) if segment.size]


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)
