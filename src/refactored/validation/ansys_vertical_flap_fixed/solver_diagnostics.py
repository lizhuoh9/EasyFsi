from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_mass_balance(
    u: np.ndarray,
    v: np.ndarray,
    inlet_mask: np.ndarray,
    outlet_mask: np.ndarray,
    ds: float,
    dy: float,
) -> dict[str, float]:
    del v, ds
    inlet = inlet_mask.astype(bool)
    outlet = outlet_mask.astype(bool)
    inlet_flux = float(np.sum(u[inlet]) * dy)
    outlet_flux = float(np.sum(u[outlet]) * dy)
    denominator = max(abs(inlet_flux), 1.0e-12)
    return {
        "inlet_flux": inlet_flux,
        "outlet_flux": outlet_flux,
        "mass_imbalance": outlet_flux - inlet_flux,
        "mass_imbalance_rel": (outlet_flux - inlet_flux) / denominator,
    }


def apply_outlet_flux_correction(
    u: np.ndarray,
    inlet_mask: np.ndarray,
    outlet_mask: np.ndarray,
    dy: float,
) -> tuple[np.ndarray, dict[str, float]]:
    inlet = inlet_mask.astype(bool)
    outlet = outlet_mask.astype(bool)
    corrected = np.array(u, dtype=np.float64, copy=True)
    inlet_flux = float(np.sum(corrected[inlet]) * dy)
    raw_outlet_flux = float(np.sum(corrected[outlet]) * dy)
    denominator = max(abs(inlet_flux), 1.0e-12)
    if outlet.any():
        delta_u = (inlet_flux - raw_outlet_flux) / (float(outlet.sum()) * dy)
        corrected[outlet] += delta_u
    else:
        delta_u = 0.0
    corrected_outlet_flux = float(np.sum(corrected[outlet]) * dy)
    return corrected, {
        "inlet_flux": inlet_flux,
        "raw_outlet_flux": raw_outlet_flux,
        "corrected_outlet_flux": corrected_outlet_flux,
        "flux_correction_delta": float(delta_u),
        "mass_imbalance_rel_raw": (raw_outlet_flux - inlet_flux) / denominator,
        "mass_imbalance_rel_corrected": (
            corrected_outlet_flux - inlet_flux
        )
        / denominator,
    }


def compute_velocity_stats(
    u: np.ndarray,
    v: np.ndarray,
    fluid_mask: np.ndarray,
    near_solid_mask: np.ndarray,
) -> dict[str, float]:
    fluid = fluid_mask.astype(bool)
    near_solid = near_solid_mask.astype(bool)
    speed = np.sqrt(u * u + v * v)
    fluid_speed = speed[fluid]
    interior = fluid & ~near_solid
    interior_speed = speed[interior]
    return {
        "max_u": float(np.max(u[fluid])) if fluid.any() else 0.0,
        "max_abs_v": float(np.max(np.abs(v[fluid]))) if fluid.any() else 0.0,
        "max_speed": float(np.max(fluid_speed)) if fluid_speed.size else 0.0,
        "interior_max_speed_excluding_near_solid": float(np.max(interior_speed))
        if interior_speed.size
        else 0.0,
        "p99_speed": float(np.percentile(fluid_speed, 99)) if fluid_speed.size else 0.0,
    }


def compute_centerline_profile(
    u: np.ndarray, y: np.ndarray, s: np.ndarray, fluid_mask: np.ndarray
) -> dict[str, float]:
    center_row = int(np.argmin(np.abs(y)))
    valid = fluid_mask.astype(bool)[center_row, :]
    if not valid.any():
        return {
            "centerline_y": float(y[center_row]),
            "centerline_max_u": 0.0,
            "centerline_max_s": 0.0,
        }
    values = u[center_row, valid]
    valid_s = s[valid]
    index = int(np.argmax(values))
    return {
        "centerline_y": float(y[center_row]),
        "centerline_max_u": float(values[index]),
        "centerline_max_s": float(valid_s[index]),
    }


def write_solver_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def write_mass_balance_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def write_poisson_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def build_step2_manifest(
    *,
    output_root: Path,
    geometry_path: Path,
    bc_path: Path,
    final_fields_path: Path,
    solver_history_path: Path,
    mass_balance_path: Path,
    manifest_path: Path,
    solver_config: dict[str, Any],
    final_summary: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    return {
        "case": "ansys_vertical_flap_fixed_flow",
        "step": "step2_fixed_flap_projection_solver",
        "scope": "fixed-flap incompressible projection solver result; no Fluent parity claim",
        "output_root": _manifest_path(output_root, project_root),
        "sources": {
            "geometry_mask": _manifest_path(geometry_path, project_root),
            "bc_map": _manifest_path(bc_path, project_root),
        },
        "forbidden_sources": {
            "traction_shared_snapshot_diagnostics": "not_used",
        },
        "generated_files": {
            "final_fields": _manifest_path(final_fields_path, project_root),
            "solver_history": _manifest_path(solver_history_path, project_root),
            "mass_balance": _manifest_path(mass_balance_path, project_root),
            "case_manifest_step2": _manifest_path(manifest_path, project_root),
        },
        "claims": {
            "fluent_parity": "not_claimed",
            "fsi": "not_claimed",
            "solver_result": "fixed_flap_projection_solver",
        },
        "variable_convention": {
            "u": "streamwise_minus_Uz",
            "v": "Uy",
            "Uz": "-u",
            "left_to_right_display_flow": "u_positive_Uz_negative",
        },
        "numerical_method": {
            "type": "fractional_step_projection",
            "advection": "upwind",
            "diffusion": "explicit",
            "pressure_poisson": "masked_weighted_jacobi",
        },
        "solver_config": solver_config,
        "final_summary": final_summary,
    }


def build_step4_manifest(
    *,
    output_root: Path,
    geometry_path: Path,
    bc_path: Path,
    final_fields_path: Path,
    solver_history_path: Path,
    mass_balance_path: Path,
    poisson_history_path: Path,
    quality_comparison_path: Path,
    initialization_sensitivity_path: Path,
    manifest_path: Path,
    solver_config: dict[str, Any],
    final_summary: dict[str, Any],
    quality: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    return {
        "case": "ansys_vertical_flap_fixed_flow",
        "step": "step4_solver_stabilization",
        "scope": "fixed-flap stabilized projection solver candidate; no Fluent parity claim",
        "output_root": _manifest_path(output_root, project_root),
        "sources": {
            "geometry_mask": _manifest_path(geometry_path, project_root),
            "bc_map": _manifest_path(bc_path, project_root),
        },
        "forbidden_sources": {
            "traction_shared_snapshot_diagnostics": "not_used",
        },
        "generated_files": {
            "final_fields_stabilized": _manifest_path(
                final_fields_path, project_root
            ),
            "solver_history_stabilized": _manifest_path(
                solver_history_path, project_root
            ),
            "mass_balance_stabilized": _manifest_path(
                mass_balance_path, project_root
            ),
            "poisson_history_stabilized": _manifest_path(
                poisson_history_path, project_root
            ),
            "quality_comparison_step2_vs_stabilized": _manifest_path(
                quality_comparison_path, project_root
            ),
            "initialization_sensitivity": _manifest_path(
                initialization_sensitivity_path, project_root
            ),
            "case_manifest_step4_solver_stabilization": _manifest_path(
                manifest_path, project_root
            ),
        },
        "claims": {
            "fluent_parity": "not_claimed",
            "fsi": "not_claimed",
            "solver_result": "step4_solver_stabilization",
        },
        "variable_convention": {
            "u": "streamwise_minus_Uz",
            "v": "Uy",
            "Uz": "-u",
            "left_to_right_display_flow": "u_positive_Uz_negative",
        },
        "numerical_method": {
            "type": "fractional_step_projection",
            "advection": "upwind",
            "diffusion": "explicit",
            "pressure_poisson": "masked_sor",
            "projection_metrics": "finite_volume",
            "outlet_flux_correction": bool(
                solver_config.get("outlet_flux_correction", False)
            ),
        },
        "solver_config": solver_config,
        "final_summary": final_summary,
        "quality": quality,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def metadata_json(payload: dict[str, Any]) -> np.ndarray:
    return np.array(json.dumps(payload, sort_keys=True), dtype=np.unicode_)
