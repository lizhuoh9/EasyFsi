from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(r"D:\working\taichi\env\python.exe")


def _float_list(csv_text: str) -> list[float]:
    values: list[float] = []
    for token in csv_text.split(","):
        stripped = token.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _run_case(
    *,
    python: Path,
    output_dir: Path,
    grid_scale: float,
    solid_density_scale: float,
    membrane_thickness_scale: float,
    extra_args: list[str],
    timeout_s: float,
    rerun: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    command = [
        str(python),
        str(REPO_ROOT / "cases" / "squid_soft_robot.py"),
        "--output-dir",
        str(output_dir),
        "--steps",
        "1",
        "--grid-scale",
        f"{grid_scale:.12g}",
        "--solid-density-scale",
        f"{solid_density_scale:.12g}",
        "--membrane-thickness-scale",
        f"{membrane_thickness_scale:.12g}",
        "--solid-model",
        "tri_mooney_shell_mpm",
        "--use-nozzle-taper",
        "--use-region14-aperture-carve",
        "--pressure-solver",
        "fv_cg",
        "--cg-tolerance",
        "1e-6",
        "--cg-preconditioner",
        "jacobi",
        "--divergence-cleanup-iterations",
        "0",
        "--fluid-substeps",
        "1",
        "--ibm-correction-iterations",
        "1",
        "--solid-mpm-velocity-damping",
        "1",
        "--fsi-coupling-iterations",
        "6",
        "--fsi-coupling-solver",
        "iqn_ils",
        "--fsi-coupling-tolerance-n",
        "1e-6",
        "--fsi-coupling-target-map-relaxation",
        "1",
        "--interface-reaction-relaxation",
        "1",
        "--no-interface-reaction-aitken",
        "--interface-reaction-robin-impedance-ns-m",
        "0",
        "--fsi-constraint-force-solid-mobility-ratio",
        "0",
        "--no-fsi-solid-response-mobility-coupling",
        "--fsi-velocity-target-solid-mobility-ratio",
        "0",
        "--no-fsi-solid-response-velocity-mobility-coupling",
        "--fsi-velocity-constraint-blend",
        "0",
        "--fsi-velocity-constraint-solid-mobility-ratio",
        "0",
        *extra_args,
    ]
    env = os.environ.copy()
    env["TI_OFFLINE_CACHE"] = "0"
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    (output_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"case {output_dir.name} failed with return code {completed.returncode}; "
            f"see {output_dir / 'command.json'}"
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json for {output_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _area_weighted_surface_mass(summary: dict[str, Any]) -> float:
    budget = summary["solid_surface_mass_budget"]
    diagnostics = summary.get("tri_surface_diagnostics", {})
    areas = diagnostics.get("diagnostic_area_m2_by_region", {})
    main_area = float(areas.get("7", 0.0))
    tail_area = float(areas.get("8", 0.0))
    total_area = main_area + tail_area
    main_mass = float(budget["main_surface_mass_kg_m2"])
    tail_mass = float(budget["tail_surface_mass_kg_m2"])
    if total_area <= 0.0:
        return 0.5 * (main_mass + tail_mass)
    return (main_area * main_mass + tail_area * tail_mass) / total_area


def _interface_spacing(summary: dict[str, Any]) -> float:
    spacing = summary.get("fluid_grid_spacing_m")
    if spacing is not None:
        return min(float(value) for value in spacing)
    return min(float(value) for value in summary["fluid_grid_min_spacing_m"])


def _case_row(
    *,
    label: str,
    grid_scale: float,
    solid_density_scale: float,
    membrane_thickness_scale: float,
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    raw_map = float(summary["max_fsi_coupling_raw_interface_map_amplification"])
    h_interface_m = _interface_spacing(summary)
    surface_mass_kg_m2 = _area_weighted_surface_mass(summary)
    return {
        "label": label,
        "output_dir": str(output_dir),
        "grid_scale": float(grid_scale),
        "solid_density_scale": float(solid_density_scale),
        "membrane_thickness_scale": float(membrane_thickness_scale),
        "h_interface_m": h_interface_m,
        "surface_mass_kg_m2": surface_mass_kg_m2,
        "surface_mass_times_h_kg_m": surface_mass_kg_m2 * h_interface_m,
        "surface_mass_over_h_kg_m3": surface_mass_kg_m2 / max(h_interface_m, 1.0e-30),
        "raw_map": raw_map,
        "converged": bool(summary["checks"]["fsi_coupling_converged"]),
        "raw_map_stable": bool(summary["checks"]["fsi_physical_interface_map_stable"]),
        "iqn_ils_least_squares_update_count": int(
            summary["max_fsi_coupling_iqn_ils_least_squares_update_count"]
        ),
        "solver_map": float(summary["max_fsi_coupling_interface_map_amplification"]),
        "selected_map": float(
            summary["max_fsi_coupling_physical_interface_map_amplification"]
        ),
        "step_wall_time_s": float(summary["timing"]["max_step_wall_time_s"]),
    }


def _fit_log_scaling(rows: list[dict[str, Any]]) -> dict[str, float]:
    import numpy as np

    finite_rows = [
        row
        for row in rows
        if row["raw_map"] > 0.0
        and row["h_interface_m"] > 0.0
        and row["surface_mass_kg_m2"] > 0.0
        and math.isfinite(row["raw_map"])
    ]
    if len(finite_rows) < 3:
        return {}
    matrix = np.asarray(
        [
            [
                1.0,
                math.log(row["h_interface_m"]),
                math.log(row["surface_mass_kg_m2"]),
            ]
            for row in finite_rows
        ],
        dtype=float,
    )
    target = np.asarray([math.log(row["raw_map"]) for row in finite_rows], dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficients
    rms_log_error = float(math.sqrt(float(np.mean(residual * residual))))
    return {
        "log_c": float(coefficients[0]),
        "h_exponent": float(coefficients[1]),
        "surface_mass_exponent": float(coefficients[2]),
        "c": float(math.exp(coefficients[0])),
        "rms_log_error": rms_log_error,
    }


def _go_no_go(rows: list[dict[str, Any]], fit: dict[str, float]) -> dict[str, Any]:
    baseline = min(rows, key=lambda row: (row["grid_scale"], row["solid_density_scale"]))
    raw = float(baseline["raw_map"])
    h = float(baseline["h_interface_m"])
    mass = float(baseline["surface_mass_kg_m2"])
    physically_plausible_mass_limit = 20.0 * mass
    physically_plausible_h_limit = 4.0 * h
    required_mass_at_baseline_h = raw * mass
    required_h_at_baseline_mass = raw * h
    required_product = raw * h * mass
    h_exponent = float(fit.get("h_exponent", float("nan")))
    mass_exponent = float(fit.get("surface_mass_exponent", float("nan")))
    c = float(fit.get("c", float("nan")))
    fitted_required_mass_at_baseline_h = None
    fitted_required_h_at_baseline_mass = None
    if c > 0.0 and math.isfinite(c):
        if mass_exponent < -1.0e-12:
            fitted_required_mass_at_baseline_h = (
                1.0 / max(c * (h ** h_exponent), 1.0e-300)
            ) ** (1.0 / mass_exponent)
        if abs(h_exponent) > 1.0e-12:
            fitted_required_h_at_baseline_mass = (
                1.0 / max(c * (mass ** mass_exponent), 1.0e-300)
            ) ** (1.0 / h_exponent)
    mass_route_possible = (
        fitted_required_mass_at_baseline_h is not None
        and fitted_required_mass_at_baseline_h <= physically_plausible_mass_limit
    )
    h_coarsening_helps = math.isfinite(h_exponent) and h_exponent < -0.5
    h_route_possible = (
        h_coarsening_helps
        and fitted_required_h_at_baseline_mass is not None
        and fitted_required_h_at_baseline_mass <= physically_plausible_h_limit
    )
    reachable_with_plausible_levers = mass_route_possible or h_route_possible
    return {
        "baseline_label": baseline["label"],
        "baseline_raw_map": raw,
        "baseline_h_interface_m": h,
        "baseline_surface_mass_kg_m2": mass,
        "required_surface_mass_times_h_kg_m_for_raw_map_1": required_product,
        "required_surface_mass_at_baseline_h_kg_m2_expected_inverse_mass": (
            required_mass_at_baseline_h
        ),
        "required_h_at_baseline_surface_mass_m_expected_inverse_h": (
            required_h_at_baseline_mass
        ),
        "fitted_required_surface_mass_at_baseline_h_kg_m2": (
            fitted_required_mass_at_baseline_h
        ),
        "fitted_required_h_at_baseline_surface_mass_m": (
            fitted_required_h_at_baseline_mass
        ),
        "h_coarsening_helps": h_coarsening_helps,
        "mass_route_possible": mass_route_possible,
        "h_route_possible": h_route_possible,
        "plausible_limit_assumption": {
            "max_surface_mass_multiplier": 20.0,
            "max_h_multiplier": 4.0,
        },
        "go_no_go": (
            "physical_parameter_route_possible"
            if reachable_with_plausible_levers
            else "semi_implicit_robin_in_matrix_required"
        ),
        "fit": fit,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]], conclusion: dict[str, Any]) -> None:
    lines = [
        "## Phase 0 Raw Interface-Map Scaling Experiment",
        "",
        "| case | h_interface_m | surface_mass_kg_m2 | raw_map | converged | IQN LS | wall_s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {h:.6g} | {mass:.6g} | {raw:.6g} | {conv} | {ls} | {wall:.3f} |".format(
                label=row["label"],
                h=row["h_interface_m"],
                mass=row["surface_mass_kg_m2"],
                raw=row["raw_map"],
                conv=row["converged"],
                ls=row["iqn_ils_least_squares_update_count"],
                wall=row["step_wall_time_s"],
            )
        )
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            "```json",
            json.dumps(conclusion, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "_codex_validation" / "phase0_raw_map_scaling",
    )
    parser.add_argument("--grid-scales", type=_float_list, default=[6.0, 8.0])
    parser.add_argument("--solid-density-scales", type=_float_list, default=[1.0, 4.0])
    parser.add_argument("--membrane-thickness-scale", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--extra-run-arg",
        action="append",
        default=[],
        help="Additional argument token forwarded to cases/squid_soft_robot.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for grid_scale in args.grid_scales:
        for solid_density_scale in args.solid_density_scales:
            label = (
                f"grid{grid_scale:g}_rho{solid_density_scale:g}_"
                f"thick{args.membrane_thickness_scale:g}"
            )
            run_dir = output_root / label
            summary = _run_case(
                python=args.python,
                output_dir=run_dir,
                grid_scale=float(grid_scale),
                solid_density_scale=float(solid_density_scale),
                membrane_thickness_scale=float(args.membrane_thickness_scale),
                extra_args=list(args.extra_run_arg),
                timeout_s=float(args.timeout_s),
                rerun=bool(args.rerun),
            )
            rows.append(
                _case_row(
                    label=label,
                    grid_scale=float(grid_scale),
                    solid_density_scale=float(solid_density_scale),
                    membrane_thickness_scale=float(args.membrane_thickness_scale),
                    output_dir=run_dir,
                    summary=summary,
                )
            )
            table_path = output_root / "phase0_scaling_table.json"
            table_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    fit = _fit_log_scaling(rows)
    conclusion = _go_no_go(rows, fit)
    result = {"rows": rows, "conclusion": conclusion}
    (output_root / "phase0_scaling_table.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _write_markdown(output_root / "phase0_scaling_table.md", rows, conclusion)
    print(json.dumps({"output_root": str(output_root), **conclusion}, indent=2))
    return result


if __name__ == "__main__":
    main()
