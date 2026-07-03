from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").exists():
            return parent
    raise RuntimeError("could not locate repo root from validation script path")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid stale/shared Taichi offline-cache locks from unrelated local runs. This
# changes compilation caching only; solver numerics are unchanged.
os.environ.setdefault("TI_OFFLINE_CACHE", "0")

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
    selected_formulation_solver_config,
    with_local_surface_force_support,
)
from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (  # noqa: E402
    save_solver_npz_from_flow_snapshot,
)


PHYSICAL_SOLID_BOUNDS = {
    "streamwise_min_m": 0.050,
    "streamwise_max_m": 0.053,
    "y_min_m": 0.0,
    "y_max_m": 0.010,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else _json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def _snapshot_meta(snapshot: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in snapshot.items():
        array = np.asarray(value)
        meta[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if array.size and np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            if finite.size:
                meta[key].update(
                    {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                    }
                )
    return meta


def _grid_summary(config: VerticalFlapFsiConfig) -> dict[str, Any]:
    nx, ny, nz = [int(v) for v in config.grid_nodes]
    modeled_height = float(
        ANSYS_VERTICAL_FLAP_CASE_METADATA["geometry"]["modeled_height_m"]
    )
    return {
        "grid_nodes": [nx, ny, nz],
        "span_cells": nx,
        "wall_normal_cells_full_solver_height": ny,
        "streamwise_cells": nz,
        "total_3d_cells": nx * ny * nz,
        "full_height_2d_equivalent_cells": ny * nz,
        "modeled_half_height_2d_equivalent_cells": int(round(ny * modeled_height / float(config.duct_height_m))) * nz,
        "cell_size_m": {
            "span": float(config.span_m) / nx,
            "wall_normal_full_height": float(config.duct_height_m) / ny,
            "streamwise": float(config.duct_length_m) / nz,
        },
        "domain_m": {
            "streamwise_length": float(config.duct_length_m),
            "full_solver_height": float(config.duct_height_m),
            "modeled_height": modeled_height,
            "span": float(config.span_m),
        },
    }


def _build_config(args: argparse.Namespace) -> VerticalFlapFsiConfig:
    config = selected_formulation_solver_config(
        step_count=int(args.steps),
        pressure_pair_provider_mode=str(args.pressure_pair_provider_mode),
        selected_anchor_markers_json=args.selected_anchor_markers_json,
    )
    config = replace(
        config,
        grid_nodes=tuple(int(v) for v in args.grid_nodes),
        solid_particle_counts=tuple(int(v) for v in args.solid_particle_counts),
        marker_count=int(args.marker_count),
        flow_projection_iterations=int(args.flow_projection_iterations),
        solid_substeps=int(args.solid_substeps),
        flow_driver_mode="sustained_boundary_predictor",
        # With flow_projection_velocity_inlet_zmax the projection keeps the
        # inlet open, so the volume-source preflow driver would double-drive
        # the field (runs away past the predictor CFL guard); boundary-driven
        # preflow is the consistent formulation.
        preflow_flow_driver_mode="sustained_boundary_predictor",
        preflow_steps=40,
        flow_projection_velocity_inlet_zmax=True,
        flow_hibm_sharp_search_radius_m=1.7e-3,
        export_final_flow_snapshot=True,
        # Campaign runs fail loud on under-seeded solids (2026-07-03 audit:
        # counts (1, 64, 12) on grid 4x256x320 leave ~2 cells between
        # wall-normal particle layers, the root clamp fractures, and the
        # flap free-falls until a particle leaves the background grid).
        enforce_solid_seeding_limit=True,
    )
    if args.flow_predictor_substeps is not None:
        config = replace(
            config,
            flow_predictor_substeps=int(args.flow_predictor_substeps),
        )
    if args.young_modulus_pa is not None:
        config = replace(
            config,
            young_modulus_pa=float(args.young_modulus_pa),
        )
    if args.hibm_search_radius_m is not None:
        config = replace(
            config,
            flow_hibm_sharp_search_radius_m=float(args.hibm_search_radius_m),
        )
    return with_local_surface_force_support(config)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "final_flow_field_snapshot"
    }


def _summary_from_report(
    *,
    report: dict[str, Any],
    config: VerticalFlapFsiConfig,
    output_dir: Path,
    elapsed_s: float,
    solver_npz_summary: dict[str, Any] | None,
    run_label: str,
) -> dict[str, Any]:
    history = list(report.get("history", []))
    final_history = history[-1] if history else {}
    return {
        "run_label": run_label,
        "status": "completed" if len(history) == int(config.step_count) else "blocked",
        "elapsed_s": float(elapsed_s),
        "output_dir": str(output_dir),
        "step_count_requested": int(config.step_count),
        "step_count_completed": len(history),
        "final_time_s": float(config.dt_s) * len(history),
        "dt_s": float(config.dt_s),
        "grid": _grid_summary(config),
        "solid_particle_counts": list(config.solid_particle_counts),
        "marker_count": int(config.marker_count),
        "flow_projection_iterations": int(config.flow_projection_iterations),
        "solid_substeps": int(config.solid_substeps),
        "max_displacement_m": report.get("max_displacement_m"),
        "max_displacement_relative_error": report.get(
            "max_displacement_relative_error"
        ),
        "local_velocity_peak_mps": report.get("local_velocity_peak_mps"),
        "max_abs_traction_pa": report.get("max_abs_traction_pa"),
        "final_history": final_history,
        "solver_npz_summary": solver_npz_summary or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", default="our_solver")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--grid-nodes", type=int, nargs=3, default=(4, 32, 64))
    parser.add_argument(
        "--solid-particle-counts", type=int, nargs=3, default=(1, 12, 4)
    )
    parser.add_argument("--marker-count", type=int, default=12)
    parser.add_argument("--flow-projection-iterations", type=int, default=1080)
    parser.add_argument("--solid-substeps", type=int, default=1600)
    parser.add_argument(
        "--flow-predictor-substeps",
        type=int,
        default=None,
        help=(
            "Override predictor advection substeps; finer wall-normal grids "
            "need more substeps to satisfy the predictor CFL guard."
        ),
    )
    parser.add_argument(
        "--hibm-search-radius-m",
        type=float,
        default=None,
        help=(
            "Override the sharp-IB row search radius; should scale with the "
            "grid (roughly 1.5x the finest in-plane spacing) so the Dirichlet "
            "row halo does not fatten the effective flap."
        ),
    )
    parser.add_argument(
        "--young-modulus-pa",
        type=float,
        default=None,
        help=(
            "Override the flap Young's modulus; a stiff value gives a "
            "rigid-proxy flap for flow-only parity runs while the structural "
            "load calibration is still open."
        ),
    )
    parser.add_argument(
        "--pressure-pair-provider-mode",
        default="runtime_anchored_cell_pair",
        choices=("runtime_anchored_cell_pair", "replay_from_diagnostics"),
    )
    parser.add_argument("--selected-anchor-markers-json")
    parser.add_argument("--span-reduction", default="mean", choices=("mean", "center"))
    parser.add_argument("--streamwise-velocity-sign", type=float, default=-1.0)
    parser.add_argument(
        "--no-reverse-streamwise-axis",
        action="store_true",
        help="Disable solver-to-Fluent streamwise axis reversal.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)
    config_payload = asdict(config)
    manifest = {
        "run_label": args.run_label,
        "repo_root": str(REPO_ROOT),
        "script": str(Path(__file__).resolve()),
        "case": "official Fluent fsi_2way vertical flap",
        "solver_entry": "cases.ansys_vertical_flap_fsi.run_vertical_flap_fsi_smoke",
        "selected_formulation": "selected_formulation_solver_config",
        "physical_solid_bounds": PHYSICAL_SOLID_BOUNDS,
        "config": config_payload,
        "grid": _grid_summary(config),
        "dry_run": bool(args.dry_run),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "our_solver_config.json", config_payload)

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "output_dir": str(output_dir)}))
        return 0

    start = time.perf_counter()
    try:
        report = run_vertical_flap_fsi_smoke(config)
    except Exception as exc:
        elapsed_s = time.perf_counter() - start
        _write_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "elapsed_s": elapsed_s,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "config": config_payload,
                "grid": _grid_summary(config),
            },
        )
        raise

    elapsed_s = time.perf_counter() - start
    report = dict(report)
    history = list(report.get("history", []))
    _write_history_csv(output_dir / "our_solver_history.csv", history)
    _write_json(output_dir / "our_solver_report_compact.json", _compact_report(report))

    snapshot = report.get("final_flow_field_snapshot")
    solver_npz_summary = None
    if isinstance(snapshot, dict):
        _write_json(output_dir / "final_flow_snapshot_meta.json", _snapshot_meta(snapshot))
        solver_npz_summary = save_solver_npz_from_flow_snapshot(
            output_dir / "our_solver_final_fields.npz",
            snapshot,
            span_reduction=args.span_reduction,
            streamwise_velocity_sign=float(args.streamwise_velocity_sign),
            reverse_streamwise_axis=not bool(args.no_reverse_streamwise_axis),
            physical_solid_bounds=PHYSICAL_SOLID_BOUNDS,
        )

    summary = _summary_from_report(
        report=report,
        config=config,
        output_dir=output_dir,
        elapsed_s=elapsed_s,
        solver_npz_summary=solver_npz_summary,
        run_label=args.run_label,
    )
    _write_json(output_dir / "our_solver_summary.json", summary)
    print(json.dumps(_json_safe(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
