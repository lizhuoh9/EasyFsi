from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "traction_shared_snapshot_diagnostics"
FIELD_PATH = OUTPUT_DIR / "step020_fields.npz"
MANIFEST_PATH = OUTPUT_DIR / "snapshot_manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "snapshot_summary.md"
VERIFICATION_PATH = OUTPUT_DIR / "verification_shared_snapshot_2026-06-26.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

CASE_ID = "ansys-vertical-flap-fsi"
PREFLOW_STEPS = 20
SCOPE_LIMIT = (
    "fixed-solid shared flow snapshot only; no coupled 50-step or Fluent parity claim"
)
PURPOSE = "fixed-solid shared flow snapshot for later traction resampling"
REFERENCE_FORMULATION_CANDIDATE = "none"
CANDIDATE_STATUS = "snapshot_only_no_reference_selection"
RUNNER_RELATIVE_PATH = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_shared_snapshot.py"
)
REQUIRED_FIELDS = (
    "pressure",
    "velocity",
    "obstacle",
    "cell_face_x_m",
    "cell_face_y_m",
    "cell_face_z_m",
    "cell_center_x_m",
    "cell_center_y_m",
    "cell_center_z_m",
    "cell_width_x_m",
    "cell_width_y_m",
    "cell_width_z_m",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive one ANSYS vertical-flap fixed-solid shared flow snapshot."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Artifact directory, defaulting to the committed validation path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    paths = _paths_for(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _snapshot_config()
    report = run_vertical_flap_fsi_smoke(config)
    snapshot = _validated_snapshot(report, config)
    arrays = _npz_arrays(snapshot, config, report)

    np.savez(paths["field"], **arrays)
    field_sha256 = _sha256_file(paths["field"])
    marker_geometry = _marker_geometry(report, config)
    marker_geometry_sha256 = _sha256_bytes(_stable_json_bytes(marker_geometry))
    summary = _field_summary(arrays)
    manifest = _manifest(
        report=report,
        config=config,
        field_path=paths["field"],
        field_sha256=field_sha256,
        field_summary=summary,
        marker_geometry=marker_geometry,
        marker_geometry_sha256=marker_geometry_sha256,
    )

    paths["manifest"].write_text(
        json.dumps(_to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["summary"].write_text(_summary_markdown(manifest), encoding="utf-8")
    paths["verification"].write_text(
        _verification_markdown(manifest),
        encoding="utf-8",
    )
    _write_checksums(output_dir, paths["checksums"])
    print(json.dumps(_to_jsonable(manifest), indent=2, sort_keys=True), flush=True)
    return 0


def _paths_for(output_dir: Path) -> dict[str, Path]:
    return {
        "field": output_dir / FIELD_PATH.name,
        "manifest": output_dir / MANIFEST_PATH.name,
        "summary": output_dir / SUMMARY_PATH.name,
        "verification": output_dir / VERIFICATION_PATH.name,
        "checksums": output_dir / CHECKSUMS_PATH.name,
    }


def _snapshot_config() -> VerticalFlapFsiConfig:
    return VerticalFlapFsiConfig(
        step_count=0,
        preflow_steps=PREFLOW_STEPS,
        apply_marker_feedback_to_fluid=False,
        flow_driver_mode="sustained_volume_source_inlet",
        flow_inlet_source_strength=0.80,
        flow_inlet_source_profile="linear_ramp",
        flow_inlet_source_ramp_steps=2,
        flow_inlet_source_schedule_scope="global",
        export_final_flow_snapshot=True,
    )


def _validated_snapshot(
    report: dict[str, Any],
    config: VerticalFlapFsiConfig,
) -> dict[str, np.ndarray]:
    completed = int(report.get("preflow_steps_completed", -1))
    if completed != PREFLOW_STEPS:
        raise RuntimeError(
            f"expected {PREFLOW_STEPS} preflow steps, got {completed}"
        )
    if int(report.get("config", {}).get("step_count", config.step_count)) != 0:
        raise RuntimeError("shared snapshot runner must keep step_count=0")

    snapshot = dict(report.get("final_flow_field_snapshot") or {})
    missing = [field for field in REQUIRED_FIELDS if field not in snapshot]
    if missing:
        raise RuntimeError(f"flow snapshot missing required fields: {missing}")

    pressure = np.asarray(snapshot["pressure"])
    velocity = np.asarray(snapshot["velocity"])
    obstacle = np.asarray(snapshot["obstacle"])
    grid_nodes = tuple(int(value) for value in config.grid_nodes)
    if pressure.shape != grid_nodes:
        raise RuntimeError(
            f"pressure shape {pressure.shape} does not match grid {grid_nodes}"
        )
    if obstacle.shape != pressure.shape:
        raise RuntimeError(
            f"obstacle shape {obstacle.shape} does not match pressure {pressure.shape}"
        )
    if velocity.shape != pressure.shape + (3,):
        raise RuntimeError(
            f"velocity shape {velocity.shape} does not match pressure + vector"
        )

    for name, array in snapshot.items():
        values = np.asarray(array)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"snapshot field {name} contains non-finite values")

    sampling_obstacle = snapshot.get("sampling_obstacle")
    if (
        sampling_obstacle is not None
        and np.asarray(sampling_obstacle).shape != pressure.shape
    ):
        raise RuntimeError("sampling_obstacle shape does not match pressure")

    return {name: np.asarray(value) for name, value in snapshot.items()}


def _npz_arrays(
    snapshot: dict[str, np.ndarray],
    config: VerticalFlapFsiConfig,
    report: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(value) for name, value in snapshot.items()}
    arrays.update(
        {
            "grid_nodes": np.asarray(config.grid_nodes, dtype=np.int32),
            "preflow_step": np.asarray([PREFLOW_STEPS], dtype=np.int32),
            "dt_s": np.asarray([config.dt_s], dtype=np.float64),
            "inlet_velocity_mps": np.asarray(
                [config.inlet_velocity_mps],
                dtype=np.float64,
            ),
            "source_strength": np.asarray(
                [config.flow_inlet_source_strength],
                dtype=np.float64,
            ),
            "source_ramp_steps": np.asarray(
                [config.flow_inlet_source_ramp_steps],
                dtype=np.int32,
            ),
            "velocity_outlet_flux_ratio": np.asarray(
                [_finite_float(report.get("velocity_outlet_flux_ratio", math.nan))],
                dtype=np.float64,
            ),
            "pressure_outlet_flux_ratio": np.asarray(
                [_finite_float(report.get("pressure_outlet_flux_ratio", math.nan))],
                dtype=np.float64,
            ),
        }
    )
    return arrays


def _field_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    pressure = np.asarray(arrays["pressure"], dtype=np.float64)
    velocity = np.asarray(arrays["velocity"], dtype=np.float64)
    obstacle = np.asarray(arrays["obstacle"])
    non_obstacle = obstacle == 0
    speed = np.linalg.norm(velocity, axis=3)
    active_pressure = pressure[non_obstacle]
    active_speed = speed[non_obstacle]
    return {
        "grid_nodes": [int(value) for value in arrays["grid_nodes"]],
        "pressure_min_pa": float(active_pressure.min(initial=0.0)),
        "pressure_max_pa": float(active_pressure.max(initial=0.0)),
        "velocity_peak_mps": float(active_speed.max(initial=0.0)),
        "velocity_p999_mps": (
            float(np.percentile(active_speed, 99.9)) if active_speed.size else 0.0
        ),
        "fluid_cell_count": int(non_obstacle.sum()),
        "obstacle_cell_count": int(np.size(obstacle) - non_obstacle.sum()),
        "field_arrays": _field_array_manifest(arrays),
    }


def _field_array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, array in sorted(arrays.items()):
        values = np.asarray(array)
        result[name] = {
            "shape": [int(dim) for dim in values.shape],
            "dtype": str(values.dtype),
        }
    return result


def _marker_geometry(
    report: dict[str, Any],
    config: VerticalFlapFsiConfig,
) -> dict[str, Any]:
    config_payload = dict(report.get("config") or asdict(config))
    return {
        "marker_layout": config_payload["traction_marker_layout"],
        "marker_face_offset_cells": float(config_payload["traction_marker_face_offset_cells"]),
        "pressure_sampling_mode": config_payload["traction_pressure_sampling_mode"],
        "traction_include_viscous": bool(config_payload["traction_include_viscous"]),
        "traction_viscosity_pa_s": float(config_payload["traction_viscosity_pa_s"]),
        "marker_count": int(report.get("marker_count_actual", 0)),
        "marker_count_per_face": int(report.get("marker_count_per_face", config.marker_count)),
        "marker_face_count": int(report.get("marker_face_count", 0)),
        "flap_box_m": dict(report.get("flap_box_m", {})),
        "grid_nodes": [int(value) for value in config_payload["grid_nodes"]],
        "solid_particle_counts": [
            int(value) for value in config_payload["solid_particle_counts"]
        ],
    }


def _manifest(
    *,
    report: dict[str, Any],
    config: VerticalFlapFsiConfig,
    field_path: Path,
    field_sha256: str,
    field_summary: dict[str, Any],
    marker_geometry: dict[str, Any],
    marker_geometry_sha256: str,
) -> dict[str, Any]:
    latest_preflow = dict(report.get("preflow_history", [{}])[-1])
    return {
        "case": CASE_ID,
        "purpose": PURPOSE,
        "scope_limit": SCOPE_LIMIT,
        "commit_sha": _git_commit_sha(),
        "source_commit": _git_commit_sha(),
        "runner": RUNNER_RELATIVE_PATH,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "preflow_steps": PREFLOW_STEPS,
        "preflow_steps_completed": int(report.get("preflow_steps_completed", 0)),
        "step_count": int(config.step_count),
        "flow_driver_mode": config.flow_driver_mode,
        "flow_inlet_source_strength": float(config.flow_inlet_source_strength),
        "flow_inlet_source_profile": config.flow_inlet_source_profile,
        "flow_inlet_source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "flow_inlet_source_schedule_scope": config.flow_inlet_source_schedule_scope,
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": config.flow_inlet_source_profile,
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "source_schedule_scope": config.flow_inlet_source_schedule_scope,
        "grid_nodes": [int(value) for value in config.grid_nodes],
        "field_path": field_path.as_posix(),
        "field_sha256": field_sha256,
        "field_arrays": field_summary["field_arrays"],
        "pressure_min_pa": field_summary["pressure_min_pa"],
        "pressure_max_pa": field_summary["pressure_max_pa"],
        "velocity_peak_mps": field_summary["velocity_peak_mps"],
        "velocity_p999_mps": field_summary["velocity_p999_mps"],
        "velocity_outlet_flux_ratio": _finite_float(
            report.get("velocity_outlet_flux_ratio", math.nan)
        ),
        "pressure_outlet_flux_ratio": _finite_float(
            report.get("pressure_outlet_flux_ratio", math.nan)
        ),
        "flow_projection_report": report.get(
            "flow_projection_report",
            latest_preflow.get("flow_projection_report", {}),
        ),
        "flow_phase": report.get("flow_phase", latest_preflow.get("flow_phase", "")),
        "flow_step_index_global": report.get(
            "flow_step_index_global",
            latest_preflow.get("flow_step_index_global", ""),
        ),
        "marker_geometry": marker_geometry,
        "marker_geometry_sha256": marker_geometry_sha256,
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "candidate_status": CANDIDATE_STATUS,
        "non_goal_statement": (
            "This snapshot archives one local EasyFsi fixed-solid flow state only; "
            "it does not choose a traction formulation, run one-sided traction, "
            "run coupled 50-step FSI, or claim Fluent parity."
        ),
        "config": asdict(config),
        "array_summary": {
            "fluid_cell_count": field_summary["fluid_cell_count"],
            "obstacle_cell_count": field_summary["obstacle_cell_count"],
        },
    }


def _summary_markdown(manifest: dict[str, Any]) -> str:
    return (
        "# ANSYS Traction Shared Snapshot - 2026-06-26\n\n"
        f"scope_limit = {manifest['scope_limit']}\n\n"
        f"field_path = {manifest['field_path']}\n\n"
        f"field_sha256 = {manifest['field_sha256']}\n\n"
        f"grid_shape = {manifest['grid_nodes']}\n\n"
        f"pressure_range_pa = [{manifest['pressure_min_pa']}, {manifest['pressure_max_pa']}]\n\n"
        f"velocity_peak_mps = {manifest['velocity_peak_mps']}\n\n"
        f"velocity_p999_mps = {manifest['velocity_p999_mps']}\n\n"
        "This snapshot exists so future formulation rows can be sampled from "
        "the exact same pressure/velocity field.\n\n"
        "It does not prove Fluent parity, does not run coupled 50-step FSI, "
        "and no reference formulation is selected.\n\n"
        "next_intended_step = snapshot resampling matrix\n"
    )


def _verification_markdown(manifest: dict[str, Any]) -> str:
    return (
        "# ANSYS Traction Shared Snapshot Verification - 2026-06-26\n\n"
        "Generated by:\n\n"
        "```powershell\n"
        '& "D:/TOOL/Anaconda/python.exe" '
        "validation_runs/ansys_vertical_flap_fsi/scripts/"
        "run_traction_shared_snapshot.py\n"
        "```\n\n"
        f"preflow_steps = {manifest['preflow_steps']}\n\n"
        f"field_sha256 = {manifest['field_sha256']}\n\n"
        f"marker_geometry_sha256 = {manifest['marker_geometry_sha256']}\n\n"
        f"reference_formulation_candidate = {manifest['reference_formulation_candidate']}\n\n"
        f"candidate_status = {manifest['candidate_status']}\n\n"
        f"scope_limit = {manifest['scope_limit']}\n"
    )


def _write_checksums(output_dir: Path, checksums_path: Path) -> None:
    lines: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == checksums_path:
            continue
        digest = _sha256_file(path)
        lines.append(f"{digest}  {path.relative_to(output_dir).as_posix()}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _finite_float(value: Any) -> float | str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(result):
        return ""
    return result


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
