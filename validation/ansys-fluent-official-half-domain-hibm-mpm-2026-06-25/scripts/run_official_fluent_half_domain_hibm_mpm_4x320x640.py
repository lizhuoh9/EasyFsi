from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = Path(__file__).resolve().with_name("base_true_sharp_fsi_runner.py")
SOURCE_DIR = (
    REPO_ROOT
    / "_codex_validation"
    / "official_ansys_fluent_fsi_2way_source_20260625"
    / "unzipped"
    / "fsi_2way"
)

GRID_NODES = tuple(
    int(part.strip())
    for part in os.environ.get("OFFICIAL_HALF_GRID_NODES", "4,320,640").split(",")
)
if len(GRID_NODES) != 3:
    raise RuntimeError("OFFICIAL_HALF_GRID_NODES must contain exactly three integers")
STEP_COUNT = int(os.environ.get("OFFICIAL_HALF_STEP_COUNT", "1"))
PROJECTION_ITERATIONS = int(os.environ.get("OFFICIAL_HALF_PROJECTION_ITERATIONS", "4096"))
FLUID_SUBSTEPS = int(os.environ.get("OFFICIAL_HALF_FLUID_SUBSTEPS", "2"))
SOLID_SUBSTEPS = int(os.environ.get("OFFICIAL_HALF_SOLID_SUBSTEPS", "1000"))
MARKERS_PER_FACE = int(os.environ.get("OFFICIAL_HALF_MARKERS_PER_FACE", "84"))
SOLID_PARTICLE_COUNTS = tuple(
    int(part.strip())
    for part in os.environ.get("OFFICIAL_HALF_SOLID_PARTICLES", "1,80,24").split(",")
)
if len(SOLID_PARTICLE_COUNTS) != 3:
    raise RuntimeError("OFFICIAL_HALF_SOLID_PARTICLES must contain exactly three integers")

MODELED_HEIGHT_M = 0.02
FULL_DUCT_HEIGHT_M = 0.04
DUCT_LENGTH_M = 0.10
SPAN_M = 0.003
FLAP_HEIGHT_M = 0.01
FLAP_THICKNESS_M = 0.003
FLAP_Z_MIN_M = 0.050
FLAP_Z_MAX_M = 0.053
INLET_VELOCITY_MPS = 10.0

RUN_LABEL = os.environ.get(
    "OFFICIAL_HALF_RUN_LABEL",
    (
        f"official_half_grid{GRID_NODES[0]}x{GRID_NODES[1]}x{GRID_NODES[2]}"
        f"_step{STEP_COUNT}_p{PROJECTION_ITERATIONS}_s{SOLID_SUBSTEPS}"
    ),
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "rerun_outputs" / RUN_LABEL


def _load_runner():
    spec = importlib.util.spec_from_file_location("official_fluent_half_runner_base", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_runner(runner: Any) -> None:
    runner.FLAP_NAMES = ("lower",)

    def full_bounds(config: Any):
        return (0.0, 0.0, 0.0), (SPAN_M, MODELED_HEIGHT_M, DUCT_LENGTH_M)

    def solid_bounds(config: Any):
        dy = MODELED_HEIGHT_M / float(config.grid_nodes[1])
        pad_y = 3.0 * dy
        return (0.0, -pad_y, 0.0), (SPAN_M, MODELED_HEIGHT_M + pad_y, DUCT_LENGTH_M)

    def flap_boxes(config: Any):
        return {
            "lower": (
                (0.0, 0.0, FLAP_Z_MIN_M),
                (SPAN_M, FLAP_HEIGHT_M, FLAP_Z_MAX_M),
            )
        }

    def build_solid(config: Any, runtime: Any):
        boxes = flap_boxes(config)
        bounds_min, bounds_max = solid_bounds(config)
        capacity = math.prod(config.solid_particle_counts)
        solid = runner.NeoHookeanMpmState(
            particle_capacity=capacity,
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            grid_nodes=config.grid_nodes,
            runtime=runtime,
        )
        solid.initialize_box(
            particle_counts=config.solid_particle_counts,
            box_min_m=boxes["lower"][0],
            box_max_m=boxes["lower"][1],
            density_kgm3=config.solid_density_kgm3,
        )
        fixed = np.zeros(capacity, dtype=np.int32)
        region = np.full(capacity, runner.PRIMARY_REGION_ID, dtype=np.int32)
        normals = np.zeros((capacity, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        areas = np.full(capacity, FLAP_HEIGHT_M * SPAN_M / float(capacity), dtype=np.float32)
        rest = solid.rest_x.to_numpy()
        row_h = FLAP_HEIGHT_M / float(config.solid_particle_counts[1])
        fixed[rest[:capacity, 1] <= boxes["lower"][0][1] + 1.01 * row_h] = 1
        solid.fixed_particle.from_numpy(fixed)
        solid.region_id.from_numpy(region)
        solid.surface_normal.from_numpy(normals)
        solid.rest_surface_normal.from_numpy(normals)
        solid.area_weight_m2.from_numpy(areas)
        solid.rest_area_weight_m2.from_numpy(areas)
        solid.external_force_n.from_numpy(np.zeros((capacity, 3), dtype=np.float32))
        solid._update_rest_center()
        masks = {
            "fixed": fixed.astype(bool),
            "lower": np.arange(0, capacity),
            "lower_tip": np.arange(0, capacity)[
                rest[:capacity, 1] >= boxes["lower"][1][1] - 1.01 * row_h
            ],
        }
        return solid, masks

    def build_markers(config: Any, runtime: Any, markers_per_face: int):
        boxes = flap_boxes(config)
        dz = DUCT_LENGTH_M / float(config.grid_nodes[2])
        x_center = 0.5 * (boxes["lower"][0][0] + boxes["lower"][1][0])
        area = FLAP_HEIGHT_M * SPAN_M / float(markers_per_face)
        positions = []
        normals = []
        areas = []
        y_min = boxes["lower"][0][1]
        y_max = boxes["lower"][1][1]
        segment = (y_max - y_min) / float(markers_per_face)
        faces = (
            ((0.0, 0.0, 1.0), boxes["lower"][1][2] + 0.51 * dz),
            ((0.0, 0.0, -1.0), boxes["lower"][0][2] - 0.51 * dz),
        )
        for normal, z in faces:
            for marker in range(markers_per_face):
                y = y_min + (float(marker) + 0.5) * segment
                positions.append((x_center, y, z))
                normals.append(normal)
                areas.append(area)
        markers = runner.HibmMpmSurfaceMarkers(marker_capacity=len(positions), runtime=runtime)
        markers.load_markers(
            positions_m=positions,
            velocities_mps=[(0.0, 0.0, 0.0)] * len(positions),
            normals=normals,
            areas_m2=areas,
            region_ids=[runner.PRIMARY_REGION_ID] * len(positions),
        )
        return markers

    def marker_face_diagnostics(markers: Any, markers_per_face: int):
        valid = markers._stress_pressure_valid.to_numpy()[: markers.marker_count]
        tractions = markers.t_gamma_pa.to_numpy()[: markers.marker_count]
        forces = markers.F_gamma_n.to_numpy()[: markers.marker_count]
        marker_diagnostics = tuple(markers.stress_marker_diagnostics())
        face_ranges = {
            "lower_plus_z": (0, markers_per_face),
            "lower_minus_z": (markers_per_face, 2 * markers_per_face),
        }
        result = {}
        for name, (start, stop) in face_ranges.items():
            face_valid = valid[start:stop] != 0
            face_forces = forces[start:stop]
            face_tractions = tractions[start:stop]
            face_marker_diagnostics = marker_diagnostics[start:stop]
            invalid_reason_counts: dict[str, int] = {}
            for diagnostic in face_marker_diagnostics:
                if diagnostic.get("valid", False):
                    continue
                reason = str(diagnostic.get("invalid_reason", "unknown"))
                invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
            traction_norm = np.linalg.norm(face_tractions, axis=1)
            result[name] = {
                "marker_count": int(stop - start),
                "valid_marker_count": int(np.count_nonzero(face_valid)),
                "invalid_marker_count": int((stop - start) - np.count_nonzero(face_valid)),
                "invalid_reason_counts": invalid_reason_counts,
                "base_pressure_found_count": int(
                    sum(1 for item in face_marker_diagnostics if item.get("base_pressure_found", False))
                ),
                "inside_pressure_found_count": int(
                    sum(1 for item in face_marker_diagnostics if item.get("inside_pressure_found", False))
                ),
                "outside_pressure_found_count": int(
                    sum(1 for item in face_marker_diagnostics if item.get("outside_pressure_found", False))
                ),
                "pressure_anchor_available_count": int(
                    sum(1 for item in face_marker_diagnostics if item.get("pressure_anchor_available", False))
                ),
                "max_abs_traction_pa": float(traction_norm.max(initial=0.0)),
                "force_n": tuple(float(v) for v in face_forces.sum(axis=0)),
            }
        return result

    def solid_displacement(solid: Any, masks: dict[str, np.ndarray]):
        current = solid.x.to_numpy()[: solid.particle_count]
        rest = solid.rest_x.to_numpy()[: solid.particle_count]
        disp = current - rest
        norm = np.linalg.norm(disp, axis=1)
        tip_mean = disp[masks["lower_tip"]].mean(axis=0)
        return {
            "max_displacement_m": float(norm.max(initial=0.0)),
            "fixed_root_max_displacement_m": float(norm[masks["fixed"]].max(initial=0.0)),
            "lower_tip_mean_displacement_m": tuple(float(v) for v in tip_mean),
        }

    runner._full_bounds = full_bounds
    runner._solid_bounds = solid_bounds
    runner._flap_boxes = flap_boxes
    runner._build_solid = build_solid
    runner._build_markers = build_markers
    runner._marker_face_diagnostics = marker_face_diagnostics
    runner._solid_displacement = solid_displacement


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = _load_runner()
    _patch_runner(runner)
    runner.ROOT = OUTPUT_DIR
    runner.LOG_PATH = OUTPUT_DIR / f"{RUN_LABEL}.log"
    runner.PROCESS_PATH = OUTPUT_DIR / f"{RUN_LABEL}_process.json"
    runner.REPORT_PATH = OUTPUT_DIR / f"{RUN_LABEL}_report.json"
    runner.HISTORY_PATH = OUTPUT_DIR / f"{RUN_LABEL}_history.csv"
    runner.FIELDS_PATH = OUTPUT_DIR / f"{RUN_LABEL}_fields.npz"

    manifest = {
        "case": "ansys-fluent-official-half-domain-single-flap",
        "solver": "local HIBM-MPM advance_hibm_mpm_sharp_mpm_step",
        "official_source": {
            "tutorial_url": "https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/flu_tg/flu_tg_fsi_2way.html",
            "mesh": str(SOURCE_DIR / "flap.msh"),
            "journal": str(SOURCE_DIR / "steady_fluid_flow.jou"),
        },
        "official_geometry_m": {
            "duct_length": DUCT_LENGTH_M,
            "full_duct_height": FULL_DUCT_HEIGHT_M,
            "modeled_half_height": MODELED_HEIGHT_M,
            "flap_height": FLAP_HEIGHT_M,
            "flap_thickness": FLAP_THICKNESS_M,
            "flap_z_min": FLAP_Z_MIN_M,
            "flap_z_max": FLAP_Z_MAX_M,
        },
        "boundary_conditions": {
            "velocity_inlet_zmax_mps": INLET_VELOCITY_MPS,
            "pressure_outlet_zmin_pa": 0.0,
            "symmetry_boundary": "modeled half-domain top boundary y=0.02 m, mirrored only for display",
        },
        "grid_nodes": list(GRID_NODES),
        "step_count": STEP_COUNT,
        "projection_iterations": PROJECTION_ITERATIONS,
        "fluid_substeps": FLUID_SUBSTEPS,
        "solid_substeps": SOLID_SUBSTEPS,
        "solid_particle_counts": list(SOLID_PARTICLE_COUNTS),
        "markers_per_face": MARKERS_PER_FACE,
        "outputs": {
            "log": str(runner.LOG_PATH),
            "process_json": str(runner.PROCESS_PATH),
            "report_json": str(runner.REPORT_PATH),
            "history_csv": str(runner.HISTORY_PATH),
            "fields_npz": str(runner.FIELDS_PATH),
        },
    }
    (OUTPUT_DIR / f"{RUN_LABEL}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    start = time.perf_counter()
    try:
        report = runner.run(
            step_count=STEP_COUNT,
            projection_iterations=PROJECTION_ITERATIONS,
            grid_nodes=GRID_NODES,
            solid_particle_counts=SOLID_PARTICLE_COUNTS,
            markers_per_face=MARKERS_PER_FACE,
            pressure_solve_failure_policy="report",
            convert_internal_nodes_to_obstacles=True,
            fluid_advection_scheme="rk2",
            fluid_substeps=FLUID_SUBSTEPS,
            solid_substeps=SOLID_SUBSTEPS,
            preflow_steps=0,
            stop_on_speed_mps=None,
        )
    except Exception as exc:
        failure = {
            **manifest,
            "status": "failed",
            "elapsed_s": time.perf_counter() - start,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (OUTPUT_DIR / f"{RUN_LABEL}_wrapper_failure.json").write_text(
            json.dumps(_jsonable(failure), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(_jsonable(failure), indent=2, sort_keys=True), flush=True)
        return 1

    latest = report["history"][-1]
    summary = {
        **manifest,
        "status": "completed",
        "elapsed_s": time.perf_counter() - start,
        "completed_steps": len(report["history"]),
        "flap_count_modeled": 1,
        "flap_count_displayed_after_symmetry_mirror": 2,
        "full_domain_two_flap": False,
        "official_half_domain": True,
        "fluid_speed_max_mps": latest.get("fluid_speed_max_mps"),
        "fluid_speed_p99_mps": latest.get("fluid_speed_p99_mps"),
        "fluid_speed_p999_mps": latest.get("fluid_speed_p999_mps"),
        "marker_force_z_n": latest.get("marker_force_z_n"),
        "max_displacement_m": latest.get("max_displacement_m"),
        "stress_valid_marker_count": latest.get("stress_valid_marker_count"),
        "stress_invalid_marker_count": latest.get("stress_invalid_marker_count"),
    }
    (OUTPUT_DIR / f"{RUN_LABEL}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
