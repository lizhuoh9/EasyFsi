from __future__ import annotations

import json
import math
import os
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path

import ansys.fluent.core as pyfluent
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


RUN_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = RUN_DIR / "figures"
MESH = Path(
    os.environ.get("EASYFSI_FLUENT_MESH", str(RUN_DIR / "flap.msh"))
).expanduser()

COARSE_CASE = RUN_DIR / "coarse_before_adapt.cas.h5"
COARSE_DATA = RUN_DIR / "coarse_before_adapt.dat.h5"
FINE_CASE = RUN_DIR / "fine_mesh_steady.cas.h5"
FINE_DATA = RUN_DIR / "fine_mesh_steady.dat.h5"
SUMMARY = RUN_DIR / "fine_mesh_summary.json"
EVENT_LOG = RUN_DIR / "fine_mesh_run_events.jsonl"
RUN_LOG = RUN_DIR / "fine_mesh_run.log"

COARSE_ITERATIONS = 100
POST_ADAPT_ITERATIONS_PER_CYCLE = 80
ADAPT_CYCLES = 4
FINAL_EXTRA_ITERATIONS = 300
MAX_REFINEMENT_LEVEL = 4
MAX_CELL_COUNT = 1_000_000
MINIMUM_EDGE_LENGTH_M = 1.0e-5
PROCESSOR_COUNT = 4


def append_event(label: str, **payload: object) -> None:
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": label, **payload}
    with EVENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def split_names(raw: np.ndarray) -> list[str]:
    if raw.size == 0:
        return []
    text = raw[0].decode("utf-8", errors="replace")
    return [item for item in text.split(";") if item]


def read_cell_zones(case_path: Path) -> list[dict[str, object]]:
    with h5py.File(case_path, "r") as case:
        z = case["meshes/1/cells/zoneTopology"]
        names = split_names(z["name"][()])
        zone_ids = z["id"][()].astype(int).tolist()
        min_ids = z["minId"][()].astype(int).tolist()
        max_ids = z["maxId"][()].astype(int).tolist()
        zone_types = z["zoneType"][()].astype(int).tolist()
    zones = []
    for index, zone_id in enumerate(zone_ids):
        name = names[index] if index < len(names) else f"zone-{zone_id}"
        min_id = min_ids[index]
        max_id = max_ids[index]
        zones.append(
            {
                "id": zone_id,
                "name": name,
                "min_cell_id": min_id,
                "max_cell_id": max_id,
                "count": max_id - min_id + 1,
                "zone_type": zone_types[index],
            }
        )
    return zones


def read_flow_summary(case_path: Path, data_path: Path) -> dict[str, object]:
    zones = read_cell_zones(case_path)
    with h5py.File(data_path, "r") as data:
        u = data["results/1/phase-1/cells/SV_U/1"][()]
        v = data["results/1/phase-1/cells/SV_V/1"][()]
        p = data["results/1/phase-1/cells/SV_P/1"][()]
        speed = np.hypot(u, v)
        residuals = {}
        residual_root = data.get("results/residuals/phase-1")
        if residual_root is not None:
            for name in residual_root.keys():
                values = residual_root[name]["data"][()]
                iterations = residual_root[name]["iterations"][()]
                residuals[name] = {
                    "iteration_count": int(len(iterations)),
                    "initial": float(values[0, 0]) if len(values) else None,
                    "final": float(values[-1, 0]) if len(values) else None,
                    "min": float(np.min(values[:, 0])) if len(values) else None,
                    "final_iteration": int(iterations[-1]) if len(iterations) else None,
                }
    fluid_cell_count = sum(int(z["count"]) for z in zones if "fluid" in str(z["name"]).lower())
    total_cell_count = sum(int(z["count"]) for z in zones)
    return {
        "cell_zones": zones,
        "total_cell_count": total_cell_count,
        "fluid_cell_count": fluid_cell_count,
        "velocity": {
            "u_min": float(np.min(u)),
            "u_max": float(np.max(u)),
            "u_mean": float(np.mean(u)),
            "v_min": float(np.min(v)),
            "v_max": float(np.max(v)),
            "v_mean": float(np.mean(v)),
            "speed_min": float(np.min(speed)),
            "speed_max": float(np.max(speed)),
            "speed_mean": float(np.mean(speed)),
        },
        "pressure": {
            "p_min": float(np.min(p)),
            "p_max": float(np.max(p)),
            "p_mean": float(np.mean(p)),
        },
        "residuals": residuals,
    }


def first_dataset(group) -> np.ndarray:
    keys = sorted(group.keys(), key=lambda value: int(value) if value.isdigit() else value)
    return group[keys[0]][()]


def build_cell_polygons(case_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    zones = read_cell_zones(case_path)
    fluid_ranges = [
        (int(zone["min_cell_id"]), int(zone["max_cell_id"]))
        for zone in zones
        if "fluid" in str(zone["name"]).lower()
    ]
    solid_ranges = [
        (int(zone["min_cell_id"]), int(zone["max_cell_id"]))
        for zone in zones
        if "solid" in str(zone["name"]).lower()
    ]

    with h5py.File(case_path, "r") as case:
        coords = first_dataset(case["meshes/1/nodes/coords"])
        face_nodes_group = case["meshes/1/faces/nodes/1"]
        nnodes = face_nodes_group["nnodes"][()].astype(int)
        flat_nodes = face_nodes_group["nodes"][()].astype(int)
        c0 = case["meshes/1/faces/c0/1"][()].astype(int)
        c1 = case["meshes/1/faces/c1/1"][()].astype(int)

    offsets = np.concatenate(([0], np.cumsum(nnodes)))
    cell_nodes: dict[int, set[int]] = defaultdict(set)
    for face_index, owner in enumerate(c0):
        nodes = flat_nodes[offsets[face_index] : offsets[face_index + 1]]
        cell_nodes[int(owner)].update(int(node) for node in nodes)
        if face_index < len(c1):
            neighbor = int(c1[face_index])
            if neighbor > 0:
                cell_nodes[neighbor].update(int(node) for node in nodes)

    def in_ranges(cell_id: int, ranges: list[tuple[int, int]]) -> bool:
        return any(lo <= cell_id <= hi for lo, hi in ranges)

    fluid_cells = sorted(cell_id for cell_id in cell_nodes if in_ranges(cell_id, fluid_ranges))
    solid_cells = sorted(cell_id for cell_id in cell_nodes if in_ranges(cell_id, solid_ranges))

    def polygon_for(cell_id: int) -> np.ndarray | None:
        node_ids = sorted(cell_nodes[cell_id])
        if len(node_ids) < 3:
            return None
        points = coords[np.array(node_ids, dtype=int) - 1]
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]

    fluid_polys = [poly for cell_id in fluid_cells if (poly := polygon_for(cell_id)) is not None]
    solid_polys = [poly for cell_id in solid_cells if (poly := polygon_for(cell_id)) is not None]
    return fluid_polys, solid_polys, fluid_cells


def render_velocity(
    case_path: Path,
    data_path: Path,
    out_path: Path,
    *,
    fixed_scale: bool,
    title: str = "Official Fluent fsi_2way fine mesh steady preflow, velocity magnitude",
) -> None:
    with h5py.File(data_path, "r") as data:
        u = data["results/1/phase-1/cells/SV_U/1"][()]
        v = data["results/1/phase-1/cells/SV_V/1"][()]
        speed = np.hypot(u, v)
    fluid_polys, solid_polys, fluid_cells = build_cell_polygons(case_path)
    if len(fluid_polys) != len(speed):
        n = min(len(fluid_polys), len(speed))
        fluid_polys = fluid_polys[:n]
        speed = speed[:n]
        append_event(
            "render_polygon_count_adjusted",
            polygon_count=len(fluid_cells),
            speed_count=int(len(speed)),
            used_count=n,
        )

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    vmin = 0.0 if fixed_scale else float(np.min(speed))
    vmax = 28.1 if fixed_scale else float(np.max(speed))
    coll = PolyCollection(
        fluid_polys,
        array=speed,
        cmap="turbo",
        edgecolors="none",
        linewidths=0,
        rasterized=True,
    )
    coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    if solid_polys:
        solid = PolyCollection(
            solid_polys,
            facecolors="white",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        ax.add_collection(solid)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.002, 0.103)
    ax.set_ylim(-0.002, 0.043)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    cbar = fig.colorbar(coll, ax=ax, shrink=0.9)
    cbar.set_label("Velocity magnitude (m/s)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def execute_tui(session, command: str) -> None:
    started = time.time()
    append_event("tui_start", command=command)
    try:
        session.execute_tui(command)
    except Exception as exc:
        append_event(
            "tui_failed",
            command=command,
            seconds=time.time() - started,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(limit=8),
        )
        raise
    append_event("tui_done", command=command, seconds=time.time() - started)


def apply_official_steady_setup(session) -> None:
    commands = [
        '/define/models/viscous/kw-sst? yes',
        '/define/boundary-conditions/zone-name wall:008 flap_attach',
        '/define/boundary-conditions/zone-name default-interior:010 flap_wall',
        '/define/boundary-conditions/zone-type solid.5 solid',
        '/define/boundary-conditions/set/velocity-inlet velocity_inlet.1 () vmag no 10 ()',
        '/define/operating-conditions/operating-pressure 1013250',
        '/define/materials/change-create air air yes constant 1.2 no no yes constant 1.8e-05 no no no',
        '/solve/set/p-v-coupling 24',
        '/solve/initialize/hyb-initialization',
        '/solve/set/pseudo-transient yes yes 1 5 0',
    ]
    for command in commands:
        execute_tui(session, command)


def run_adapt_cycle(session, cycle: int) -> None:
    adapt = session.tui.mesh.adapt
    append_event("adapt_cycle_start", cycle=cycle)
    adapt.set.maximum_cell_count(MAX_CELL_COUNT)
    adapt.set.maximum_refinement_level(MAX_REFINEMENT_LEVEL)
    adapt.set.minimum_edge_length(MINIMUM_EDGE_LENGTH_M)
    adapt.predefined_criteria.aerodynamics.error_based.pressure_hessian_indicator()
    adapt.adapt_mesh()
    append_event("adapt_cycle_mesh_done", cycle=cycle)
    execute_tui(session, f"/solve/iterate {POST_ADAPT_ITERATIONS_PER_CYCLE}")
    append_event("adapt_cycle_done", cycle=cycle)


def copy_latest_transcript() -> str | None:
    transcripts = sorted(RUN_DIR.glob("fluent-*.trn"), key=lambda path: path.stat().st_mtime)
    if not transcripts:
        return None
    target = RUN_DIR / "fine_mesh_run.trn"
    shutil.copy2(transcripts[-1], target)
    return str(target)


def write_run_log(summary: dict[str, object]) -> None:
    lines = [
        "Official Fluent fsi_2way fine mesh steady preflow",
        f"source mesh: {MESH}",
        f"fine case: {FINE_CASE}",
        f"fine data: {FINE_DATA}",
        f"summary: {SUMMARY}",
        f"coarse iterations: {COARSE_ITERATIONS}",
        f"adapt cycles: {ADAPT_CYCLES}",
        f"post-adapt iterations per cycle: {POST_ADAPT_ITERATIONS_PER_CYCLE}",
        f"final extra iterations: {FINAL_EXTRA_ITERATIONS}",
        f"max refinement level: {MAX_REFINEMENT_LEVEL}",
        f"max cell count: {MAX_CELL_COUNT}",
        f"minimum edge length m: {MINIMUM_EDGE_LENGTH_M}",
        "",
        json.dumps(summary, indent=2, sort_keys=True),
    ]
    RUN_LOG.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not MESH.is_file():
        raise FileNotFoundError(
            "Fluent mesh not found; set EASYFSI_FLUENT_MESH to the public "
            f"fsi_2way flap.msh path (resolved={MESH})"
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_LOG.write_text("", encoding="utf-8")
    started = time.time()
    session = None
    summary: dict[str, object] = {
        "source": "Local Ansys Fluent 2025 R1 run of public fsi_2way tutorial steady preflow",
        "scope_note": "Steady official Fluent preflow with Fluent mesh adaption; not transient two-way FSI monitor validation.",
        "source_mesh": str(MESH),
        "run_dir": str(RUN_DIR),
        "processor_count": PROCESSOR_COUNT,
        "coarse_iterations": COARSE_ITERATIONS,
        "adapt_cycles": ADAPT_CYCLES,
        "post_adapt_iterations_per_cycle": POST_ADAPT_ITERATIONS_PER_CYCLE,
        "final_extra_iterations": FINAL_EXTRA_ITERATIONS,
        "max_refinement_level": MAX_REFINEMENT_LEVEL,
        "max_cell_count": MAX_CELL_COUNT,
        "minimum_edge_length_m": MINIMUM_EDGE_LENGTH_M,
    }
    try:
        session = pyfluent.launch_fluent(
            mode="solver",
            precision="double",
            dimension=2,
            processor_count=PROCESSOR_COUNT,
            start_timeout=240,
            cwd=str(RUN_DIR),
            ui_mode="no_gui",
        )
        summary["fluent_version"] = str(session.get_fluent_version())
        append_event("fluent_started", version=summary["fluent_version"])
        session.file.read_case(file_name=str(MESH))
        append_event("mesh_read", mesh=str(MESH))

        apply_official_steady_setup(session)
        execute_tui(session, f"/solve/iterate {COARSE_ITERATIONS}")
        session.file.write_case_data(file_name=str(COARSE_CASE))
        summary["coarse_case"] = str(COARSE_CASE)
        summary["coarse_data"] = str(COARSE_DATA)
        summary["coarse"] = read_flow_summary(COARSE_CASE, COARSE_DATA)
        SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        for cycle in range(1, ADAPT_CYCLES + 1):
            run_adapt_cycle(session, cycle)
            cycle_case = RUN_DIR / f"fine_mesh_after_adapt_cycle_{cycle}.cas.h5"
            session.file.write_case_data(file_name=str(cycle_case))
            summary[f"adapt_cycle_{cycle}"] = read_flow_summary(
                cycle_case,
                RUN_DIR / f"fine_mesh_after_adapt_cycle_{cycle}.dat.h5",
            )
            SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        execute_tui(session, f"/solve/iterate {FINAL_EXTRA_ITERATIONS}")
        append_event("final_extra_iterations_done", iterations=FINAL_EXTRA_ITERATIONS)
        session.file.write_case_data(file_name=str(FINE_CASE))
        summary["fine_case"] = str(FINE_CASE)
        summary["fine_data"] = str(FINE_DATA)
        summary["fine"] = read_flow_summary(FINE_CASE, FINE_DATA)

        fixed_figure = FIGURE_DIR / "velocity_magnitude_fluent_scale_0_28p1.png"
        auto_figure = FIGURE_DIR / "velocity_magnitude_autoscale.png"
        render_velocity(FINE_CASE, FINE_DATA, fixed_figure, fixed_scale=True)
        render_velocity(FINE_CASE, FINE_DATA, auto_figure, fixed_scale=False)
        summary["figures"] = {
            "velocity_magnitude_fluent_scale_0_28p1": str(fixed_figure),
            "velocity_magnitude_autoscale": str(auto_figure),
        }
        summary["elapsed_seconds"] = time.time() - started
        summary["transcript"] = copy_latest_transcript()
        SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_run_log(summary)
        append_event("run_complete", elapsed_seconds=summary["elapsed_seconds"])
        return 0
    except Exception as exc:
        summary["failed"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        summary["elapsed_seconds"] = time.time() - started
        summary["transcript"] = copy_latest_transcript()
        SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_run_log(summary)
        append_event("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if session is not None:
            session.exit()


if __name__ == "__main__":
    raise SystemExit(main())
