from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import ansys.fluent.core as pyfluent

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_official_fluent_fine_steady import read_flow_summary, render_velocity


ROOT = SCRIPT_DIR.parents[0]
RUN_DIR = Path(os.environ.get("FLUENT_FSI_RUN_DIR", str(ROOT / "fsi_50step_from_fine_mesh"))).resolve()
FIGURE_DIR = RUN_DIR / "figures"
STEADY_FINE_CASE = Path(
    os.environ.get("FLUENT_FSI_START_CASE", str(ROOT / "fine_mesh_steady.cas.h5"))
).resolve()
FINAL_CASE = RUN_DIR / "fine_fsi_50step_final.cas.h5"
FINAL_DATA = RUN_DIR / "fine_fsi_50step_final.dat.h5"
SETUP_CASE = RUN_DIR / "fine_fsi_setup.cas.h5"
SUMMARY = RUN_DIR / "fine_fsi_50step_summary.json"
TIMESERIES = RUN_DIR / "fine_fsi_50step_steps.csv"
EVENT_LOG = RUN_DIR / "fine_fsi_50step_events.jsonl"

DT = 5.0e-4
N_STEPS = int(os.environ.get("FLUENT_FSI_N_STEPS", "50"))
MAX_ITER_PER_STEP = int(os.environ.get("FLUENT_FSI_MAX_ITER_PER_STEP", "40"))
PROCESSOR_COUNT = int(os.environ.get("FLUENT_FSI_PROCESSOR_COUNT", "4"))


def append_event(label: str, **payload: object) -> None:
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": label, **payload}
    with EVENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def apply_official_fsi_setup(session) -> None:
    session.setup.general.solver.time.set_state("unsteady-1st-order")
    session.setup.models.structure.model.set_state("linear-elasticity")
    try:
        session.setup.materials.solid.make_a_copy(from_="aluminum", to="silicone-rubber")
    except Exception:
        pass
    rubber = session.setup.materials.solid["silicone-rubber"]
    rubber.density.value.set_state(1600.0)
    rubber.struct_youngs_modulus.value.set_state(1.0e6)
    rubber.struct_poisson_ratio.value.set_state(0.47)
    session.setup.cell_zone_conditions.solid["solid.5"].general.material.set_state("silicone-rubber")

    attach = session.setup.boundary_conditions.wall["flap_attach"].structure
    attach.x_disp_boundary_condition.set_state("Node X-Displacement")
    attach.x_disp_boundary_value.set_state(0.0)
    attach.y_disp_boundary_condition.set_state("Node Y-Displacement")
    attach.y_disp_boundary_value.set_state(0.0)

    flap_wall = session.setup.boundary_conditions.wall["flap_wall"].structure
    flap_wall.x_disp_boundary_condition.set_state("Intrinsic FSI")
    flap_wall.y_disp_boundary_condition.set_state("Intrinsic FSI")

    session.setup.dynamic_mesh.enabled.set_state(True)
    for zone_name in ["po.3", "symmetry.2", "velocity_inlet.1", "wall"]:
        try:
            session.tui.define.dynamic_mesh.zones.create(zone_name)
        except Exception as exc:
            append_event("dynamic_mesh_zone_create_skipped", zone=zone_name, error=str(exc))
    try:
        session.tui.define.dynamic_mesh.zones.create("flap_wall-shadow", "intrinsic-fsi")
    except Exception as exc:
        append_event("intrinsic_fsi_zone_create_skipped", zone="flap_wall-shadow", error=str(exc))

    controls = session.solution.run_calculation.transient_controls
    controls.time_step_size.set_state(DT)
    controls.max_iter_per_time_step.set_state(MAX_ITER_PER_STEP)


def copy_latest_transcript() -> str | None:
    transcripts = sorted(RUN_DIR.glob("fluent-*.trn"), key=lambda path: path.stat().st_mtime)
    if not transcripts:
        return None
    target = RUN_DIR / "fine_fsi_50step_run.trn"
    target.write_bytes(transcripts[-1].read_bytes())
    return str(target)


def write_timeseries(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with TIMESERIES.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_LOG.write_text("", encoding="utf-8")
    started = time.time()
    session = None
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "source": "Local Ansys Fluent 2025 R1 official fsi_2way transient run from fine adapted steady preflow",
        "scope_note": "Transient two-way FSI continuation from the saved fine steady preflow; final case/data only, not per-step data dumps.",
        "steady_fine_case": str(STEADY_FINE_CASE),
        "dt_s": DT,
        "n_steps_requested": N_STEPS,
        "max_iter_per_step": MAX_ITER_PER_STEP,
        "processor_count": PROCESSOR_COUNT,
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
        session.file.read_case_data(file_name=str(STEADY_FINE_CASE))
        append_event("steady_fine_case_read", case=str(STEADY_FINE_CASE))
        apply_official_fsi_setup(session)
        session.file.write_case(file_name=str(SETUP_CASE))
        summary["setup_case"] = str(SETUP_CASE)

        for step_index in range(1, N_STEPS + 1):
            step_started = time.time()
            session.solution.run_calculation.dual_time_iterate(
                time_step_count=1,
                max_iter_per_step=MAX_ITER_PER_STEP,
            )
            row = {
                "step": step_index,
                "time_s": step_index * DT,
                "seconds": time.time() - step_started,
            }
            rows.append(row)
            write_timeseries(rows)
            summary["n_steps_completed"] = step_index
            summary["last_step_seconds"] = row["seconds"]
            SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            append_event("step_done", **row)

        session.file.write_case_data(file_name=str(FINAL_CASE))
        summary["final_case"] = str(FINAL_CASE)
        summary["final_data"] = str(FINAL_DATA)
        summary["final"] = read_flow_summary(FINAL_CASE, FINAL_DATA)
        fixed_figure = FIGURE_DIR / "velocity_magnitude_t0p025_fluent_scale_0_28p1.png"
        auto_figure = FIGURE_DIR / "velocity_magnitude_t0p025_autoscale.png"
        render_velocity(
            FINAL_CASE,
            FINAL_DATA,
            fixed_figure,
            fixed_scale=True,
            title="Official Fluent fsi_2way fine mesh, velocity magnitude at t = 0.025 s",
        )
        render_velocity(
            FINAL_CASE,
            FINAL_DATA,
            auto_figure,
            fixed_scale=False,
            title="Official Fluent fsi_2way fine mesh, velocity magnitude at t = 0.025 s",
        )
        summary["figures"] = {
            "velocity_magnitude_t0p025_fluent_scale_0_28p1": str(fixed_figure),
            "velocity_magnitude_t0p025_autoscale": str(auto_figure),
        }
        summary["timeseries"] = str(TIMESERIES)
        summary["elapsed_seconds"] = time.time() - started
        summary["transcript"] = copy_latest_transcript()
        SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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
        append_event("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if session is not None:
            session.exit()


if __name__ == "__main__":
    raise SystemExit(main())
