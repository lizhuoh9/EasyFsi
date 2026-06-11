"""Extract honesty fields from the _010 150-step run for the progress file."""
from __future__ import annotations

import csv
import json

RUN = (
    r"_codex_validation"
    r"\hibm_sharp_smoke_20260611_010_150step_per_substep_anchor"
)
with open(RUN + r"\history.csv", newline="") as f:
    rows = list(csv.DictReader(f))
print("rows:", len(rows))
keys = [
    "step",
    "interior_divergence_l2",
    "unreached_divergence_l2",
    "unreached_divergence_max_abs",
    "unreached_divergence_cell_count",
    "hibm_pressure_disconnected_nonprojectable_cell_count",
    "hibm_marker_total_force_z_n",
    "hibm_full_stress_valid_marker_count",
    "hibm_velocity_dirichlet_active_rows",
    "cfl",
    "max_fluid_speed_mps",
    "solid_mpm_total_force_z_n",
    "pressure_load_pa",
]
for step_index in (1, 24, 74, 99, 149):
    r = rows[step_index]
    print("|".join(f"{k}={r.get(k, '<MISSING>')}" for k in keys))
maxes = {
    "max_cfl": max(float(r["cfl"]) for r in rows),
    "max_interior_div": max(float(r["interior_divergence_l2"]) for r in rows),
    "max_unreached_div": max(
        float(r.get("unreached_divergence_l2", 0.0) or 0.0) for r in rows
    ),
    "max_marker_force_z_abs": max(
        abs(float(r["hibm_marker_total_force_z_n"])) for r in rows
    ),
    "max_valid_markers": max(
        int(float(r["hibm_full_stress_valid_marker_count"])) for r in rows
    ),
}
print(maxes)
with open(RUN + r"\summary.json") as f:
    summary = json.load(f)
for k in (
    "completed_steps",
    "max_cfl",
    "max_divergence_l2",
    "max_interior_divergence_l2",
    "completed_step_checks_passed",
):
    print(k, "=", summary.get(k, "<MISSING>"))
checks = summary.get("completed_step_checks", {})
print(
    "reconstruction_valid =",
    checks.get("hibm_velocity_dirichlet_reconstruction_valid"),
)
