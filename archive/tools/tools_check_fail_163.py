"""Dissect the step-163 guard failure rows of the 2s run."""
from __future__ import annotations

import csv

RUN = r"_codex_validation/hibm_sharp_run_20260611_2s_full_waveform"
with open(RUN + r"/history.csv", newline="") as f:
    rows = list(csv.DictReader(f))
print("rows:", len(rows))
keys = [
    "step",
    "interior_divergence_l2",
    "divergence_l2",
    "unreached_divergence_l2",
    "unreached_divergence_cell_count",
    "hibm_pressure_disconnected_nonprojectable_cell_count",
    "hibm_solid_band_nonprojectable_cell_count",
    "hibm_internal_obstacle_cell_count",
    "hibm_velocity_dirichlet_active_rows",
    "pressure_projection_cg_converged_all",
    "pressure_projection_cg_max_relative_residual",
    "cfl",
    "max_fluid_speed_mps",
    "hibm_velocity_dirichlet_near_divergence_l2",
    "hibm_velocity_dirichlet_far_divergence_l2",
    "pressure_load_pa",
]
for index in (155, 158, 160, 161, 162):
    if index < len(rows):
        r = rows[index]
        print(" | ".join(f"{k}={r.get(k, '<MISSING>')}" for k in keys))
