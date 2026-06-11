"""Acceptance-field check for the 2026-06-11 sharp probes."""
from __future__ import annotations

import csv
import json
import sys

run_dir = sys.argv[1]
with open(run_dir + r"\history.csv", newline="") as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
keys = [
    "step",
    "interior_divergence_l2",
    "divergence_l2",
    "cfl",
    "hibm_full_stress_valid_marker_count",
    "hibm_full_stress_invalid_marker_count",
    "hibm_marker_total_force_z_n",
    "hibm_mpm_scatter_active_marker_count",
    "hibm_mpm_scatter_action_reaction_residual_n",
    "hibm_velocity_dirichlet_active_rows",
    "hibm_velocity_dirichlet_relocated_rows",
    "hibm_velocity_dirichlet_relocation_merged_rows",
    "hibm_velocity_dirichlet_relocation_blocked_rows",
    "hibm_velocity_dirichlet_narrow_gap_count",
    "hibm_velocity_dirichlet_invalid_reconstruction_count",
    "hibm_pressure_neumann_active_rows",
    "hibm_pressure_neumann_skipped_velocity_dirichlet_count",
    "hibm_pressure_neumann_skipped_obstacle_owner_count",
    "hibm_boundary_pressure_neumann_count",
    "hibm_solid_band_nonprojectable_cell_count",
    "hibm_pressure_disconnected_nonprojectable_cell_count",
    "hibm_no_slip_residual_l2_mps",
    "hibm_no_slip_residual_max_mps",
    "max_fluid_speed_mps",
    "solid_mpm_total_force_z_n",
    "pressure_solve_failed",
    "pressure_projection_cg_converged_all",
    "pressure_projection_cg_max_relative_residual",
    "pressure_load_pa",
]
for k in keys:
    print(k, "=", last.get(k, "<MISSING>"))
try:
    with open(run_dir + r"\run_process.json") as f:
        proc = json.load(f)
    print("run_status =", proc.get("status"), "error =", proc.get("error"))
except FileNotFoundError:
    print("run_process.json missing")
