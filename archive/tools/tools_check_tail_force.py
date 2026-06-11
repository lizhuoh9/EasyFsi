"""Verify the tail membrane (secondary region) participates in sharp FSI."""
from __future__ import annotations

import csv
import sys

run_dir = sys.argv[1]
with open(run_dir + r"\history.csv", newline="") as f:
    rows = list(csv.DictReader(f))
keys = [
    "step",
    "hibm_marker_primary_count",
    "hibm_marker_secondary_count",
    "hibm_marker_primary_force_n",
    "hibm_marker_secondary_force_n",
    "tail_fsi_fluid_force_z_n",
    "main_fsi_fluid_force_z_n",
    "hibm_full_stress_valid_marker_count",
    "hibm_marker_total_force_z_n",
]
for r in rows:
    print(" | ".join(f"{k}={r.get(k, '<MISSING>')}" for k in keys))
