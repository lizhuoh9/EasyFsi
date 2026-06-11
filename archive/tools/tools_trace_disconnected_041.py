"""0c diagnostic: trace disconnected-count jumps across the _041 history."""
from __future__ import annotations

import csv

PATH = (
    r"D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM"
    r"\_codex_validation"
    r"\hibm_sharp_smoke_20260610_041_150step_p512_sub3_cleanup32_device_disconnected_mask"
    r"\history.csv"
)

with open(PATH, newline="") as f:
    rows = list(csv.DictReader(f))
print("rows:", len(rows))
prev = None
for r in rows:
    d = int(float(r["hibm_pressure_disconnected_nonprojectable_cell_count"]))
    if prev is None or abs(d - prev) > 500:
        print(
            "step", r["step"],
            "disconnected=", d,
            "band=", r["hibm_solid_band_nonprojectable_cell_count"],
            "internal=", r["hibm_internal_obstacle_cell_count"],
            "dirichlet_rows=", r["hibm_velocity_dirichlet_active_rows"],
            "valid_markers=", r["hibm_full_stress_valid_marker_count"],
        )
    prev = d
last = rows[-1]
print(
    "final step", last["step"],
    "disconnected=", last["hibm_pressure_disconnected_nonprojectable_cell_count"],
)
