"""Locate divergence hotspots in the step-163 failure VTI."""
from __future__ import annotations

import re

import numpy as np

RUN = r"_codex_validation/hibm_sharp_run_20260611_2s_full_waveform"
PATH = RUN + r"/sharp_failure_step_000163_fluid.vti"
text = open(PATH, encoding="utf-8", errors="replace").read()


def read_array(name: str) -> np.ndarray:
    match = re.search(
        rf'Name="{name}"[^>]*>\s*([^<]+)<',
        text,
        re.DOTALL,
    )
    if match is None:
        raise SystemExit(f"array {name} not found")
    return np.fromstring(match.group(1), sep=" ", dtype=np.float64)


extent = re.search(r'WholeExtent="([^"]+)"', text).group(1)
x0, x1, y0, y1, z0, z1 = (int(v) for v in extent.split())
nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
print("grid:", nx, ny, nz)

divergence = read_array("divergence").reshape((nz, ny, nx)).transpose(2, 1, 0)
obstacle = read_array("obstacle").reshape((nz, ny, nx)).transpose(2, 1, 0)
active = read_array("active_fluid").reshape((nz, ny, nx)).transpose(2, 1, 0)

fluid_mask = obstacle < 0.5
print("active fluid cells:", int(fluid_mask.sum()))
magnitude = np.abs(divergence) * fluid_mask
flat_order = np.argsort(magnitude.ravel())[::-1][:25]
print("top |div| cells (i, j, k, |div|, near-obstacle?):")
for flat_index in flat_order:
    i, j, k = np.unravel_index(flat_index, magnitude.shape)
    neighborhood = obstacle[
        max(0, i - 1) : i + 2, max(0, j - 1) : j + 2, max(0, k - 1) : k + 2
    ]
    print(
        int(i),
        int(j),
        int(k),
        f"{magnitude[i, j, k]:.4e}",
        "near_obstacle" if neighborhood.max() > 0.5 else "open_fluid",
    )
print(
    "hotspot bbox:",
    [
        (
            int(np.unravel_index(flat_order, magnitude.shape)[axis].min()),
            int(np.unravel_index(flat_order, magnitude.shape)[axis].max()),
        )
        for axis in range(3)
    ],
)
