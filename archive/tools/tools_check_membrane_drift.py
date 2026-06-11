"""Quantify rigid drift of each membrane between first and latest snapshot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RUN = Path(r"_codex_validation/hibm_sharp_run_20260611_2s_full_waveform")
paths = sorted((RUN / "snapshots").glob("snapshot_*.npz"))
first, last = paths[0], paths[-1]
with np.load(first) as data:
    p0 = np.asarray(data["marker_positions_m"], dtype=np.float64)
with np.load(last) as data:
    p1 = np.asarray(data["marker_positions_m"], dtype=np.float64)
    step = int(data["step"])
PRIMARY = 7782
regions = {
    "main(primary)": (slice(0, PRIMARY)),
    "tail(secondary)": (slice(PRIMARY, None)),
}
print(f"first={first.name} last={last.name} (step {step})")
for name, region_slice in regions.items():
    delta = p1[region_slice] - p0[region_slice]
    mean_delta = delta.mean(axis=0)
    norms = np.linalg.norm(delta, axis=1)
    base = p0[region_slice]
    radius = np.hypot(base[:, 0] - base[:, 0].mean(), base[:, 1] - base[:, 1].mean())
    rim = radius >= np.percentile(radius, 90)
    rim_norm = np.linalg.norm(delta[rim], axis=1)
    print(
        f"{name}: mean d=({mean_delta[0]:+.2e},{mean_delta[1]:+.2e},{mean_delta[2]:+.2e}) m, "
        f"max |d|={norms.max():.2e}, mean |d|={norms.mean():.2e}, "
        f"rim(10% outermost) mean |d|={rim_norm.mean():.2e} max={rim_norm.max():.2e}"
    )
