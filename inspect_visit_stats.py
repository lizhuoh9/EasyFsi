from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _complete_records(visit_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((visit_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for record in manifest.get("records") or []:
        files = record.get("files") or {}
        fluid = files.get("fluid_vti")
        if not fluid:
            continue
        fluid_path = visit_dir / str(fluid)
        if fluid_path.exists() and fluid_path.stat().st_size > 0:
            records.append(record)
    return sorted(records, key=lambda item: int(item.get("step_index", -1)))


def _physical_mask(dataset: Any, speed: np.ndarray) -> np.ndarray:
    array_names = set(dataset.array_names)
    excluded = np.zeros(speed.shape, dtype=bool)
    for name in ("visual_excluded_mask", "solid_extension_mask", "solid_obstacle_full_mask", "inactive_water_mask"):
        if name in array_names:
            excluded |= np.asarray(dataset[name], dtype=np.float64) > 0.5
    if "active_water_mask" in array_names:
        active = np.asarray(dataset["active_water_mask"], dtype=np.float64) > 0.5
    else:
        active = ~excluded
    return active & ~excluded & np.isfinite(speed)


def _safe_percentiles_and_max(values: np.ndarray) -> tuple[list[float], float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0], 0.0
    return [float(x) for x in np.percentile(finite, [50.0, 95.0, 99.0])], float(np.max(finite))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("visit_dir", type=Path)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()

    import pyvista as pv

    visit_dir = args.visit_dir.resolve()
    records = _complete_records(visit_dir)
    if not records:
        raise RuntimeError(f"No complete fluid records in {visit_dir}")
    if args.step is None:
        record = records[-1]
    else:
        matches = [item for item in records if int(item.get("step_index", -1)) == args.step]
        if not matches:
            raise RuntimeError(f"Step {args.step} not found in {visit_dir}")
        record = matches[0]

    fluid = pv.read(visit_dir / record["files"]["fluid_vti"])
    speed = np.asarray(fluid["speed_mps"], dtype=np.float64)
    velocity = np.asarray(fluid["velocity_mps"], dtype=np.float64)
    physical = _physical_mask(fluid, speed)
    downward = -velocity[:, 2]
    finite_speed = speed[physical]
    finite_down = downward[physical & np.isfinite(downward)]
    speed_percentiles, speed_max = _safe_percentiles_and_max(finite_speed)
    downward_percentiles, downward_max = _safe_percentiles_and_max(finite_down)
    result = {
        "visit_dir": str(visit_dir),
        "step": int(record.get("step_index", -1)),
        "time_s": record.get("time_s"),
        "array_names": list(fluid.array_names),
        "physical_cell_count": int(np.sum(physical)),
        "speed_mps": {
            "p50_p95_p99": speed_percentiles,
            "max": speed_max,
            "above": {str(t): int(np.sum(finite_speed > t)) for t in (0.005, 0.01, 0.05, 0.1)},
        },
        "downward_velocity_mps": {
            "p50_p95_p99": downward_percentiles,
            "max": downward_max,
            "above": {str(t): int(np.sum(finite_down > t)) for t in (0.005, 0.01, 0.05, 0.1)},
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
