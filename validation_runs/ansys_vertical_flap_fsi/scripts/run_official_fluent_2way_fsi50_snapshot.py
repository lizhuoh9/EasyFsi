from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)
from refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (  # noqa: E402
    save_solver_npz_from_flow_snapshot,
)


CASE_NAME = "ansys_vertical_flap_fsi"
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
REFERENCE_ROOT = ROOT / "official_fluent_2way_reference"
OUTPUT_ROOT = ROOT / "official_fluent_2way_solver_snapshot"
SOLVER_FIELDS = OUTPUT_ROOT / "step50_solver_u_v_p_fields.npz"
SOLVER_HISTORY = OUTPUT_ROOT / "solver_history.csv"
RUN_MANIFEST = OUTPUT_ROOT / "solver_snapshot_manifest.json"
SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_official_fluent_2way_fsi50_snapshot.py"
)


def run(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
    grid_nodes: tuple[int, int, int] | None = None,
    step_count: int = 50,
) -> dict[str, Any]:
    reference_root = Path(reference_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = VerticalFlapFsiConfig(
        step_count=int(step_count),
        grid_nodes=grid_nodes
        if grid_nodes is not None
        else _official_reference_grid_nodes(reference_root),
        export_final_flow_snapshot=True,
    )
    flap_min, flap_max = _official_reference_flap_streamwise_bounds(reference_root)
    config = replace(
        config,
        flap_streamwise_min_m=float(flap_min),
        flap_streamwise_max_m=float(flap_max),
    )
    report = run_vertical_flap_fsi_smoke(config)
    snapshot = dict(report.get("final_flow_field_snapshot") or {})
    if not snapshot:
        raise RuntimeError("solver report did not include final_flow_field_snapshot")

    fields_summary = save_solver_npz_from_flow_snapshot(
        output_root / SOLVER_FIELDS.name,
        snapshot,
        physical_solid_bounds={
            "streamwise_min_m": float(config.flap_streamwise_min_m),
            "streamwise_max_m": float(config.flap_streamwise_max_m),
            "y_min_m": 0.0,
            "y_max_m": float(config.flap_height_m),
        },
    )
    history_summary = _write_solver_history_csv(
        output_root / SOLVER_HISTORY.name,
        report.get("history", []),
    )
    manifest = {
        "case": CASE_NAME,
        "source_script": SOURCE_SCRIPT,
        "reference_root": _repo_relative(reference_root),
        "output_root": _repo_relative(output_root),
        "grid_nodes": list(config.grid_nodes),
        "step_count": int(config.step_count),
        "flap_streamwise_bounds_m": [
            float(config.flap_streamwise_min_m),
            float(config.flap_streamwise_max_m),
        ],
        "fluent_parity_claimed": False,
        "solver_fields": fields_summary,
        "solver_history": history_summary,
    }
    _write_json(output_root / RUN_MANIFEST.name, manifest)
    return manifest


def _official_reference_flap_streamwise_bounds(
    reference_root: str | Path,
) -> tuple[float, float]:
    mesh_summary = _reference_mesh_summary(Path(reference_root))
    cell_bounds = mesh_summary.get("cell_zone_center_bounds", {})
    solid_bounds = _first_named_bounds(cell_bounds, "solid")
    if solid_bounds is None:
        face_bounds = mesh_summary.get("face_zone_node_bounds", {})
        solid_bounds = _first_named_bounds(face_bounds, "flap_wall", "wall")
    if solid_bounds is None:
        raise KeyError("official reference mesh summary does not contain solid flap bounds")
    return float(solid_bounds["x_min"]), float(solid_bounds["x_max"])


def _official_reference_grid_nodes(
    reference_root: str | Path,
    *,
    span_nodes: int = 4,
) -> tuple[int, int, int]:
    return _grid_nodes_from_fluent_mesh_summary(
        _reference_mesh_summary(Path(reference_root)),
        span_nodes=span_nodes,
    )


def _grid_nodes_from_fluent_mesh_summary(
    mesh_summary: Mapping[str, Any],
    *,
    span_nodes: int = 4,
) -> tuple[int, int, int]:
    fluid_bounds = _first_named_bounds(
        mesh_summary.get("cell_zone_center_bounds", {}),
        "fluid",
    )
    if fluid_bounds is None:
        raise KeyError("mesh summary does not contain fluid cell-zone bounds")
    fluid_cell_count = int(fluid_bounds["count"])
    face_bounds = mesh_summary.get("face_zone_node_bounds", {})
    symmetry = _first_named_bounds(face_bounds, "symmetry")
    inlet = _first_named_bounds(face_bounds, "velocity_inlet", "inlet")

    ny = 0
    if symmetry is not None and int(symmetry.get("count", 0)) > 0:
        ny = max(1, int(symmetry["count"]) // 2)
    elif inlet is not None and int(inlet.get("count", 0)) > 0:
        ny = int(inlet["count"])
    if ny <= 0 or fluid_cell_count % ny != 0:
        ny = max(1, int(round(fluid_cell_count**0.5)))
    nz = max(1, fluid_cell_count // ny)
    return int(span_nodes), int(ny), int(nz)


def _parse_int_tuple3(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated integers")
    parsed = tuple(int(part) for part in parts)
    if any(item <= 0 for item in parsed):
        raise ValueError("grid tuple values must be positive")
    return parsed


def _write_solver_history_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_history_row(row) for row in rows]
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["row"])
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {"path": _repo_relative(path), "row_count": len(flat_rows)}
    for key in (
        "local_velocity_peak_mps",
        "fluid_speed_p99_mps",
        "fluid_speed_p999_mps",
        "pressure_min_pa",
        "pressure_max_pa",
    ):
        values = [float(row[key]) for row in flat_rows if _is_number(row.get(key))]
        if values:
            summary[key] = max(values) if key.endswith(("mps", "pa")) else values[-1]
    return summary


def _flatten_history_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple)) and len(value) == 3:
            base, suffix = _vector_column_base_and_suffix(key)
            flat[f"{base}_x{suffix}"] = value[0]
            flat[f"{base}_y{suffix}"] = value[1]
            flat[f"{base}_z{suffix}"] = value[2]
        else:
            flat[key] = value
    return flat


def _vector_column_base_and_suffix(key: str) -> tuple[str, str]:
    for suffix in ("_mps", "_m", "_n", "_N"):
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return key, ""


def _reference_mesh_summary(reference_root: Path) -> dict[str, Any]:
    for name in ("steady_fluent_fields.npz", "fsi50_final_fluent_fields.npz"):
        path = reference_root / name
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            if "mesh_summary_json" not in data.files:
                if "x" in data.files and "y" in data.files:
                    x = np.asarray(data["x"], dtype=np.float64)
                    y = np.asarray(data["y"], dtype=np.float64)
                    x_unique = np.unique(x)
                    y_unique = np.unique(y)
                    return {
                        "cell_zone_center_bounds": {
                            "fluid.4": {"count": int(x.size)}
                        },
                        "face_zone_node_bounds": {
                            "symmetry.2": {"count": int(2 * y_unique.size)},
                            "velocity_inlet.1": {"count": int(y_unique.size)},
                        },
                        "derived_from_field_coordinates": {
                            "x_unique_count": int(x_unique.size),
                            "y_unique_count": int(y_unique.size),
                        },
                    }
                continue
            return json.loads(str(data["mesh_summary_json"]))
    raise FileNotFoundError(f"no official Fluent field NPZ with mesh summary under {reference_root}")


def _first_named_bounds(
    bounds_by_name: Mapping[str, Any],
    *needles: str,
) -> Mapping[str, Any] | None:
    lowered_needles = tuple(needle.lower() for needle in needles)
    for name, bounds in bounds_by_name.items():
        lowered = str(name).lower()
        if any(needle in lowered for needle in lowered_needles):
            return bounds
    return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repo_relative(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--grid-nodes", type=_parse_int_tuple3, default=None)
    parser.add_argument("--steps", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    payload = run(
        reference_root=args.reference_root,
        output_root=args.output_root,
        grid_nodes=args.grid_nodes,
        step_count=args.steps,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
