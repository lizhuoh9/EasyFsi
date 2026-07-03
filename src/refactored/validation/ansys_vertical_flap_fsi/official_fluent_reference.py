from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OFFICIAL_SOURCE_ROOT = Path(
    r"D:\working\squid robot\LBM\MPM-LBM\benchmarks\private"
    r"\ansys_fsi_2way_public_tutorial\fsi_2way"
)

STEADY_CASE_NAME = "steady_fluid_flow_export.cas.h5"
STEADY_DATA_NAME = "steady_fluid_flow_export.dat.h5"
FSI50_FINAL_CASE_NAME = "official_fsi_50step_final.cas.h5"
FSI50_FINAL_DATA_NAME = "official_fsi_50step_final.dat.h5"
FSI50_MONITOR_NAME = "official_fsi_50step_monitor_timeseries.csv"
FSI50_SUMMARY_NAME = "official_fsi_50step_summary.json"
STEADY_SUMMARY_NAME = "steady_reference_summary.json"
STEADY_JOURNAL_NAME = "steady_fluid_flow.jou"

CELL_FIELD_PATHS = {
    "u": "results/1/phase-1/cells/SV_U/1",
    "v": "results/1/phase-1/cells/SV_V/1",
    "p": "results/1/phase-1/cells/SV_P/1",
}

CASE_FIELD_PATHS = {
    "node_coordinates": "meshes/1/nodes/coords/8",
    "cell_zone_name": "meshes/1/cells/zoneTopology/name",
    "cell_zone_id": "meshes/1/cells/zoneTopology/id",
    "cell_zone_min_id": "meshes/1/cells/zoneTopology/minId",
    "cell_zone_max_id": "meshes/1/cells/zoneTopology/maxId",
    "cell_zone_type": "meshes/1/cells/zoneTopology/zoneType",
    "face_zone_name": "meshes/1/faces/zoneTopology/name",
    "face_zone_id": "meshes/1/faces/zoneTopology/id",
    "face_zone_min_id": "meshes/1/faces/zoneTopology/minId",
    "face_zone_max_id": "meshes/1/faces/zoneTopology/maxId",
    "face_zone_type": "meshes/1/faces/zoneTopology/zoneType",
    "face_zone_face_type": "meshes/1/faces/zoneTopology/faceType",
    "face_nodes": "meshes/1/faces/nodes/1/nodes",
    "face_c0": "meshes/1/faces/c0/1",
    "face_c1": "meshes/1/faces/c1/1",
}

STRUCTURE_PATHS = {
    "direct_svars": "special/structure-direct-data/data",
    "node_data": "special/structure-node-data/nodes/8/data",
    "node_ids": "special/structure-node-data/nodes/8/elemids",
    "node_data_width": "special/structure-node-data/nodes/8/ndata",
}


@dataclass(frozen=True)
class FluentFieldBundle:
    case_path: Path
    data_path: Path
    cell_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    speed: np.ndarray
    mesh_summary: dict[str, Any]
    field_summary: dict[str, Any]


def import_official_fluent_reference(
    source_root: str | Path = DEFAULT_OFFICIAL_SOURCE_ROOT,
    output_root: str | Path = (
        Path("validation_runs")
        / "ansys_vertical_flap_fsi"
        / "official_fluent_2way_reference"
    ),
) -> dict[str, Any]:
    """Import the official Fluent run into repo-controlled reference artifacts."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    _require_source_files(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    steady = read_fluent_cell_fields(
        source_root / STEADY_CASE_NAME,
        source_root / STEADY_DATA_NAME,
    )
    fsi50 = read_fluent_cell_fields(
        source_root / FSI50_FINAL_CASE_NAME,
        source_root / FSI50_FINAL_DATA_NAME,
    )
    monitor_rows = read_monitor_timeseries(source_root / FSI50_MONITOR_NAME)
    structure_summary = summarize_monitor_timeseries(monitor_rows)
    final_structure = read_final_structure_summary(source_root / FSI50_FINAL_DATA_NAME)

    _write_field_npz(output_root / "steady_fluent_fields.npz", steady)
    _write_field_npz(output_root / "fsi50_final_fluent_fields.npz", fsi50)
    _write_monitor_csv(output_root / "fsi50_structure_monitor.csv", monitor_rows)

    field_map = {
        "case_paths": CASE_FIELD_PATHS,
        "data_cell_fields": CELL_FIELD_PATHS,
        "structure_paths": STRUCTURE_PATHS,
    }
    _write_json(output_root / "fluent_hdf5_field_map.json", field_map)

    source_summaries = _read_optional_source_summaries(source_root)
    summary = {
        "source": "local Ansys Fluent public fsi_2way tutorial run",
        "claim_boundary": "official Fluent reference import only; solver parity not claimed",
        "steady": steady.field_summary,
        "fsi50_final": fsi50.field_summary,
        "structure_monitor": structure_summary,
        "final_structure_hdf5": final_structure,
        "source_summaries": source_summaries,
    }
    _write_json(output_root / "fluent_reference_summary.json", summary)

    manifest = {
        "artifact_schema": "official_fluent_2way_reference_v1",
        "source_root": str(source_root.resolve()),
        "output_root": _repo_display_path(output_root),
        "source_files": {
            "steady_case": str((source_root / STEADY_CASE_NAME).resolve()),
            "steady_data": str((source_root / STEADY_DATA_NAME).resolve()),
            "fsi50_final_case": str((source_root / FSI50_FINAL_CASE_NAME).resolve()),
            "fsi50_final_data": str((source_root / FSI50_FINAL_DATA_NAME).resolve()),
            "fsi50_monitor": str((source_root / FSI50_MONITOR_NAME).resolve()),
            "fsi50_summary": str((source_root / FSI50_SUMMARY_NAME).resolve()),
            "steady_summary": str((source_root / STEADY_SUMMARY_NAME).resolve()),
            "steady_journal": str((source_root / STEADY_JOURNAL_NAME).resolve()),
        },
        "outputs": {
            "steady_fluent_fields": _repo_display_path(
                output_root / "steady_fluent_fields.npz"
            ),
            "fsi50_final_fluent_fields": _repo_display_path(
                output_root / "fsi50_final_fluent_fields.npz"
            ),
            "fsi50_structure_monitor": _repo_display_path(
                output_root / "fsi50_structure_monitor.csv"
            ),
            "fluent_hdf5_field_map": _repo_display_path(
                output_root / "fluent_hdf5_field_map.json"
            ),
            "fluent_reference_summary": _repo_display_path(
                output_root / "fluent_reference_summary.json"
            ),
        },
        "fluent_parity_claimed": False,
    }
    _write_json(output_root / "official_reference_manifest.json", manifest)

    return {
        "manifest": manifest,
        "summary": summary,
        "output_root": str(output_root),
    }


def read_fluent_cell_fields(case_path: str | Path, data_path: str | Path) -> FluentFieldBundle:
    h5py = _require_h5py()
    case_path = Path(case_path)
    data_path = Path(data_path)
    with h5py.File(case_path, "r") as case_file, h5py.File(data_path, "r") as data_file:
        mesh = _read_mesh(case_file)
        fluid_zone = _find_zone(mesh["cell_zones"], "fluid")
        first_cell = int(fluid_zone["min_id"])
        last_cell = int(fluid_zone["max_id"])
        cell_ids = np.arange(first_cell, last_cell + 1, dtype=np.int64)
        centers = mesh["cell_centers_by_id"][cell_ids]
        if np.isnan(centers).any():
            raise ValueError(f"could not reconstruct all fluid cell centers in {case_path}")

        fields = {
            name: _read_required_dataset(data_file, path).astype(np.float64)
            for name, path in CELL_FIELD_PATHS.items()
        }
        for name, values in fields.items():
            if values.shape != cell_ids.shape:
                raise ValueError(
                    f"{data_path}:{CELL_FIELD_PATHS[name]} shape {values.shape} "
                    f"does not match fluid cells {cell_ids.shape}"
                )
            _require_finite(values, f"{data_path}:{CELL_FIELD_PATHS[name]}")

        speed = np.hypot(fields["u"], fields["v"])
        field_summary = {
            "cell_count": int(cell_ids.size),
            "x_min": float(np.min(centers[:, 0])),
            "x_max": float(np.max(centers[:, 0])),
            "y_min": float(np.min(centers[:, 1])),
            "y_max": float(np.max(centers[:, 1])),
            "u_min": float(np.min(fields["u"])),
            "u_mean": float(np.mean(fields["u"])),
            "u_max": float(np.max(fields["u"])),
            "v_min": float(np.min(fields["v"])),
            "v_mean": float(np.mean(fields["v"])),
            "v_max": float(np.max(fields["v"])),
            "speed_min": float(np.min(speed)),
            "speed_mean": float(np.mean(speed)),
            "speed_max": float(np.max(speed)),
            "p_min": float(np.min(fields["p"])),
            "p_mean": float(np.mean(fields["p"])),
            "p_max": float(np.max(fields["p"])),
            "pressure_range": float(np.max(fields["p"]) - np.min(fields["p"])),
        }
        return FluentFieldBundle(
            case_path=case_path,
            data_path=data_path,
            cell_ids=cell_ids,
            x=centers[:, 0].astype(np.float64),
            y=centers[:, 1].astype(np.float64),
            u=fields["u"],
            v=fields["v"],
            p=fields["p"],
            speed=speed,
            mesh_summary=mesh["summary"],
            field_summary=field_summary,
        )


def read_monitor_timeseries(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"dat_path", "direct_svars", "selected_node_ids"}:
                    parsed[key] = value
                else:
                    parsed[key] = _parse_float_or_int(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"monitor CSV is empty: {path}")
    return rows


def summarize_monitor_timeseries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array(
        [float(row["monitor_avg_total_col0_col6_m"]) for row in rows],
        dtype=np.float64,
    )
    col0 = np.array([float(row["monitor_col0_svar_m"]) for row in rows])
    col6 = np.array([float(row["monitor_col6_svar_m"]) for row in rows])
    solid_max = np.array([float(row["solid_max_total_col0_col6_m"]) for row in rows])
    _require_finite(values, "monitor_avg_total_col0_col6_m")
    peak_index = int(np.argmax(values))
    final = rows[-1]
    return {
        "step_count": int(len(rows)),
        "dt_s": _infer_dt(rows),
        "target_x": float(final.get("col12_svar11455_mean", math.nan)),
        "target_y": float(final.get("col13_svar11456_mean", math.nan)),
        "monitor_displacement_peak_m": float(values[peak_index]),
        "monitor_displacement_peak_step": int(rows[peak_index]["step"]),
        "monitor_displacement_peak_time_s": float(rows[peak_index]["time_s"]),
        "monitor_final_displacement_m": float(values[-1]),
        "monitor_col0_peak_m": float(np.max(col0)),
        "monitor_col6_peak_m": float(np.max(col6)),
        "solid_max_peak_displacement_m": float(np.max(solid_max)),
        "solid_max_peak_step": int(rows[int(np.argmax(solid_max))]["step"]),
        "solid_max_final_displacement_m": float(solid_max[-1]),
    }


def read_final_structure_summary(data_path: str | Path) -> dict[str, Any]:
    h5py = _require_h5py()
    data_path = Path(data_path)
    with h5py.File(data_path, "r") as data_file:
        if STRUCTURE_PATHS["direct_svars"] not in data_file:
            return {"available": False, "reason": "structure paths absent"}
        direct_svars = _read_required_dataset(
            data_file, STRUCTURE_PATHS["direct_svars"]
        ).astype(int)
        node_ids = _read_required_dataset(data_file, STRUCTURE_PATHS["node_ids"]).astype(
            int
        )
        widths = _read_required_dataset(
            data_file, STRUCTURE_PATHS["node_data_width"]
        ).astype(int)
        if len(set(widths.tolist())) != 1:
            raise ValueError(f"unexpected variable-width node data in {data_path}")
        width = int(widths[0])
        rows = _read_required_dataset(data_file, STRUCTURE_PATHS["node_data"]).reshape(
            len(node_ids), width
        )
        displacement = np.hypot(rows[:, 0], rows[:, 6])
        max_index = int(np.argmax(displacement))
        return {
            "available": True,
            "direct_svars": direct_svars.tolist(),
            "node_count": int(len(node_ids)),
            "column_count": width,
            "displacement_col0_col6_min_m": float(np.min(displacement)),
            "displacement_col0_col6_mean_m": float(np.mean(displacement)),
            "displacement_col0_col6_max_m": float(np.max(displacement)),
            "max_node_id": int(node_ids[max_index]),
            "max_node_col0_m": float(rows[max_index, 0]),
            "max_node_col6_m": float(rows[max_index, 6]),
        }


def _read_mesh(case_file: Any) -> dict[str, Any]:
    coords = _read_required_dataset(case_file, CASE_FIELD_PATHS["node_coordinates"])
    _require_finite(coords, "node coordinates")
    cell_zones = _read_zones(case_file, "cell")
    face_zones = _read_zones(case_file, "face")
    max_cell_id = max(int(zone["max_id"]) for zone in cell_zones)
    cell_centers = _reconstruct_cell_centers(case_file, coords, max_cell_id)
    summary = {
        "node_count": int(coords.shape[0]),
        "cell_zones": _json_safe_zones(cell_zones),
        "cell_zone_center_bounds": _cell_zone_center_bounds(
            cell_zones,
            cell_centers,
        ),
        "face_zones": _json_safe_zones(face_zones),
        "face_zone_node_bounds": _face_zone_node_bounds(
            case_file,
            face_zones,
            coords,
        ),
        "x_min": float(np.min(coords[:, 0])),
        "x_max": float(np.max(coords[:, 0])),
        "y_min": float(np.min(coords[:, 1])),
        "y_max": float(np.max(coords[:, 1])),
    }
    return {
        "coords": coords,
        "cell_zones": cell_zones,
        "face_zones": face_zones,
        "cell_centers_by_id": cell_centers,
        "summary": summary,
    }


def _face_zone_node_bounds(
    case_file: Any,
    face_zones: list[dict[str, Any]],
    coords: np.ndarray,
) -> dict[str, dict[str, Any]]:
    face_nodes = _read_required_dataset(case_file, CASE_FIELD_PATHS["face_nodes"])
    if face_nodes.size % 2 != 0:
        raise ValueError("expected 2 node ids per face in 2D Fluent mesh")
    face_nodes = face_nodes.reshape((-1, 2)).astype(np.int64)
    bounds: dict[str, dict[str, Any]] = {}
    for zone in face_zones:
        first_face = int(zone["min_id"]) - 1
        last_face = int(zone["max_id"])
        zone_faces = face_nodes[first_face:last_face]
        node_ids = np.unique(zone_faces.reshape(-1))
        zone_summary: dict[str, Any] = {
            "count": int(zone["count"]),
            "node_count": int(node_ids.size),
        }
        if node_ids.size:
            node_indices = node_ids.astype(np.int64) - 1
            if np.any(node_indices < 0) or np.any(node_indices >= coords.shape[0]):
                raise ValueError(f"face zone {zone['name']!r} references invalid nodes")
            points = np.asarray(coords[node_indices], dtype=np.float64)
            finite = np.all(np.isfinite(points), axis=1)
            zone_summary["finite_node_count"] = int(np.count_nonzero(finite))
            if np.any(finite):
                finite_points = points[finite]
                zone_summary.update(
                    {
                        "x_min": float(np.min(finite_points[:, 0])),
                        "x_max": float(np.max(finite_points[:, 0])),
                        "y_min": float(np.min(finite_points[:, 1])),
                        "y_max": float(np.max(finite_points[:, 1])),
                    }
                )
        else:
            zone_summary["finite_node_count"] = 0
        bounds[str(zone["name"])] = zone_summary
    return bounds


def _cell_zone_center_bounds(
    cell_zones: list[dict[str, Any]],
    cell_centers: np.ndarray,
) -> dict[str, dict[str, Any]]:
    bounds: dict[str, dict[str, Any]] = {}
    for zone in cell_zones:
        first = int(zone["min_id"])
        last = int(zone["max_id"])
        centers = np.asarray(cell_centers[first : last + 1], dtype=np.float64)
        finite = np.all(np.isfinite(centers), axis=1)
        zone_summary: dict[str, Any] = {
            "count": int(zone["count"]),
            "finite_center_count": int(np.count_nonzero(finite)),
        }
        if np.any(finite):
            finite_centers = centers[finite]
            zone_summary.update(
                {
                    "x_min": float(np.min(finite_centers[:, 0])),
                    "x_max": float(np.max(finite_centers[:, 0])),
                    "y_min": float(np.min(finite_centers[:, 1])),
                    "y_max": float(np.max(finite_centers[:, 1])),
                }
            )
        bounds[str(zone["name"])] = zone_summary
    return bounds


def _reconstruct_cell_centers(
    case_file: Any, coords: np.ndarray, max_cell_id: int
) -> np.ndarray:
    face_nodes = _read_required_dataset(case_file, CASE_FIELD_PATHS["face_nodes"])
    if face_nodes.size % 2 != 0:
        raise ValueError("expected 2 node ids per face in 2D Fluent mesh")
    face_nodes = face_nodes.reshape((-1, 2)).astype(np.int64)
    c0 = _read_required_dataset(case_file, CASE_FIELD_PATHS["face_c0"]).astype(np.int64)
    c1 = _read_required_dataset(case_file, CASE_FIELD_PATHS["face_c1"]).astype(np.int64)
    if len(c0) != len(face_nodes):
        raise ValueError("face c0 length does not match face-node length")

    nodes_by_cell: list[set[int]] = [set() for _ in range(max_cell_id + 1)]
    for face_index, (node_a, node_b) in enumerate(face_nodes):
        first_cell = int(c0[face_index])
        if first_cell > 0:
            nodes_by_cell[first_cell].update((int(node_a), int(node_b)))
        if face_index < len(c1):
            second_cell = int(c1[face_index])
            if second_cell > 0:
                nodes_by_cell[second_cell].update((int(node_a), int(node_b)))

    centers = np.full((max_cell_id + 1, coords.shape[1]), np.nan, dtype=np.float64)
    for cell_id in range(1, max_cell_id + 1):
        node_ids = sorted(nodes_by_cell[cell_id])
        if node_ids:
            centers[cell_id] = np.mean(coords[np.array(node_ids) - 1], axis=0)
    return centers


def _read_zones(case_file: Any, kind: str) -> list[dict[str, Any]]:
    prefix = "meshes/1/cells/zoneTopology" if kind == "cell" else "meshes/1/faces/zoneTopology"
    names = _decode_semicolon_dataset(_read_required_dataset(case_file, f"{prefix}/name"))
    ids = _read_required_dataset(case_file, f"{prefix}/id")
    min_ids = _read_required_dataset(case_file, f"{prefix}/minId")
    max_ids = _read_required_dataset(case_file, f"{prefix}/maxId")
    zone_types = _read_required_dataset(case_file, f"{prefix}/zoneType")
    if not (len(names) == len(ids) == len(min_ids) == len(max_ids) == len(zone_types)):
        raise ValueError(f"{kind} zone topology arrays have inconsistent lengths")
    return [
        {
            "name": names[index],
            "id": int(ids[index]),
            "min_id": int(min_ids[index]),
            "max_id": int(max_ids[index]),
            "count": int(max_ids[index] - min_ids[index] + 1),
            "zone_type": int(zone_types[index]),
        }
        for index in range(len(names))
    ]


def _find_zone(zones: list[dict[str, Any]], name_fragment: str) -> dict[str, Any]:
    matches = [zone for zone in zones if name_fragment in str(zone["name"])]
    if not matches:
        raise ValueError(f"could not find zone matching {name_fragment!r}")
    if len(matches) > 1:
        exact = [zone for zone in matches if str(zone["name"]).startswith(name_fragment)]
        if len(exact) == 1:
            return exact[0]
    return matches[0]


def _write_field_npz(path: Path, bundle: FluentFieldBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cell_ids=bundle.cell_ids,
        x=bundle.x,
        y=bundle.y,
        u=bundle.u,
        v=bundle.v,
        p=bundle.p,
        speed=bundle.speed,
        field_summary_json=json.dumps(bundle.field_summary, sort_keys=True),
        mesh_summary_json=json.dumps(bundle.mesh_summary, sort_keys=True),
        case_path=str(bundle.case_path),
        data_path=str(bundle.data_path),
    )


def _write_monitor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_optional_source_summaries(source_root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key, filename in (
        ("steady_reference_summary", STEADY_SUMMARY_NAME),
        ("official_fsi_50step_summary", FSI50_SUMMARY_NAME),
    ):
        path = source_root / filename
        if path.exists():
            summaries[key] = json.loads(path.read_text(encoding="utf-8"))
    journal = source_root / STEADY_JOURNAL_NAME
    if journal.exists():
        summaries["steady_journal_text"] = journal.read_text(encoding="utf-8")
    return summaries


def _require_source_files(source_root: Path) -> None:
    required = [
        STEADY_CASE_NAME,
        STEADY_DATA_NAME,
        FSI50_FINAL_CASE_NAME,
        FSI50_FINAL_DATA_NAME,
        FSI50_MONITOR_NAME,
    ]
    missing = [name for name in required if not (source_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing Fluent reference files under {source_root}: {missing}"
        )


def _read_required_dataset(handle: Any, path: str) -> np.ndarray:
    if path not in handle:
        raise KeyError(f"missing required Fluent HDF5 dataset: {path}")
    return handle[path][()]


def _decode_semicolon_dataset(value: np.ndarray) -> list[str]:
    if value.size != 1:
        raise ValueError("expected semicolon zone dataset with one row")
    raw = value[0]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    return [part for part in text.split(";") if part]


def _json_safe_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(zone) for zone in zones]


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values in {label}")


def _parse_float_or_int(value: str) -> float | int | str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _infer_dt(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return math.nan
    return float(rows[1]["time_s"]) - float(rows[0]["time_s"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repo_display_path(path: Path) -> str:
    try:
        return path.as_posix()
    except ValueError:
        return str(path)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read official Fluent HDF5 reference files"
        ) from exc
    return h5py


def copy_private_reference_artifacts_for_debug(
    source_root: str | Path, destination_root: str | Path
) -> None:
    """Explicit opt-in helper for local debugging; not used by the importer."""

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in (
        STEADY_CASE_NAME,
        STEADY_DATA_NAME,
        FSI50_FINAL_CASE_NAME,
        FSI50_FINAL_DATA_NAME,
        FSI50_MONITOR_NAME,
    ):
        shutil.copy2(source_root / name, destination_root / name)
