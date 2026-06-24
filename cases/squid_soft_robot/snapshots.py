from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .history import write_csv

def _ascii_vtk_numbers(values: np.ndarray, *, precision: int = 9) -> str:
    flat = np.asarray(values).reshape(-1)
    return " ".join(f"{float(value):.{precision}g}" for value in flat)

def _write_fluid_snapshot_npz(
    *,
    snapshot_dir: Path,
    step: int,
    fluid,
    markers,
    marker_count: int,
    time_s: float,
    pressure_pa: float,
) -> Path | None:
    """Dump a compact per-step visualization snapshot (case-level host read)."""
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        velocity = np.asarray(fluid.velocity.to_numpy(), dtype=np.float32)
        obstacle = np.asarray(fluid.obstacle.to_numpy(), dtype=np.int8)
        speed = np.linalg.norm(velocity, axis=-1).astype(np.float32)
        positions = np.asarray(
            markers.x_gamma_m.to_numpy()[: int(marker_count)],
            dtype=np.float32,
        )
        nx, ny, nz = speed.shape
        path = snapshot_dir / f"snapshot_{int(step):06d}.npz"
        np.savez_compressed(
            path,
            step=np.int64(step),
            time_s=np.float64(time_s),
            pressure_pa=np.float64(pressure_pa),
            speed_xz=speed[:, ny // 2, :],
            speed_yz=speed[nx // 2, :, :],
            obstacle_xz=obstacle[:, ny // 2, :],
            marker_positions_m=positions,
            cell_center_x_m=np.asarray(
                fluid.cell_center_x_m.to_numpy(), dtype=np.float32
            ),
            cell_center_y_m=np.asarray(
                fluid.cell_center_y_m.to_numpy(), dtype=np.float32
            ),
            cell_center_z_m=np.asarray(
                fluid.cell_center_z_m.to_numpy(), dtype=np.float32
            ),
        )
        return path
    except Exception as exc:  # noqa: BLE001 - snapshot must not kill a long run
        print(f"[snapshot] step {step} failed: {exc}", flush=True)
        return None

def _write_minimal_fluid_vti(
    *,
    output_dir: Path,
    step: int,
    fluid,
) -> Path | None:
    try:
        velocity = np.asarray(fluid.velocity.to_numpy(), dtype=np.float32)
        obstacle = np.asarray(fluid.obstacle.to_numpy(), dtype=np.int32)
        divergence = np.asarray(fluid.divergence.to_numpy(), dtype=np.float32)
        if velocity.ndim != 4 or velocity.shape[-1] != 3:
            return None
        if obstacle.shape != velocity.shape[:3] or divergence.shape != velocity.shape[:3]:
            return None
        nx, ny, nz = (int(value) for value in velocity.shape[:3])
        if nx <= 0 or ny <= 0 or nz <= 0:
            return None
        x_centers = np.asarray(fluid.cell_center_x_m.to_numpy(), dtype=np.float64)
        y_centers = np.asarray(fluid.cell_center_y_m.to_numpy(), dtype=np.float64)
        z_centers = np.asarray(fluid.cell_center_z_m.to_numpy(), dtype=np.float64)
        width_x = np.asarray(fluid.cell_width_x_m.to_numpy(), dtype=np.float64)
        width_y = np.asarray(fluid.cell_width_y_m.to_numpy(), dtype=np.float64)
        width_z = np.asarray(fluid.cell_width_z_m.to_numpy(), dtype=np.float64)
        spacing = (
            float(np.mean(width_x)) if width_x.size else 1.0,
            float(np.mean(width_y)) if width_y.size else 1.0,
            float(np.mean(width_z)) if width_z.size else 1.0,
        )
        origin = (
            float(x_centers[0]) if x_centers.size else 0.0,
            float(y_centers[0]) if y_centers.size else 0.0,
            float(z_centers[0]) if z_centers.size else 0.0,
        )
        speed = np.linalg.norm(velocity, axis=3).astype(np.float32)
        active_fluid = (obstacle == 0).astype(np.int32)
        path = output_dir / f"sharp_failure_step_{int(step):06d}_fluid.vti"
        extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
        text = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
            f'  <ImageData WholeExtent="{extent}" '
            f'Origin="{origin[0]:.9g} {origin[1]:.9g} {origin[2]:.9g}" '
            f'Spacing="{spacing[0]:.9g} {spacing[1]:.9g} {spacing[2]:.9g}">\n'
            f'    <Piece Extent="{extent}">\n'
            '      <PointData Scalars="speed_mps" Vectors="velocity_mps">\n'
            '        <DataArray type="Float32" Name="velocity_mps" '
            'NumberOfComponents="3" format="ascii">\n'
            f'          {_ascii_vtk_numbers(velocity)}\n'
            '        </DataArray>\n'
            '        <DataArray type="Float32" Name="speed_mps" format="ascii">\n'
            f'          {_ascii_vtk_numbers(speed)}\n'
            '        </DataArray>\n'
            '        <DataArray type="Int32" Name="obstacle" format="ascii">\n'
            f'          {" ".join(str(int(value)) for value in obstacle.reshape(-1))}\n'
            '        </DataArray>\n'
            '        <DataArray type="Int32" Name="active_fluid" format="ascii">\n'
            f'          {" ".join(str(int(value)) for value in active_fluid.reshape(-1))}\n'
            '        </DataArray>\n'
            '        <DataArray type="Float32" Name="divergence" format="ascii">\n'
            f'          {_ascii_vtk_numbers(divergence)}\n'
            '        </DataArray>\n'
            '      </PointData>\n'
            '      <CellData/>\n'
            '    </Piece>\n'
            '  </ImageData>\n'
            '</VTKFile>\n'
        )
        path.write_text(text, encoding="utf-8")
        return path
    except (AttributeError, OSError, ValueError, TypeError):
        return None

def _write_step_failure_artifacts(
    *,
    process_path: Path,
    output_dir: Path,
    rows: list[dict[str, object]],
    step: int,
    exc: Exception,
    fluid=None,
    markers=None,
    pressure_outlet_zmin: bool = True,
) -> Path:
    partial_history_path = output_dir / "history.csv"
    write_csv(partial_history_path, rows)
    failure_fluid_vti = (
        _write_minimal_fluid_vti(output_dir=output_dir, step=step, fluid=fluid)
        if fluid is not None
        else None
    )
    high_residual_summary = None
    pressure_interface_matrix_report = None
    if fluid is not None:
        try:
            high_residual_summary = _write_hibm_high_residual_cell_dump(
                output_dir=output_dir,
                step=step,
                fluid=fluid,
                markers=markers,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        except (AttributeError, OSError, ValueError, TypeError):
            high_residual_summary = None
        try:
            pressure_interface_matrix_report = (
                fluid.pressure_interface_matrix_terms_report()
            )
        except (AttributeError, OSError, ValueError, TypeError):
            pressure_interface_matrix_report = None
    process_payload: dict[str, object] = {}
    if process_path.exists():
        try:
            parsed = json.loads(process_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                process_payload.update(parsed)
        except (OSError, json.JSONDecodeError):
            pass
    process_payload.update(
        {
            "pid": os.getpid(),
            "status": "failed",
            "failed_at_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_step": int(step),
            "history_csv": str(partial_history_path),
        }
    )
    if failure_fluid_vti is not None:
        process_payload["failure_fluid_vti"] = str(failure_fluid_vti)
    if high_residual_summary is not None:
        process_payload["failure_high_residual_cells"] = high_residual_summary
    if pressure_interface_matrix_report is not None:
        process_payload["failure_pressure_interface_matrix"] = (
            pressure_interface_matrix_report
        )
    process_path.write_text(
        json.dumps(process_payload, indent=2),
        encoding="utf-8",
    )
    return partial_history_path

def _pressure_correctable_mask_from_host_fields(
    *,
    obstacle: np.ndarray,
    velocity_dirichlet_active: np.ndarray,
    pressure_outlet_zmin: bool,
) -> np.ndarray:
    obstacle_mask = obstacle != 0
    fixed = velocity_dirichlet_active != 0
    nx, ny, nz = obstacle_mask.shape
    correctable = np.zeros(obstacle_mask.shape, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if obstacle_mask[i, j, k]:
                    continue
                if i > 0 and not obstacle_mask[i - 1, j, k] and not fixed[i, j, k]:
                    correctable[i, j, k] = True
                if (
                    i < nx - 1
                    and not obstacle_mask[i + 1, j, k]
                    and not fixed[i + 1, j, k]
                ):
                    correctable[i, j, k] = True
                if j > 0 and not obstacle_mask[i, j - 1, k] and not fixed[i, j, k]:
                    correctable[i, j, k] = True
                if (
                    j < ny - 1
                    and not obstacle_mask[i, j + 1, k]
                    and not fixed[i, j + 1, k]
                ):
                    correctable[i, j, k] = True
                if k > 0 and not obstacle_mask[i, j, k - 1] and not fixed[i, j, k]:
                    correctable[i, j, k] = True
                if (
                    k < nz - 1
                    and not obstacle_mask[i, j, k + 1]
                    and not fixed[i, j, k + 1]
                ):
                    correctable[i, j, k] = True
                if pressure_outlet_zmin and k == 0 and not fixed[i, j, k]:
                    correctable[i, j, k] = True
    return correctable

def _write_hibm_zero_correctable_cell_dump(
    *,
    output_dir: Path,
    step: int,
    fluid,
    markers,
    pressure_outlet_zmin: bool,
) -> dict[str, object]:
    obstacle = fluid.obstacle.to_numpy()
    velocity_dirichlet_active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    correctable = _pressure_correctable_mask_from_host_fields(
        obstacle=obstacle,
        velocity_dirichlet_active=velocity_dirichlet_active,
        pressure_outlet_zmin=pressure_outlet_zmin,
    )
    active_fluid = obstacle == 0
    interior = np.zeros(active_fluid.shape, dtype=bool)
    if all(axis_size > 2 for axis_size in active_fluid.shape):
        interior[1:-1, 1:-1, 1:-1] = True
    zero_correctable = active_fluid & interior & ~correctable
    indices = np.argwhere(zero_correctable)

    x_centers = fluid.cell_center_x_m.to_numpy()
    y_centers = fluid.cell_center_y_m.to_numpy()
    z_centers = fluid.cell_center_z_m.to_numpy()
    width_x = fluid.cell_width_x_m.to_numpy()
    width_y = fluid.cell_width_y_m.to_numpy()
    width_z = fluid.cell_width_z_m.to_numpy()
    divergence = fluid.divergence.to_numpy()
    volume_source = fluid.volume_source_s.to_numpy()

    marker_count = int(markers.marker_count)
    marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
    marker_normals = markers.n_gamma.to_numpy()[:marker_count]
    marker_regions = markers.region_id.to_numpy()[:marker_count]
    nearest_index = np.full(indices.shape[0], -1, dtype=np.int64)
    nearest_distance = np.full(indices.shape[0], math.nan, dtype=np.float64)
    nearest_signed_distance = np.full(indices.shape[0], math.nan, dtype=np.float64)
    nearest_region = np.full(indices.shape[0], -1, dtype=np.int64)
    if marker_count > 0 and len(indices) > 0:
        positions = np.column_stack(
            (
                x_centers[indices[:, 0]],
                y_centers[indices[:, 1]],
                z_centers[indices[:, 2]],
            )
        )
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            delta = positions[start:end, None, :] - marker_positions[None, :, :]
            distance2 = np.einsum("cmq,cmq->cm", delta, delta)
            local_index = np.argmin(distance2, axis=1)
            global_index = local_index.astype(np.int64)
            local_delta = delta[np.arange(end - start), local_index, :]
            local_normals = marker_normals[global_index]
            nearest_index[start:end] = global_index
            nearest_distance[start:end] = np.sqrt(distance2[np.arange(end - start), local_index])
            nearest_signed_distance[start:end] = np.einsum(
                "cq,cq->c",
                local_delta,
                local_normals,
            )
            nearest_region[start:end] = marker_regions[global_index]

    dump_dir = output_dir / "hibm_zero_correctable_cells"
    dump_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dump_dir / f"step_{int(step):06d}_interior_zero_correctable_cells.csv"
    fieldnames = (
        "i",
        "j",
        "k",
        "x_m",
        "y_m",
        "z_m",
        "divergence_s",
        "volume_source_s",
        "residual_s",
        "nearest_marker_index",
        "nearest_marker_region_id",
        "nearest_marker_distance_m",
        "nearest_marker_signed_distance_m",
        "local_cell_diagonal_m",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (i, j, k) in enumerate(indices):
            div = float(divergence[i, j, k])
            src = float(volume_source[i, j, k])
            writer.writerow(
                {
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "x_m": float(x_centers[i]),
                    "y_m": float(y_centers[j]),
                    "z_m": float(z_centers[k]),
                    "divergence_s": div,
                    "volume_source_s": src,
                    "residual_s": div - src,
                    "nearest_marker_index": int(nearest_index[row_index]),
                    "nearest_marker_region_id": int(nearest_region[row_index]),
                    "nearest_marker_distance_m": float(nearest_distance[row_index]),
                    "nearest_marker_signed_distance_m": float(
                        nearest_signed_distance[row_index]
                    ),
                    "local_cell_diagonal_m": math.sqrt(
                        float(width_x[i]) ** 2
                        + float(width_y[j]) ** 2
                        + float(width_z[k]) ** 2
                    ),
                }
            )

    local_diagonal = np.sqrt(
        width_x[indices[:, 0]] ** 2
        + width_y[indices[:, 1]] ** 2
        + width_z[indices[:, 2]] ** 2
    ) if len(indices) > 0 else np.array([], dtype=np.float64)
    shell_band_candidate = (
        np.isfinite(nearest_distance)
        & (nearest_distance <= local_diagonal)
        if len(indices) > 0
        else np.array([], dtype=bool)
    )
    region_counts: dict[str, int] = {}
    for region in nearest_region:
        key = str(int(region))
        region_counts[key] = region_counts.get(key, 0) + 1
    summary = {
        "step": int(step),
        "pressure_outlet_zmin": bool(pressure_outlet_zmin),
        "zero_correctable_interior_cell_count": int(len(indices)),
        "active_fluid_cell_count": int(np.count_nonzero(active_fluid)),
        "pressure_correctable_cell_count": int(np.count_nonzero(active_fluid & correctable)),
        "nearest_marker_count": int(marker_count),
        "nearest_marker_region_counts": region_counts,
        "shell_band_candidate_cell_count": int(np.count_nonzero(shell_band_candidate)),
        "shell_band_candidate_rule": (
            "nearest marker distance <= local cell diagonal"
        ),
        "nearest_marker_distance_min_m": (
            float(np.nanmin(nearest_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_distance_mean_m": (
            float(np.nanmean(nearest_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_distance_max_m": (
            float(np.nanmax(nearest_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_signed_distance_min_m": (
            float(np.nanmin(nearest_signed_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_signed_distance_mean_m": (
            float(np.nanmean(nearest_signed_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_signed_distance_max_m": (
            float(np.nanmax(nearest_signed_distance)) if len(indices) > 0 else math.nan
        ),
        "negative_signed_distance_count": int(
            np.count_nonzero(nearest_signed_distance < 0.0)
        ),
        "positive_signed_distance_count": int(
            np.count_nonzero(nearest_signed_distance > 0.0)
        ),
        "csv_path": str(csv_path),
    }
    if len(indices) > 0:
        summary["i_min"] = int(np.min(indices[:, 0]))
        summary["i_max"] = int(np.max(indices[:, 0]))
        summary["j_min"] = int(np.min(indices[:, 1]))
        summary["j_max"] = int(np.max(indices[:, 1]))
        summary["k_min"] = int(np.min(indices[:, 2]))
        summary["k_max"] = int(np.max(indices[:, 2]))
    summary_path = dump_dir / f"step_{int(step):06d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
def _write_hibm_pressure_neumann_invalid_row_dump(
    *,
    output_dir: Path,
    step: int,
    ib_boundary=None,
    search=None,
    markers=None,
    fluid=None,
    rows=None,
    stage: str = "latest",
    limit: int = 1024,
) -> dict[str, object]:
    rows_provided = rows is not None
    if rows is None:
        if ib_boundary is None:
            raise ValueError("ib_boundary or rows must be provided")
        rows = ib_boundary.pressure_neumann_invalid_diagnostic_rows(
            search=search,
            markers=markers,
            fluid=fluid,
            limit=limit,
        )
    else:
        rows = list(rows)[: max(0, int(limit))]
    dump_dir = output_dir / "hibm_pressure_neumann_invalid_rows"
    dump_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(stage)
    )
    csv_path = (
        dump_dir
        / f"step_{int(step):06d}_{safe_stage}_invalid_pressure_neumann_rows.csv"
    )
    fieldnames = tuple(rows[0].keys()) if rows else (
        "row_index",
        "reason_code",
        "reason",
        "node_i",
        "node_j",
        "node_k",
        "owner_i",
        "owner_j",
        "owner_k",
        "neighbor_i",
        "neighbor_j",
        "neighbor_k",
        "anchor_i",
        "anchor_j",
        "anchor_k",
        "marker_index",
        "marker_region_id",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    total_count = len(rows)
    count_field = getattr(ib_boundary, "pressure_neumann_invalid_diag_count", None)
    if count_field is not None and not rows_provided:
        try:
            total_count = int(count_field[None])
        except Exception:
            total_count = len(rows)

    reason_counts: dict[str, int] = {}
    marker_region_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        region = str(int(row.get("marker_region_id", -1)))
        marker_region_counts[region] = marker_region_counts.get(region, 0) + 1

    summary = {
        "step": int(step),
        "stage": str(stage),
        "captured_invalid_row_count": int(len(rows)),
        "total_invalid_row_count": int(total_count),
        "diagnostic_capacity": int(limit),
        "reason_counts": reason_counts,
        "marker_region_counts": marker_region_counts,
        "csv_path": str(csv_path),
    }
    summary_path = dump_dir / f"step_{int(step):06d}_{safe_stage}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def _write_hibm_high_residual_cell_dump(
    *,
    output_dir: Path,
    step: int,
    fluid,
    markers=None,
    pressure_outlet_zmin: bool,
    limit: int = 256,
) -> dict[str, object]:
    obstacle = fluid.obstacle.to_numpy()
    velocity_dirichlet_active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    velocity_dirichlet_value = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
    velocity_dirichlet_projection_weight = (
        fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
    )
    correctable = _pressure_correctable_mask_from_host_fields(
        obstacle=obstacle,
        velocity_dirichlet_active=velocity_dirichlet_active,
        pressure_outlet_zmin=pressure_outlet_zmin,
    )
    active_fluid = obstacle == 0
    interior = np.zeros(active_fluid.shape, dtype=bool)
    if all(axis_size > 2 for axis_size in active_fluid.shape):
        interior[1:-1, 1:-1, 1:-1] = True
    candidates = active_fluid & interior
    indices = np.argwhere(candidates)

    divergence = fluid.divergence.to_numpy()
    volume_source = fluid.volume_source_s.to_numpy()
    residual = divergence - volume_source
    top_limit = max(0, int(limit))
    if len(indices) > 0 and top_limit > 0:
        candidate_abs = np.abs(residual[candidates])
        selected = np.argpartition(
            candidate_abs,
            -min(top_limit, len(candidate_abs)),
        )[-min(top_limit, len(candidate_abs)) :]
        selected = selected[np.argsort(candidate_abs[selected])[::-1]]
        indices = indices[selected]
    else:
        indices = np.empty((0, 3), dtype=np.int64)

    x_centers = fluid.cell_center_x_m.to_numpy()
    y_centers = fluid.cell_center_y_m.to_numpy()
    z_centers = fluid.cell_center_z_m.to_numpy()
    width_x = fluid.cell_width_x_m.to_numpy()
    width_y = fluid.cell_width_y_m.to_numpy()
    width_z = fluid.cell_width_z_m.to_numpy()
    pressure = fluid.pressure.to_numpy()
    pressure_diag = fluid.pressure_interface_matrix_diagonal.to_numpy()
    pressure_rhs = fluid.pressure_interface_matrix_rhs.to_numpy()
    pressure_outlet_reachable = fluid.hibm_pressure_outlet_reachable.to_numpy()
    pressure_unreached_component_label = (
        fluid.hibm_pressure_unreached_component_label.to_numpy()
    )
    velocity_dirichlet_marker_region = getattr(
        fluid,
        "velocity_dirichlet_boundary_marker_region_id",
        None,
    )
    if velocity_dirichlet_marker_region is None:
        velocity_dirichlet_marker_region_id = np.full(
            obstacle.shape,
            -1,
            dtype=np.int32,
        )
    else:
        velocity_dirichlet_marker_region_id = (
            velocity_dirichlet_marker_region.to_numpy()
        )

    marker_count = 0 if markers is None else int(markers.marker_count)
    nearest_index = np.full(indices.shape[0], -1, dtype=np.int64)
    nearest_distance = np.full(indices.shape[0], math.nan, dtype=np.float64)
    nearest_signed_distance = np.full(indices.shape[0], math.nan, dtype=np.float64)
    nearest_region = np.full(indices.shape[0], -1, dtype=np.int64)
    if marker_count > 0 and len(indices) > 0:
        marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
        marker_normals = markers.n_gamma.to_numpy()[:marker_count]
        marker_regions = markers.region_id.to_numpy()[:marker_count]
        positions = np.column_stack(
            (
                x_centers[indices[:, 0]],
                y_centers[indices[:, 1]],
                z_centers[indices[:, 2]],
            )
        )
        for start in range(0, len(indices), 128):
            end = min(start + 128, len(indices))
            delta = positions[start:end, None, :] - marker_positions[None, :, :]
            distance2 = np.einsum("cmq,cmq->cm", delta, delta)
            local_index = np.argmin(distance2, axis=1)
            global_index = local_index.astype(np.int64)
            local_delta = delta[np.arange(end - start), local_index, :]
            local_normals = marker_normals[global_index]
            nearest_index[start:end] = global_index
            nearest_distance[start:end] = np.sqrt(
                distance2[np.arange(end - start), local_index]
            )
            nearest_signed_distance[start:end] = np.einsum(
                "cq,cq->c",
                local_delta,
                local_normals,
            )
            nearest_region[start:end] = marker_regions[global_index]

    dump_dir = output_dir / "hibm_high_residual_cells"
    dump_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dump_dir / f"step_{int(step):06d}_top_residual_cells.csv"
    fieldnames = (
        "rank",
        "i",
        "j",
        "k",
        "x_m",
        "y_m",
        "z_m",
        "divergence_s",
        "volume_source_s",
        "residual_s",
        "abs_residual_s",
        "pressure_pa",
        "pressure_interface_diagonal_per_s2",
        "pressure_interface_rhs_pa_per_m2",
        "pressure_outlet_reachable",
        "pressure_unreached_component_label",
        "pressure_correctable",
        "velocity_dirichlet_active",
        "velocity_dirichlet_marker_region_id",
        "velocity_dirichlet_projection_weight",
        "velocity_dirichlet_value_x_mps",
        "velocity_dirichlet_value_y_mps",
        "velocity_dirichlet_value_z_mps",
        "x_left_dirichlet",
        "x_right_dirichlet",
        "y_back_dirichlet",
        "y_front_dirichlet",
        "z_bottom_dirichlet",
        "z_top_dirichlet",
        "nearest_marker_index",
        "nearest_marker_region_id",
        "nearest_marker_distance_m",
        "nearest_marker_signed_distance_m",
        "local_cell_diagonal_m",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (i, j, k) in enumerate(indices, start=1):
            i = int(i)
            j = int(j)
            k = int(k)
            cell_residual = float(residual[i, j, k])
            velocity_target = velocity_dirichlet_value[i, j, k]
            writer.writerow(
                {
                    "rank": int(rank),
                    "i": i,
                    "j": j,
                    "k": k,
                    "x_m": float(x_centers[i]),
                    "y_m": float(y_centers[j]),
                    "z_m": float(z_centers[k]),
                    "divergence_s": float(divergence[i, j, k]),
                    "volume_source_s": float(volume_source[i, j, k]),
                    "residual_s": cell_residual,
                    "abs_residual_s": abs(cell_residual),
                    "pressure_pa": float(pressure[i, j, k]),
                    "pressure_interface_diagonal_per_s2": float(pressure_diag[i, j, k]),
                    "pressure_interface_rhs_pa_per_m2": float(pressure_rhs[i, j, k]),
                    "pressure_outlet_reachable": int(
                        pressure_outlet_reachable[i, j, k]
                    ),
                    "pressure_unreached_component_label": int(
                        pressure_unreached_component_label[i, j, k]
                    ),
                    "pressure_correctable": int(correctable[i, j, k]),
                    "velocity_dirichlet_active": int(velocity_dirichlet_active[i, j, k]),
                    "velocity_dirichlet_marker_region_id": int(
                        velocity_dirichlet_marker_region_id[i, j, k]
                    ),
                    "velocity_dirichlet_projection_weight": float(
                        velocity_dirichlet_projection_weight[i, j, k]
                    ),
                    "velocity_dirichlet_value_x_mps": float(velocity_target[0]),
                    "velocity_dirichlet_value_y_mps": float(velocity_target[1]),
                    "velocity_dirichlet_value_z_mps": float(velocity_target[2]),
                    "x_left_dirichlet": int(
                        i > 0 and velocity_dirichlet_active[i, j, k] != 0
                    ),
                    "x_right_dirichlet": int(
                        i < velocity_dirichlet_active.shape[0] - 1
                        and velocity_dirichlet_active[i + 1, j, k] != 0
                    ),
                    "y_back_dirichlet": int(
                        j > 0 and velocity_dirichlet_active[i, j, k] != 0
                    ),
                    "y_front_dirichlet": int(
                        j < velocity_dirichlet_active.shape[1] - 1
                        and velocity_dirichlet_active[i, j + 1, k] != 0
                    ),
                    "z_bottom_dirichlet": int(
                        k > 0 and velocity_dirichlet_active[i, j, k] != 0
                    ),
                    "z_top_dirichlet": int(
                        k < velocity_dirichlet_active.shape[2] - 1
                        and velocity_dirichlet_active[i, j, k + 1] != 0
                    ),
                    "nearest_marker_index": int(nearest_index[rank - 1]),
                    "nearest_marker_region_id": int(nearest_region[rank - 1]),
                    "nearest_marker_distance_m": float(nearest_distance[rank - 1]),
                    "nearest_marker_signed_distance_m": float(
                        nearest_signed_distance[rank - 1]
                    ),
                    "local_cell_diagonal_m": math.sqrt(
                        float(width_x[i]) ** 2
                        + float(width_y[j]) ** 2
                        + float(width_z[k]) ** 2
                    ),
                }
            )

    selected_abs_residual = np.abs(residual[tuple(indices.T)]) if len(indices) else np.array([])
    selected_regions: dict[str, int] = {}
    for region in nearest_region:
        key = str(int(region))
        selected_regions[key] = selected_regions.get(key, 0) + 1
    selected_correctable = (
        correctable[tuple(indices.T)] if len(indices) else np.array([], dtype=bool)
    )
    selected_dirichlet = (
        velocity_dirichlet_active[tuple(indices.T)] != 0
        if len(indices)
        else np.array([], dtype=bool)
    )
    selected_pressure_diag = (
        pressure_diag[tuple(indices.T)] if len(indices) else np.array([], dtype=np.float32)
    )
    selected_pressure_rhs = (
        pressure_rhs[tuple(indices.T)] if len(indices) else np.array([], dtype=np.float32)
    )
    selected_reachable = (
        pressure_outlet_reachable[tuple(indices.T)]
        if len(indices)
        else np.array([], dtype=np.int32)
    )
    selected_unreached_labels = (
        pressure_unreached_component_label[tuple(indices.T)]
        if len(indices)
        else np.array([], dtype=np.int32)
    )
    selected_dirichlet_regions = (
        velocity_dirichlet_marker_region_id[tuple(indices.T)]
        if len(indices)
        else np.array([], dtype=np.int32)
    )
    selected_dirichlet_region_counts: dict[str, int] = {}
    for region in selected_dirichlet_regions:
        key = str(int(region))
        selected_dirichlet_region_counts[key] = (
            selected_dirichlet_region_counts.get(key, 0) + 1
        )
    summary = {
        "step": int(step),
        "pressure_outlet_zmin": bool(pressure_outlet_zmin),
        "candidate_interior_active_cell_count": int(np.count_nonzero(candidates)),
        "dumped_cell_count": int(len(indices)),
        "limit": int(top_limit),
        "active_fluid_cell_count": int(np.count_nonzero(active_fluid)),
        "pressure_correctable_cell_count": int(np.count_nonzero(active_fluid & correctable)),
        "dumped_pressure_correctable_cell_count": int(np.count_nonzero(selected_correctable)),
        "dumped_velocity_dirichlet_cell_count": int(np.count_nonzero(selected_dirichlet)),
        "dumped_pressure_outlet_reachable_cell_count": int(
            np.count_nonzero(selected_reachable != 0)
        ),
        "dumped_unreached_labeled_cell_count": int(
            np.count_nonzero(
                (selected_unreached_labels >= -32)
                & (selected_unreached_labels <= -1)
            )
        ),
        "dumped_pressure_interface_diagonal_cell_count": int(
            np.count_nonzero(selected_pressure_diag != 0.0)
        ),
        "dumped_pressure_interface_rhs_cell_count": int(
            np.count_nonzero(selected_pressure_rhs != 0.0)
        ),
        "max_abs_residual_s": (
            float(np.max(selected_abs_residual)) if len(selected_abs_residual) else 0.0
        ),
        "nearest_marker_count": int(marker_count),
        "nearest_marker_region_counts": selected_regions,
        "velocity_dirichlet_marker_region_counts": selected_dirichlet_region_counts,
        "csv_path": str(csv_path),
    }
    if len(indices) > 0:
        summary["i_min"] = int(np.min(indices[:, 0]))
        summary["i_max"] = int(np.max(indices[:, 0]))
        summary["j_min"] = int(np.min(indices[:, 1]))
        summary["j_max"] = int(np.max(indices[:, 1]))
        summary["k_min"] = int(np.min(indices[:, 2]))
        summary["k_max"] = int(np.max(indices[:, 2]))
    summary_path = dump_dir / f"step_{int(step):06d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
