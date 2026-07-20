"""Offline post-processing for a paired Fluent fine-grid FSI campaign.

The campaign contract is intentionally strict: every ``step_XXXX.cas.h5`` must
have a matching ``step_XXXX.dat.h5``; fields and structural displacements must
be finite; and every step must contain a non-zero structural displacement.
Nothing in this module launches Fluent or mutates the source run directory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .official_fluent_reference import (
    FluentFieldBundle,
    _write_field_npz,
    read_fluent_cell_fields,
)


STEP_FILE_RE = re.compile(r"^step_(?P<step>\d{4,})\.(?P<kind>cas|dat)\.h5$")
DEFAULT_DT_S = 5.0e-4
DEFAULT_TARGET_X_M = 0.0505
DEFAULT_TARGET_Y_M = 0.0095
DEFAULT_VELOCITY_VMAX_MPS = 28.1
DEFAULT_GIF_DURATION_MS = 120
DEFAULT_GIF_MAX_WIDTH_PX = 1600
DEFAULT_NONZERO_DISPLACEMENT_TOLERANCE_M = 1.0e-18
DEFAULT_EXPECTED_STEP_COUNT = 50


class CampaignValidationError(RuntimeError):
    """Raised when an offline campaign artifact violates the strict contract."""


@dataclass(frozen=True)
class StepPair:
    """One immutable, paired Fluent case/data time-step artifact."""

    step: int
    case_path: Path
    data_path: Path


@dataclass(frozen=True)
class LegacyRenderer:
    """Optional reference to the existing Fluent polygon renderer."""

    function: Callable[..., None] | None
    name: str
    unavailable_reason: str | None = None


def discover_step_pairs(run_dir: str | Path) -> list[StepPair]:
    """Return strictly consecutive case/data pairs from ``run_dir/steps``."""

    run_dir = Path(run_dir).resolve()
    steps_dir = run_dir / "steps"
    if not steps_dir.is_dir():
        raise CampaignValidationError(f"steps directory is missing: {steps_dir}")

    by_kind: dict[str, dict[int, Path]] = {"cas": {}, "dat": {}}
    unexpected_h5: list[str] = []
    for path in sorted(steps_dir.iterdir()):
        if not path.is_file():
            continue
        match = STEP_FILE_RE.fullmatch(path.name)
        if match is None:
            if path.name.endswith((".cas.h5", ".dat.h5")):
                unexpected_h5.append(path.name)
            continue
        kind = match.group("kind")
        step = int(match.group("step"))
        if step in by_kind[kind]:
            raise CampaignValidationError(
                f"duplicate {kind} artifact for step {step}: {path}"
            )
        by_kind[kind][step] = path.resolve()

    if unexpected_h5:
        raise CampaignValidationError(
            "unexpected Fluent step filename(s): " + ", ".join(unexpected_h5)
        )

    case_steps = set(by_kind["cas"])
    data_steps = set(by_kind["dat"])
    if case_steps != data_steps:
        missing_cases = sorted(data_steps - case_steps)
        missing_data = sorted(case_steps - data_steps)
        raise CampaignValidationError(
            "case/data step mismatch: "
            f"missing_cases={missing_cases}, missing_data={missing_data}"
        )
    if not case_steps:
        raise CampaignValidationError(f"no paired Fluent steps found under {steps_dir}")

    ordered_steps = sorted(case_steps)
    expected_steps = list(range(1, ordered_steps[-1] + 1))
    if ordered_steps != expected_steps:
        raise CampaignValidationError(
            "step sequence must start at 1 and be consecutive: "
            f"observed={ordered_steps}, expected={expected_steps}"
        )

    return [
        StepPair(
            step=step,
            case_path=by_kind["cas"][step],
            data_path=by_kind["dat"][step],
        )
        for step in ordered_steps
    ]


def validate_phase_manifest(
    run_dir: str | Path,
    pairs: Sequence[StepPair],
    *,
    expected_steps: int,
    dt_s: float,
) -> dict[str, Any]:
    """Require a passed launcher manifest aligned with every paired artifact."""

    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if len(pairs) != expected_steps:
        raise CampaignValidationError(
            f"expected {expected_steps} paired steps, found {len(pairs)}"
        )

    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CampaignValidationError(f"phase manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignValidationError(
            f"phase manifest is unreadable: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise CampaignValidationError(f"phase manifest must be an object: {manifest_path}")
    if manifest.get("status") != "passed":
        raise CampaignValidationError(
            f"phase manifest status is not passed: {manifest.get('status')!r}"
        )
    for key in ("requested_steps", "completed_steps"):
        try:
            value = int(manifest[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignValidationError(
                f"phase manifest has invalid {key}: {manifest.get(key)!r}"
            ) from exc
        if value != expected_steps:
            raise CampaignValidationError(
                f"phase manifest {key}={value}, expected {expected_steps}"
            )

    manifest_steps = manifest.get("steps")
    if not isinstance(manifest_steps, list) or len(manifest_steps) != expected_steps:
        raise CampaignValidationError(
            "phase manifest step records do not match expected count: "
            f"observed={len(manifest_steps) if isinstance(manifest_steps, list) else None}, "
            f"expected={expected_steps}"
        )
    for pair, row in zip(pairs, manifest_steps):
        if not isinstance(row, dict) or int(row.get("step", -1)) != pair.step:
            raise CampaignValidationError(
                f"phase manifest step mismatch at paired step {pair.step}: {row!r}"
            )
        try:
            time_s = float(row["time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignValidationError(
                f"phase manifest time is invalid at step {pair.step}: {row.get('time_s')!r}"
            ) from exc
        expected_time_s = pair.step * dt_s
        if not math.isfinite(time_s) or not math.isclose(
            time_s,
            expected_time_s,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise CampaignValidationError(
                f"phase manifest time mismatch at step {pair.step}: "
                f"observed={time_s!r}, expected={expected_time_s!r}"
            )
        for key, expected_path in (
            ("case_path", pair.case_path),
            ("data_path", pair.data_path),
        ):
            raw_path = row.get(key)
            if not isinstance(raw_path, str) or Path(raw_path).resolve() != expected_path:
                raise CampaignValidationError(
                    f"phase manifest {key} mismatch at step {pair.step}: "
                    f"observed={raw_path!r}, expected={str(expected_path)!r}"
                )
    return manifest


def prepare_new_output_dir(output_dir: str | Path) -> Path:
    """Create a new output directory, refusing to reuse any existing path."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise CampaignValidationError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def default_output_dir(run_dir: str | Path) -> Path:
    """Return a timestamped, non-overwriting default output location."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(run_dir).resolve() / f"postprocess_{stamp}"


def field_statistics(label: str, values: np.ndarray) -> dict[str, float | int]:
    """Return finite scalar statistics for a non-empty numerical field."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise CampaignValidationError(f"{label} field is empty")
    if not np.all(np.isfinite(array)):
        nonfinite = int(np.count_nonzero(~np.isfinite(array)))
        raise CampaignValidationError(
            f"{label} field is not finite: nonfinite_count={nonfinite}"
        )
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p01": float(np.percentile(array, 1.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
    }


def read_structure_snapshot(
    case_path: str | Path,
    data_path: str | Path,
    *,
    target_x_m: float = DEFAULT_TARGET_X_M,
    target_y_m: float = DEFAULT_TARGET_Y_M,
    nearest_node_count: int = 4,
    nonzero_tolerance_m: float = DEFAULT_NONZERO_DISPLACEMENT_TOLERANCE_M,
) -> dict[str, Any]:
    """Read and validate Fluent intrinsic-structure displacement data."""

    if nearest_node_count <= 0:
        raise ValueError("nearest_node_count must be positive")
    if nonzero_tolerance_m < 0.0:
        raise ValueError("nonzero_tolerance_m must be non-negative")

    h5py = _require_h5py()
    case_path = Path(case_path)
    data_path = Path(data_path)
    with h5py.File(case_path, "r") as case_file, h5py.File(
        data_path, "r"
    ) as data_file:
        direct_svars = _required_dataset(
            data_file,
            "special/structure-direct-data/data",
            data_path,
        ).astype(np.int64)
        node_groups = _required_group(
            data_file,
            "special/structure-node-data/nodes",
            data_path,
        )
        node_group = _first_group(node_groups)
        node_group_name = node_group.name.rsplit("/", 1)[-1]
        coordinate_groups = _required_group(
            case_file,
            "meshes/1/nodes/coords",
            case_path,
        )
        if node_group_name not in coordinate_groups:
            raise CampaignValidationError(
                "structure node group has no matching coordinate section: "
                f"nodes/{node_group_name}, case={case_path}"
            )
        coordinate_dataset = coordinate_groups[node_group_name]
        if not hasattr(coordinate_dataset, "shape"):
            raise CampaignValidationError(
                "matching structure coordinate section is not a dataset: "
                f"{coordinate_dataset.name}"
            )
        coordinates = np.asarray(coordinate_dataset[()], dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise CampaignValidationError(
                f"invalid node coordinate shape in {case_path}: {coordinates.shape}"
            )
        _require_finite("structure node coordinates", coordinates)
        node_ids = np.asarray(node_group["elemids"][()], dtype=np.int64)
        widths = np.asarray(node_group["ndata"][()], dtype=np.int64)
        raw = np.asarray(node_group["data"][()], dtype=np.float64)
        if node_ids.size == 0:
            raise CampaignValidationError(f"structure node data is empty: {data_path}")
        if widths.size != node_ids.size or len(set(widths.tolist())) != 1:
            raise CampaignValidationError(
                f"variable or inconsistent structure row width in {data_path}"
            )
        width = int(widths[0])
        if width <= 6:
            raise CampaignValidationError(
                f"structure data needs displacement columns 0 and 6: width={width}"
            )
        if raw.size != node_ids.size * width:
            raise CampaignValidationError(
                f"structure data size mismatch in {data_path}: "
                f"raw={raw.size}, expected={node_ids.size * width}"
            )
        rows = raw.reshape(node_ids.size, width)
        _require_finite("structure node data", rows)
        if np.any(node_ids <= 0) or np.any(node_ids > coordinates.shape[0]):
            raise CampaignValidationError(
                f"structure node ids are outside case coordinate bounds: {data_path}"
            )

        displacement_x = rows[:, 0]
        displacement_y = rows[:, 6]
        displacement_norm = np.hypot(displacement_x, displacement_y)
        max_index = int(np.argmax(displacement_norm))
        max_displacement = float(displacement_norm[max_index])
        if max_displacement <= nonzero_tolerance_m:
            raise CampaignValidationError(
                "structure displacement is zero or below tolerance: "
                f"max={max_displacement:.17g} m, "
                f"tolerance={nonzero_tolerance_m:.17g} m, data={data_path}"
            )

        node_coordinates = coordinates[node_ids - 1, :2]
        target = np.array([target_x_m, target_y_m], dtype=np.float64)
        distances = np.linalg.norm(node_coordinates - target, axis=1)
        selection_count = min(nearest_node_count, node_ids.size)
        selected_indices = np.argsort(distances)[:selection_count]
        selected_x = displacement_x[selected_indices]
        selected_y = displacement_y[selected_indices]
        selected_norm = displacement_norm[selected_indices]
        mean_x = float(np.mean(selected_x))
        mean_y = float(np.mean(selected_y))

        return {
            "target_x_m": float(target_x_m),
            "target_y_m": float(target_y_m),
            "selected_node_ids": ";".join(
                str(int(node_ids[index])) for index in selected_indices
            ),
            "selected_node_count": int(selection_count),
            "selected_max_initial_distance_m": float(
                np.max(distances[selected_indices])
            ),
            "tip_displacement_x_m": mean_x,
            "tip_displacement_y_m": mean_y,
            "tip_displacement_norm_m": float(np.mean(selected_norm)),
            "tip_mean_vector_norm_m": float(math.hypot(mean_x, mean_y)),
            "max_displacement_m": max_displacement,
            "max_displacement_node_id": int(node_ids[max_index]),
            "structure_node_count": int(node_ids.size),
            "nonzero_displacement_node_count": int(
                np.count_nonzero(displacement_norm > nonzero_tolerance_m)
            ),
            "structure_column_count": width,
            "direct_svars": ";".join(str(int(value)) for value in direct_svars),
        }


def read_residual_snapshot(data_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read every stored residual sample from one Fluent data snapshot."""

    h5py = _require_h5py()
    data_path = Path(data_path)
    with h5py.File(data_path, "r") as data_file:
        root = data_file.get("results/residuals/phase-1")
        if root is None:
            raise CampaignValidationError(
                f"residual group is missing: {data_path}:results/residuals/phase-1"
            )
        output: dict[str, dict[str, Any]] = {}
        for equation in sorted(root.keys()):
            group = root[equation]
            if "iterations" not in group or "data" not in group:
                raise CampaignValidationError(
                    f"incomplete residual datasets for {equation}: {data_path}"
                )
            iterations = np.asarray(group["iterations"][()], dtype=np.float64).reshape(-1)
            values = np.asarray(group["data"][()], dtype=np.float64)
            if values.ndim == 1:
                values = values[:, np.newaxis]
            if values.ndim != 2 or values.shape[0] != iterations.size:
                raise CampaignValidationError(
                    f"residual shape mismatch for {equation} in {data_path}: "
                    f"iterations={iterations.shape}, values={values.shape}"
                )
            if iterations.size == 0:
                raise CampaignValidationError(
                    f"empty residual history for {equation}: {data_path}"
                )
            _require_finite(f"{equation} residual iterations", iterations)
            _require_finite(f"{equation} residual values", values)
            output[equation] = {
                "iterations": iterations,
                "values": values,
            }
        if not output:
            raise CampaignValidationError(f"residual group is empty: {data_path}")
        return output


def postprocess_campaign(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    dt_s: float = DEFAULT_DT_S,
    velocity_vmax_mps: float = DEFAULT_VELOCITY_VMAX_MPS,
    gif_duration_ms: int = DEFAULT_GIF_DURATION_MS,
    gif_max_width_px: int = DEFAULT_GIF_MAX_WIDTH_PX,
    target_x_m: float = DEFAULT_TARGET_X_M,
    target_y_m: float = DEFAULT_TARGET_Y_M,
    nearest_node_count: int = 4,
    nonzero_tolerance_m: float = DEFAULT_NONZERO_DISPLACEMENT_TOLERANCE_M,
    expected_steps: int = DEFAULT_EXPECTED_STEP_COUNT,
    prefer_existing_polygon_renderer: bool = True,
) -> dict[str, Any]:
    """Create a complete, non-overwriting offline post-process artifact set."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(velocity_vmax_mps) or velocity_vmax_mps <= 0.0:
        raise ValueError("velocity_vmax_mps must be finite and positive")
    if gif_duration_ms <= 0 or gif_max_width_px <= 0:
        raise ValueError("GIF duration and maximum width must be positive")

    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    pairs = discover_step_pairs(run_dir)
    phase_manifest = validate_phase_manifest(
        run_dir,
        pairs,
        expected_steps=expected_steps,
        dt_s=dt_s,
    )
    output_dir = prepare_new_output_dir(output_dir)
    histories_dir = output_dir / "histories"
    figures_dir = output_dir / "figures" / "velocity"
    fields_dir = output_dir / "fields"
    histories_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    fields_dir.mkdir(parents=True)

    legacy_renderer = (
        _load_existing_polygon_renderer(repo_root)
        if prefer_existing_polygon_renderer
        else LegacyRenderer(None, "offline_cell_center_scatter")
    )

    velocity_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    residual_summary_rows: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    input_pairs: list[dict[str, Any]] = []
    final_bundle: FluentFieldBundle | None = None

    try:
        for pair in pairs:
            _require_nonempty_file(pair.case_path)
            _require_nonempty_file(pair.data_path)
            time_s = pair.step * dt_s
            try:
                bundle = read_fluent_cell_fields(pair.case_path, pair.data_path)
            except Exception as exc:
                raise CampaignValidationError(
                    f"failed to read paired Fluent fields for step {pair.step}: {exc}"
                ) from exc
            _validate_bundle(bundle, pair)
            final_bundle = bundle

            u_stats = field_statistics("u velocity", bundle.u)
            v_stats = field_statistics("v velocity", bundle.v)
            speed_stats = field_statistics("velocity magnitude", bundle.speed)
            pressure_stats = field_statistics("pressure", bundle.p)
            frame_path = figures_dir / f"velocity_step_{pair.step:04d}.png"
            _render_velocity_frame(
                bundle,
                frame_path,
                step=pair.step,
                time_s=time_s,
                velocity_vmax_mps=velocity_vmax_mps,
                legacy_renderer=legacy_renderer,
            )
            frame_paths.append(frame_path)

            velocity_rows.append(
                {
                    "step": pair.step,
                    "time_s": time_s,
                    "cell_count": speed_stats["count"],
                    **_prefixed_statistics("u", u_stats),
                    **_prefixed_statistics("v", v_stats),
                    **_prefixed_statistics("speed", speed_stats),
                    "case_path": str(pair.case_path),
                    "data_path": str(pair.data_path),
                    "velocity_png": str(frame_path),
                }
            )
            pressure_rows.append(
                {
                    "step": pair.step,
                    "time_s": time_s,
                    **_prefixed_statistics("pressure", pressure_stats),
                    "pressure_range_pa": float(
                        pressure_stats["max"] - pressure_stats["min"]
                    ),
                    "case_path": str(pair.case_path),
                    "data_path": str(pair.data_path),
                }
            )

            structure = read_structure_snapshot(
                pair.case_path,
                pair.data_path,
                target_x_m=target_x_m,
                target_y_m=target_y_m,
                nearest_node_count=nearest_node_count,
                nonzero_tolerance_m=nonzero_tolerance_m,
            )
            structure_rows.append(
                {
                    "step": pair.step,
                    "time_s": time_s,
                    **structure,
                    "case_path": str(pair.case_path),
                    "data_path": str(pair.data_path),
                }
            )

            residuals = read_residual_snapshot(pair.data_path)
            for equation, snapshot in residuals.items():
                iterations = snapshot["iterations"]
                values = snapshot["values"]
                for index, iteration in enumerate(iterations):
                    residual_rows.append(
                        {
                            "snapshot_step": pair.step,
                            "snapshot_time_s": time_s,
                            "equation": equation,
                            "sample_index": index,
                            "iteration": float(iteration),
                            "values": tuple(float(value) for value in values[index]),
                            "data_path": str(pair.data_path),
                        }
                    )
                primary = values[:, 0]
                residual_summary_rows.append(
                    {
                        "step": pair.step,
                        "time_s": time_s,
                        "equation": equation,
                        "sample_count": int(iterations.size),
                        "first_iteration": float(iterations[0]),
                        "last_iteration": float(iterations[-1]),
                        "primary_initial": float(primary[0]),
                        "primary_final": float(primary[-1]),
                        "primary_min": float(np.min(primary)),
                        "primary_max": float(np.max(primary)),
                        "stored_value_column_count": int(values.shape[1]),
                        "data_path": str(pair.data_path),
                    }
                )

            input_pairs.append(
                {
                    "step": pair.step,
                    "case_path": str(pair.case_path),
                    "case_size_bytes": pair.case_path.stat().st_size,
                    "case_sha256": sha256_file(pair.case_path),
                    "data_path": str(pair.data_path),
                    "data_size_bytes": pair.data_path.stat().st_size,
                    "data_sha256": sha256_file(pair.data_path),
                }
            )

        if final_bundle is None:
            raise CampaignValidationError("campaign produced no final field bundle")

        velocity_csv = histories_dir / "velocity_history.csv"
        pressure_csv = histories_dir / "pressure_history.csv"
        structure_csv = histories_dir / "structure_displacement_history.csv"
        residual_csv = histories_dir / "residual_history.csv"
        residual_summary_csv = histories_dir / "residual_snapshot_summary.csv"
        _write_dict_csv(velocity_csv, velocity_rows)
        _write_dict_csv(pressure_csv, pressure_rows)
        _write_dict_csv(structure_csv, structure_rows)
        _write_residual_csv(residual_csv, residual_rows)
        _write_dict_csv(residual_summary_csv, residual_summary_rows)

        final_fields_npz = fields_dir / "final_fields.npz"
        _write_field_npz(final_fields_npz, final_bundle)
        input_manifest = output_dir / "input_manifest.json"
        _write_json(
            input_manifest,
            {
                "schema": "fluent_fine_fsi_input_pairs_v1",
                "run_dir": str(run_dir),
                "step_count": len(pairs),
                "pairs": input_pairs,
            },
        )

        velocity_gif = output_dir / "velocity_magnitude_fixed_0_28p1.gif"
        _build_gif(
            frame_paths,
            velocity_gif,
            duration_ms=gif_duration_ms,
            max_width_px=gif_max_width_px,
        )

        summary_path = output_dir / "summary.json"
        summary = {
            "schema": "fluent_fine_fsi_offline_postprocess_v1",
            "status": "complete",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "offline_only": True,
            "fluent_launched": False,
            "source_artifacts_modified": False,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "step_count": len(pairs),
            "expected_step_count": expected_steps,
            "first_step": pairs[0].step,
            "final_step": pairs[-1].step,
            "dt_s": dt_s,
            "final_time_s": pairs[-1].step * dt_s,
            "velocity_display_range_mps": [0.0, velocity_vmax_mps],
            "target_point_m": [target_x_m, target_y_m],
            "nearest_structure_node_count": nearest_node_count,
            "nonzero_displacement_tolerance_m": nonzero_tolerance_m,
            "all_structure_steps_nonzero": True,
            "phase_manifest_status": phase_manifest["status"],
            "renderer": legacy_renderer.name,
            "renderer_unavailable_reason": legacy_renderer.unavailable_reason,
            "residual_scope_note": (
                "residual_history.csv preserves every residual sample stored in "
                "each data snapshot; cumulative samples may therefore repeat across steps"
            ),
            "final_velocity": velocity_rows[-1],
            "final_pressure": pressure_rows[-1],
            "final_structure": structure_rows[-1],
            "peak_tip_displacement_m": max(
                float(row["tip_displacement_norm_m"]) for row in structure_rows
            ),
            "peak_solid_displacement_m": max(
                float(row["max_displacement_m"]) for row in structure_rows
            ),
            "outputs": {
                "velocity_history_csv": _relative_to(velocity_csv, output_dir),
                "pressure_history_csv": _relative_to(pressure_csv, output_dir),
                "structure_displacement_history_csv": _relative_to(
                    structure_csv, output_dir
                ),
                "residual_history_csv": _relative_to(residual_csv, output_dir),
                "residual_snapshot_summary_csv": _relative_to(
                    residual_summary_csv, output_dir
                ),
                "velocity_frames_dir": _relative_to(figures_dir, output_dir),
                "velocity_gif": _relative_to(velocity_gif, output_dir),
                "final_fields_npz": _relative_to(final_fields_npz, output_dir),
                "input_manifest": _relative_to(input_manifest, output_dir),
                "summary": _relative_to(summary_path, output_dir),
                "checksums": "CHECKSUMS.sha256",
            },
        }
        _write_json(summary_path, summary)
        write_checksums(output_dir)
        return summary
    except Exception as exc:
        _write_json(
            output_dir / "failure.json",
            {
                "schema": "fluent_fine_fsi_offline_postprocess_failure_v1",
                "status": "failed",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "fluent_launched": False,
            },
        )
        raise


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: str | Path) -> Path:
    """Write sorted SHA-256 checksums for every completed output artifact."""

    output_dir = Path(output_dir).resolve()
    checksum_path = output_dir / "CHECKSUMS.sha256"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _validate_bundle(bundle: FluentFieldBundle, pair: StepPair) -> None:
    expected = bundle.cell_ids.shape
    arrays = {
        "x": bundle.x,
        "y": bundle.y,
        "u": bundle.u,
        "v": bundle.v,
        "p": bundle.p,
        "speed": bundle.speed,
    }
    for name, values in arrays.items():
        if values.shape != expected:
            raise CampaignValidationError(
                f"step {pair.step} {name} shape {values.shape} != {expected}"
            )
        _require_finite(f"step {pair.step} {name}", values)
    recomputed_speed = np.hypot(bundle.u, bundle.v)
    if not np.allclose(bundle.speed, recomputed_speed, rtol=1.0e-12, atol=1.0e-14):
        raise CampaignValidationError(
            f"step {pair.step} stored speed is inconsistent with u/v"
        )


def _render_velocity_frame(
    bundle: FluentFieldBundle,
    output_path: Path,
    *,
    step: int,
    time_s: float,
    velocity_vmax_mps: float,
    legacy_renderer: LegacyRenderer,
) -> None:
    title = (
        "Official Fluent fine-grid FSI velocity magnitude | "
        f"step {step:04d}, t = {time_s:.6f} s"
    )
    if legacy_renderer.function is not None:
        legacy_renderer.function(
            bundle.case_path,
            bundle.data_path,
            output_path,
            fixed_scale=True,
            title=title,
        )
    else:
        _render_cell_center_velocity(
            bundle,
            output_path,
            title=title,
            velocity_vmax_mps=velocity_vmax_mps,
        )
    _require_nonempty_file(output_path)


def _render_cell_center_velocity(
    bundle: FluentFieldBundle,
    output_path: Path,
    *,
    title: str,
    velocity_vmax_mps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    marker_size = max(0.1, min(4.0, 20_000.0 / bundle.speed.size))
    cloud = axis.scatter(
        bundle.x,
        bundle.y,
        c=bundle.speed,
        s=marker_size,
        cmap="turbo",
        vmin=0.0,
        vmax=velocity_vmax_mps,
        linewidths=0,
        rasterized=True,
    )
    x_span = float(np.max(bundle.x) - np.min(bundle.x))
    y_span = float(np.max(bundle.y) - np.min(bundle.y))
    axis.set_xlim(float(np.min(bundle.x) - 0.02 * x_span), float(np.max(bundle.x) + 0.02 * x_span))
    axis.set_ylim(float(np.min(bundle.y) - 0.02 * y_span), float(np.max(bundle.y) + 0.02 * y_span))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title)
    colorbar = figure.colorbar(cloud, ax=axis, shrink=0.9)
    colorbar.set_label("Velocity magnitude (m/s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _load_existing_polygon_renderer(repo_root: Path) -> LegacyRenderer:
    helper_path = (
        repo_root
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "official_fluent_fine_mesh_steady_2026-07-01"
        / "scripts"
        / "run_official_fluent_fine_steady.py"
    )
    if not helper_path.is_file():
        return LegacyRenderer(
            None,
            "offline_cell_center_scatter",
            f"existing polygon renderer not found: {helper_path}",
        )
    try:
        spec = importlib.util.spec_from_file_location(
            "_fine_fsi_existing_render_helpers",
            helper_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create import spec for {helper_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        return LegacyRenderer(
            None,
            "offline_cell_center_scatter",
            f"existing polygon renderer unavailable: {type(exc).__name__}: {exc}",
        )

    def strict_append_event(label: str, **payload: object) -> None:
        if label == "render_polygon_count_adjusted":
            raise CampaignValidationError(
                "existing renderer detected polygon/field count mismatch: "
                + json.dumps(payload, sort_keys=True)
            )

    module.append_event = strict_append_event
    renderer = getattr(module, "render_velocity", None)
    if not callable(renderer):
        return LegacyRenderer(
            None,
            "offline_cell_center_scatter",
            f"render_velocity is unavailable in {helper_path}",
        )
    return LegacyRenderer(renderer, "existing_fluent_cell_polygon_renderer")


def _build_gif(
    frame_paths: Sequence[Path],
    output_path: Path,
    *,
    duration_ms: int,
    max_width_px: int,
) -> None:
    if not frame_paths:
        raise CampaignValidationError("cannot build GIF without velocity frames")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to build the velocity GIF") from exc

    rgb_frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frame = image.convert("RGB")
            if frame.width > max_width_px:
                height = max(1, round(frame.height * max_width_px / frame.width))
                frame = frame.resize(
                    (max_width_px, height),
                    Image.Resampling.LANCZOS,
                )
            rgb_frames.append(frame.copy())
    expected_size = rgb_frames[0].size
    if any(frame.size != expected_size for frame in rgb_frames):
        raise CampaignValidationError("velocity GIF frame dimensions are inconsistent")

    palette_frame = rgb_frames[0].convert(
        "P",
        palette=Image.Palette.ADAPTIVE,
        colors=256,
    )
    quantized = [palette_frame]
    quantized.extend(
        frame.quantize(palette=palette_frame, dither=Image.Dither.NONE)
        for frame in rgb_frames[1:]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    for frame in rgb_frames:
        frame.close()
    for frame in quantized:
        frame.close()
    _require_nonempty_file(output_path)


def _write_residual_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise CampaignValidationError("residual CSV cannot be empty")
    value_width = max(len(row["values"]) for row in rows)
    value_columns = [f"value_col{index}" for index in range(value_width)]
    fieldnames = [
        "snapshot_step",
        "snapshot_time_s",
        "equation",
        "sample_index",
        "iteration",
        *value_columns,
        "data_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fieldnames if key in row}
            for index, value in enumerate(row["values"]):
                output[f"value_col{index}"] = value
            writer.writerow(output)


def _write_dict_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise CampaignValidationError(f"CSV cannot be empty: {path}")
    fieldnames = list(rows[0].keys())
    expected = set(fieldnames)
    for row in rows[1:]:
        if set(row) != expected:
            raise CampaignValidationError(
                f"inconsistent CSV schema for {path}: {sorted(set(row) ^ expected)}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prefixed_statistics(
    prefix: str,
    statistics: dict[str, float | int],
) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in statistics.items()}


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file():
        raise CampaignValidationError(f"required file is missing: {path}")
    if path.stat().st_size <= 0:
        raise CampaignValidationError(f"required file is empty: {path}")


def _require_finite(label: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        count = int(np.count_nonzero(~np.isfinite(values)))
        raise CampaignValidationError(
            f"{label} is not finite: nonfinite_count={count}"
        )


def _required_group(handle: Any, path: str, source_path: Path) -> Any:
    group = handle.get(path)
    if group is None:
        raise CampaignValidationError(f"required HDF5 group missing: {source_path}:{path}")
    return group


def _required_dataset(handle: Any, path: str, source_path: Path) -> np.ndarray:
    dataset = handle.get(path)
    if dataset is None:
        raise CampaignValidationError(
            f"required HDF5 dataset missing: {source_path}:{path}"
        )
    return np.asarray(dataset[()])


def _first_dataset(group: Any) -> np.ndarray:
    for key in sorted(group.keys(), key=_numeric_sort_key):
        child = group[key]
        if hasattr(child, "shape"):
            return np.asarray(child[()])
    raise CampaignValidationError(f"HDF5 group has no dataset: {group.name}")


def _first_group(group: Any) -> Any:
    for key in sorted(group.keys(), key=_numeric_sort_key):
        child = group[key]
        if hasattr(child, "keys"):
            return child
    raise CampaignValidationError(f"HDF5 group has no child group: {group.name}")


def _numeric_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for offline Fluent post-processing") from exc
    return h5py


def _relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "CampaignValidationError",
    "StepPair",
    "default_output_dir",
    "discover_step_pairs",
    "field_statistics",
    "postprocess_campaign",
    "prepare_new_output_dir",
    "read_residual_snapshot",
    "read_structure_snapshot",
    "sha256_file",
    "validate_phase_manifest",
    "write_checksums",
]
