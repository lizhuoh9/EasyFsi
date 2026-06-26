"""Sweep ANSYS vertical-flap marker offsets and pressure-probe origins.

This diagnostic reuses the archived shared preflow snapshot and samples marker
tractions only. It does not advance fluid, structure, or a coupled FSI loop.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_formulation_validation_matrix as base_matrix,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_snapshot_resampling_matrix as snapshot_resampling,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
OUTPUT_DIR = CASE_ROOT / "traction_probe_offset_decoupling_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_probe_offset_decoupling_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_probe_offset_decoupling_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_probe_offset_decoupling_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_probe_offset_decoupling_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH

SCOPE_LIMIT = (
    "shared snapshot sampling-only traction probe offset decoupling diagnostic "
    "on archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

BASELINE_SCENARIO = "fixed_marker0p51_probe0p51"
CANDIDATE_STATUS = "probe_offset_decoupling_diagnostic_only"
SCENARIOS = (
    ("fixed_marker0p51_probe0p00", 0.51, 0.00, "fixed_marker"),
    ("fixed_marker0p51_probe0p25", 0.51, 0.25, "fixed_marker"),
    (BASELINE_SCENARIO, 0.51, 0.51, "fixed_marker"),
    ("fixed_marker0p51_probe1p00", 0.51, 1.00, "fixed_marker"),
    ("fixed_probe0p51_marker0p00", 0.00, 0.51, "fixed_probe"),
    ("fixed_probe0p51_marker0p25", 0.25, 0.51, "fixed_probe"),
    ("fixed_probe0p51_marker0p51", 0.51, 0.51, "fixed_probe"),
    ("fixed_probe0p51_marker1p00", 1.00, 0.51, "fixed_probe"),
)

MATRIX_COLUMNS = [
    "scenario",
    "run_status",
    "formulation_status",
    "scenario_group",
    "marker_face_offset_cells",
    "pressure_probe_origin_offset_cells",
    "total_force_z_N",
    "force_ratio_to_baseline",
    "marker_geometry_sha256",
    "pressure_probe_origin_sha256",
    "marker_diagnostics_json",
    "flow_snapshot_sha256",
    "flow_snapshot_source_commit",
    "scope_limit",
]

MARKER_REQUIRED_FIELDS = [
    *snapshot_resampling.MARKER_REQUIRED_FIELDS,
    "pressure_probe_origin_m",
    "pressure_probe_origin_source",
    "pressure_probe_origin_explicit",
]


class ProbeOffsetDecouplingError(RuntimeError):
    """Raised when the probe-offset decoupling diagnostic cannot run."""


def _repo_relative(path: Path) -> str:
    return snapshot_resampling._repo_relative(path)


def _json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    return snapshot_resampling._json_dumps(payload, indent=indent)


def _write_json(path: Path, payload: Any) -> None:
    snapshot_resampling._write_json(path, payload)


def _sha256_file(path: Path) -> str:
    return snapshot_resampling._sha256_file(path)


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return snapshot_resampling._sha256_payload(payload)


def _source_config(marker_offset: float, probe_offset: float) -> Any:
    return base_matrix._source_config(
        traction_marker_layout="dual_physical_faces",
        traction_pressure_sampling_mode="two_sided_pressure_jump",
        traction_include_viscous=False,
        traction_marker_face_offset_cells=float(marker_offset),
        traction_pressure_probe_origin_mode="physical_face_offset",
        traction_pressure_probe_origin_offset_cells=float(probe_offset),
    )


def _scenario_specs() -> list[tuple[str, Any, str]]:
    return [
        (name, _source_config(marker_offset, probe_offset), group)
        for name, marker_offset, probe_offset, group in SCENARIOS
    ]


def _marker_geometry_identity(markers: Any) -> dict[str, Any]:
    marker_count = int(markers.marker_count)
    return {
        "positions_m": markers.x_gamma_m.to_numpy()[:marker_count].tolist(),
        "normals": markers.n_gamma.to_numpy()[:marker_count].tolist(),
        "areas_m2": markers.A_gamma_m2.to_numpy()[:marker_count].tolist(),
        "region_ids": markers.region_id.to_numpy()[:marker_count].tolist(),
    }


def _pressure_probe_origin_identity(markers: Any) -> dict[str, Any]:
    marker_count = int(markers.marker_count)
    origins = []
    explicit = markers.pressure_probe_origin_explicit.to_numpy()[:marker_count]
    stored_origins = markers.pressure_probe_origin_m.to_numpy()[:marker_count]
    positions = markers.x_gamma_m.to_numpy()[:marker_count]
    for marker in range(marker_count):
        origin = stored_origins[marker] if int(explicit[marker]) else positions[marker]
        origins.append([float(value) for value in origin])
    return {"pressure_probe_origins_m": origins}


def _marker_required_subset(marker: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in marker]
    if missing:
        raise ProbeOffsetDecouplingError(
            "Marker diagnostic is missing required fields: " + ", ".join(missing)
        )
    return {field: marker[field] for field in MARKER_REQUIRED_FIELDS}


def _write_marker_diagnostics(
    *,
    scenario: str,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    marker_geometry: Mapping[str, Any],
    marker_geometry_sha256: str,
    pressure_probe_origin: Mapping[str, Any],
    pressure_probe_origin_sha256: str,
) -> Path:
    marker_subset = [
        _marker_required_subset(marker) for marker in markers.stress_marker_diagnostics()
    ]
    payload = {
        "schema_version": 1,
        "case": CASE_NAME,
        "scenario": scenario,
        "purpose": "shared_flow_snapshot_traction_probe_offset_decoupling_marker_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_geometry": marker_geometry,
        "pressure_probe_origin": pressure_probe_origin,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": marker_subset,
        "face_diagnostics": markers.stress_face_diagnostics(
            primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
            secondary_region_id=solid_mpm_fsi_runner.SECONDARY_REGION_ID,
        ),
        "force_report": solid_mpm_fsi_runner._marker_force_report_fields(
            force_report
        ),
        "stress_report": solid_mpm_fsi_runner._stress_sampling_report_fields(
            stress_report
        ),
    }
    path = MARKER_DIAGNOSTICS_DIR / f"{scenario}_markers.json"
    _write_json(path, payload)
    return path


def _complete_row(
    *,
    scenario: str,
    scenario_group: str,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    elapsed_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_geometry = _marker_geometry_identity(markers)
    marker_geometry_sha256 = _sha256_payload(marker_geometry)
    pressure_probe_origin = _pressure_probe_origin_identity(markers)
    pressure_probe_origin_sha256 = _sha256_payload(pressure_probe_origin)
    marker_path = _write_marker_diagnostics(
        scenario=scenario,
        config=config,
        markers=markers,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        marker_geometry=marker_geometry,
        marker_geometry_sha256=marker_geometry_sha256,
        pressure_probe_origin=pressure_probe_origin,
        pressure_probe_origin_sha256=pressure_probe_origin_sha256,
    )
    force_fields = solid_mpm_fsi_runner._marker_force_report_fields(force_report)
    stress_fields = solid_mpm_fsi_runner._stress_sampling_report_fields(stress_report)
    face_fields = solid_mpm_fsi_runner._marker_traction_report_fields(markers)
    row = {
        "scenario": scenario,
        "run_status": "completed",
        "formulation_status": "completed",
        "scenario_group": scenario_group,
        "worker_mode": "shared_snapshot_probe_offset_decoupling",
        "worker_elapsed_s": elapsed_s,
        "scope_limit": SCOPE_LIMIT,
        "solid_advanced": False,
        "feedback_applied": False,
        "marker_layout": config.traction_marker_layout,
        "pressure_sampling_mode": config.traction_pressure_sampling_mode,
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_mode": config.traction_pressure_probe_origin_mode,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_diagnostics_json": _repo_relative(marker_path),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
    }
    row.update(force_fields)
    row.update(stress_fields)
    row.update(face_fields)
    row["total_force_z_N"] = force_fields.get("marker_force_z_N", "")
    history_row = {
        "step": 0,
        "flow_phase": "shared_snapshot_probe_offset_decoupling",
        "scenario": scenario,
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
    }
    history_row.update(force_fields)
    history_row.update(stress_fields)
    history_row.update(face_fields)
    return row, history_row


def _resample_scenario(
    *,
    scenario: str,
    scenario_group: str,
    config: Any,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        raise ProbeOffsetDecouplingError(f"{scenario} unsupported: {reason}")
    started = time.perf_counter()
    markers = solid_mpm_fsi_runner._build_markers(config, runtime)
    stress_report = solid_mpm_fsi_runner._sample_stress_to_marker_forces(
        markers,
        fluid,
        config,
    )
    force_report = markers.aggregate_region_forces(
        primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
        secondary_region_id=solid_mpm_fsi_runner.SECONDARY_REGION_ID,
    )
    elapsed_s = time.perf_counter() - started
    return _complete_row(
        scenario=scenario,
        scenario_group=scenario_group,
        config=config,
        markers=markers,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        elapsed_s=elapsed_s,
    )


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _ratio_span(values: Sequence[float]) -> dict[str, float | str]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"min": "", "max": "", "relative_span": ""}
    minimum = min(finite)
    maximum = max(finite)
    if minimum == 0.0:
        relative_span: float | str = ""
    else:
        relative_span = (maximum - minimum) / abs(minimum)
    return {"min": minimum, "max": maximum, "relative_span": relative_span}


def _apply_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["scenario"]: row for row in rows}
    baseline_force = _float_or_none(by_name[BASELINE_SCENARIO]["total_force_z_N"])
    fixed_probe_baseline = _float_or_none(
        by_name["fixed_probe0p51_marker0p51"]["total_force_z_N"]
    )
    if baseline_force is None or baseline_force == 0.0:
        raise ProbeOffsetDecouplingError("baseline force is unavailable")
    if fixed_probe_baseline is None or fixed_probe_baseline == 0.0:
        raise ProbeOffsetDecouplingError("fixed-probe baseline force is unavailable")

    fixed_marker_ratios: list[float] = []
    fixed_probe_ratios: list[float] = []
    for row in rows:
        force = _float_or_none(row.get("total_force_z_N"))
        if force is None:
            row["force_ratio_to_baseline"] = ""
            continue
        row["force_ratio_to_baseline"] = force / baseline_force
        if row["scenario_group"] == "fixed_marker":
            ratio = force / baseline_force
            row["force_ratio_to_group_baseline"] = ratio
            fixed_marker_ratios.append(ratio)
        elif row["scenario_group"] == "fixed_probe":
            ratio = force / fixed_probe_baseline
            row["force_ratio_to_group_baseline"] = ratio
            fixed_probe_ratios.append(ratio)
    return {
        "fixed_marker_probe_origin_ratio_span": _ratio_span(fixed_marker_ratios),
        "fixed_probe_marker_ratio_span": _ratio_span(fixed_probe_ratios),
    }


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    values = {
        str(row.get("flow_snapshot_sha256", ""))
        for row in rows
        if row.get("run_status") == "completed"
    }
    if values == {sha}:
        return "shared_snapshot_sha256_identical_completed_rows"
    return "shared_snapshot_sha256_mismatch"


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_summary = _apply_ratios(rows)
    expected_sha = str(manifest.get("field_sha256", ""))
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_traction_probe_offset_decoupling_matrix",
        "scope_limit": SCOPE_LIMIT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": _repo_relative(Path(__file__).resolve()),
        "input_shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "input_shared_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": expected_sha,
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
        "flow_snapshot_identity_status": _shared_snapshot_identity_status(
            rows,
            expected_sha,
        ),
        "candidate_status": CANDIDATE_STATUS,
        "reference_formulation_candidate": None,
        "candidate_blockers": [
            {
                "blocker": "reference_selection_deferred",
                "detail": "Probe offset decoupling is diagnostic-only evidence.",
            },
            {
                "blocker": "dual_face_one_sided_unsupported",
                "detail": "Per-face one-sided pressure support is not implemented yet.",
            },
            {
                "blocker": "probe_offset_decoupling_diagnostic_only",
                "detail": "Rows isolate marker/probe sensitivity without selecting a formulation.",
            },
            {
                "blocker": "sampling_only_no_coupled_fsi",
                "detail": "Rows reuse one flow snapshot and do not advance coupled FSI.",
            },
        ],
        "baseline_scenario": BASELINE_SCENARIO,
        "completed_formulation_count": len(rows),
        "scenario_count": len(rows),
        "scenarios": [row["scenario"] for row in rows],
        "fixed_marker_scenarios": [
            row["scenario"] for row in rows if row["scenario_group"] == "fixed_marker"
        ],
        "fixed_probe_scenarios": [
            row["scenario"] for row in rows if row["scenario_group"] == "fixed_probe"
        ],
        "histories": histories,
        "rows": rows,
        **ratio_summary,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}g}"


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# ANSYS vertical-flap traction probe offset decoupling",
        "",
        "## Scope",
        "",
        (
            "This artifact reuses one archived shared preflow snapshot and re-runs "
            "only marker traction sampling. It does not advance the flow, the "
            "structure, or a coupled FSI loop, and it does not claim Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        "- reference_formulation_candidate: none",
        f"- candidate_status: `{payload.get('candidate_status', '')}`",
        "- candidate_blockers:",
    ]
    for blocker in payload.get("candidate_blockers", []):
        lines.append(f"  - {blocker.get('blocker', '')}")
    lines.extend(
        [
            "",
            "## Shared snapshot",
            "",
            f"- Manifest: `{payload.get('input_shared_snapshot_manifest', '')}`",
            f"- Fields: `{payload.get('input_shared_snapshot_npz', '')}`",
            f"- Source commit: `{payload.get('flow_snapshot_source_commit', '')}`",
            f"- Field SHA-256: `{payload.get('flow_snapshot_sha256', '')}`",
            "",
            "## Ratio spans",
            "",
            "- Fixed marker, swept pressure-probe origin relative span: "
            f"{_fmt(payload['fixed_marker_probe_origin_ratio_span']['relative_span'])}",
            "- Fixed pressure-probe origin, swept marker relative span: "
            f"{_fmt(payload['fixed_probe_marker_ratio_span']['relative_span'])}",
            "",
            "## Scenarios",
            "",
            "| scenario | group | marker offset | probe-origin offset | ratio |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['scenario_group']} | "
            f"{row['marker_face_offset_cells']} | "
            f"{row['pressure_probe_origin_offset_cells']} | "
            f"{_fmt(row.get('force_ratio_to_group_baseline'))} |"
        )
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "- Does not select a reference formulation.",
            "",
            "## Next step",
            "",
            (
                "Use these fixed-marker and fixed-probe sweeps to decide whether "
                "the offset pathology is dominated by pressure-probe ladder origin, "
                "force marker geometry, or both before implementing per-face "
                "one-sided pressure support."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_PATH.name
    )
    lines = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}")
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS_PATH):
        if path.exists():
            path.unlink()
    for path in MARKER_DIAGNOSTICS_DIR.glob("*.json"):
        path.unlink()


def run() -> dict[str, Any]:
    _prepare_output_dir()
    manifest = snapshot_resampling._load_manifest()
    fields = snapshot_resampling._load_snapshot_fields()
    baseline_config = _source_config(0.51, 0.51)
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    for scenario, config, scenario_group in _scenario_specs():
        row, history = _resample_scenario(
            scenario=scenario,
            scenario_group=scenario_group,
            config=config,
            fluid=fluid,
            runtime=runtime,
            manifest=manifest,
        )
        rows.append(row)
        histories[scenario] = history

    payload = _payload(rows, histories, manifest)
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_traction_probe_offset_decoupling_history",
            "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": manifest.get("field_sha256", ""),
            "histories": histories,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[traction_probe_offset_decoupling] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_probe_offset_decoupling] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to "
        f"{OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
