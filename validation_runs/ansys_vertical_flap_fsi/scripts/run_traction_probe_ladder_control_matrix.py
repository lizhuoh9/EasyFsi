"""Evaluate ANSYS vertical-flap pressure-probe ladder controls.

This diagnostic reuses the archived shared preflow snapshot and samples marker
tractions only. It does not advance fluid, structure, or a coupled FSI loop.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_probe_ladder_stability_matrix as ladder_stability,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_probe_offset_decoupling_matrix as offset_decoupling,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_snapshot_resampling_matrix as snapshot_resampling,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
OUTPUT_DIR = CASE_ROOT / "traction_probe_ladder_control_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_probe_ladder_control_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_probe_ladder_control_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_probe_ladder_control_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_probe_ladder_control_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH

SCOPE_LIMIT = (
    "shared snapshot sampling-only pressure probe ladder control diagnostic "
    "on archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

MARKER_FACE_OFFSET_CELLS = 0.51
BASELINE_PROBE_OFFSET = 0.51
STABLE_FORCE_RATIO_SPAN_MAX = 0.10
TRACTION_DECOMPOSITION_RESIDUAL_MAX = 1.0e-8
LADDER_MODE = "current_normal_cell_ladder"

PROBE_ORIGIN_OFFSETS = (
    ("probe0p51", 0.51),
    ("probe0p625", 0.625),
    ("probe1p00", 1.0),
)

LADDER_STRATEGIES = (
    {
        "strategy": "current_control_baseline",
        "pressure_probe_start_offset_cells": None,
        "pressure_probe_ladder_spacing_cells": 0.5,
        "pressure_probe_ladder_rung_count": 5,
    },
    {
        "strategy": "origin0p51_start1p00_spacing0p50",
        "pressure_probe_start_offset_cells": 1.0,
        "pressure_probe_ladder_spacing_cells": 0.5,
        "pressure_probe_ladder_rung_count": 5,
    },
    {
        "strategy": "origin0p51_start0p75_spacing0p50",
        "pressure_probe_start_offset_cells": 0.75,
        "pressure_probe_ladder_spacing_cells": 0.5,
        "pressure_probe_ladder_rung_count": 5,
    },
    {
        "strategy": "origin0p51_start0p625_spacing0p25",
        "pressure_probe_start_offset_cells": 0.625,
        "pressure_probe_ladder_spacing_cells": 0.25,
        "pressure_probe_ladder_rung_count": 5,
    },
    {
        "strategy": "origin0p51_start0p51_spacing0p25",
        "pressure_probe_start_offset_cells": 0.51,
        "pressure_probe_ladder_spacing_cells": 0.25,
        "pressure_probe_ladder_rung_count": 5,
    },
)

DEFERRED_STRATEGIES = (
    {
        "strategy": "origin0p51_symmetric_cell_pair_policy",
        "status": "report_only_deferred",
        "reason": "Symmetric cell-pair selection requires a separate implementation and artifact proof.",
    },
)

MATRIX_COLUMNS = [
    "scenario",
    "strategy",
    "run_status",
    "formulation_status",
    "probe_origin_offset_cells",
    "marker_face_offset_cells",
    "pressure_probe_start_offset_cells",
    "pressure_probe_ladder_spacing_cells",
    "pressure_probe_ladder_rung_count",
    "total_force_z_N",
    "force_ratio_to_strategy_baseline",
    "primary_face_mean_pressure_jump_pa",
    "secondary_face_mean_pressure_jump_pa",
    "primary_face_pressure_complete_marker_count",
    "secondary_face_pressure_complete_marker_count",
    "primary_face_invalid_marker_count",
    "secondary_face_invalid_marker_count",
    "max_face_traction_decomposition_residual_pa",
    "marker_geometry_sha256",
    "pressure_probe_origin_sha256",
    "marker_diagnostics_json",
    "flow_snapshot_sha256",
    "flow_snapshot_source_commit",
    "scope_limit",
]

MARKER_REQUIRED_FIELDS = ladder_stability.MARKER_REQUIRED_FIELDS
PRIMARY_REGION_ID = solid_mpm_fsi_runner.PRIMARY_REGION_ID
SECONDARY_REGION_ID = solid_mpm_fsi_runner.SECONDARY_REGION_ID


class ProbeLadderControlError(RuntimeError):
    """Raised when the pressure-probe ladder control diagnostic cannot run."""


def _repo_relative(path: Path) -> str:
    return snapshot_resampling._repo_relative(path)


def _write_json(path: Path, payload: Any) -> None:
    snapshot_resampling._write_json(path, payload)


def _sha256_file(path: Path) -> str:
    return snapshot_resampling._sha256_file(path)


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return snapshot_resampling._sha256_payload(payload)


def _json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return snapshot_resampling._json_dumps(payload, indent=indent)


def _float_or_none(value: Any) -> float | None:
    return ladder_stability._float_or_none(value)


def _source_config(strategy: Mapping[str, Any], probe_offset: float) -> Any:
    base = offset_decoupling._source_config(
        MARKER_FACE_OFFSET_CELLS,
        float(probe_offset),
    )
    return replace(
        base,
        traction_pressure_probe_start_offset_cells=strategy[
            "pressure_probe_start_offset_cells"
        ],
        traction_pressure_probe_ladder_spacing_cells=float(
            strategy["pressure_probe_ladder_spacing_cells"]
        ),
        traction_pressure_probe_ladder_rung_count=int(
            strategy["pressure_probe_ladder_rung_count"]
        ),
        traction_pressure_probe_ladder_mode=LADDER_MODE,
    )


def _scenario_specs() -> list[tuple[str, str, Any, float, Mapping[str, Any]]]:
    specs = []
    for strategy in LADDER_STRATEGIES:
        strategy_name = str(strategy["strategy"])
        for probe_label, probe_offset in PROBE_ORIGIN_OFFSETS:
            scenario = f"{strategy_name}_{probe_label}"
            specs.append(
                (
                    scenario,
                    strategy_name,
                    _source_config(strategy, probe_offset),
                    float(probe_offset),
                    strategy,
                )
            )
    return specs


def _marker_required_subset(marker: Mapping[str, Any]) -> dict[str, Any]:
    return ladder_stability._marker_required_subset(marker)


def _max_face_residual(fields: Mapping[str, Any]) -> float | str:
    values = []
    for key in (
        "primary_face_traction_decomposition_max_abs_residual_pa",
        "secondary_face_traction_decomposition_max_abs_residual_pa",
    ):
        value = _float_or_none(fields.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return ""
    return max(values)


def _write_marker_diagnostics(
    *,
    scenario: str,
    strategy_name: str,
    config: Any,
    markers: Any,
    marker_subset: Sequence[Mapping[str, Any]],
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    marker_geometry: Mapping[str, Any],
    marker_geometry_sha256: str,
    pressure_probe_origin: Mapping[str, Any],
    pressure_probe_origin_sha256: str,
    transition_fields: Mapping[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "case": CASE_NAME,
        "scenario": scenario,
        "strategy": strategy_name,
        "purpose": "shared_flow_snapshot_traction_probe_ladder_control_marker_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "pressure_probe_start_offset_cells": (
            config.traction_pressure_probe_start_offset_cells
        ),
        "pressure_probe_ladder_spacing_cells": (
            config.traction_pressure_probe_ladder_spacing_cells
        ),
        "pressure_probe_ladder_rung_count": (
            config.traction_pressure_probe_ladder_rung_count
        ),
        "pressure_probe_ladder_mode": config.traction_pressure_probe_ladder_mode,
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_geometry": marker_geometry,
        "pressure_probe_origin": pressure_probe_origin,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": list(marker_subset),
        "transition_fields": transition_fields,
        "face_diagnostics": markers.stress_face_diagnostics(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
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
    strategy_name: str,
    probe_offset: float,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    elapsed_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_subset = [
        _marker_required_subset(marker) for marker in markers.stress_marker_diagnostics()
    ]
    marker_geometry = ladder_stability._marker_geometry_identity(markers)
    marker_geometry_sha256 = _sha256_payload(marker_geometry)
    pressure_probe_origin = ladder_stability._pressure_probe_origin_identity(markers)
    pressure_probe_origin_sha256 = _sha256_payload(pressure_probe_origin)
    transition_fields = ladder_stability._transition_face_fields(marker_subset)
    marker_path = _write_marker_diagnostics(
        scenario=scenario,
        strategy_name=strategy_name,
        config=config,
        markers=markers,
        marker_subset=marker_subset,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        marker_geometry=marker_geometry,
        marker_geometry_sha256=marker_geometry_sha256,
        pressure_probe_origin=pressure_probe_origin,
        pressure_probe_origin_sha256=pressure_probe_origin_sha256,
        transition_fields=transition_fields,
    )
    force_fields = solid_mpm_fsi_runner._marker_force_report_fields(force_report)
    stress_fields = solid_mpm_fsi_runner._stress_sampling_report_fields(stress_report)
    traction_fields = solid_mpm_fsi_runner._marker_traction_report_fields(markers)
    row = {
        "scenario": scenario,
        "strategy": strategy_name,
        "run_status": "completed",
        "formulation_status": "completed",
        "worker_mode": "shared_snapshot_probe_ladder_control",
        "worker_elapsed_s": elapsed_s,
        "scope_limit": SCOPE_LIMIT,
        "solid_advanced": False,
        "feedback_applied": False,
        "marker_layout": config.traction_marker_layout,
        "pressure_sampling_mode": config.traction_pressure_sampling_mode,
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_mode": config.traction_pressure_probe_origin_mode,
        "probe_origin_offset_cells": probe_offset,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "pressure_probe_start_offset_cells": (
            ""
            if config.traction_pressure_probe_start_offset_cells is None
            else config.traction_pressure_probe_start_offset_cells
        ),
        "pressure_probe_ladder_spacing_cells": (
            config.traction_pressure_probe_ladder_spacing_cells
        ),
        "pressure_probe_ladder_rung_count": (
            config.traction_pressure_probe_ladder_rung_count
        ),
        "pressure_probe_ladder_mode": config.traction_pressure_probe_ladder_mode,
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_diagnostics_json": _repo_relative(marker_path),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
    }
    row.update(force_fields)
    row.update(stress_fields)
    row.update(traction_fields)
    row.update(ladder_stability._row_face_fields(marker_subset))
    row["total_force_z_N"] = force_fields.get("marker_force_z_N", "")
    row["max_face_traction_decomposition_residual_pa"] = _max_face_residual(row)

    history_row = {
        "step": 0,
        "flow_phase": "shared_snapshot_probe_ladder_control",
        "scenario": scenario,
        "strategy": strategy_name,
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "probe_origin_offset_cells": probe_offset,
        "pressure_probe_start_offset_cells": row["pressure_probe_start_offset_cells"],
        "pressure_probe_ladder_spacing_cells": (
            config.traction_pressure_probe_ladder_spacing_cells
        ),
        "pressure_probe_ladder_rung_count": (
            config.traction_pressure_probe_ladder_rung_count
        ),
    }
    history_row.update(force_fields)
    history_row.update(stress_fields)
    history_row.update(traction_fields)
    return row, history_row


def _sample_scenario(
    *,
    scenario: str,
    strategy_name: str,
    probe_offset: float,
    config: Any,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        raise ProbeLadderControlError(f"{scenario} unsupported: {reason}")
    started = time.perf_counter()
    markers = solid_mpm_fsi_runner._build_markers(config, runtime)
    stress_report = solid_mpm_fsi_runner._sample_stress_to_marker_forces(
        markers,
        fluid,
        config,
    )
    force_report = markers.aggregate_region_forces(
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
    )
    elapsed_s = time.perf_counter() - started
    return _complete_row(
        scenario=scenario,
        strategy_name=strategy_name,
        probe_offset=probe_offset,
        config=config,
        markers=markers,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        elapsed_s=elapsed_s,
    )


def _ratio_span(values: Sequence[float]) -> dict[str, float | str]:
    return ladder_stability._ratio_span(values)


def _apply_strategy_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_summaries: dict[str, dict[str, Any]] = {}
    for strategy in LADDER_STRATEGIES:
        strategy_name = str(strategy["strategy"])
        group = [row for row in rows if row["strategy"] == strategy_name]
        baseline = next(
            (
                row
                for row in group
                if math.isclose(
                    float(row["probe_origin_offset_cells"]),
                    BASELINE_PROBE_OFFSET,
                )
            ),
            None,
        )
        baseline_force = None if baseline is None else _float_or_none(
            baseline.get("total_force_z_N")
        )
        ratios: list[float] = []
        for row in group:
            force = _float_or_none(row.get("total_force_z_N"))
            if baseline_force is None or baseline_force == 0.0 or force is None:
                row["force_ratio_to_strategy_baseline"] = ""
                continue
            ratio = force / baseline_force
            row["force_ratio_to_strategy_baseline"] = ratio
            ratios.append(ratio)
        strategy_summaries[strategy_name] = {
            "row_count": len(group),
            "completed_row_count": sum(1 for row in group if row["run_status"] == "completed"),
            "baseline_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
            "force_ratio_span": _ratio_span(ratios),
        }
    all_ratios = [
        float(row["force_ratio_to_strategy_baseline"])
        for row in rows
        if _float_or_none(row.get("force_ratio_to_strategy_baseline")) is not None
    ]
    return {
        "strategy_summaries": strategy_summaries,
        "all_strategy_force_ratio_span": _ratio_span(all_ratios),
    }


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    return ladder_stability._shared_snapshot_identity_status(rows, sha)


def _strategy_acceptance(
    rows: Sequence[Mapping[str, Any]],
    strategy_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    accepted: list[str] = []
    details: dict[str, Any] = {}
    expected_rows = len(PROBE_ORIGIN_OFFSETS)
    for strategy in LADDER_STRATEGIES:
        strategy_name = str(strategy["strategy"])
        group = [row for row in rows if row["strategy"] == strategy_name]
        summary = strategy_summaries[strategy_name]
        force_span = _float_or_none(summary["force_ratio_span"].get("relative_span"))
        max_residual = max(
            (
                _float_or_none(row.get("max_face_traction_decomposition_residual_pa"))
                or 0.0
                for row in group
            ),
            default=0.0,
        )
        pressure_complete = all(
            int(row["primary_face_pressure_complete_marker_count"])
            == int(row["primary_face_marker_count"])
            and int(row["secondary_face_pressure_complete_marker_count"])
            == int(row["secondary_face_marker_count"])
            for row in group
        )
        invalid_zero = all(
            int(row["primary_face_invalid_marker_count"]) == 0
            and int(row["secondary_face_invalid_marker_count"]) == 0
            for row in group
        )
        row_scope_ok = all(
            row["run_status"] == "completed"
            and not bool(row["solid_advanced"])
            and not bool(row["feedback_applied"])
            for row in group
        )
        accepted_strategy = (
            len(group) == expected_rows
            and row_scope_ok
            and pressure_complete
            and invalid_zero
            and force_span is not None
            and force_span <= STABLE_FORCE_RATIO_SPAN_MAX
            and max_residual <= TRACTION_DECOMPOSITION_RESIDUAL_MAX
        )
        details[strategy_name] = {
            "accepted": accepted_strategy,
            "row_count": len(group),
            "expected_row_count": expected_rows,
            "force_ratio_relative_span": "" if force_span is None else force_span,
            "max_face_traction_decomposition_residual_pa": max_residual,
            "pressure_complete": pressure_complete,
            "invalid_marker_counts_zero": invalid_zero,
            "scope_sampling_only": row_scope_ok,
        }
        if accepted_strategy:
            accepted.append(strategy_name)
    return {
        "accepted_strategies": accepted,
        "strategy_acceptance_details": details,
        "stable_ladder_candidate": accepted[0] if accepted else None,
    }


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_summary = _apply_strategy_ratios(rows)
    acceptance = _strategy_acceptance(rows, ratio_summary["strategy_summaries"])
    expected_sha = str(manifest.get("field_sha256", ""))
    stable_candidate = acceptance["stable_ladder_candidate"]
    candidate_status = (
        "probe_ladder_control_stable_candidate_found"
        if stable_candidate is not None
        else "probe_ladder_control_no_stable_candidate"
    )
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_traction_probe_ladder_control_matrix",
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
        "candidate_status": candidate_status,
        "stable_ladder_candidate": stable_candidate,
        "reference_formulation_candidate": None,
        "candidate_blockers": [
            {
                "blocker": "reference_selection_deferred",
                "detail": "Pressure-probe ladder control is diagnostic-only evidence.",
            },
            {
                "blocker": "dual_face_one_sided_unsupported",
                "detail": "Per-face one-sided pressure support is not implemented yet.",
            },
            {
                "blocker": "sampling_only_no_coupled_fsi",
                "detail": "Rows reuse one flow snapshot and do not advance coupled FSI.",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "No coupled or Fluent comparison run is part of this artifact.",
            },
        ],
        "stable_candidate_gate": {
            "force_ratio_relative_span_max": STABLE_FORCE_RATIO_SPAN_MAX,
            "traction_decomposition_residual_max": (
                TRACTION_DECOMPOSITION_RESIDUAL_MAX
            ),
        },
        "marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
        "probe_origin_offsets_cells": [offset for _, offset in PROBE_ORIGIN_OFFSETS],
        "strategies": [strategy["strategy"] for strategy in LADDER_STRATEGIES],
        "deferred_strategies": DEFERRED_STRATEGIES,
        "completed_formulation_count": len(rows),
        "scenario_count": len(rows),
        "histories": histories,
        "rows": rows,
        **ratio_summary,
        **acceptance,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key, value in list(csv_row.items()):
                if isinstance(value, (dict, list, tuple)):
                    csv_row[key] = _json_dumps(value, indent=None)
            writer.writerow(csv_row)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}g}"


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# ANSYS vertical-flap pressure-probe ladder control",
        "",
        "## Scope",
        "",
        (
            "This artifact reuses one archived shared preflow snapshot and "
            "re-runs only marker traction sampling. It does not advance the "
            "flow, the structure, or a coupled FSI loop, and it does not claim "
            "Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        "- reference_formulation_candidate: none",
        f"- candidate_status: `{payload.get('candidate_status', '')}`",
        f"- stable_ladder_candidate: `{payload.get('stable_ladder_candidate')}`",
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
            "## Strategy Acceptance",
            "",
            "| strategy | accepted | force span | max residual |",
            "|---|---:|---:|---:|",
        ]
    )
    details = payload["strategy_acceptance_details"]
    for strategy in payload["strategies"]:
        detail = details[strategy]
        lines.append(
            "| "
            f"{strategy} | "
            f"{detail['accepted']} | "
            f"{_fmt(detail['force_ratio_relative_span'])} | "
            f"{_fmt(detail['max_face_traction_decomposition_residual_pa'])} |"
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| scenario | probe-origin offset | start | spacing | ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['probe_origin_offset_cells']} | "
            f"{row['pressure_probe_start_offset_cells']} | "
            f"{row['pressure_probe_ladder_spacing_cells']} | "
            f"{_fmt(row.get('force_ratio_to_strategy_baseline'))} |"
        )
    lines.extend(
        [
            "",
            "## Deferred strategies",
            "",
        ]
    )
    for item in payload.get("deferred_strategies", []):
        lines.append(f"- {item['strategy']}: {item['status']}")
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "- Does not select a reference formulation.",
            "- Does not implement per-face one-sided pressure.",
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
    baseline_config = _source_config(LADDER_STRATEGIES[0], BASELINE_PROBE_OFFSET)
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    for scenario, strategy_name, config, probe_offset, _strategy in _scenario_specs():
        row, history = _sample_scenario(
            scenario=scenario,
            strategy_name=strategy_name,
            probe_offset=probe_offset,
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
            "purpose": "shared_flow_snapshot_traction_probe_ladder_control_history",
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
        print(f"[traction_probe_ladder_control] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_probe_ladder_control] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to "
        f"{OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
