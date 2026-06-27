"""Evaluate ANSYS vertical-flap symmetric pressure-pair sampling.

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
    run_traction_probe_ladder_control_matrix as ladder_control,
)
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
OUTPUT_DIR = CASE_ROOT / "traction_symmetric_pressure_pair_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_symmetric_pressure_pair_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_symmetric_pressure_pair_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_symmetric_pressure_pair_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_symmetric_pressure_pair_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH

SCOPE_LIMIT = (
    "shared snapshot sampling-only symmetric pressure cell-pair diagnostic "
    "on archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

MARKER_FACE_OFFSET_CELLS = 0.51
BASELINE_PROBE_OFFSET = 0.51
STABLE_FORCE_RATIO_SPAN_MAX = 0.10
TRACTION_DECOMPOSITION_RESIDUAL_MAX = 1.0e-8
PAIR_SYMMETRY_RESIDUAL_MAX_CELLS = 1.0e-8
LADDER_MODE = "current_normal_cell_ladder"
PAIR_MAX_CELL_DELTA = 1
PAIR_REQUIRE_OPPOSITE_SIDES = True
INDEPENDENT_POLICY = "independent_ladder"
SYMMETRIC_POLICY = "symmetric_cell_pair"

INDEPENDENT_SCENARIOS = (
    ("independent_ladder_baseline_probe0p51", 0.51),
    ("independent_ladder_baseline_probe0p625", 0.625),
    ("independent_ladder_baseline_probe1p00", 1.0),
)

SYMMETRIC_SCENARIOS = (
    ("symmetric_pair_probe0p51", 0.51),
    ("symmetric_pair_probe0p625", 0.625),
    ("symmetric_pair_probe1p00", 1.0),
    ("symmetric_pair_probe0p00", 0.0),
    ("symmetric_pair_probe0p25", 0.25),
    ("symmetric_pair_probe0p375", 0.375),
    ("symmetric_pair_probe0p75", 0.75),
    ("symmetric_pair_probe1p50", 1.5),
)

PAIR_MARKER_FIELDS = [
    "pressure_pair_policy",
    "pressure_pair_selected",
    "pressure_pair_fallback_used",
    "pressure_pair_inside_cell",
    "pressure_pair_outside_cell",
    "pressure_pair_cell_delta",
    "pressure_pair_symmetry_residual_cells",
]

MARKER_REQUIRED_FIELDS = [
    *ladder_control.MARKER_REQUIRED_FIELDS,
    *PAIR_MARKER_FIELDS,
]

MATRIX_COLUMNS = [
    "scenario",
    "pressure_pair_policy",
    "run_status",
    "formulation_status",
    "probe_origin_offset_cells",
    "marker_face_offset_cells",
    "pressure_pair_selected_marker_count",
    "pressure_pair_fallback_marker_count",
    "pressure_pair_max_symmetry_residual_cells",
    "total_force_z_N",
    "force_ratio_to_policy_baseline",
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

PRIMARY_REGION_ID = solid_mpm_fsi_runner.PRIMARY_REGION_ID
SECONDARY_REGION_ID = solid_mpm_fsi_runner.SECONDARY_REGION_ID


class SymmetricPressurePairError(RuntimeError):
    """Raised when the symmetric pressure-pair diagnostic cannot run."""


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


def _source_config(policy: str, probe_offset: float) -> Any:
    base = offset_decoupling._source_config(
        MARKER_FACE_OFFSET_CELLS,
        float(probe_offset),
    )
    return replace(
        base,
        traction_pressure_probe_start_offset_cells=None,
        traction_pressure_probe_ladder_spacing_cells=0.5,
        traction_pressure_probe_ladder_rung_count=5,
        traction_pressure_probe_ladder_mode=LADDER_MODE,
        traction_pressure_pair_policy=policy,
        traction_pressure_pair_max_cell_delta=PAIR_MAX_CELL_DELTA,
        traction_pressure_pair_require_opposite_sides=PAIR_REQUIRE_OPPOSITE_SIDES,
    )


def _scenario_specs() -> list[tuple[str, str, float, Any]]:
    specs: list[tuple[str, str, float, Any]] = []
    for scenario, probe_offset in INDEPENDENT_SCENARIOS:
        specs.append(
            (
                scenario,
                INDEPENDENT_POLICY,
                float(probe_offset),
                _source_config(INDEPENDENT_POLICY, probe_offset),
            )
        )
    for scenario, probe_offset in SYMMETRIC_SCENARIOS:
        specs.append(
            (
                scenario,
                SYMMETRIC_POLICY,
                float(probe_offset),
                _source_config(SYMMETRIC_POLICY, probe_offset),
            )
        )
    return specs


def _marker_required_subset(marker: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in marker]
    if missing:
        raise SymmetricPressurePairError(
            "Marker diagnostic is missing required fields: " + ", ".join(missing)
        )
    return {field: marker[field] for field in MARKER_REQUIRED_FIELDS}


def _pair_stats(marker_subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    residuals = [
        float(marker["pressure_pair_symmetry_residual_cells"])
        for marker in marker_subset
        if _float_or_none(marker.get("pressure_pair_symmetry_residual_cells"))
        is not None
        and float(marker["pressure_pair_symmetry_residual_cells"]) >= 0.0
    ]
    return {
        "pressure_pair_selected_marker_count": sum(
            1 for marker in marker_subset if bool(marker["pressure_pair_selected"])
        ),
        "pressure_pair_fallback_marker_count": sum(
            1 for marker in marker_subset if bool(marker["pressure_pair_fallback_used"])
        ),
        "pressure_pair_max_symmetry_residual_cells": (
            max(residuals) if residuals else ""
        ),
    }


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
    policy: str,
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
        "pressure_pair_policy": policy,
        "purpose": "shared_flow_snapshot_symmetric_pressure_pair_marker_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "pressure_probe_ladder_mode": config.traction_pressure_probe_ladder_mode,
        "pressure_pair_max_cell_delta": config.traction_pressure_pair_max_cell_delta,
        "pressure_pair_require_opposite_sides": (
            config.traction_pressure_pair_require_opposite_sides
        ),
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_geometry": marker_geometry,
        "pressure_probe_origin": pressure_probe_origin,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": list(marker_subset),
        "pair_stats": _pair_stats(marker_subset),
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
    policy: str,
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
        policy=policy,
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
        "pressure_pair_policy": policy,
        "run_status": "completed",
        "formulation_status": "completed",
        "worker_mode": "shared_snapshot_symmetric_pressure_pair",
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
        "pressure_probe_ladder_mode": config.traction_pressure_probe_ladder_mode,
        "pressure_pair_max_cell_delta": config.traction_pressure_pair_max_cell_delta,
        "pressure_pair_require_opposite_sides": (
            config.traction_pressure_pair_require_opposite_sides
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
    row.update(traction_fields)
    row.update(ladder_stability._row_face_fields(marker_subset))
    row.update(_pair_stats(marker_subset))
    row["total_force_z_N"] = force_fields.get("marker_force_z_N", "")
    row["max_face_traction_decomposition_residual_pa"] = _max_face_residual(row)

    history_row = {
        "step": 0,
        "flow_phase": "shared_snapshot_symmetric_pressure_pair",
        "scenario": scenario,
        "pressure_pair_policy": policy,
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "probe_origin_offset_cells": probe_offset,
    }
    history_row.update(force_fields)
    history_row.update(stress_fields)
    history_row.update(traction_fields)
    history_row.update(_pair_stats(marker_subset))
    return row, history_row


def _sample_scenario(
    *,
    scenario: str,
    policy: str,
    probe_offset: float,
    config: Any,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    solid_mpm_fsi_runner._validate_rectangular_solid_config(config)
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        raise SymmetricPressurePairError(f"{scenario} unsupported: {reason}")
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
        policy=policy,
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


def _apply_policy_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_summaries: dict[str, dict[str, Any]] = {}
    for policy in (INDEPENDENT_POLICY, SYMMETRIC_POLICY):
        group = [row for row in rows if row["pressure_pair_policy"] == policy]
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
                row["force_ratio_to_policy_baseline"] = ""
                continue
            ratio = force / baseline_force
            row["force_ratio_to_policy_baseline"] = ratio
            ratios.append(ratio)
        policy_summaries[policy] = {
            "row_count": len(group),
            "completed_row_count": sum(
                1 for row in group if row["run_status"] == "completed"
            ),
            "baseline_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
            "force_ratio_span": _ratio_span(ratios),
        }
    return {"policy_summaries": policy_summaries}


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    return ladder_stability._shared_snapshot_identity_status(rows, sha)


def _stable_candidate_acceptance(
    rows: Sequence[Mapping[str, Any]],
    policy_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    symmetric_rows = [
        row for row in rows if row["pressure_pair_policy"] == SYMMETRIC_POLICY
    ]
    force_span = _float_or_none(
        policy_summaries[SYMMETRIC_POLICY]["force_ratio_span"].get("relative_span")
    )
    max_residual = max(
        (
            _float_or_none(row.get("max_face_traction_decomposition_residual_pa"))
            or 0.0
            for row in symmetric_rows
        ),
        default=0.0,
    )
    max_pair_residual = max(
        (
            _float_or_none(row.get("pressure_pair_max_symmetry_residual_cells"))
            or 0.0
            for row in symmetric_rows
        ),
        default=0.0,
    )
    pressure_complete = all(
        int(row["primary_face_pressure_complete_marker_count"])
        == int(row["primary_face_marker_count"])
        and int(row["secondary_face_pressure_complete_marker_count"])
        == int(row["secondary_face_marker_count"])
        for row in symmetric_rows
    )
    invalid_zero = all(
        int(row["primary_face_invalid_marker_count"]) == 0
        and int(row["secondary_face_invalid_marker_count"]) == 0
        for row in symmetric_rows
    )
    pair_selected = all(
        int(row["pressure_pair_selected_marker_count"])
        == int(row["total_marker_count"])
        for row in symmetric_rows
    )
    pair_fallback_zero = all(
        int(row["pressure_pair_fallback_marker_count"]) == 0
        for row in symmetric_rows
    )
    row_scope_ok = all(
        row["run_status"] == "completed"
        and not bool(row["solid_advanced"])
        and not bool(row["feedback_applied"])
        for row in symmetric_rows
    )
    accepted = (
        len(symmetric_rows) == len(SYMMETRIC_SCENARIOS)
        and row_scope_ok
        and pressure_complete
        and invalid_zero
        and pair_selected
        and pair_fallback_zero
        and force_span is not None
        and force_span <= STABLE_FORCE_RATIO_SPAN_MAX
        and max_residual <= TRACTION_DECOMPOSITION_RESIDUAL_MAX
        and max_pair_residual <= PAIR_SYMMETRY_RESIDUAL_MAX_CELLS
    )
    candidate = SYMMETRIC_POLICY if accepted else None
    return {
        "symmetric_pair_acceptance": {
            "accepted": accepted,
            "row_count": len(symmetric_rows),
            "expected_row_count": len(SYMMETRIC_SCENARIOS),
            "force_ratio_relative_span": "" if force_span is None else force_span,
            "max_face_traction_decomposition_residual_pa": max_residual,
            "max_pressure_pair_symmetry_residual_cells": max_pair_residual,
            "pressure_complete": pressure_complete,
            "invalid_marker_counts_zero": invalid_zero,
            "pair_selected_all_markers": pair_selected,
            "pair_fallback_zero": pair_fallback_zero,
            "scope_sampling_only": row_scope_ok,
        },
        "accepted_symmetric_pair_candidate": candidate,
    }


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_summary = _apply_policy_ratios(rows)
    acceptance = _stable_candidate_acceptance(
        rows,
        ratio_summary["policy_summaries"],
    )
    expected_sha = str(manifest.get("field_sha256", ""))
    stable_candidate = acceptance["accepted_symmetric_pair_candidate"]
    candidate_status = (
        "symmetric_pressure_pair_stable_candidate_found"
        if stable_candidate is not None
        else "symmetric_pressure_pair_no_stable_candidate"
    )
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_symmetric_pressure_pair_matrix",
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
        "stable_symmetric_pressure_pair_candidate": stable_candidate,
        "reference_formulation_candidate": None,
        "candidate_blockers": [
            {
                "blocker": "reference_selection_deferred",
                "detail": "Symmetric pressure-pair sampling is diagnostic evidence only.",
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
            "pressure_pair_symmetry_residual_max_cells": (
                PAIR_SYMMETRY_RESIDUAL_MAX_CELLS
            ),
        },
        "marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
        "baseline_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
        "independent_scenarios": [name for name, _ in INDEPENDENT_SCENARIOS],
        "symmetric_scenarios": [name for name, _ in SYMMETRIC_SCENARIOS],
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
    acceptance = payload["symmetric_pair_acceptance"]
    lines = [
        "# ANSYS vertical-flap symmetric pressure pair",
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
        (
            "- stable_symmetric_pressure_pair_candidate: "
            f"`{payload.get('stable_symmetric_pressure_pair_candidate')}`"
        ),
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
            "## Symmetric Pair Gate",
            "",
            "| gate | value |",
            "|---|---:|",
            f"| accepted | {acceptance['accepted']} |",
            (
                "| force span | "
                f"{_fmt(acceptance['force_ratio_relative_span'])} |"
            ),
            (
                "| max traction residual | "
                f"{_fmt(acceptance['max_face_traction_decomposition_residual_pa'])} |"
            ),
            (
                "| max pair residual | "
                f"{_fmt(acceptance['max_pressure_pair_symmetry_residual_cells'])} |"
            ),
            f"| pair selected all markers | {acceptance['pair_selected_all_markers']} |",
            f"| pair fallback zero | {acceptance['pair_fallback_zero']} |",
            "",
            "## Rows",
            "",
            "| scenario | policy | probe-origin offset | ratio | pair selected |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['pressure_pair_policy']} | "
            f"{row['probe_origin_offset_cells']} | "
            f"{_fmt(row.get('force_ratio_to_policy_baseline'))} | "
            f"{row['pressure_pair_selected_marker_count']} |"
        )
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
    baseline_config = _source_config(INDEPENDENT_POLICY, BASELINE_PROBE_OFFSET)
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    for scenario, policy, probe_offset, config in _scenario_specs():
        row, history = _sample_scenario(
            scenario=scenario,
            policy=policy,
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
            "purpose": "shared_flow_snapshot_symmetric_pressure_pair_history",
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
        print(f"[traction_symmetric_pressure_pair] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_symmetric_pressure_pair] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to "
        f"{OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
