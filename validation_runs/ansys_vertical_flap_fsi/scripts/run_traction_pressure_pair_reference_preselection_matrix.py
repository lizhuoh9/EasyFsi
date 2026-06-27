"""Preselect the ANSYS vertical-flap pressure-pair policy component.

This diagnostic reuses the archived shared preflow snapshot and samples marker
tractions only. It promotes the existing baseline-anchored pressure-pair result
to a pressure-pair policy component candidate while keeping full reference
formulation selection deferred.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_pressure_pair_anchor_map_matrix as anchor_map_matrix,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_probe_ladder_stability_matrix as ladder_stability,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_snapshot_resampling_matrix as snapshot_resampling,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
OUTPUT_DIR = CASE_ROOT / "traction_pressure_pair_reference_preselection_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_pressure_pair_reference_preselection_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_pressure_pair_reference_preselection_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_pressure_pair_reference_preselection_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_pressure_pair_reference_preselection_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH

SCOPE_LIMIT = (
    "shared snapshot sampling-only pressure-pair reference preselection on "
    "archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

BASELINE_SCENARIO = "baseline_independent_ladder_probe0p51"
UNSUPPORTED_SCENARIO = (
    "dual_one_sided_offset0p51_pressure_only_unsupported_confirmed"
)
UNSUPPORTED_REASON = (
    "dual-face one-sided pressure needs per-face one-sided region support"
)
INDEPENDENT_POLICY = anchor_map_matrix.INDEPENDENT_POLICY
ANCHORED_POLICY = anchor_map_matrix.ANCHORED_POLICY
PRESSURE_PAIR_POLICY_CANDIDATE = ANCHORED_POLICY
MARKER_FACE_OFFSET_CELLS = anchor_map_matrix.MARKER_FACE_OFFSET_CELLS
BASELINE_PROBE_OFFSET = anchor_map_matrix.BASELINE_PROBE_OFFSET
STABLE_FORCE_RATIO_SPAN_MAX = 0.10
ABSOLUTE_BASELINE_BIAS_MAX = 0.01
TRACTION_DECOMPOSITION_RESIDUAL_MAX = 1.0e-8

ANCHORED_SCENARIOS = (
    ("anchored_pair_dual_faces_probe0p00", 0.0),
    ("anchored_pair_dual_faces_probe0p25", 0.25),
    ("anchored_pair_dual_faces_probe0p375", 0.375),
    ("anchored_pair_dual_faces_probe0p51", 0.51),
    ("anchored_pair_dual_faces_probe0p625", 0.625),
    ("anchored_pair_dual_faces_probe0p75", 0.75),
    ("anchored_pair_dual_faces_probe1p00", 1.0),
    ("anchored_pair_dual_faces_probe1p50", 1.5),
)

MATRIX_COLUMNS = [
    "scenario",
    "preselection_role",
    "pressure_pair_policy",
    "run_status",
    "formulation_status",
    "unsupported_reason",
    "probe_origin_offset_cells",
    "marker_face_offset_cells",
    "anchor_source_scenario",
    "anchor_map_sha256",
    "pressure_pair_anchor_selected_marker_count",
    "pressure_pair_anchor_fallback_marker_count",
    "total_force_z_N",
    "force_ratio_to_preselection_baseline",
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


class PressurePairReferencePreselectionError(RuntimeError):
    """Raised when the pressure-pair preselection diagnostic cannot run."""


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


def _ratio_span(values: Sequence[float]) -> dict[str, float | str]:
    return ladder_stability._ratio_span(values)


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS_PATH):
        if path.exists():
            path.unlink()
    for path in MARKER_DIAGNOSTICS_DIR.glob("*.json"):
        path.unlink()


def _set_anchor_map_output_dir() -> None:
    anchor_map_matrix.MARKER_DIAGNOSTICS_DIR = MARKER_DIAGNOSTICS_DIR


def _scenario_config(policy: str, probe_offset: float) -> Any:
    return anchor_map_matrix._source_config(policy, probe_offset)


def _completed_row_role(scenario: str) -> str:
    if scenario == BASELINE_SCENARIO:
        return "baseline"
    return "anchored_pressure_pair_candidate"


def _normalize_completed_row(row: dict[str, Any], *, role: str) -> dict[str, Any]:
    row["preselection_role"] = role
    row["worker_mode"] = "shared_snapshot_pressure_pair_reference_preselection"
    row["scope_limit"] = SCOPE_LIMIT
    row["unsupported_reason"] = ""
    row["force_ratio_to_preselection_baseline"] = ""
    return row


def _normalize_completed_history(
    history: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, Any]:
    normalized = dict(history)
    normalized["flow_phase"] = "shared_snapshot_pressure_pair_reference_preselection"
    normalized["scenario"] = scenario
    return normalized


def _enrich_anchor_map(
    *,
    base_anchor_map: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(base_anchor_map["payload"])
    payload.update(
        {
            "source_scenario": BASELINE_SCENARIO,
            "source_policy": INDEPENDENT_POLICY,
            "anchor_source_scenario": BASELINE_SCENARIO,
            "anchor_source_policy": INDEPENDENT_POLICY,
            "anchor_source_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
            "anchor_source_marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
            "anchor_source_flow_snapshot_sha256": manifest.get("field_sha256", ""),
            "anchor_source_marker_geometry_sha256": baseline_row[
                "marker_geometry_sha256"
            ],
            "anchor_source_pressure_probe_origin_sha256": baseline_row[
                "pressure_probe_origin_sha256"
            ],
        }
    )
    anchor_sha = _sha256_payload(payload)
    return {
        "source_scenario": BASELINE_SCENARIO,
        "anchor_map_sha256": anchor_sha,
        "inside_cells": base_anchor_map["inside_cells"],
        "outside_cells": base_anchor_map["outside_cells"],
        "payload": payload,
    }


def _apply_preselection_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["scenario"] == BASELINE_SCENARIO)
    baseline_force = _float_or_none(baseline.get("total_force_z_N"))
    anchored_ratios: list[float] = []
    ratio_at_baseline_probe = None
    for row in rows:
        if row.get("run_status") != "completed":
            row["force_ratio_to_preselection_baseline"] = ""
            continue
        force = _float_or_none(row.get("total_force_z_N"))
        if (
            baseline_force is None
            or baseline_force == 0.0
            or force is None
            or row.get("pressure_pair_policy") != ANCHORED_POLICY
        ):
            row["force_ratio_to_preselection_baseline"] = ""
            continue
        ratio = force / baseline_force
        row["force_ratio_to_preselection_baseline"] = ratio
        anchored_ratios.append(ratio)
        if row["scenario"] == "anchored_pair_dual_faces_probe0p51":
            ratio_at_baseline_probe = ratio
    absolute_bias = (
        ""
        if ratio_at_baseline_probe is None
        else abs(float(ratio_at_baseline_probe) - 1.0)
    )
    return {
        "baseline_force_z_N": "" if baseline_force is None else baseline_force,
        "anchored_force_ratio_span": _ratio_span(anchored_ratios),
        "anchor_ratio_at_baseline_probe": (
            "" if ratio_at_baseline_probe is None else ratio_at_baseline_probe
        ),
        "absolute_baseline_bias": absolute_bias,
    }


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    if not completed:
        return "no_completed_rows"
    if all(row.get("flow_snapshot_sha256") == sha for row in completed):
        return "shared_snapshot_sha256_identical_completed_rows"
    return "shared_snapshot_sha256_mismatch"


def _unsupported_row(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario": UNSUPPORTED_SCENARIO,
        "preselection_role": "unsupported_blocker",
        "pressure_pair_policy": "per_face_one_sided_pressure",
        "run_status": "unsupported",
        "formulation_status": "unsupported",
        "unsupported_reason": UNSUPPORTED_REASON,
        "worker_mode": "shared_snapshot_pressure_pair_reference_preselection",
        "worker_elapsed_s": 0.0,
        "scope_limit": SCOPE_LIMIT,
        "solid_advanced": False,
        "feedback_applied": False,
        "marker_layout": "dual_physical_faces",
        "pressure_sampling_mode": "one_sided_pressure_jump",
        "marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
        "pressure_probe_origin_mode": "physical_face_offset",
        "probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
        "pressure_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
        "anchor_source_scenario": "",
        "anchor_map_sha256": "",
        "pressure_pair_anchor_selected_marker_count": 0,
        "pressure_pair_anchor_fallback_marker_count": 0,
        "total_force_z_N": "",
        "force_ratio_to_preselection_baseline": "",
        "primary_face_marker_count": 0,
        "secondary_face_marker_count": 0,
        "primary_face_pressure_complete_marker_count": 0,
        "secondary_face_pressure_complete_marker_count": 0,
        "primary_face_invalid_marker_count": 0,
        "secondary_face_invalid_marker_count": 0,
        "max_face_traction_decomposition_residual_pa": "",
        "marker_geometry_sha256": "",
        "pressure_probe_origin_sha256": "",
        "marker_diagnostics_json": "",
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
    }


def _unsupported_history(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": 0,
        "flow_phase": "shared_snapshot_pressure_pair_reference_preselection",
        "scenario": row["scenario"],
        "run_status": row["run_status"],
        "formulation_status": row["formulation_status"],
        "unsupported_reason": row["unsupported_reason"],
        "flow_snapshot_sha256": row["flow_snapshot_sha256"],
    }


def _preselection_acceptance(
    rows: Sequence[Mapping[str, Any]],
    ratio_summary: Mapping[str, Any],
) -> dict[str, Any]:
    anchored_rows = [
        row
        for row in rows
        if row.get("run_status") == "completed"
        and row.get("pressure_pair_policy") == ANCHORED_POLICY
    ]
    completed_rows = [row for row in rows if row.get("run_status") == "completed"]
    unsupported_rows = [
        row for row in rows if row.get("scenario") == UNSUPPORTED_SCENARIO
    ]
    force_span = _float_or_none(
        ratio_summary["anchored_force_ratio_span"].get("relative_span")
    )
    absolute_bias = _float_or_none(ratio_summary.get("absolute_baseline_bias"))
    max_residual = max(
        (
            _float_or_none(row.get("max_face_traction_decomposition_residual_pa"))
            or 0.0
            for row in completed_rows
        ),
        default=0.0,
    )
    pressure_complete = all(
        int(row["primary_face_pressure_complete_marker_count"])
        == int(row["primary_face_marker_count"])
        and int(row["secondary_face_pressure_complete_marker_count"])
        == int(row["secondary_face_marker_count"])
        for row in completed_rows
    )
    invalid_zero = all(
        int(row["primary_face_invalid_marker_count"]) == 0
        and int(row["secondary_face_invalid_marker_count"]) == 0
        for row in completed_rows
    )
    anchor_selected = all(
        int(row["pressure_pair_anchor_selected_marker_count"])
        == int(row["total_marker_count"])
        for row in anchored_rows
    )
    anchor_fallback_zero = all(
        int(row["pressure_pair_anchor_fallback_marker_count"]) == 0
        for row in anchored_rows
    )
    scope_sampling_only = all(
        not bool(row["solid_advanced"]) and not bool(row["feedback_applied"])
        for row in completed_rows
    )
    unsupported_confirmed = (
        len(unsupported_rows) == 1
        and unsupported_rows[0].get("run_status") == "unsupported"
        and unsupported_rows[0].get("formulation_status") == "unsupported"
        and unsupported_rows[0].get("unsupported_reason") == UNSUPPORTED_REASON
    )
    accepted = (
        len(anchored_rows) == len(ANCHORED_SCENARIOS)
        and len(completed_rows) == len(ANCHORED_SCENARIOS) + 1
        and force_span is not None
        and force_span <= STABLE_FORCE_RATIO_SPAN_MAX
        and absolute_bias is not None
        and absolute_bias <= ABSOLUTE_BASELINE_BIAS_MAX
        and max_residual <= TRACTION_DECOMPOSITION_RESIDUAL_MAX
        and pressure_complete
        and invalid_zero
        and anchor_selected
        and anchor_fallback_zero
        and scope_sampling_only
        and unsupported_confirmed
    )
    return {
        "accepted": accepted,
        "completed_row_count": len(completed_rows),
        "expected_completed_row_count": len(ANCHORED_SCENARIOS) + 1,
        "anchored_row_count": len(anchored_rows),
        "expected_anchored_row_count": len(ANCHORED_SCENARIOS),
        "force_ratio_relative_span": "" if force_span is None else force_span,
        "absolute_baseline_bias": "" if absolute_bias is None else absolute_bias,
        "max_face_traction_decomposition_residual_pa": max_residual,
        "pressure_complete": pressure_complete,
        "invalid_marker_counts_zero": invalid_zero,
        "anchor_selected_all_markers": anchor_selected,
        "anchor_fallback_zero": anchor_fallback_zero,
        "scope_sampling_only": scope_sampling_only,
        "dual_face_one_sided_unsupported_confirmed": unsupported_confirmed,
    }


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    anchor_map: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_summary = _apply_preselection_ratios(rows)
    acceptance = _preselection_acceptance(rows, ratio_summary)
    expected_sha = str(manifest.get("field_sha256", ""))
    candidate = PRESSURE_PAIR_POLICY_CANDIDATE if acceptance["accepted"] else None
    status = (
        "pressure_pair_policy_preselection_candidate_found"
        if candidate is not None
        else "pressure_pair_policy_preselection_blocked"
    )
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_pressure_pair_reference_preselection_matrix",
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
        "candidate_status": status,
        "pressure_pair_policy_candidate": candidate,
        "reference_formulation_candidate": None,
        "baseline_scenario": BASELINE_SCENARIO,
        "anchored_scenarios": [name for name, _ in ANCHORED_SCENARIOS],
        "unsupported_scenarios": [UNSUPPORTED_SCENARIO],
        "anchor_map": anchor_map["payload"],
        "anchor_map_sha256": anchor_map["anchor_map_sha256"],
        "candidate_blockers": [
            {
                "blocker": "dual_face_one_sided_unsupported",
                "detail": UNSUPPORTED_REASON,
            },
            {
                "blocker": "sampling_only_no_coupled_fsi",
                "detail": "Rows reuse one flow snapshot and do not advance coupled FSI.",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "No coupled or Fluent comparison run is part of this artifact.",
            },
            {
                "blocker": "reference_selection_deferred",
                "detail": "This artifact preselects only a pressure-pair component.",
            },
        ],
        "stable_candidate_gate": {
            "force_ratio_relative_span_max": STABLE_FORCE_RATIO_SPAN_MAX,
            "absolute_baseline_bias_max": ABSOLUTE_BASELINE_BIAS_MAX,
            "traction_decomposition_residual_max": (
                TRACTION_DECOMPOSITION_RESIDUAL_MAX
            ),
        },
        "preselection_acceptance": acceptance,
        "anchor_force_ratio_summary": ratio_summary,
        "marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
        "baseline_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
        "completed_formulation_count": sum(
            1 for row in rows if row.get("run_status") == "completed"
        ),
        "unsupported_formulation_count": sum(
            1 for row in rows if row.get("run_status") == "unsupported"
        ),
        "scenario_count": len(rows),
        "histories": histories,
        "rows": rows,
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
    acceptance = payload["preselection_acceptance"]
    ratio_summary = payload["anchor_force_ratio_summary"]
    lines = [
        "# ANSYS vertical-flap pressure-pair reference preselection",
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
        f"- candidate_status: `{payload.get('candidate_status', '')}`",
        (
            "- pressure_pair_policy_candidate: "
            f"`{payload.get('pressure_pair_policy_candidate')}`"
        ),
        "- reference_formulation_candidate: none",
        (
            "- dual_one_sided_offset0p51_pressure_only_unsupported_confirmed: "
            f"{acceptance['dual_face_one_sided_unsupported_confirmed']}"
        ),
        "",
        "## Gates",
        "",
        "| gate | value |",
        "|---|---:|",
        f"| accepted | {acceptance['accepted']} |",
        (
            "| force span | "
            f"{_fmt(acceptance['force_ratio_relative_span'])} |"
        ),
        (
            "| absolute baseline bias | "
            f"{_fmt(ratio_summary['absolute_baseline_bias'])} |"
        ),
        (
            "| max traction residual | "
            f"{_fmt(acceptance['max_face_traction_decomposition_residual_pa'])} |"
        ),
        f"| anchor selected all markers | {acceptance['anchor_selected_all_markers']} |",
        f"| anchor fallback zero | {acceptance['anchor_fallback_zero']} |",
        "",
        "## Rows",
        "",
        "| scenario | status | policy | ratio | anchor selected |",
        "|---|---|---|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['run_status']} | "
            f"{row['pressure_pair_policy']} | "
            f"{_fmt(row.get('force_ratio_to_preselection_baseline'))} | "
            f"{row.get('pressure_pair_anchor_selected_marker_count', '')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate blockers",
            "",
        ]
    )
    for blocker in payload.get("candidate_blockers", []):
        lines.append(f"- {blocker.get('blocker', '')}: {blocker.get('detail', '')}")
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "- Does not select a complete reference formulation.",
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


def run() -> dict[str, Any]:
    _prepare_output_dir()
    _set_anchor_map_output_dir()
    manifest = snapshot_resampling._load_manifest()
    fields = snapshot_resampling._load_snapshot_fields()
    baseline_config = _scenario_config(INDEPENDENT_POLICY, BASELINE_PROBE_OFFSET)
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    baseline_row, baseline_history, baseline_markers = anchor_map_matrix._sample_scenario(
        scenario=BASELINE_SCENARIO,
        policy=INDEPENDENT_POLICY,
        probe_offset=BASELINE_PROBE_OFFSET,
        config=baseline_config,
        fluid=fluid,
        runtime=runtime,
        manifest=manifest,
        anchor_map=None,
    )
    baseline_row = _normalize_completed_row(baseline_row, role="baseline")
    rows.append(baseline_row)
    histories[BASELINE_SCENARIO] = _normalize_completed_history(
        baseline_history,
        scenario=BASELINE_SCENARIO,
    )

    base_anchor = anchor_map_matrix._anchor_map_from_baseline(baseline_markers)
    anchor_map = _enrich_anchor_map(
        base_anchor_map=base_anchor,
        baseline_row=baseline_row,
        manifest=manifest,
    )

    for scenario, probe_offset in ANCHORED_SCENARIOS:
        config = _scenario_config(ANCHORED_POLICY, probe_offset)
        row, history, _marker_subset = anchor_map_matrix._sample_scenario(
            scenario=scenario,
            policy=ANCHORED_POLICY,
            probe_offset=probe_offset,
            config=config,
            fluid=fluid,
            runtime=runtime,
            manifest=manifest,
            anchor_map=anchor_map,
        )
        row = _normalize_completed_row(
            row,
            role=_completed_row_role(scenario),
        )
        rows.append(row)
        histories[scenario] = _normalize_completed_history(
            history,
            scenario=scenario,
        )

    unsupported = _unsupported_row(manifest)
    rows.append(unsupported)
    histories[UNSUPPORTED_SCENARIO] = _unsupported_history(unsupported)

    payload = _payload(rows, histories, manifest, anchor_map)
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_pressure_pair_reference_preselection_history",
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
        print(
            f"[traction_pressure_pair_reference_preselection] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[traction_pressure_pair_reference_preselection] wrote "
        f"{payload.get('scenario_count', 0)} rows to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        from validation_runs.ansys_vertical_flap_fsi.scripts import (
            run_traction_pressure_pair_reference_preselection_matrix as module_entry,
        )

        raise SystemExit(module_entry.main())
    raise SystemExit(main())
