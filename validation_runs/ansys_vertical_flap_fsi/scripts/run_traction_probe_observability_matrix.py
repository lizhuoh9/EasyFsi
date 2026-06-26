from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from cases.ansys_vertical_flap_fsi import run_vertical_flap_fsi_smoke  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_formulation_validation_matrix as base,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "traction_probe_observability_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_probe_observability_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_probe_observability_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_probe_observability_history.json"
SUMMARY_PATH = OUTPUT_DIR / "traction_probe_observability_summary.md"
VERIFICATION_PATH = (
    OUTPUT_DIR / "verification_traction_probe_observability_2026-06-26.md"
)
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"
HISTORIES_DIR = OUTPUT_DIR / "histories"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
WORKER_LOGS_DIR = OUTPUT_DIR / "worker_logs"
FAILURES_DIR = OUTPUT_DIR / "failures"

OBSERVABILITY_SCOPE_LIMIT = (
    "fixed-solid traction probe observability only; no coupled 50-step or "
    "Fluent parity claim"
)
OBSERVABILITY_PURPOSE = "fixed-solid traction probe observability diagnostics"
MATRIX_COLUMNS = [*base.MATRIX_COLUMNS, "marker_diagnostics_json"]

MARKER_REQUIRED_FIELDS = (
    "marker_index",
    "region_id",
    "position_m",
    "normal",
    "valid",
    "invalid_reason_code",
    "invalid_reason",
    "base_pressure_found",
    "inside_pressure_found",
    "outside_pressure_found",
    "base_pressure_pa",
    "inside_pressure_pa",
    "outside_pressure_pa",
    "pressure_jump_pa",
    "fluid_side_pressure_defined",
    "fluid_side_pressure_pa",
    "reference_pressure_pa",
    "inside_probe_ladder_mode",
    "outside_probe_ladder_mode",
    "inside_probe_rung",
    "outside_probe_rung",
    "inside_probe_multiplier",
    "outside_probe_multiplier",
    "inside_probe_distance_m",
    "outside_probe_distance_m",
    "inside_probe_grid_coordinate",
    "outside_probe_grid_coordinate",
    "inside_probe_nearest_cell",
    "outside_probe_nearest_cell",
    "inside_probe_fluid_weight",
    "outside_probe_fluid_weight",
    "pressure_traction_pa",
    "viscous_traction_pa",
    "total_traction_pa",
    "traction_decomposition_residual_pa",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ANSYS traction probe observability matrix."
    )
    parser.add_argument("--single-scenario")
    parser.add_argument(
        "--single-output",
        default=str(OUTPUT_DIR / ".single_worker.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.single_scenario:
        scenario, config = base._scenario_spec(args.single_scenario)
        row, history, marker_payload = _run_config_or_unsupported(scenario, config)
        Path(args.single_output).write_text(
            json.dumps(
                {
                    "row": row,
                    "history": history,
                    "marker_diagnostics": marker_payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    for directory in (
        OUTPUT_DIR,
        HISTORIES_DIR,
        MARKER_DIAGNOSTICS_DIR,
        WORKER_LOGS_DIR,
        FAILURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    marker_payloads: dict[str, dict[str, Any]] = {}

    for scenario in base.REQUIRED_SCENARIOS:
        row, history, marker_payload = _run_scenario_subprocess(scenario)
        rows.append(row)
        histories[scenario] = history
        if history:
            base._write_csv(HISTORIES_DIR / f"{scenario}_history.csv", history, base.HISTORY_COLUMNS)
        if marker_payload:
            marker_path = _write_marker_payload(scenario, marker_payload)
            row["marker_diagnostics_json"] = str(marker_path)
            marker_payloads[scenario] = marker_payload

    rows = base._hydrate_rows_from_histories(rows, histories)
    rows = _with_observability_scope_rows(base._apply_baseline_comparisons(rows))
    payload = base._payload(rows)
    payload.update(
        {
            "purpose": OBSERVABILITY_PURPOSE,
            "scope_limit": OBSERVABILITY_SCOPE_LIMIT,
            "artifact_directory": str(OUTPUT_DIR),
            "marker_diagnostics_directory": str(MARKER_DIAGNOSTICS_DIR),
            "old_artifact_directory_kept_separate": str(
                ROOT / "traction_formulation_diagnostics"
            ),
            "marker_required_fields": list(MARKER_REQUIRED_FIELDS),
        }
    )
    payload["offset_pathology"] = _offset_pathology_report(rows, marker_payloads)
    _write_payload_artifacts(payload, histories, rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _run_config_or_unsupported(
    scenario: str,
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        return _with_observability_scope_row(
            base._unsupported_row(scenario, config, reason)
        ), [], None
    return _run_config(scenario, config)


def _run_config(
    scenario: str,
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    started = time.perf_counter()
    try:
        report = run_vertical_flap_fsi_smoke(config)
        raw_history = list(report.get("preflow_history", []))
        history = [
            base._history_row(scenario, row)
            for row in raw_history
        ]
        row = _with_observability_scope_row(
            base._summary_row(scenario, config, report, history)
        )
        if raw_history:
            row["flow_driver_uses_full_velocity_reset"] = raw_history[-1].get(
                "flow_driver_uses_full_velocity_reset",
                "",
            )
        _merge_final_face_diagnostics(row, history, report)
        row["elapsed_s"] = time.perf_counter() - started
        row["error"] = ""
        marker_payload = _marker_payload(scenario, report, row)
        return row, history, marker_payload
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        return (
            _with_observability_scope_row(
                base._failed_worker_row(
                    scenario,
                    config,
                    time.perf_counter() - started,
                    f"{exc}\n{traceback.format_exc()}",
                    "",
                    False,
                    "",
                    "",
                )
            ),
            [],
            None,
        )


def _run_scenario_subprocess(
    scenario: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    _, config = base._scenario_spec(scenario)
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        return _with_observability_scope_row(
            base._unsupported_row(scenario, config, reason)
        ), [], None

    output_path = OUTPUT_DIR / f".{scenario}_worker.json"
    if output_path.exists():
        output_path.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-scenario",
        scenario,
        "--single-output",
        str(output_path),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=base.WORKER_TIMEOUT_S,
        )
        elapsed_s = time.perf_counter() - started
        stdout_log = _write_worker_log(
            scenario,
            "stdout",
            result.stdout,
            failed=result.returncode != 0,
        )
        stderr_log = _write_worker_log(
            scenario,
            "stderr",
            result.stderr,
            failed=result.returncode != 0,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - started
        stdout_log = _write_worker_log(
            scenario,
            "stdout",
            base._timeout_stream_text(exc.stdout),
            failed=True,
        )
        stderr_log = _write_worker_log(
            scenario,
            "stderr",
            base._timeout_stream_text(exc.stderr),
            failed=True,
        )
        row = base._failed_worker_row(
            scenario,
            config,
            elapsed_s,
            f"worker timed out after {base.WORKER_TIMEOUT_S} s",
            "timeout",
            True,
            stdout_log,
            stderr_log,
        )
        _write_failure(scenario, row)
        return _with_observability_scope_row(row), [], None

    if result.returncode == 0 and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        output_path.unlink()
        row = dict(payload["row"])
        row.update(
            base._worker_fields(
                result.returncode,
                False,
                elapsed_s,
                stdout_log,
                stderr_log,
            )
        )
        return (
            _with_observability_scope_row(row),
            [dict(item) for item in payload["history"]],
            payload.get("marker_diagnostics"),
        )

    row = base._failed_worker_row(
        scenario,
        config,
        elapsed_s,
        (
            f"worker exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ),
        result.returncode,
        False,
        stdout_log,
        stderr_log,
    )
    _write_failure(scenario, row)
    return _with_observability_scope_row(row), [], None


def _marker_payload(
    scenario: str,
    report: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    markers = [
        _marker_required_subset(dict(marker))
        for marker in report.get("final_stress_marker_diagnostics", [])
    ]
    return {
        "scenario": scenario,
        "preflow_step": int(report.get("preflow_steps_completed", 0)),
        "scope_limit": OBSERVABILITY_SCOPE_LIMIT,
        "marker_count": len(markers),
        "marker_required_fields": list(MARKER_REQUIRED_FIELDS),
        "face_diagnostics": dict(report.get("final_stress_face_diagnostics", {})),
        "row_force_summary": {
            "total_force_z_N": row.get("total_force_z_N", ""),
            "primary_face_force_z_N": row.get("primary_face_force_z_N", ""),
            "secondary_face_force_z_N": row.get("secondary_face_force_z_N", ""),
            "force_ratio_to_baseline": row.get("force_ratio_to_baseline", ""),
        },
        "markers": markers,
    }


def _merge_final_face_diagnostics(
    row: dict[str, Any],
    history: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    face_diagnostics = dict(report.get("final_stress_face_diagnostics", {}))
    if not face_diagnostics:
        return
    for field, value in face_diagnostics.items():
        row[field] = value
        if history:
            history[-1][field] = value


def _marker_required_subset(marker: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in marker]
    if missing:
        raise ValueError(f"marker diagnostics missing fields: {missing}")
    return {field: marker[field] for field in MARKER_REQUIRED_FIELDS}


def _write_marker_payload(scenario: str, payload: dict[str, Any]) -> Path:
    path = MARKER_DIAGNOSTICS_DIR / f"{scenario}_step020_markers.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _with_observability_scope_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_with_observability_scope_row(row) for row in rows]


def _with_observability_scope_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["scope_limit"] = OBSERVABILITY_SCOPE_LIMIT
    return updated


def _offset_pathology_report(
    rows: list[dict[str, Any]],
    marker_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_name = {str(row["scenario"]): row for row in rows}
    return {
        "offset0p25": _offset_mechanism(
            "dual_two_sided_offset0p25_pressure_only",
            by_name,
            marker_payloads,
        ),
        "offset0p51": _offset_mechanism(
            base.BASELINE_SCENARIO,
            by_name,
            marker_payloads,
        ),
        "offset1p00": _offset_mechanism(
            "dual_two_sided_offset1p00_pressure_only",
            by_name,
            marker_payloads,
        ),
    }


def _offset_mechanism(
    scenario: str,
    rows: dict[str, dict[str, Any]],
    marker_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = rows.get(scenario, {})
    payload = marker_payloads.get(scenario, {})
    markers = payload.get("markers", [])
    primary = _face_marker_stats(
        markers,
        solid_mpm_fsi_runner.PRIMARY_REGION_ID,
    )
    secondary = _face_marker_stats(
        markers,
        solid_mpm_fsi_runner.SECONDARY_REGION_ID,
    )
    ratio = row.get("force_ratio_to_baseline", "")
    return {
        "scenario": scenario,
        "force_ratio_to_baseline": ratio,
        "primary": primary,
        "secondary": secondary,
        "interpretation": _interpret_offset(scenario, ratio, primary, secondary),
    }


def _face_marker_stats(markers: list[dict[str, Any]], region_id: int) -> dict[str, Any]:
    face_markers = [
        marker for marker in markers if int(marker.get("region_id", -1)) == region_id
    ]
    valid = [marker for marker in face_markers if bool(marker.get("valid"))]
    return {
        "marker_count": len(face_markers),
        "valid_marker_count": len(valid),
        "mean_inside_pressure_pa": _mean(valid, "inside_pressure_pa"),
        "mean_outside_pressure_pa": _mean(valid, "outside_pressure_pa"),
        "mean_pressure_jump_pa": _mean(valid, "pressure_jump_pa"),
        "inside_rung_histogram": _histogram(valid, "inside_probe_rung"),
        "outside_rung_histogram": _histogram(valid, "outside_probe_rung"),
        "inside_unique_nearest_cell_count": _unique_cell_count(
            valid,
            "inside_probe_nearest_cell",
        ),
        "outside_unique_nearest_cell_count": _unique_cell_count(
            valid,
            "outside_probe_nearest_cell",
        ),
        "mean_total_traction_z_pa": _mean_vector_component(valid, "total_traction_pa", 2),
    }


def _mean(markers: list[dict[str, Any]], field: str) -> float | str:
    values = [float(marker[field]) for marker in markers if field in marker]
    if not values:
        return ""
    return sum(values) / len(values)


def _mean_vector_component(
    markers: list[dict[str, Any]],
    field: str,
    component: int,
) -> float | str:
    values = [
        float(marker[field][component])
        for marker in markers
        if field in marker and len(marker[field]) > component
    ]
    if not values:
        return ""
    return sum(values) / len(values)


def _histogram(markers: list[dict[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for marker in markers:
        value = int(marker[field])
        if value < 0:
            continue
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _unique_cell_count(markers: list[dict[str, Any]], field: str) -> int:
    cells = {
        tuple(int(value) for value in marker[field])
        for marker in markers
        if all(int(value) >= 0 for value in marker[field])
    }
    return len(cells)


def _interpret_offset(
    scenario: str,
    ratio: Any,
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> str:
    primary_jump = abs(float(primary.get("mean_pressure_jump_pa") or 0.0))
    secondary_jump = abs(float(secondary.get("mean_pressure_jump_pa") or 0.0))
    secondary_same_side = secondary_jump <= max(primary_jump * 0.2, 1.0e-12)
    try:
        ratio_value = float(ratio)
    except (TypeError, ValueError):
        ratio_value = float("nan")

    if scenario.endswith("offset0p25_pressure_only"):
        if secondary_jump > 0.5 * primary_jump:
            return (
                "offset0p25 duplicates pressure jump across both physical faces: "
                "primary and secondary both report non-trivial two-sided jumps."
            )
        return (
            "offset0p25 keeps probe evidence complete but does not show equal "
            "primary/secondary jumps in this archived run."
        )
    if scenario == base.BASELINE_SCENARIO:
        if secondary_same_side:
            return (
                "offset0p51 leaves the secondary face near zero because its "
                "inside/outside probes sample nearly equal pressure regions; "
                "the primary face carries the dominant jump."
            )
        return (
            "offset0p51 remains asymmetric: primary and secondary probe evidence "
            "do not support a physical per-face one-sided interpretation."
        )
    if scenario.endswith("offset1p00_pressure_only"):
        if ratio_value < 0.2 or secondary_same_side:
            return (
                "offset1p00 loses the thin-wall pressure jump: nearest-cell and "
                "pressure evidence show probes no longer straddle the jump cleanly."
            )
        return (
            "offset1p00 stays offset-sensitive and cannot be promoted without "
            "separating marker location from probe start."
        )
    return "not an offset sensitivity scenario"


def _write_payload_artifacts(
    payload: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> None:
    MATRIX_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    base._write_csv(MATRIX_CSV, rows, MATRIX_COLUMNS)
    HISTORY_JSON.write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(_verification_markdown(payload), encoding="utf-8")
    _write_checksums()


def _summary_markdown(payload: dict[str, Any]) -> str:
    pathology = payload["offset_pathology"]
    blockers = ", ".join(payload["candidate_blockers"]) or "none"
    return (
        "# ANSYS Traction Probe Observability Summary - 2026-06-26\n\n"
        f"scope_limit = {OBSERVABILITY_SCOPE_LIMIT}\n\n"
        f"reference_formulation_candidate = {payload['reference_formulation_candidate']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        f"candidate_blockers = {blockers}\n\n"
        "## offset0p25 mechanism\n\n"
        f"{pathology['offset0p25']['interpretation']}\n\n"
        f"primary = {json.dumps(pathology['offset0p25']['primary'], sort_keys=True)}\n\n"
        f"secondary = {json.dumps(pathology['offset0p25']['secondary'], sort_keys=True)}\n\n"
        "## offset0p51 mechanism\n\n"
        f"{pathology['offset0p51']['interpretation']}\n\n"
        f"primary = {json.dumps(pathology['offset0p51']['primary'], sort_keys=True)}\n\n"
        f"secondary = {json.dumps(pathology['offset0p51']['secondary'], sort_keys=True)}\n\n"
        "## offset1p00 mechanism\n\n"
        f"{pathology['offset1p00']['interpretation']}\n\n"
        f"primary = {json.dumps(pathology['offset1p00']['primary'], sort_keys=True)}\n\n"
        f"secondary = {json.dumps(pathology['offset1p00']['secondary'], sort_keys=True)}\n\n"
        "## Conclusion\n\n"
        "The observability rerun archives marker-level inside/outside pressure, "
        "probe rung, distance, nearest-cell, fluid-weight, and traction "
        "decomposition evidence. It explains the dual/two-sided offset "
        "sensitivity but does not select a reference formulation and does not "
        "claim Fluent parity.\n"
    )


def _verification_markdown(payload: dict[str, Any]) -> str:
    completed = [
        row["scenario"]
        for row in payload["rows"]
        if row.get("run_status") == "completed"
    ]
    unsupported = [
        row["scenario"]
        for row in payload["rows"]
        if row.get("run_status") == "unsupported"
    ]
    return (
        "# ANSYS Traction Probe Observability Verification - 2026-06-26\n\n"
        "Generated by:\n\n"
        "```powershell\n"
        "& \"D:/TOOL/Anaconda/python.exe\" "
        "validation_runs/ansys_vertical_flap_fsi/scripts/"
        "run_traction_probe_observability_matrix.py\n"
        "```\n\n"
        f"completed_scenarios = {json.dumps(completed, sort_keys=True)}\n\n"
        f"unsupported_scenarios = {json.dumps(unsupported, sort_keys=True)}\n\n"
        f"reference_formulation_candidate = {payload['reference_formulation_candidate']}\n\n"
        f"candidate_blockers = {json.dumps(payload['candidate_blockers'], sort_keys=True)}\n"
    )


def _write_worker_log(
    scenario: str,
    stream: str,
    text: str,
    *,
    failed: bool,
) -> str:
    directory = FAILURES_DIR if failed else WORKER_LOGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scenario}_{stream}.log"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_failure(scenario: str, row: dict[str, Any]) -> None:
    path = FAILURES_DIR / f"{scenario}_failure.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checksums() -> None:
    lines: list[str] = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file() or path == CHECKSUMS_PATH:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(OUTPUT_DIR).as_posix()}")
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
