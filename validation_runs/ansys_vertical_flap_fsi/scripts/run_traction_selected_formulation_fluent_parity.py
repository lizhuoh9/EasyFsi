from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fluent_reference_contract_schema import validate_fluent_reference_contract


CASE_NAME = "ansys_vertical_flap_fsi"
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "traction_selected_formulation_fluent_parity_diagnostics"
SCENARIO_DIAGNOSTICS_DIR = OUTPUT_DIR / "scenario_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_selected_formulation_fluent_parity_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_selected_formulation_fluent_parity_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_selected_formulation_fluent_parity_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_selected_formulation_fluent_parity_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"
SOURCE_STEP50_MATRIX_JSON = (
    ROOT
    / "traction_selected_formulation_coupled_step50_diagnostics"
    / "traction_selected_formulation_coupled_step50_matrix.json"
)
SOURCE_STEP50_HISTORY_JSON = (
    ROOT
    / "traction_selected_formulation_coupled_step50_diagnostics"
    / "traction_selected_formulation_coupled_step50_history.json"
)
FLUENT_REFERENCE_CONTRACT_JSON = (
    ROOT / "fluent_reference" / "fluent_reference_contract_2026-06-27.json"
)
ACTIVE_CONTRACT_MANIFEST_JSON = (
    ROOT / "fluent_reference" / "active_fluent_reference_contract.json"
)

SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_selected_formulation_fluent_parity.py"
)
SCENARIO = "selected_formulation_fluent_parity"
SOURCE_STEP50_SCENARIO = "selected_formulation_coupled_step50"
REFERENCE_FORMULATION_CANDIDATE = (
    "anchored_dual_face_pressure_pair_with_per_face_one_sided"
)
BLOCKER_REFERENCE_INCOMPLETE = "fluent_reference_incomplete"
BLOCKER_NO_FLUENT_PARITY = "no_fluent_parity_claim"
RELATIVE_ERROR_DENOMINATOR_FLOOR = 1.0e-12

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "parity_status",
    "candidate_status",
    "reference_contract_status",
    "source_step50_candidate_status",
    "reference_formulation_candidate",
    "source_step50_matrix_sha256",
    "source_step50_history_sha256",
    "fluent_reference_contract_sha256",
    "displacement_gate_status",
    "force_gate_status",
    "flow_outlet_gate_status",
    "pressure_gate_status",
    "metadata_gate_status",
    "active_blockers",
    "scenario_diagnostics_json",
]


def run() -> dict[str, Any]:
    _prepare_output_dir()
    source_matrix = _read_json(SOURCE_STEP50_MATRIX_JSON)
    source_history = _read_json(SOURCE_STEP50_HISTORY_JSON)
    active_contract_manifest = _active_contract_manifest()
    fluent_reference_contract_json = _active_reference_contract_path(
        active_contract_manifest
    )
    reference_contract = _read_json(fluent_reference_contract_json)
    contract_schema_validation = validate_fluent_reference_contract(
        reference_contract
    )
    source_row = _source_step50_row(source_matrix)
    source_history_rows = _source_step50_history_rows(source_history)
    metrics = _parity_metrics(
        source_matrix=source_matrix,
        source_row=source_row,
        source_history_rows=source_history_rows,
        reference_contract=reference_contract,
        reference_contract_path=fluent_reference_contract_json,
        contract_schema_validation=contract_schema_validation,
    )
    candidate_status = _candidate_status(reference_contract, metrics)
    blockers = _candidate_blockers(candidate_status, metrics)
    row = _row(
        candidate_status=candidate_status,
        blockers=blockers,
        source_matrix=source_matrix,
        source_row=source_row,
        reference_contract=reference_contract,
        reference_contract_path=fluent_reference_contract_json,
        active_contract_manifest=active_contract_manifest,
        metrics=metrics,
    )
    diagnostics_path = SCENARIO_DIAGNOSTICS_DIR / f"{SCENARIO}.json"
    row["scenario_diagnostics_json"] = _repo_relative(diagnostics_path)
    diagnostics = {
        "case": CASE_NAME,
        "scenario": SCENARIO,
        "purpose": "selected_formulation_fluent_parity_diagnostics",
        "candidate_status": candidate_status,
        "candidate_blockers": blockers,
        "parity_metrics": metrics,
        "fluent_reference_contract_schema_validation": contract_schema_validation,
        "source_step50_row": source_row,
        "active_fluent_reference_contract_manifest": active_contract_manifest,
        "fluent_reference_contract": reference_contract,
    }
    _write_json(diagnostics_path, diagnostics)
    payload = _payload(
        candidate_status=candidate_status,
        blockers=blockers,
        row=row,
        source_matrix=source_matrix,
        reference_contract=reference_contract,
        reference_contract_path=fluent_reference_contract_json,
        active_contract_manifest=active_contract_manifest,
        metrics=metrics,
    )
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, payload["rows"])
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "selected_formulation_fluent_parity_history",
            "source_script": SOURCE_SCRIPT,
            "source_step50_matrix": _repo_relative(SOURCE_STEP50_MATRIX_JSON),
            "source_step50_matrix_sha256": _sha256_file(SOURCE_STEP50_MATRIX_JSON),
            "source_step50_history": _repo_relative(SOURCE_STEP50_HISTORY_JSON),
            "source_step50_history_sha256": _sha256_file(SOURCE_STEP50_HISTORY_JSON),
            "fluent_reference_contract": _repo_relative(
                fluent_reference_contract_json
            ),
            "fluent_reference_contract_sha256": _sha256_file(
                fluent_reference_contract_json
            ),
            "active_fluent_reference_contract_manifest": _repo_relative(
                ACTIVE_CONTRACT_MANIFEST_JSON
            ),
            "active_fluent_reference_contract_manifest_sha256": (
                _active_contract_manifest_sha256()
            ),
            "histories": {
                SCENARIO: {
                    "scenario": SCENARIO,
                    "candidate_status": candidate_status,
                    "fluent_parity_claimed": (
                        candidate_status == "fluent_parity_validated"
                    ),
                    "candidate_blockers": blockers,
                    "parity_metrics": metrics,
                    "source_step50_completed_step_count": int(
                        source_row["completed_step_count"]
                    ),
                }
            },
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def _row(
    *,
    candidate_status: str,
    blockers: list[dict[str, str]],
    source_matrix: Mapping[str, Any],
    source_row: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    reference_contract_path: Path,
    active_contract_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "scenario": SCENARIO,
        "run_status": "blocked"
        if candidate_status != "fluent_parity_validated"
        else "completed",
        "parity_status": candidate_status,
        "candidate_status": candidate_status,
        "fluent_parity_claimed": candidate_status == "fluent_parity_validated",
        "reference_contract_status": _schema_contract_status(
            reference_contract,
            metrics,
        ),
        "source_script": SOURCE_SCRIPT,
        "source_step50_matrix": _repo_relative(SOURCE_STEP50_MATRIX_JSON),
        "source_step50_matrix_sha256": _sha256_file(SOURCE_STEP50_MATRIX_JSON),
        "source_step50_history": _repo_relative(SOURCE_STEP50_HISTORY_JSON),
        "source_step50_history_sha256": _sha256_file(SOURCE_STEP50_HISTORY_JSON),
        "fluent_reference_contract": _repo_relative(reference_contract_path),
        "fluent_reference_contract_sha256": _sha256_file(
            reference_contract_path
        ),
        "active_fluent_reference_contract_manifest": _repo_relative(
            ACTIVE_CONTRACT_MANIFEST_JSON
        ),
        "active_fluent_reference_contract_manifest_sha256": (
            _active_contract_manifest_sha256()
        ),
        "active_contract_status": active_contract_manifest["active_contract_status"],
        "active_contract_promotion_status": active_contract_manifest[
            "promotion_status"
        ],
        "source_step50_candidate_status": source_matrix["candidate_status"],
        "reference_formulation_candidate": source_matrix[
            "reference_formulation_candidate"
        ],
        "pressure_pair_policy_candidate": source_matrix[
            "pressure_pair_policy_candidate"
        ],
        "one_sided_pressure_policy_candidate": source_matrix[
            "one_sided_pressure_policy_candidate"
        ],
        "source_step50_completed_step_count": int(source_row["completed_step_count"]),
        "source_step50_smoke_status": source_row["smoke_status"],
        "selected_anchor_markers_source": source_row["selected_anchor_markers_source"],
        "selected_anchor_markers_source_sha256": source_row[
            "selected_anchor_markers_source_sha256"
        ],
        "pressure_pair_anchor_map_sha256": source_row[
            "pressure_pair_anchor_map_sha256"
        ],
        "displacement_gate_status": metrics["displacement"]["gate_status"],
        "force_gate_status": metrics["force"]["gate_status"],
        "flow_outlet_gate_status": metrics["flow_outlet"]["gate_status"],
        "pressure_gate_status": metrics["pressure"]["gate_status"],
        "metadata_gate_status": metrics["metadata"]["gate_status"],
        "active_blockers": [item["blocker"] for item in blockers],
        "parity_metrics": metrics,
    }


def _payload(
    *,
    candidate_status: str,
    blockers: list[dict[str, str]],
    row: Mapping[str, Any],
    source_matrix: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    reference_contract_path: Path,
    active_contract_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "purpose": "selected_formulation_fluent_parity_matrix",
        "source_script": SOURCE_SCRIPT,
        "scenario_count": 1,
        "candidate_status": candidate_status,
        "fluent_parity_claimed": candidate_status == "fluent_parity_validated",
        "candidate_blockers": blockers,
        "historical_blockers_retired": (
            [BLOCKER_NO_FLUENT_PARITY]
            if candidate_status == "fluent_parity_validated"
            else []
        ),
        "reference_contract_status": _schema_contract_status(
            reference_contract,
            metrics,
        ),
        "source_step50_matrix": _repo_relative(SOURCE_STEP50_MATRIX_JSON),
        "source_step50_matrix_sha256": _sha256_file(SOURCE_STEP50_MATRIX_JSON),
        "source_step50_history": _repo_relative(SOURCE_STEP50_HISTORY_JSON),
        "source_step50_history_sha256": _sha256_file(SOURCE_STEP50_HISTORY_JSON),
        "source_step50_candidate_status": source_matrix["candidate_status"],
        "fluent_reference_contract": _repo_relative(reference_contract_path),
        "fluent_reference_contract_sha256": _sha256_file(
            reference_contract_path
        ),
        "active_fluent_reference_contract_manifest": _repo_relative(
            ACTIVE_CONTRACT_MANIFEST_JSON
        ),
        "active_fluent_reference_contract_manifest_sha256": (
            _active_contract_manifest_sha256()
        ),
        "active_contract_status": active_contract_manifest["active_contract_status"],
        "active_contract_promotion_status": active_contract_manifest[
            "promotion_status"
        ],
        "reference_formulation_candidate": source_matrix[
            "reference_formulation_candidate"
        ],
        "pressure_pair_policy_candidate": source_matrix[
            "pressure_pair_policy_candidate"
        ],
        "one_sided_pressure_policy_candidate": source_matrix[
            "one_sided_pressure_policy_candidate"
        ],
        "parity_metrics": metrics,
        "rows": [dict(row)],
    }


def _parity_metrics(
    *,
    source_matrix: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_history_rows: list[Mapping[str, Any]],
    reference_contract: Mapping[str, Any],
    reference_contract_path: Path = FLUENT_REFERENCE_CONTRACT_JSON,
    contract_schema_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    final_step = source_history_rows[-1] if source_history_rows else {}
    reference_metrics = dict(reference_contract.get("reference_metrics", {}))
    tolerances = dict(reference_contract.get("tolerances", {}))
    if contract_schema_validation is None:
        contract_schema_validation = {
            "contract_status": str(
                reference_contract.get("contract_status", "unknown")
            ),
            "blockers": [],
            "validated_metric_count": None,
            "required_metric_count": None,
            "missing_required_metrics": [],
        }
    tip_displacement = _relative_comparison(
        source_value=_vector_norm_or_numeric(final_step.get("tip_mean_displacement_m")),
        reference_value=_reference_value(reference_metrics, "tip_displacement_m"),
        tolerance=_tolerance_value(tolerances, "tip_displacement_relative"),
    )
    max_displacement = _relative_comparison(
        source_value=_numeric_value(final_step.get("max_displacement_m")),
        reference_value=_reference_value(reference_metrics, "max_displacement_m"),
        tolerance=_tolerance_value(tolerances, "max_displacement_relative"),
    )
    force_z = _relative_comparison(
        source_value=_numeric_value(final_step.get("marker_force_z_N")),
        reference_value=_reference_value(reference_metrics, "force_z_N"),
        tolerance=_tolerance_value(tolerances, "force_z_relative"),
    )
    flow_rate = _relative_comparison(
        source_value=_numeric_value(final_step.get("zmin_velocity_outlet_flux_m3s")),
        reference_value=_reference_value(reference_metrics, "flow_rate_m3s"),
        tolerance=_tolerance_value(tolerances, "flow_rate_relative"),
    )
    pressure_range = _absolute_comparison(
        source_value=_pressure_range(final_step),
        reference_value=_reference_value(reference_metrics, "pressure_range_pa"),
        tolerance=_tolerance_value(tolerances, "pressure_sanity_absolute"),
    )
    return {
        "displacement": {
            "gate_status": _combined_gate_status(tip_displacement, max_displacement),
            "comparison_status": _combined_comparison_status(
                tip_displacement,
                max_displacement,
            ),
            "source_max_displacement_m": source_row["max_displacement_m"],
            "source_step50_max_displacement_m": final_step.get(
                "max_displacement_m"
            ),
            "source_step50_tip_mean_displacement_m": final_step.get(
                "tip_mean_displacement_m"
            ),
            "fluent_tip_displacement_m": _reference_value(
                reference_metrics,
                "tip_displacement_m",
            ),
            "fluent_max_displacement_m": _reference_value(
                reference_metrics,
                "max_displacement_m",
            ),
            "tip_displacement_relative_error": tip_displacement["relative_error"],
            "max_displacement_relative_error": max_displacement["relative_error"],
            "tip_displacement_tolerance": tip_displacement["tolerance"],
            "max_displacement_tolerance": max_displacement["tolerance"],
            "relative_error": _max_present(
                tip_displacement["relative_error"],
                max_displacement["relative_error"],
            ),
            "comparisons": {
                "tip_displacement": tip_displacement,
                "max_displacement": max_displacement,
            },
            "comparison_definition": reference_contract.get(
                "displacement_definition",
                {},
            ),
        },
        "force": {
            "gate_status": force_z["gate_status"],
            "comparison_status": force_z["comparison_status"],
            "source_marker_force_z_history": source_row["marker_force_z_by_step"],
            "source_step50_marker_force_z_N": final_step.get("marker_force_z_N"),
            "source_step50_primary_face_force_z_N": final_step.get(
                "primary_face_force_z_N"
            ),
            "source_step50_secondary_face_force_z_N": final_step.get(
                "secondary_face_force_z_N"
            ),
            "source_force_sign_flip_count": source_row["force_sign_flip_count"],
            "fluent_force_z_N": _reference_value(reference_metrics, "force_z_N"),
            "relative_error": force_z["relative_error"],
            "absolute_error": force_z["absolute_error"],
            "tolerance": force_z["tolerance"],
            "force_sign_matches": _same_sign(
                _numeric_value(final_step.get("marker_force_z_N")),
                _reference_value(reference_metrics, "force_z_N"),
            ),
            "sign_convention": reference_contract.get("sign_conventions", {}).get(
                "force_z_positive",
                "",
            ),
            "comparison": force_z,
        },
        "flow_outlet": {
            "gate_status": flow_rate["gate_status"],
            "comparison_status": flow_rate["comparison_status"],
            "source_fluid_finite": bool(source_row["fluid_finite"]),
            "source_max_velocity_mps": source_row["max_velocity_mps"],
            "source_step50_outlet_flow_rate_m3s": final_step.get(
                "zmin_velocity_outlet_flux_m3s"
            ),
            "fluent_flow_rate_m3s": _reference_value(
                reference_metrics,
                "flow_rate_m3s",
            ),
            "relative_error": flow_rate["relative_error"],
            "absolute_error": flow_rate["absolute_error"],
            "tolerance": flow_rate["tolerance"],
            "flow_sign_matches": _same_sign(
                _numeric_value(final_step.get("zmin_velocity_outlet_flux_m3s")),
                _reference_value(reference_metrics, "flow_rate_m3s"),
            ),
            "sign_convention": reference_contract.get("sign_conventions", {}).get(
                "flow_rate_positive",
                "",
            ),
            "comparison": flow_rate,
        },
        "pressure": {
            "gate_status": pressure_range["gate_status"],
            "comparison_status": pressure_range["comparison_status"],
            "source_pressure_finite": bool(source_row["pressure_finite"]),
            "source_max_pressure_pa": source_row["max_pressure_pa"],
            "source_max_pressure_growth_ratio": source_row[
                "max_pressure_growth_ratio"
            ],
            "source_step50_pressure_min_pa": final_step.get("pressure_min_pa"),
            "source_step50_pressure_max_pa": final_step.get("pressure_max_pa"),
            "source_step50_pressure_range_pa": _pressure_range(final_step),
            "fluent_pressure_range_pa": _reference_value(
                reference_metrics,
                "pressure_range_pa",
            ),
            "absolute_error": pressure_range["absolute_error"],
            "relative_error": pressure_range["relative_error"],
            "tolerance": pressure_range["tolerance"],
            "pressure_reference_convention": reference_contract.get(
                "sign_conventions",
                {},
            ).get("pressure_reference", ""),
            "comparison": pressure_range,
        },
        "metadata": {
            "gate_status": _metadata_gate_status(reference_contract),
            "source_step50_candidate_status": source_matrix["candidate_status"],
            "source_step50_completed_step_count": source_row["completed_step_count"],
            "reference_formulation_candidate": source_matrix[
                "reference_formulation_candidate"
            ],
            "source_step50_matrix_sha256": _sha256_file(SOURCE_STEP50_MATRIX_JSON),
            "fluent_reference_contract_sha256": _sha256_file(
                reference_contract_path
            ),
            "active_fluent_reference_contract_manifest": _repo_relative(
                ACTIVE_CONTRACT_MANIFEST_JSON
            ),
            "active_fluent_reference_contract_manifest_sha256": (
                _active_contract_manifest_sha256()
            ),
            "selected_anchor_markers_source": source_row[
                "selected_anchor_markers_source"
            ],
            "selected_anchor_markers_source_sha256": source_row[
                "selected_anchor_markers_source_sha256"
            ],
            "pressure_pair_anchor_map_sha256": source_row[
                "pressure_pair_anchor_map_sha256"
            ],
            "contract_step_count": reference_contract["step_count"],
            "contract_time_step_s": reference_contract["time_step_s"],
            "contract_simulation": reference_contract.get("simulation", {}),
            "contract_source_provenance": reference_contract.get(
                "source_provenance",
                {},
            ),
            "contract_geometry": reference_contract.get("geometry", {}),
            "contract_material": reference_contract.get("material", {}),
            "contract_displacement_definition": reference_contract.get(
                "displacement_definition",
                {},
            ),
            "contract_sign_conventions": reference_contract.get(
                "sign_conventions",
                {},
            ),
            "contract_schema_validation": dict(contract_schema_validation),
        },
    }


def _candidate_status(
    reference_contract: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    if _schema_contract_status(reference_contract, metrics) != "fluent_reference_complete":
        return "fluent_parity_blocked_reference_incomplete"
    failed = [
        key
        for key, metric in metrics.items()
        if metric.get("gate_status") not in {"passed", "report_only"}
    ]
    if failed:
        return "fluent_parity_failed"
    return "fluent_parity_validated"


def _schema_contract_status(
    reference_contract: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> str:
    if metrics is not None:
        metadata = metrics.get("metadata", {})
        if isinstance(metadata, Mapping):
            validation = metadata.get("contract_schema_validation", {})
            if isinstance(validation, Mapping):
                status = validation.get("contract_status")
                if status:
                    return str(status)
    return str(reference_contract.get("contract_status", "unknown"))


def _candidate_blockers(
    candidate_status: str,
    metrics: Mapping[str, Any],
) -> list[dict[str, str]]:
    if candidate_status == "fluent_parity_validated":
        return []
    if candidate_status == "fluent_parity_blocked_reference_incomplete":
        blockers = [BLOCKER_REFERENCE_INCOMPLETE, BLOCKER_NO_FLUENT_PARITY]
    else:
        blockers = [
            _mismatch_blocker(name)
            for name, metric in metrics.items()
            if metric.get("gate_status") != "passed"
        ]
        blockers.append(BLOCKER_NO_FLUENT_PARITY)
    deduped = list(dict.fromkeys(blockers))
    return [{"blocker": blocker, "detail": _blocker_detail(blocker)} for blocker in deduped]


def _mismatch_blocker(metric_name: str) -> str:
    return {
        "displacement": "fluent_displacement_mismatch",
        "force": "fluent_force_mismatch",
        "flow_outlet": "fluent_flow_mismatch",
        "pressure": "fluent_pressure_mismatch",
        "metadata": "fluent_metadata_mismatch",
    }[metric_name]


def _blocker_detail(blocker: str) -> str:
    details = {
        BLOCKER_REFERENCE_INCOMPLETE: (
            "Fluent reference contract lacks provenance-backed displacement, "
            "force, flow, and pressure targets"
        ),
        BLOCKER_NO_FLUENT_PARITY: (
            "Fluent parity remains unclaimed until reference-backed metrics pass"
        ),
        "fluent_displacement_mismatch": "displacement parity gate failed",
        "fluent_force_mismatch": "force parity gate failed",
        "fluent_flow_mismatch": "flow/outlet parity gate failed",
        "fluent_pressure_mismatch": "pressure parity gate failed",
        "fluent_metadata_mismatch": "metadata provenance parity gate failed",
    }
    return details[blocker]


def _gate_status(reference_metrics: Mapping[str, Any], metric_name: str) -> str:
    if _reference_value(reference_metrics, metric_name) is None:
        return "blocked_reference_missing"
    return "pending_comparison"


def _reference_value(reference_metrics: Mapping[str, Any], metric_name: str) -> Any:
    metric = reference_metrics.get(metric_name, {})
    if not isinstance(metric, Mapping) or metric.get("status") == "missing":
        return None
    return _numeric_value(metric.get("value"))


def _tolerance_value(tolerances: Mapping[str, Any], tolerance_name: str) -> float | None:
    tolerance = tolerances.get(tolerance_name, {})
    if not isinstance(tolerance, Mapping) or tolerance.get("status") != "available":
        return None
    return _numeric_value(tolerance.get("value"))


def _relative_comparison(
    *,
    source_value: float | None,
    reference_value: float | None,
    tolerance: float | None,
) -> dict[str, Any]:
    comparison = _comparison_values(
        source_value=source_value,
        reference_value=reference_value,
        tolerance=tolerance,
    )
    if comparison["gate_status"] != "pending":
        return comparison
    comparison["gate_status"] = (
        "passed"
        if comparison["relative_error"] <= comparison["tolerance"]
        else "failed"
    )
    comparison["comparison_status"] = "compared"
    return comparison


def _absolute_comparison(
    *,
    source_value: float | None,
    reference_value: float | None,
    tolerance: float | None,
) -> dict[str, Any]:
    comparison = _comparison_values(
        source_value=source_value,
        reference_value=reference_value,
        tolerance=tolerance,
    )
    if comparison["gate_status"] != "pending":
        return comparison
    comparison["gate_status"] = (
        "passed"
        if comparison["absolute_error"] <= comparison["tolerance"]
        else "failed"
    )
    comparison["comparison_status"] = "compared"
    return comparison


def _comparison_values(
    *,
    source_value: float | None,
    reference_value: float | None,
    tolerance: float | None,
) -> dict[str, Any]:
    result = {
        "source_value": source_value,
        "fluent_reference_value": reference_value,
        "tolerance": tolerance,
        "absolute_error": None,
        "relative_error": None,
        "gate_status": "pending",
        "comparison_status": "pending",
    }
    if reference_value is None:
        result["gate_status"] = "blocked_reference_missing"
        result["comparison_status"] = "blocked_reference_missing"
        return result
    if source_value is None:
        result["gate_status"] = "blocked_source_missing"
        result["comparison_status"] = "blocked_source_missing"
        return result
    if tolerance is None:
        result["gate_status"] = "blocked_tolerance_missing"
        result["comparison_status"] = "blocked_tolerance_missing"
        return result
    absolute_error = abs(source_value - reference_value)
    denominator = max(abs(reference_value), RELATIVE_ERROR_DENOMINATOR_FLOOR)
    result["absolute_error"] = absolute_error
    result["relative_error"] = absolute_error / denominator
    return result


def _combined_gate_status(*comparisons: Mapping[str, Any]) -> str:
    statuses = [str(comparison["gate_status"]) for comparison in comparisons]
    if any(status == "failed" for status in statuses):
        return "failed"
    for blocked in (
        "blocked_reference_missing",
        "blocked_source_missing",
        "blocked_tolerance_missing",
    ):
        if any(status == blocked for status in statuses):
            return blocked
    if all(status == "passed" for status in statuses):
        return "passed"
    return "pending_comparison"


def _combined_comparison_status(*comparisons: Mapping[str, Any]) -> str:
    statuses = [str(comparison["comparison_status"]) for comparison in comparisons]
    if all(status == "compared" for status in statuses):
        return "compared"
    for blocked in (
        "blocked_reference_missing",
        "blocked_source_missing",
        "blocked_tolerance_missing",
    ):
        if any(status == blocked for status in statuses):
            return blocked
    return "pending"


def _pressure_range(final_step: Mapping[str, Any]) -> float | None:
    pressure_min = _numeric_value(final_step.get("pressure_min_pa"))
    pressure_max = _numeric_value(final_step.get("pressure_max_pa"))
    if pressure_min is None or pressure_max is None:
        return None
    return pressure_max - pressure_min


def _same_sign(lhs: float | None, rhs: float | None) -> bool | None:
    if lhs is None or rhs is None:
        return None
    if abs(lhs) <= RELATIVE_ERROR_DENOMINATOR_FLOOR:
        return abs(rhs) <= RELATIVE_ERROR_DENOMINATOR_FLOOR
    if abs(rhs) <= RELATIVE_ERROR_DENOMINATOR_FLOOR:
        return abs(lhs) <= RELATIVE_ERROR_DENOMINATOR_FLOOR
    return lhs * rhs > 0.0


def _metadata_gate_status(reference_contract: Mapping[str, Any]) -> str:
    if reference_contract.get("contract_status") != "fluent_reference_complete":
        return "passed"
    simulation = reference_contract.get("simulation", {})
    provenance = reference_contract.get("source_provenance", {})
    displacement_definition = reference_contract.get("displacement_definition", {})
    sign_conventions = reference_contract.get("sign_conventions", {})
    step_count = _numeric_value(simulation.get("step_count"))
    time_step_s = _numeric_value(simulation.get("time_step_s"))
    total_time_s = _numeric_value(simulation.get("total_time_s"))
    checks = [
        step_count == 50,
        time_step_s is not None and abs(time_step_s - 0.0005) <= 1.0e-12,
        total_time_s is not None and abs(total_time_s - 0.025) <= 1.0e-12,
        provenance.get("status") == "complete",
        displacement_definition.get("status") == "complete",
        sign_conventions.get("status") == "complete",
    ]
    return "passed" if all(checks) else "failed"


def _max_present(*values: Any) -> float | None:
    numeric_values = [
        numeric
        for numeric in (_numeric_value(value) for value in values)
        if numeric is not None
    ]
    return max(numeric_values) if numeric_values else None


def _numeric_value(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _vector_norm_or_numeric(value: Any) -> float | None:
    numeric = _numeric_value(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, (list, tuple)):
        return None
    components = [_numeric_value(component) for component in value]
    if any(component is None for component in components):
        return None
    return math.sqrt(sum(float(component) ** 2 for component in components))


def _source_step50_row(source_matrix: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in source_matrix["rows"]:
        if row["scenario"] == SOURCE_STEP50_SCENARIO:
            return row
    raise ValueError(f"missing {SOURCE_STEP50_SCENARIO} row")


def _source_step50_history_rows(source_history: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(source_history["histories"][SOURCE_STEP50_SCENARIO]["history"])


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    blockers = ", ".join(item["blocker"] for item in payload["candidate_blockers"])
    metrics = payload["parity_metrics"]
    lines = [
        "# ANSYS vertical-flap selected-formulation Fluent parity",
        "",
        "## Scope",
        "",
        (
            "This artifact compares committed selected-formulation step50 evidence "
            "against the explicit Fluent reference contract. It does not claim "
            "Fluent parity while the reference contract is incomplete."
        ),
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload['candidate_status']}`",
        f"- active_blockers: `{blockers}`",
        (
            "- source_step50_candidate_status: "
            f"`{payload['source_step50_candidate_status']}`"
        ),
        (
            "- reference_contract_status: "
            f"`{payload['reference_contract_status']}`"
        ),
        "",
        "## Metric gates",
        "",
        "metric | gate status",
        "--- | ---",
    ]
    for name in ("displacement", "force", "flow_outlet", "pressure", "metadata"):
        lines.append(f"{name} | {metrics[name]['gate_status']}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Matrix JSON: `{_repo_relative(MATRIX_JSON)}`",
            f"- Matrix CSV: `{_repo_relative(MATRIX_CSV)}`",
            f"- History JSON: `{_repo_relative(HISTORY_JSON)}`",
            f"- Scenario diagnostics: `{_repo_relative(SCENARIO_DIAGNOSTICS_DIR)}`",
            f"- Checksums: `{_repo_relative(CHECKSUMS_PATH)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_contract_manifest() -> dict[str, Any]:
    if not ACTIVE_CONTRACT_MANIFEST_JSON.exists():
        return {
            "case": CASE_NAME,
            "purpose": "active_fluent_reference_contract_manifest",
            "active_contract": _repo_relative(FLUENT_REFERENCE_CONTRACT_JSON),
            "active_contract_sha256": _sha256_file(FLUENT_REFERENCE_CONTRACT_JSON),
            "active_contract_status": "unknown_manifest_missing",
            "promotion_status": "manifest_missing_fallback",
            "recommended_action": "use_default_incomplete_contract",
            "promotion_blockers": [
                {
                    "blocker": "active_contract_manifest_missing",
                    "detail": "Falling back to the default incomplete contract",
                }
            ],
            "no_fluent_parity_claim_retired": False,
        }
    manifest = _read_json(ACTIVE_CONTRACT_MANIFEST_JSON)
    _validate_repo_relative_path(str(manifest["active_contract"]))
    return manifest


def _active_reference_contract_path(manifest: Mapping[str, Any]) -> Path:
    path_text = str(manifest.get("active_contract", ""))
    _validate_repo_relative_path(path_text)
    return Path(path_text)


def _active_contract_manifest_sha256() -> str:
    if not ACTIVE_CONTRACT_MANIFEST_JSON.exists():
        return ""
    return _sha256_file(ACTIVE_CONTRACT_MANIFEST_JSON)


def _validate_repo_relative_path(path_text: str) -> None:
    path = Path(path_text)
    if path.is_absolute() or "\\" in path_text or path_text.startswith(".."):
        raise ValueError(f"active Fluent reference contract path is not repo-relative: {path_text}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _csv_value(row.get(column, "")) for column in CSV_COLUMNS}
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != CHECKSUMS_PATH
    )
    lines = []
    for path in files:
        lines.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    SCENARIO_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _repo_relative(path: Path | str) -> str:
    return Path(path).as_posix()


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[traction_selected_formulation_fluent_parity] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_selected_formulation_fluent_parity] wrote "
        f"{payload['candidate_status']} to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
