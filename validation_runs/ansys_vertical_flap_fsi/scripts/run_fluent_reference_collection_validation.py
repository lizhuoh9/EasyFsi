from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fluent_reference_contract_schema import (
    EXPECTED_METRIC_UNITS,
    validate_fluent_reference_contract,
)
from fluent_source_export_schema import validate_source_export_csv


CASE_NAME = "ansys_vertical_flap_fsi"
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
REFERENCE_ROOT = ROOT / "fluent_reference"
SOURCE_EXPORTS_ROOT = REFERENCE_ROOT / "source_exports"
PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON = (
    SOURCE_EXPORTS_ROOT / "public_tutorial_evidence_map.json"
)
OUTPUT_DIR = REFERENCE_ROOT / "validation_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "fluent_reference_collection_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "fluent_reference_collection_matrix.csv"
SUMMARY_MD = OUTPUT_DIR / "fluent_reference_collection_summary.md"
CANDIDATE_CONTRACT_JSON = OUTPUT_DIR / "fluent_reference_collection_candidate_contract.json"
ARTIFACT_MANIFEST_JSON = OUTPUT_DIR / "ARTIFACT_MANIFEST.json"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"
CURRENT_CONTRACT_JSON = REFERENCE_ROOT / "fluent_reference_contract_2026-06-27.json"
ACTIVE_CONTRACT_MANIFEST_JSON = (
    REFERENCE_ROOT / "active_fluent_reference_contract.json"
)
ACTIVE_MANIFEST_SCHEMA_VERSION = "active_fluent_reference_contract_manifest_v1"
REAL_FLUENT_IMPORT_GATE_SCHEMA_VERSION = (
    "ansys_vertical_flap_real_fluent_import_gate_v1"
)
ALLOW_TEST_SOURCES = False

SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_fluent_reference_collection_validation.py"
)
DEFAULT_GENERATED_FROM_COMMIT = "c94332888fe09d792a119086a4969f78b03bb134"
DEFAULT_GENERATED_FROM_REF = (
    "solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25"
)
DEFAULT_ARTIFACT_COMMITTED_IN_REVIEW_HEAD = (
    "25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9"
)
GENERATED_FROM_COMMIT_ENV = "EASYFSI_VALIDATION_COMMIT"
GENERATED_FROM_REF_ENV = "EASYFSI_VALIDATION_REF"
ARTIFACT_COMMITTED_IN_REVIEW_HEAD_ENV = "EASYFSI_VALIDATION_REVIEW_HEAD"

EXPECTED_STEP_COUNT = 50
EXPECTED_TIME_STEP_S = 0.0005
EXPECTED_TOTAL_TIME_S = 0.025

STATUS_PENDING = "fluent_reference_collection_pending"
STATUS_COMPLETE = "fluent_reference_collection_complete"
CONTRACT_INCOMPLETE = "fluent_reference_incomplete"
CONTRACT_COMPLETE = "fluent_reference_complete"

BLOCKER_DISPLACEMENT = "fluent_displacement_reference_missing"
BLOCKER_FORCE = "fluent_force_reference_missing"
BLOCKER_FLOW = "fluent_flow_reference_missing"
BLOCKER_PRESSURE = "fluent_pressure_reference_missing"
BLOCKER_PROVENANCE = "fluent_reference_provenance_incomplete"
BLOCKER_METADATA_DISALLOWED_PROVENANCE = (
    "fluent_reference_metadata_disallowed_provenance"
)
BLOCKER_TOLERANCES = "fluent_reference_tolerances_incomplete"

MISSING_VALUES = {"", "missing", "todo", "tbd", "n/a", "na", "null", "none"}
DISALLOWED_METADATA_PROVENANCE_TERMS = (
    "easyfsi",
    "hibm-mpm",
    "synthetic",
    "fixture",
    "placeholder",
    "not fluent truth",
    "validation_runs",
    "not_collected",
    "public tutorial",
    "web tutorial",
    "tutorial page",
    "web contour",
    "official web baseline",
)

METRIC_SPECS = [
    {
        "artifact": "fluent_tip_displacement_history.csv",
        "metric_group": "displacement",
        "required_columns": [
            "step",
            "time_s",
            "tip_displacement_x_m",
            "tip_displacement_y_m",
            "tip_displacement_z_m",
            "tip_displacement_norm_m",
            "max_displacement_m",
            "source",
        ],
        "reference_values": {
            "tip_displacement_m": "tip_displacement_norm_m",
            "max_displacement_m": "max_displacement_m",
        },
        "blocker": BLOCKER_DISPLACEMENT,
    },
    {
        "artifact": "fluent_force_history.csv",
        "metric_group": "force",
        "required_columns": [
            "step",
            "time_s",
            "force_x_N",
            "force_y_N",
            "force_z_N",
            "primary_force_z_N",
            "secondary_force_z_N",
            "source",
        ],
        "reference_values": {
            "force_z_N": "force_z_N",
        },
        "blocker": BLOCKER_FORCE,
    },
    {
        "artifact": "fluent_flow_balance_history.csv",
        "metric_group": "flow",
        "required_columns": [
            "step",
            "time_s",
            "inlet_flow_rate_m3s",
            "outlet_flow_rate_m3s",
            "pressure_outlet_flux_m3s",
            "velocity_outlet_flux_m3s",
            "source",
        ],
        "reference_values": {
            "flow_rate_m3s": "outlet_flow_rate_m3s",
        },
        "blocker": BLOCKER_FLOW,
    },
    {
        "artifact": "fluent_pressure_summary_history.csv",
        "metric_group": "pressure",
        "required_columns": [
            "step",
            "time_s",
            "pressure_min_pa",
            "pressure_max_pa",
            "pressure_range_pa",
            "source",
        ],
        "reference_values": {
            "pressure_range_pa": "pressure_range_pa",
        },
        "blocker": BLOCKER_PRESSURE,
    },
]

REQUIRED_METADATA_FIELDS = [
    "Source document",
    "Fluent run id",
    "Export author",
    "Export date",
    "Fluent version",
    "mesh/domain source",
    "geometry units",
    "material model",
    "boundary conditions",
    "time step",
    "number of steps",
    "coupling settings if applicable",
    "export procedure",
    "who/when/how generated",
    "force_z_positive",
    "flow_rate_positive",
    "pressure_reference",
    "displacement_definition",
]

CSV_COLUMNS = [
    "artifact",
    "metric_group",
    "source_path",
    "file_status",
    "header_status",
    "final_step_status",
    "metric_status",
    "blocker",
]


def run() -> dict[str, Any]:
    return _run()


def run_with_paths(
    *,
    source_exports_root: Path,
    current_contract_json: Path,
    output_dir: Path,
    active_manifest_json: Path,
    allow_test_sources: bool = False,
) -> dict[str, Any]:
    original = {
        "SOURCE_EXPORTS_ROOT": SOURCE_EXPORTS_ROOT,
        "PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON": PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MATRIX_JSON": MATRIX_JSON,
        "MATRIX_CSV": MATRIX_CSV,
        "SUMMARY_MD": SUMMARY_MD,
        "CANDIDATE_CONTRACT_JSON": CANDIDATE_CONTRACT_JSON,
        "ARTIFACT_MANIFEST_JSON": ARTIFACT_MANIFEST_JSON,
        "CHECKSUMS_PATH": CHECKSUMS_PATH,
        "CURRENT_CONTRACT_JSON": CURRENT_CONTRACT_JSON,
        "ACTIVE_CONTRACT_MANIFEST_JSON": ACTIVE_CONTRACT_MANIFEST_JSON,
        "ALLOW_TEST_SOURCES": ALLOW_TEST_SOURCES,
    }
    try:
        _set_paths(
            source_exports_root=source_exports_root,
            current_contract_json=current_contract_json,
            output_dir=output_dir,
            active_manifest_json=active_manifest_json,
            allow_test_sources=allow_test_sources,
        )
        return _run()
    finally:
        globals().update(original)


def _set_paths(
    *,
    source_exports_root: Path,
    current_contract_json: Path,
    output_dir: Path,
    active_manifest_json: Path,
    allow_test_sources: bool,
) -> None:
    globals().update(
        {
            "SOURCE_EXPORTS_ROOT": source_exports_root,
            "PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON": (
                source_exports_root / "public_tutorial_evidence_map.json"
            ),
            "OUTPUT_DIR": output_dir,
            "MATRIX_JSON": output_dir / "fluent_reference_collection_matrix.json",
            "MATRIX_CSV": output_dir / "fluent_reference_collection_matrix.csv",
            "SUMMARY_MD": output_dir / "fluent_reference_collection_summary.md",
            "CANDIDATE_CONTRACT_JSON": (
                output_dir / "fluent_reference_collection_candidate_contract.json"
            ),
            "ARTIFACT_MANIFEST_JSON": output_dir / "ARTIFACT_MANIFEST.json",
            "CHECKSUMS_PATH": output_dir / "CHECKSUMS.sha256",
            "CURRENT_CONTRACT_JSON": current_contract_json,
            "ACTIVE_CONTRACT_MANIFEST_JSON": active_manifest_json,
            "ALLOW_TEST_SOURCES": allow_test_sources,
        }
    )


def _run() -> dict[str, Any]:
    _prepare_output_dir()
    current_contract = _read_json(CURRENT_CONTRACT_JSON)
    source_checks = [_source_check(spec) for spec in METRIC_SPECS]
    metadata_check = _metadata_check()
    candidate_contract = _candidate_contract(
        current_contract=current_contract,
        source_checks=source_checks,
        metadata_check=metadata_check,
    )
    _write_json(CANDIDATE_CONTRACT_JSON, candidate_contract)
    active_manifest = _active_contract_manifest(
        current_contract=current_contract,
        candidate_contract=candidate_contract,
        source_checks=source_checks,
        metadata_check=metadata_check,
    )
    _write_json(ACTIVE_CONTRACT_MANIFEST_JSON, active_manifest)

    blockers = _candidate_blockers(
        candidate_contract=candidate_contract,
        source_checks=source_checks,
        metadata_check=metadata_check,
    )
    candidate_status = (
        STATUS_COMPLETE
        if candidate_contract["contract_status"] == CONTRACT_COMPLETE
        else STATUS_PENDING
    )
    rows = _csv_rows(source_checks=source_checks, metadata_check=metadata_check)
    payload = _payload(
        candidate_status=candidate_status,
        blockers=blockers,
        source_checks=source_checks,
        metadata_check=metadata_check,
        candidate_contract=candidate_contract,
        active_manifest=active_manifest,
        rows=rows,
    )
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_json(ARTIFACT_MANIFEST_JSON, _artifact_manifest(payload))
    _write_checksums(OUTPUT_DIR)
    return payload


def _source_check(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = SOURCE_EXPORTS_ROOT / str(spec["artifact"])
    required_columns = list(spec["required_columns"])
    validation = validate_source_export_csv(
        path,
        required_columns,
        required_final_step=EXPECTED_STEP_COUNT,
        expected_final_time=EXPECTED_TOTAL_TIME_S,
        reference_value_columns=spec["reference_values"],
        allow_test_sources=ALLOW_TEST_SOURCES,
    )
    values_complete = validation["metric_status"] == "available"

    return {
        "artifact": spec["artifact"],
        "metric_group": spec["metric_group"],
        "source_path": _repo_relative(path),
        "exists": bool(validation["exists"]),
        "file_status": validation["file_status"],
        "header_status": validation["header_status"],
        "final_step_status": validation["final_step_status"],
        "metric_status": validation["metric_status"],
        "blocker": None if values_complete else spec["blocker"],
        "required_columns": required_columns,
        "observed_columns": validation["observed_columns"],
        "reference_values": validation["reference_values"] if values_complete else {},
        "row_count": validation["row_count"],
        "final_time_s": validation["final_time_s"],
        "schema_blockers": validation["blockers"],
    }


def _metadata_check() -> dict[str, Any]:
    path = SOURCE_EXPORTS_ROOT / "fluent_metadata_2026-06-28.md"
    if not path.exists():
        return {
            "artifact": path.name,
            "source_path": _repo_relative(path),
            "file_status": "missing_file",
            "provenance_status": "incomplete",
            "blocker": BLOCKER_PROVENANCE,
            "required_fields": REQUIRED_METADATA_FIELDS,
            "observed_fields": {},
            "missing_fields": REQUIRED_METADATA_FIELDS,
            "semantic_mismatches": [],
            "disallowed_provenance": [],
            "source_provenance": _missing_source_provenance(),
        }

    observed = _metadata_fields(path.read_text(encoding="utf-8"))
    missing = [
        field
        for field in REQUIRED_METADATA_FIELDS
        if _is_missing(observed.get(_normalize_field(field), ""))
    ]
    semantic_mismatches = _metadata_semantic_mismatches(observed, missing)
    disallowed_provenance = (
        [] if ALLOW_TEST_SOURCES else _metadata_disallowed_provenance(observed)
    )
    complete = not missing and not semantic_mismatches and not disallowed_provenance
    blocker = _metadata_blocker(
        complete=complete,
        disallowed_provenance=disallowed_provenance,
    )
    return {
        "artifact": path.name,
        "source_path": _repo_relative(path),
        "file_status": "present_complete" if complete else "present_incomplete",
        "provenance_status": "complete" if complete else "incomplete",
        "blocker": blocker,
        "required_fields": REQUIRED_METADATA_FIELDS,
        "observed_fields": observed,
        "missing_fields": missing,
        "semantic_mismatches": semantic_mismatches,
        "disallowed_provenance": disallowed_provenance,
        "source_provenance": (
            {
                "document": observed[_normalize_field("Source document")],
                "run_id": observed[_normalize_field("Fluent run id")],
                "author": observed[_normalize_field("Export author")],
                "date": observed[_normalize_field("Export date")],
                "status": "complete",
            }
            if complete
            else _missing_source_provenance()
        ),
    }


def _candidate_contract(
    *,
    current_contract: Mapping[str, Any],
    source_checks: Iterable[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
) -> dict[str, Any]:
    contract = json.loads(json.dumps(current_contract))
    reference_metrics = {
        metric: {
            "status": "missing",
            "value": None,
            "source": "not_collected",
            "unit": EXPECTED_METRIC_UNITS[metric],
            "extraction_method": "not_collected",
            "time_s": EXPECTED_TOTAL_TIME_S,
        }
        for metric in (
            "tip_displacement_m",
            "max_displacement_m",
            "force_z_N",
            "flow_rate_m3s",
            "pressure_range_pa",
        )
    }
    for check in source_checks:
        for metric, value in check["reference_values"].items():
            reference_metrics[metric] = {
                "status": "available",
                "value": value,
                "unit": EXPECTED_METRIC_UNITS[metric],
                "source": check["source_path"],
                "extraction_method": "fluent_source_export_csv",
                "time_s": check.get("final_time_s", EXPECTED_TOTAL_TIME_S),
            }

    missing_metrics = [
        metric
        for metric, payload in reference_metrics.items()
        if payload["status"] != "available"
    ]
    comparison_metadata = _comparison_metadata(metadata_check)
    tolerances_complete = _tolerances_complete(contract.get("tolerances", {}))
    provenance_complete = metadata_check["provenance_status"] == "complete"
    contract_complete = (
        not missing_metrics
        and provenance_complete
        and tolerances_complete
        and comparison_metadata["status"] == "complete"
    )

    contract["source"] = {
        "description": (
            "Candidate Fluent reference values generated from committed source "
            "exports by the collection validator."
        ),
        "source_type": "fluent_reference_collection_validator",
        "provenance": metadata_check["provenance_status"],
    }
    contract["source_provenance"] = metadata_check["source_provenance"]
    contract["provenance_status"] = metadata_check["provenance_status"]
    contract["reference_metrics"] = reference_metrics
    contract["displacement_definition"] = comparison_metadata[
        "displacement_definition"
    ]
    contract["sign_conventions"] = comparison_metadata["sign_conventions"]
    contract["missing_reference_metrics"] = missing_metrics
    contract["contract_status"] = (
        CONTRACT_COMPLETE if contract_complete else CONTRACT_INCOMPLETE
    )
    contract["active_contract_recommendation"] = _active_contract_recommendation(
        contract_complete=contract_complete,
        missing_metrics=missing_metrics,
        metadata_check=metadata_check,
        comparison_metadata=comparison_metadata,
        tolerances_complete=tolerances_complete,
    )
    contract["collection_validator"] = {
        "source_script": SOURCE_SCRIPT,
        "source_exports_root": _repo_relative(SOURCE_EXPORTS_ROOT),
        "current_contract": _repo_relative(CURRENT_CONTRACT_JSON),
        "current_contract_sha256": _sha256_file(CURRENT_CONTRACT_JSON),
        "tolerances_complete": tolerances_complete,
        "comparison_metadata_complete": comparison_metadata["status"] == "complete",
    }
    schema_validation = validate_fluent_reference_contract(contract)
    contract["schema_validation"] = schema_validation
    contract["missing_reference_metrics"] = schema_validation[
        "missing_required_metrics"
    ]
    contract["contract_status"] = schema_validation["contract_status"]
    return contract


def _candidate_blockers(
    *,
    candidate_contract: Mapping[str, Any],
    source_checks: Iterable[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
) -> list[dict[str, str]]:
    if candidate_contract["contract_status"] == CONTRACT_COMPLETE:
        return []

    blockers = [
        str(check["blocker"])
        for check in source_checks
        if check.get("blocker")
    ]
    if metadata_check.get("blocker"):
        blockers.append(str(metadata_check["blocker"]))
    if (
        not candidate_contract["missing_reference_metrics"]
        and metadata_check["provenance_status"] == "complete"
        and not candidate_contract["collection_validator"]["tolerances_complete"]
    ):
        blockers.append(BLOCKER_TOLERANCES)
    return [
        {"blocker": blocker, "detail": _blocker_detail(blocker)}
        for blocker in dict.fromkeys(blockers)
    ]


def _active_contract_manifest(
    *,
    current_contract: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    source_checks: Iterable[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
) -> dict[str, Any]:
    promotion_blockers = _promotion_blockers(
        candidate_contract=candidate_contract,
        source_checks=source_checks,
        metadata_check=metadata_check,
    )
    ready = candidate_contract["contract_status"] == CONTRACT_COMPLETE
    return {
        "case": CASE_NAME,
        "purpose": "active_fluent_reference_contract_manifest",
        "manifest_schema_version": ACTIVE_MANIFEST_SCHEMA_VERSION,
        "source_script": SOURCE_SCRIPT,
        "active_contract": _repo_relative(CURRENT_CONTRACT_JSON),
        "active_contract_sha256": _sha256_file(CURRENT_CONTRACT_JSON),
        "active_contract_status": str(current_contract["contract_status"]),
        "active_contract_schema_validation": validate_fluent_reference_contract(
            current_contract
        ),
        "candidate_contract": _repo_relative(CANDIDATE_CONTRACT_JSON),
        "candidate_contract_sha256": _sha256_file(CANDIDATE_CONTRACT_JSON),
        "candidate_contract_status": str(candidate_contract["contract_status"]),
        "candidate_contract_schema_validation": candidate_contract[
            "schema_validation"
        ],
        "promotion_status": (
            "ready_for_versioned_contract_promotion"
            if ready
            else "blocked_reference_incomplete"
        ),
        "recommended_action": (
            "promote_versioned_contract"
            if ready
            else "keep_current_incomplete_contract"
        ),
        "promotion_blockers": promotion_blockers,
        "no_fluent_parity_claim_retired": False,
    }


def _promotion_blockers(
    *,
    candidate_contract: Mapping[str, Any],
    source_checks: Iterable[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
) -> list[dict[str, str]]:
    if candidate_contract["contract_status"] == CONTRACT_COMPLETE:
        return []
    blockers = [
        str(check["blocker"])
        for check in source_checks
        if check.get("blocker")
    ]
    if metadata_check.get("blocker"):
        blockers.append(str(metadata_check["blocker"]))
    if not candidate_contract["collection_validator"]["comparison_metadata_complete"]:
        blockers.append("fluent_reference_comparison_metadata_incomplete")
    if not candidate_contract["collection_validator"]["tolerances_complete"]:
        blockers.append(BLOCKER_TOLERANCES)
    return [
        {"blocker": blocker, "detail": _blocker_detail(blocker)}
        for blocker in dict.fromkeys(blockers)
    ]


def _payload(
    *,
    candidate_status: str,
    blockers: list[dict[str, str]],
    source_checks: list[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    active_manifest: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "purpose": "fluent_reference_collection_matrix",
        "source_script": SOURCE_SCRIPT,
        "source_exports_root": _repo_relative(SOURCE_EXPORTS_ROOT),
        "public_reference_evidence": _repo_relative(
            PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON
        ),
        "public_reference_evidence_sha256": _sha256_file(
            PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON
        ),
        "public_reference_use_policy": _public_reference_use_policy(),
        "current_reference_contract": _repo_relative(CURRENT_CONTRACT_JSON),
        "current_reference_contract_sha256": _sha256_file(CURRENT_CONTRACT_JSON),
        "candidate_contract": _repo_relative(CANDIDATE_CONTRACT_JSON),
        "candidate_contract_sha256": _sha256_file(CANDIDATE_CONTRACT_JSON),
        "active_fluent_reference_contract_manifest": _repo_relative(
            ACTIVE_CONTRACT_MANIFEST_JSON
        ),
        "active_fluent_reference_contract_manifest_sha256": _sha256_file(
            ACTIVE_CONTRACT_MANIFEST_JSON
        ),
        "active_contract": active_manifest["active_contract"],
        "active_contract_sha256": active_manifest["active_contract_sha256"],
        "promotion_status": active_manifest["promotion_status"],
        "recommended_action": active_manifest["recommended_action"],
        "promotion_blockers": active_manifest["promotion_blockers"],
        "real_fluent_import_gate": _real_fluent_import_gate(
            source_checks=source_checks,
            metadata_check=metadata_check,
            candidate_contract=candidate_contract,
            active_manifest=active_manifest,
        ),
        "candidate_status": candidate_status,
        "candidate_contract_status": candidate_contract["contract_status"],
        "candidate_blockers": blockers,
        "expected_step_count": EXPECTED_STEP_COUNT,
        "expected_time_step_s": EXPECTED_TIME_STEP_S,
        "expected_total_time_s": EXPECTED_TOTAL_TIME_S,
        "source_checks": source_checks,
        "metadata_check": metadata_check,
        "missing_reference_metrics": candidate_contract["missing_reference_metrics"],
        "reference_metrics": candidate_contract["reference_metrics"],
        "schema_validation": candidate_contract["schema_validation"],
        "rows": rows,
    }


def _real_fluent_import_gate(
    *,
    source_checks: list[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    active_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_exports = [
        _real_fluent_source_export_gate_row(check) for check in source_checks
    ]
    metadata_gate = _real_fluent_metadata_gate(metadata_check)
    blockers = _real_fluent_import_blockers(
        source_exports=source_exports,
        metadata_gate=metadata_gate,
        candidate_contract=candidate_contract,
        active_manifest=active_manifest,
    )
    ready = not blockers
    return {
        "gate_schema_version": REAL_FLUENT_IMPORT_GATE_SCHEMA_VERSION,
        "status": (
            "ready_for_real_fluent_import"
            if ready
            else "blocked_real_fluent_import_incomplete"
        ),
        "can_import_real_fluent_reference": ready,
        "can_run_solver_evaluation": ready,
        "fluent_parity_claimed": False,
        "source_exports": source_exports,
        "metadata": metadata_gate,
        "candidate_contract_status": str(candidate_contract["contract_status"]),
        "promotion_status": str(active_manifest["promotion_status"]),
        "blockers": blockers,
    }


def _real_fluent_source_export_gate_row(
    check: Mapping[str, Any],
) -> dict[str, Any]:
    schema_blockers = [str(blocker) for blocker in check.get("schema_blockers", [])]
    ready = check.get("metric_status") == "available" and not schema_blockers
    return {
        "artifact": str(check["artifact"]),
        "metric_group": str(check["metric_group"]),
        "source_path": str(check["source_path"]),
        "file_status": str(check["file_status"]),
        "final_step_status": str(check["final_step_status"]),
        "metric_status": str(check["metric_status"]),
        "schema_blockers": schema_blockers,
        "ready": ready,
    }


def _real_fluent_metadata_gate(
    metadata_check: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_mismatches = list(metadata_check.get("semantic_mismatches", []))
    ready = (
        metadata_check.get("provenance_status") == "complete"
        and not semantic_mismatches
    )
    return {
        "artifact": str(metadata_check["artifact"]),
        "source_path": str(metadata_check["source_path"]),
        "file_status": str(metadata_check["file_status"]),
        "provenance_status": str(metadata_check["provenance_status"]),
        "blocker": metadata_check.get("blocker"),
        "missing_fields": list(metadata_check.get("missing_fields", [])),
        "semantic_mismatches": semantic_mismatches,
        "disallowed_provenance": list(
            metadata_check.get("disallowed_provenance", [])
        ),
        "ready": ready,
    }


def _real_fluent_import_blockers(
    *,
    source_exports: list[Mapping[str, Any]],
    metadata_gate: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    active_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    for source_export in source_exports:
        if source_export["ready"]:
            continue
        blockers.append(
            {
                "blocker": "source_export_not_ready",
                "artifact": str(source_export["artifact"]),
                "detail": _source_export_gate_detail(source_export),
            }
        )

    if not metadata_gate["ready"]:
        blockers.append(
            {
                "blocker": "metadata_not_ready",
                "artifact": str(metadata_gate["artifact"]),
                "detail": _metadata_gate_detail(metadata_gate),
            }
        )

    if ALLOW_TEST_SOURCES:
        blockers.append(
            {
                "blocker": "test_source_allowance_enabled",
                "artifact": "source_exports",
                "detail": (
                    "allow_test_sources=True is reserved for temp-only "
                    "comparison mechanics"
                ),
            }
        )

    if candidate_contract["contract_status"] != CONTRACT_COMPLETE:
        blockers.append(
            {
                "blocker": "candidate_contract_incomplete",
                "artifact": _repo_relative(CANDIDATE_CONTRACT_JSON),
                "detail": (
                    "candidate contract is "
                    f"{candidate_contract['contract_status']}"
                ),
            }
        )

    if active_manifest["promotion_status"] != "ready_for_versioned_contract_promotion":
        blockers.append(
            {
                "blocker": "active_manifest_promotion_blocked",
                "artifact": _repo_relative(ACTIVE_CONTRACT_MANIFEST_JSON),
                "detail": (
                    "active manifest promotion is "
                    f"{active_manifest['promotion_status']}"
                ),
            }
        )
    return blockers


def _source_export_gate_detail(source_export: Mapping[str, Any]) -> str:
    schema_blockers = ",".join(source_export.get("schema_blockers", []))
    return (
        f"file_status={source_export['file_status']}; "
        f"final_step_status={source_export['final_step_status']}; "
        f"metric_status={source_export['metric_status']}; "
        f"schema_blockers={schema_blockers}"
    )


def _metadata_gate_detail(metadata_gate: Mapping[str, Any]) -> str:
    missing_fields = ",".join(metadata_gate.get("missing_fields", []))
    semantic_mismatch_count = len(metadata_gate.get("semantic_mismatches", []))
    disallowed_terms = ",".join(
        sorted(
            {
                str(match.get("term", ""))
                for match in metadata_gate.get("disallowed_provenance", [])
            }
        )
    )
    return (
        f"provenance_status={metadata_gate['provenance_status']}; "
        f"missing_fields={missing_fields}; "
        f"semantic_mismatch_count={semantic_mismatch_count}; "
        f"disallowed_terms={disallowed_terms}"
    )


def _csv_rows(
    *,
    source_checks: Iterable[Mapping[str, Any]],
    metadata_check: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "artifact": check["artifact"],
            "metric_group": check["metric_group"],
            "source_path": check["source_path"],
            "file_status": check["file_status"],
            "header_status": check["header_status"],
            "final_step_status": check["final_step_status"],
            "metric_status": check["metric_status"],
            "blocker": check["blocker"] or "",
        }
        for check in source_checks
    ]
    rows.append(
        {
            "artifact": metadata_check["artifact"],
            "metric_group": "provenance",
            "source_path": metadata_check["source_path"],
            "file_status": metadata_check["file_status"],
            "header_status": "not_applicable",
            "final_step_status": "not_applicable",
            "metric_status": metadata_check["provenance_status"],
            "blocker": metadata_check["blocker"] or "",
        }
    )
    return rows


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    blockers = ", ".join(item["blocker"] for item in payload["candidate_blockers"])
    lines = [
        "# ANSYS vertical-flap Fluent reference collection",
        "",
        "## Scope",
        "",
        (
            "This artifact validates committed Fluent reference source export "
            "schemas and provenance. It does not run Fluent, does not run "
            "EasyFsi, and does not claim Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload['candidate_status']}`",
        f"- candidate_contract_status: `{payload['candidate_contract_status']}`",
        f"- active_blockers: `{blockers}`",
        "",
        "## Source exports",
        "",
        "artifact | file status | header status | final step | metric status",
        "--- | --- | --- | --- | ---",
    ]
    for row in payload["rows"]:
        lines.append(
            " | ".join(
                [
                    str(row["artifact"]),
                    str(row["file_status"]),
                    str(row["header_status"]),
                    str(row["final_step_status"]),
                    str(row["metric_status"]),
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Matrix JSON: `{_repo_relative(MATRIX_JSON)}`",
            f"- Matrix CSV: `{_repo_relative(MATRIX_CSV)}`",
            f"- Candidate contract: `{_repo_relative(CANDIDATE_CONTRACT_JSON)}`",
            (
                "- Active contract manifest: "
                f"`{_repo_relative(ACTIVE_CONTRACT_MANIFEST_JSON)}`"
            ),
            (
                "- Public tutorial evidence map: "
                f"`{_repo_relative(PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON)}`"
            ),
            f"- Artifact manifest: `{_repo_relative(ARTIFACT_MANIFEST_JSON)}`",
            f"- Checksums: `{_repo_relative(CHECKSUMS_PATH)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    reference_complete = (
        payload.get("candidate_contract_status") == CONTRACT_COMPLETE
    )
    return {
        "manifest_schema_version": "validation_artifact_manifest_v1",
        "artifact_group": "fluent_reference_collection",
        "source_script": SOURCE_SCRIPT,
        "generated_from_commit": _generated_from_commit(),
        "generated_from_ref": _generated_from_ref(),
        "artifact_generation_source_commit": _generated_from_commit(),
        "artifact_generation_source_ref": _generated_from_ref(),
        "artifact_committed_in_review_head": _artifact_committed_in_review_head(),
        "inputs": {
            "current_contract": _repo_relative(CURRENT_CONTRACT_JSON),
            "source_exports_root": _repo_relative(SOURCE_EXPORTS_ROOT),
            "public_tutorial_evidence_map": _repo_relative(
                PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON
            ),
        },
        "outputs": {
            "matrix_json": _manifest_output(MATRIX_JSON),
            "matrix_csv": _manifest_output(MATRIX_CSV),
            "candidate_contract": _manifest_output(CANDIDATE_CONTRACT_JSON),
            "summary_md": _manifest_output(SUMMARY_MD),
        },
        "claim_policy": {
            "fluent_parity_claimed": False,
            "reason": (
                "reference complete but collection does not evaluate parity"
                if reference_complete
                else "reference incomplete"
            ),
        },
    }


def _generated_from_commit() -> str:
    return os.environ.get(
        GENERATED_FROM_COMMIT_ENV,
        DEFAULT_GENERATED_FROM_COMMIT,
    )


def _generated_from_ref() -> str:
    return os.environ.get(
        GENERATED_FROM_REF_ENV,
        DEFAULT_GENERATED_FROM_REF,
    )


def _artifact_committed_in_review_head() -> str:
    return os.environ.get(
        ARTIFACT_COMMITTED_IN_REVIEW_HEAD_ENV,
        DEFAULT_ARTIFACT_COMMITTED_IN_REVIEW_HEAD,
    )


def _manifest_output(path: Path) -> dict[str, str]:
    return {
        "path": _repo_relative(path),
        "sha256": _sha256_file(path),
    }


def _reference_values(
    spec: Mapping[str, Any],
    final_row: Mapping[str, Any] | None,
) -> dict[str, float]:
    if final_row is None:
        return {}
    values = {}
    for metric, column in spec["reference_values"].items():
        value = _float_value(final_row.get(column))
        if value is None:
            continue
        values[str(metric)] = value
    return values


def _final_step_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    final_rows = [
        row for row in rows if _int_value(row.get("step")) == EXPECTED_STEP_COUNT
    ]
    return final_rows[-1] if final_rows else None


def _metadata_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        fields[_normalize_field(key)] = value.strip()
    return fields


def _metadata_semantic_mismatches(
    observed: Mapping[str, str],
    missing: Iterable[str],
) -> list[dict[str, Any]]:
    missing_keys = {_normalize_field(field) for field in missing}
    checks = [
        {
            "field": "time step",
            "expected": EXPECTED_TIME_STEP_S,
            "actual": _float_value(observed.get(_normalize_field("time step"))),
        },
        {
            "field": "number of steps",
            "expected": EXPECTED_STEP_COUNT,
            "actual": _float_value(observed.get(_normalize_field("number of steps"))),
        },
    ]
    mismatches: list[dict[str, Any]] = []
    for check in checks:
        field = str(check["field"])
        if _normalize_field(field) in missing_keys:
            continue
        actual = check["actual"]
        expected = float(check["expected"])
        if not _float_equals(actual, expected):
            mismatches.append(
                {
                    "field": field,
                    "expected": check["expected"],
                    "actual": actual,
                }
            )
    return mismatches


def _metadata_disallowed_provenance(
    observed: Mapping[str, str],
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for field, value in observed.items():
        normalized = _normalize_provenance_text(value)
        for term in DISALLOWED_METADATA_PROVENANCE_TERMS:
            if term not in normalized:
                continue
            matches.append(
                {
                    "field": field,
                    "term": term,
                    "value": str(value),
                }
            )
    return matches


def _metadata_blocker(
    *,
    complete: bool,
    disallowed_provenance: Iterable[Mapping[str, str]],
) -> str | None:
    if complete:
        return None
    if list(disallowed_provenance):
        return BLOCKER_METADATA_DISALLOWED_PROVENANCE
    return BLOCKER_PROVENANCE


def _normalize_field(value: str) -> str:
    return " ".join(value.lower().split())


def _normalize_provenance_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("\\", "/").split())


def _is_missing(value: Any) -> bool:
    return str(value).strip().lower() in MISSING_VALUES


def _comparison_metadata(metadata_check: Mapping[str, Any]) -> dict[str, Any]:
    observed = metadata_check.get("observed_fields", {})
    if not isinstance(observed, Mapping):
        observed = {}
    status = (
        "complete"
        if metadata_check.get("provenance_status") == "complete"
        else "missing"
    )
    return {
        "status": status,
        "displacement_definition": {
            "metric": "tip_displacement_norm_m",
            "source_step50_metric": "tip_mean_displacement_m",
            "point": str(
                observed.get(_normalize_field("displacement_definition"), "")
            ),
            "status": status,
        },
        "sign_conventions": {
            "force_z_positive": str(
                observed.get(_normalize_field("force_z_positive"), "")
            ),
            "flow_rate_positive": str(
                observed.get(_normalize_field("flow_rate_positive"), "")
            ),
            "pressure_reference": str(
                observed.get(_normalize_field("pressure_reference"), "")
            ),
            "status": status,
        },
    }


def _active_contract_recommendation(
    *,
    contract_complete: bool,
    missing_metrics: Iterable[str],
    metadata_check: Mapping[str, Any],
    comparison_metadata: Mapping[str, Any],
    tolerances_complete: bool,
) -> dict[str, Any]:
    if contract_complete:
        reason = (
            "Candidate contract has complete source metrics, provenance, "
            "comparison metadata, and tolerances."
        )
        action = "promote_versioned_contract"
    else:
        reason_parts = []
        missing = list(missing_metrics)
        if missing:
            reason_parts.append("missing_metrics=" + ",".join(missing))
        if metadata_check.get("provenance_status") != "complete":
            reason_parts.append("provenance_incomplete")
        if comparison_metadata.get("status") != "complete":
            reason_parts.append("comparison_metadata_incomplete")
        if not tolerances_complete:
            reason_parts.append("tolerances_incomplete")
        reason = "; ".join(reason_parts)
        action = "keep_current_incomplete_contract"
    return {
        "recommended_action": action,
        "current_contract": _repo_relative(CURRENT_CONTRACT_JSON),
        "candidate_contract": _repo_relative(CANDIDATE_CONTRACT_JSON),
        "reason": reason,
    }


def _missing_source_provenance() -> dict[str, str]:
    return {
        "document": "",
        "run_id": "",
        "author": "",
        "date": "",
        "status": "missing",
    }


def _tolerances_complete(tolerances: Mapping[str, Any]) -> bool:
    for payload in tolerances.values():
        if not isinstance(payload, Mapping):
            return False
        if payload.get("status") != "available":
            return False
        if _float_value(payload.get("value")) is None:
            return False
        if _is_missing(payload.get("comparator", "")):
            return False
        if _is_missing(payload.get("source", "")):
            return False
        if _is_missing(payload.get("rationale", "")):
            return False
    return bool(tolerances)


def _public_reference_use_policy() -> str:
    if not PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON.exists():
        return "missing_public_reference_evidence"
    evidence = _read_json(PUBLIC_TUTORIAL_EVIDENCE_MAP_JSON)
    return str(evidence.get("use_policy", "missing_public_reference_evidence"))


def _blocker_detail(blocker: str) -> str:
    details = {
        BLOCKER_DISPLACEMENT: "Fluent displacement source export is missing final-step values",
        BLOCKER_FORCE: "Fluent force source export is missing final-step values",
        BLOCKER_FLOW: "Fluent flow/outlet source export is missing final-step values",
        BLOCKER_PRESSURE: "Fluent pressure source export is missing final-step values",
        BLOCKER_PROVENANCE: "Fluent source provenance metadata is incomplete",
        BLOCKER_METADATA_DISALLOWED_PROVENANCE: (
            "Fluent metadata points at non-Fluent source provenance"
        ),
        "fluent_reference_comparison_metadata_incomplete": (
            "Fluent displacement/sign/pressure convention metadata is incomplete"
        ),
        BLOCKER_TOLERANCES: "Reference metrics exist but comparison tolerances are incomplete",
    }
    return details[blocker]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _repo_relative(path: Path | str) -> str:
    return Path(path).as_posix()


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int_value(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _float_equals(lhs: float | None, rhs: float) -> bool:
    return lhs is not None and abs(lhs - rhs) <= 1.0e-12


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[fluent_reference_collection] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[fluent_reference_collection] wrote "
        f"{payload['candidate_status']} to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
