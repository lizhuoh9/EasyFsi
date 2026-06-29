from __future__ import annotations

import math
from typing import Any, Mapping


CASE_NAME = "ansys_vertical_flap_fsi"
SCHEMA_VERSION = "ansys_vertical_flap_fluent_reference_contract_v1"
CONTRACT_COMPLETE = "fluent_reference_complete"
CONTRACT_INCOMPLETE = "fluent_reference_incomplete"
EXPECTED_STEP_COUNT = 50
EXPECTED_TIME_STEP_S = 0.0005
EXPECTED_TOTAL_TIME_S = 0.025

REQUIRED_REFERENCE_METRICS = (
    "tip_displacement_m",
    "max_displacement_m",
    "force_z_N",
    "flow_rate_m3s",
    "pressure_range_pa",
)

EXPECTED_METRIC_UNITS = {
    "tip_displacement_m": "m",
    "max_displacement_m": "m",
    "force_z_N": "N",
    "flow_rate_m3s": "m3/s",
    "pressure_range_pa": "Pa",
}

REQUIRED_TOLERANCES = (
    "tip_displacement_relative",
    "max_displacement_relative",
    "force_z_relative",
    "flow_rate_relative",
    "pressure_sanity_absolute",
)

ALLOWED_TOLERANCE_COMPARATORS = {
    "relative_error",
    "absolute_error",
    "range_contains",
    "sign_matches",
    "report_only",
}

REQUIRED_SAMPLING_DEFINITIONS = (
    "tip_displacement",
    "max_displacement",
    "force_z",
    "flow_rate",
    "pressure_range",
)

REQUIRED_SAMPLING_FIELDS = (
    "definition",
    "unit",
    "status",
)

REQUIRED_COMPARISON_POLICY_FIELDS = (
    "status",
    "reference_complete_required",
    "parity_claim_requires_all_gates",
)

REQUIRED_GEOMETRY_FIELDS = (
    "duct_length_m",
    "duct_height_m",
    "modeled_domain",
    "flap_height_m",
    "flap_thickness_m",
    "flap_streamwise_min_m",
    "flap_streamwise_max_m",
)

REQUIRED_MATERIAL_FIELDS = (
    "air_density_kgm3",
    "air_viscosity_pa_s",
    "solid_density_kgm3",
    "youngs_modulus_pa",
    "poisson_ratio",
)

REQUIRED_SOURCE_PROVENANCE_FIELDS = (
    "document",
    "run_id",
    "author",
    "date",
)

REQUIRED_SIGN_CONVENTIONS = (
    "force_z_positive",
    "flow_rate_positive",
    "pressure_reference",
)

REQUIRED_DISPLACEMENT_DEFINITION = (
    "metric",
    "source_step50_metric",
    "point",
)

MISSING_VALUES = {"", "missing", "todo", "tbd", "n/a", "na", "null", "none"}


def validate_fluent_reference_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    missing_metrics: list[str] = []
    validated_metric_count = 0

    if contract.get("schema_version") != SCHEMA_VERSION:
        _append_blocker(
            blockers,
            "fluent_reference_schema_version_missing",
            f"expected schema_version={SCHEMA_VERSION}",
        )
    if _is_missing(contract.get("contract_id")):
        _append_blocker(
            blockers,
            "fluent_reference_contract_id_missing",
            "contract_id must be present",
        )
    if contract.get("case") != CASE_NAME:
        _append_blocker(
            blockers,
            "fluent_reference_case_mismatch",
            f"expected case {CASE_NAME}",
        )

    _validate_time_integration(contract, blockers)
    _validate_mapping_fields(
        contract.get("geometry"),
        REQUIRED_GEOMETRY_FIELDS,
        blockers,
        "fluent_reference_geometry_incomplete",
        require_numeric_fields={
            "duct_length_m",
            "duct_height_m",
            "flap_height_m",
            "flap_thickness_m",
            "flap_streamwise_min_m",
            "flap_streamwise_max_m",
        },
    )
    _validate_mapping_fields(
        contract.get("material"),
        REQUIRED_MATERIAL_FIELDS,
        blockers,
        "fluent_reference_material_incomplete",
        require_numeric_fields=set(REQUIRED_MATERIAL_FIELDS),
    )
    _validate_source_provenance(contract.get("source_provenance"), blockers)
    _validate_status_mapping(
        contract.get("sign_conventions"),
        REQUIRED_SIGN_CONVENTIONS,
        blockers,
        "fluent_reference_sign_conventions_incomplete",
    )
    _validate_status_mapping(
        contract.get("displacement_definition"),
        REQUIRED_DISPLACEMENT_DEFINITION,
        blockers,
        "fluent_reference_displacement_definition_incomplete",
    )
    _validate_nested_status_mapping(
        contract.get("sampling_definitions"),
        REQUIRED_SAMPLING_DEFINITIONS,
        REQUIRED_SAMPLING_FIELDS,
        blockers,
        "fluent_reference_sampling_definitions_incomplete",
    )
    _validate_comparison_policy(contract.get("comparison_policy"), blockers)

    reference_metrics = contract.get("reference_metrics")
    if not isinstance(reference_metrics, Mapping):
        reference_metrics = {}
        _append_blocker(
            blockers,
            "fluent_reference_metrics_incomplete",
            "reference_metrics must be a mapping",
        )
    for metric in REQUIRED_REFERENCE_METRICS:
        payload = reference_metrics.get(metric)
        if _metric_payload_available(metric, payload):
            validated_metric_count += 1
        else:
            missing_metrics.append(metric)
            _append_blocker(
                blockers,
                f"{metric}_missing",
                f"{metric} must be available, numeric, and finite",
            )

    tolerances = contract.get("tolerances")
    if not isinstance(tolerances, Mapping):
        tolerances = {}
        _append_blocker(
            blockers,
            "fluent_reference_tolerances_incomplete",
            "tolerances must be a mapping",
        )
    for tolerance in REQUIRED_TOLERANCES:
        payload = tolerances.get(tolerance)
        if not _tolerance_payload_available(tolerance, payload):
            _append_blocker(
                blockers,
                "fluent_reference_tolerances_incomplete",
                (
                    f"{tolerance} must be available, numeric, finite, "
                    "sourced, rationalized, and use a supported comparator"
                ),
            )

    deduped = _dedupe_blockers(blockers)
    return {
        "contract_status": (
            CONTRACT_COMPLETE if not deduped else CONTRACT_INCOMPLETE
        ),
        "blockers": deduped,
        "validated_metric_count": validated_metric_count,
        "required_metric_count": len(REQUIRED_REFERENCE_METRICS),
        "missing_required_metrics": missing_metrics,
    }


def _validate_time_integration(
    contract: Mapping[str, Any],
    blockers: list[dict[str, str]],
) -> None:
    simulation = contract.get("simulation")
    if not isinstance(simulation, Mapping):
        simulation = {}
        _append_blocker(
            blockers,
            "fluent_reference_time_integration_incomplete",
            "simulation must be a mapping",
        )

    step_count = _int_value(simulation.get("step_count", contract.get("step_count")))
    if step_count != EXPECTED_STEP_COUNT:
        _append_blocker(
            blockers,
            "fluent_reference_step_count_mismatch",
            f"expected {EXPECTED_STEP_COUNT} steps",
        )

    time_step_s = _float_value(
        simulation.get("time_step_s", contract.get("time_step_s"))
    )
    if time_step_s is None or abs(time_step_s - EXPECTED_TIME_STEP_S) > 1.0e-12:
        _append_blocker(
            blockers,
            "fluent_reference_time_step_mismatch",
            f"expected time_step_s={EXPECTED_TIME_STEP_S}",
        )

    total_time_s = _float_value(simulation.get("total_time_s"))
    if total_time_s is None or abs(total_time_s - EXPECTED_TOTAL_TIME_S) > 1.0e-12:
        _append_blocker(
            blockers,
            "fluent_reference_total_time_mismatch",
            f"expected total_time_s={EXPECTED_TOTAL_TIME_S}",
        )


def _validate_mapping_fields(
    payload: Any,
    required_fields: tuple[str, ...],
    blockers: list[dict[str, str]],
    blocker: str,
    *,
    require_numeric_fields: set[str],
) -> None:
    if not isinstance(payload, Mapping):
        _append_blocker(blockers, blocker, "required mapping is missing")
        return
    missing = []
    for field in required_fields:
        value = payload.get(field)
        if _is_missing(value):
            missing.append(field)
        elif field in require_numeric_fields and _float_value(value) is None:
            missing.append(field)
    if missing:
        _append_blocker(blockers, blocker, "missing fields: " + ",".join(missing))


def _validate_source_provenance(
    payload: Any,
    blockers: list[dict[str, str]],
) -> None:
    if not isinstance(payload, Mapping):
        _append_blocker(
            blockers,
            "fluent_reference_provenance_incomplete",
            "source_provenance must be a mapping",
        )
        return
    missing = [
        field for field in REQUIRED_SOURCE_PROVENANCE_FIELDS if _is_missing(payload.get(field))
    ]
    if payload.get("status") != "complete":
        missing.append("status")
    if missing:
        _append_blocker(
            blockers,
            "fluent_reference_provenance_incomplete",
            "missing fields: " + ",".join(dict.fromkeys(missing)),
        )


def _validate_status_mapping(
    payload: Any,
    required_fields: tuple[str, ...],
    blockers: list[dict[str, str]],
    blocker: str,
) -> None:
    if not isinstance(payload, Mapping):
        _append_blocker(blockers, blocker, "required mapping is missing")
        return
    missing = [field for field in required_fields if _is_missing(payload.get(field))]
    if payload.get("status") != "complete":
        missing.append("status")
    if missing:
        _append_blocker(blockers, blocker, "missing fields: " + ",".join(dict.fromkeys(missing)))


def _validate_nested_status_mapping(
    payload: Any,
    required_keys: tuple[str, ...],
    required_fields: tuple[str, ...],
    blockers: list[dict[str, str]],
    blocker: str,
) -> None:
    if not isinstance(payload, Mapping):
        _append_blocker(blockers, blocker, "required mapping is missing")
        return
    missing = []
    for key in required_keys:
        item = payload.get(key)
        if not isinstance(item, Mapping):
            missing.append(key)
            continue
        for field in required_fields:
            if _is_missing(item.get(field)):
                missing.append(f"{key}.{field}")
        if item.get("status") != "complete":
            missing.append(f"{key}.status")
    if missing:
        _append_blocker(
            blockers,
            blocker,
            "missing fields: " + ",".join(dict.fromkeys(missing)),
        )


def _validate_comparison_policy(
    payload: Any,
    blockers: list[dict[str, str]],
) -> None:
    if not isinstance(payload, Mapping):
        _append_blocker(
            blockers,
            "fluent_reference_comparison_policy_incomplete",
            "comparison_policy must be a mapping",
        )
        return
    missing = [
        field
        for field in REQUIRED_COMPARISON_POLICY_FIELDS
        if _is_missing(payload.get(field))
    ]
    if payload.get("status") != "complete":
        missing.append("status")
    if payload.get("reference_complete_required") is not True:
        missing.append("reference_complete_required")
    if payload.get("parity_claim_requires_all_gates") is not True:
        missing.append("parity_claim_requires_all_gates")
    if missing:
        _append_blocker(
            blockers,
            "fluent_reference_comparison_policy_incomplete",
            "missing fields: " + ",".join(dict.fromkeys(missing)),
        )


def _metric_payload_available(metric_name: str, payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("status") == "available"
        and _float_value(payload.get("value")) is not None
        and payload.get("unit") == EXPECTED_METRIC_UNITS[metric_name]
        and not _is_missing(payload.get("source"))
        and not _is_missing(payload.get("extraction_method"))
        and _float_equals(_float_value(payload.get("time_s")), EXPECTED_TOTAL_TIME_S)
    )


def _tolerance_payload_available(tolerance_name: str, payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    comparator = str(payload.get("comparator", ""))
    if comparator not in ALLOWED_TOLERANCE_COMPARATORS:
        return False
    if tolerance_name == "pressure_sanity_absolute" and comparator not in {
        "absolute_error",
        "range_contains",
    }:
        return False
    return (
        payload.get("status") == "available"
        and _float_value(payload.get("value")) is not None
        and not _is_missing(payload.get("source"))
        and not _is_missing(payload.get("rationale"))
    )


def _append_blocker(
    blockers: list[dict[str, str]],
    blocker: str,
    detail: str,
) -> None:
    blockers.append({"blocker": blocker, "detail": detail})


def _dedupe_blockers(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for blocker in blockers:
        deduped.setdefault(blocker["blocker"], blocker)
    return list(deduped.values())


def _is_missing(value: Any) -> bool:
    return str(value).strip().lower() in MISSING_VALUES


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
