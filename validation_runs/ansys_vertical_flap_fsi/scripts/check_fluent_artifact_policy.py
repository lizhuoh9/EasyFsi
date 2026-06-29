from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DEFAULT_ROOTS = (
    ROOT / "fluent_reference" / "validation_diagnostics",
    ROOT / "traction_selected_formulation_fluent_parity_diagnostics",
)

ALLOWED_GATE_STATUSES = {"passed", "report_only"}
SYNTHETIC_MARKER = "synthetic-test-only"


def check_fluent_artifact_policy(
    roots: Iterable[Path] = DEFAULT_ROOTS,
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    checked_files: list[str] = []

    for root in roots:
        for path in sorted(root.rglob("*.json")):
            checked_files.append(path.as_posix())
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                _add_violation(
                    violations,
                    path,
                    "json_decode_failed",
                    str(exc),
                )
                continue
            _check_payload(path, payload, violations)

    return {
        "policy": "fluent_artifact_policy_v1",
        "checked_file_count": len(checked_files),
        "checked_files": checked_files,
        "status": "passed" if not violations else "failed",
        "violations": violations,
    }


def _check_payload(
    path: Path,
    payload: Any,
    violations: list[dict[str, str]],
) -> None:
    if _contains_string(payload, SYNTHETIC_MARKER):
        _add_violation(
            violations,
            path,
            "synthetic_marker_in_real_artifact",
            "real generated artifacts must not contain synthetic-test-only data",
        )

    for policy in _find_values(payload, "public_reference_use_policy"):
        if policy != "metadata_only_not_parity_truth":
            _add_violation(
                violations,
                path,
                "public_reference_policy_violation",
                f"unexpected public_reference_use_policy={policy}",
            )

    claimed = any(value is True for value in _find_values(payload, "fluent_parity_claimed"))
    candidate_status = _first_string(payload, "candidate_status")
    reference_status = _first_string(payload, "reference_contract_status")

    if claimed:
        if candidate_status != "fluent_parity_validated":
            _add_violation(
                violations,
                path,
                "claimed_parity_without_validated_candidate",
                f"candidate_status={candidate_status}",
            )
        if reference_status != "fluent_reference_complete":
            _add_violation(
                violations,
                path,
                "claimed_parity_without_complete_reference",
                f"reference_contract_status={reference_status}",
            )
        gate_statuses = _gate_statuses(_first_mapping(payload, "parity_metrics"))
        if not gate_statuses:
            _add_violation(
                violations,
                path,
                "claimed_parity_without_metric_gates",
                "parity_metrics gate_status values are missing",
            )
        for gate, status in gate_statuses.items():
            if status not in ALLOWED_GATE_STATUSES:
                _add_violation(
                    violations,
                    path,
                    "claimed_parity_with_failed_metric_gate",
                    f"{gate} gate_status={status}",
                )

    if reference_status and reference_status != "fluent_reference_complete":
        if candidate_status == "fluent_parity_validated":
            _add_violation(
                violations,
                path,
                "validated_candidate_with_incomplete_reference",
                f"reference_contract_status={reference_status}",
            )
        if claimed:
            _add_violation(
                violations,
                path,
                "claimed_parity_with_incomplete_reference",
                f"reference_contract_status={reference_status}",
            )


def _gate_statuses(payload: Mapping[str, Any] | None) -> dict[str, str]:
    if payload is None:
        return {}
    statuses = {}
    for key, value in payload.items():
        if isinstance(value, Mapping) and "gate_status" in value:
            statuses[str(key)] = str(value["gate_status"])
    return statuses


def _find_values(payload: Any, key: str) -> list[Any]:
    values = []
    if isinstance(payload, Mapping):
        for item_key, item_value in payload.items():
            if item_key == key:
                values.append(item_value)
            values.extend(_find_values(item_value, key))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_find_values(item, key))
    return values


def _first_string(payload: Any, key: str) -> str | None:
    for value in _find_values(payload, key):
        if isinstance(value, str):
            return value
    return None


def _first_mapping(payload: Any, key: str) -> Mapping[str, Any] | None:
    for value in _find_values(payload, key):
        if isinstance(value, Mapping):
            return value
    return None


def _contains_string(payload: Any, needle: str) -> bool:
    if isinstance(payload, str):
        return needle in payload
    if isinstance(payload, Mapping):
        return any(
            _contains_string(key, needle) or _contains_string(value, needle)
            for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_contains_string(item, needle) for item in payload)
    return False


def _add_violation(
    violations: list[dict[str, str]],
    path: Path,
    rule: str,
    detail: str,
) -> None:
    violations.append(
        {
            "path": path.as_posix(),
            "rule": rule,
            "detail": detail,
        }
    )


def main() -> int:
    result = check_fluent_artifact_policy()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
