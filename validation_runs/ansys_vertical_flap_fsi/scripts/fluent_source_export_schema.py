from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MISSING_VALUES = {"", "missing", "todo", "tbd", "n/a", "na", "null", "none"}
DISALLOWED_SOURCE_PROVENANCE_TERMS = (
    "easyfsi",
    "hibm-mpm",
    "synthetic",
    "fixture",
    "placeholder",
    "public tutorial",
    "not fluent truth",
    "validation_runs",
    "not_collected",
)


def validate_source_export_csv(
    path: Path,
    expected_header: Sequence[str],
    *,
    required_final_step: int = 50,
    expected_final_time: float = 0.025,
    reference_value_columns: Mapping[str, str] | None = None,
    allow_test_sources: bool = False,
) -> dict[str, Any]:
    reference_value_columns = reference_value_columns or {}
    if not path.exists():
        return {
            "exists": False,
            "file_status": "missing_file",
            "header_status": "missing_file",
            "final_step_status": "missing_file",
            "metric_status": "missing",
            "observed_columns": [],
            "reference_values": {},
            "row_count": 0,
            "final_time_s": None,
            "blockers": ["missing_file"],
        }

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        observed_columns = list(reader.fieldnames or [])
        rows = list(reader)

    expected = list(expected_header)
    if observed_columns != expected:
        return {
            "exists": True,
            "file_status": "present_header_mismatch",
            "header_status": "failed",
            "final_step_status": "not_checked",
            "metric_status": "missing",
            "observed_columns": observed_columns,
            "reference_values": {},
            "row_count": len(rows),
            "final_time_s": None,
            "blockers": ["present_header_mismatch"],
        }

    if not rows:
        return {
            "exists": True,
            "file_status": "schema_only",
            "header_status": "passed",
            "final_step_status": "missing_final_step",
            "metric_status": "missing",
            "observed_columns": observed_columns,
            "reference_values": {},
            "row_count": 0,
            "final_time_s": None,
            "blockers": ["missing_final_step"],
        }

    final_row = _final_step_row(rows, required_final_step)
    if final_row is None:
        return _present_missing(
            observed_columns=observed_columns,
            row_count=len(rows),
            file_status="present_missing_final_step",
            final_step_status="missing_final_step",
            blocker="missing_final_step",
        )

    final_time = _float_value(final_row.get("time_s"))
    if not _float_equals(final_time, expected_final_time):
        return _present_missing(
            observed_columns=observed_columns,
            row_count=len(rows),
            file_status="present_final_time_mismatch",
            final_step_status="final_time_mismatch",
            blocker="final_time_mismatch",
            final_time_s=final_time,
        )

    if _is_missing(final_row.get("source")):
        return _present_missing(
            observed_columns=observed_columns,
            row_count=len(rows),
            file_status="present_missing_source",
            final_step_status="passed",
            blocker="missing_source",
            final_time_s=final_time,
        )

    source = str(final_row.get("source", ""))
    if not allow_test_sources and _has_disallowed_source_provenance(source):
        return _present_missing(
            observed_columns=observed_columns,
            row_count=len(rows),
            file_status="present_disallowed_source_provenance",
            final_step_status="passed",
            blocker="disallowed_source_provenance",
            final_time_s=final_time,
        )

    reference_values = {}
    for metric, column in reference_value_columns.items():
        value = _float_value(final_row.get(column))
        if value is None:
            return _present_missing(
                observed_columns=observed_columns,
                row_count=len(rows),
                file_status="present_missing_metric_value",
                final_step_status="passed",
                blocker="missing_metric_value",
                final_time_s=final_time,
            )
        reference_values[str(metric)] = value

    return {
        "exists": True,
        "file_status": "present_complete",
        "header_status": "passed",
        "final_step_status": "passed",
        "metric_status": "available",
        "observed_columns": observed_columns,
        "reference_values": reference_values,
        "row_count": len(rows),
        "final_time_s": final_time,
        "blockers": [],
    }


def _present_missing(
    *,
    observed_columns: list[str],
    row_count: int,
    file_status: str,
    final_step_status: str,
    blocker: str,
    final_time_s: float | None = None,
) -> dict[str, Any]:
    return {
        "exists": True,
        "file_status": file_status,
        "header_status": "passed",
        "final_step_status": final_step_status,
        "metric_status": "missing",
        "observed_columns": observed_columns,
        "reference_values": {},
        "row_count": row_count,
        "final_time_s": final_time_s,
        "blockers": [blocker],
    }


def _final_step_row(
    rows: Sequence[Mapping[str, Any]],
    required_final_step: int,
) -> Mapping[str, Any] | None:
    final_rows = [
        row for row in rows if _int_value(row.get("step")) == required_final_step
    ]
    return final_rows[-1] if final_rows else None


def _is_missing(value: Any) -> bool:
    return str(value).strip().lower() in MISSING_VALUES


def _has_disallowed_source_provenance(value: str) -> bool:
    normalized = " ".join(value.strip().lower().replace("\\", "/").split())
    return any(term in normalized for term in DISALLOWED_SOURCE_PROVENANCE_TERMS)


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
