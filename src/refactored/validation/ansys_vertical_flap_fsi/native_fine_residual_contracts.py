"""Streaming contracts for native Fluent residual-history exports."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


LOCKED_NATIVE_FLUENT_EQUATIONS = (
    "continuity",
    "k",
    "omega",
    "x-displacement",
    "x-velocity",
    "y-displacement",
    "y-velocity",
)


class NativeFineResidualContractError(RuntimeError):
    """Raised when native Fluent residual artifacts are incomplete or inconsistent."""


def validate_fluent_residual_histories(
    residual_history_path: str | Path,
    residual_snapshot_summary_path: str | Path,
    *,
    expected_steps: int,
    dt_s: float,
) -> dict[str, Any]:
    """Stream the large residual CSV and cross-check every snapshot summary."""

    residual_history_path = Path(residual_history_path)
    residual_snapshot_summary_path = Path(residual_snapshot_summary_path)
    for label, path in (
        ("residual history", residual_history_path),
        ("residual snapshot summary", residual_snapshot_summary_path),
    ):
        if not path.is_file():
            raise NativeFineResidualContractError(
                f"native Fluent {label} is missing: {path}"
            )

    stats_by_group: dict[tuple[int, str], dict[str, Any]] = {}
    covered_steps: set[int] = set()
    residual_row_count = 0
    with residual_history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {
            "snapshot_step",
            "snapshot_time_s",
            "equation",
            "sample_index",
            "iteration",
            "value_col0",
            "data_path",
        }
        missing = sorted(required - set(fields))
        if missing:
            raise NativeFineResidualContractError(
                f"native Fluent residual history is missing columns: {missing}"
            )
        value_fields = sorted(
            (field for field in fields if re.fullmatch(r"value_col\d+", field)),
            key=lambda field: int(field.removeprefix("value_col")),
        )
        if value_fields != [f"value_col{index}" for index in range(len(value_fields))]:
            raise NativeFineResidualContractError(
                "native Fluent residual value columns are not exact and contiguous"
            )
        for row_number, row in enumerate(reader, start=2):
            step = _csv_integer(
                row.get("snapshot_step"),
                f"native Fluent residual step at row {row_number}",
                minimum=1,
            )
            if step > expected_steps:
                raise NativeFineResidualContractError(
                    f"native Fluent residual step exceeds {expected_steps}: {step}"
                )
            _validate_residual_time(
                row.get("snapshot_time_s"), step=step, dt_s=dt_s, label="history"
            )
            equation = str(row.get("equation") or "").strip()
            if not equation:
                raise NativeFineResidualContractError(
                    f"native Fluent residual equation is empty at row {row_number}"
                )
            sample_index = _csv_integer(
                row.get("sample_index"),
                f"native Fluent residual sample_index at row {row_number}",
                minimum=0,
            )
            iteration = _finite_float(
                row.get("iteration"),
                f"native Fluent residual iteration at row {row_number}",
            )
            values = [
                _finite_float(row.get(field), f"residual {field} at row {row_number}")
                for field in value_fields
            ]
            data_path = str(row.get("data_path") or "").strip().replace("\\", "/")
            if not values or not data_path:
                raise NativeFineResidualContractError(
                    f"native Fluent residual row {row_number} is incomplete"
                )
            key = (step, equation)
            stats = stats_by_group.setdefault(
                key,
                {
                    "count": 0,
                    "first_iteration": iteration,
                    "last_iteration": iteration,
                    "primary_initial": values[0],
                    "primary_final": values[0],
                    "primary_min": values[0],
                    "primary_max": values[0],
                    "value_column_count": len(value_fields),
                    "data_path": data_path,
                },
            )
            if sample_index != stats["count"]:
                raise NativeFineResidualContractError(
                    "native Fluent residual sample_index sequence is not contiguous "
                    f"for step {step}, equation {equation!r}"
                )
            if data_path != stats["data_path"]:
                raise NativeFineResidualContractError(
                    f"native Fluent residual data_path changes within group {key!r}"
                )
            stats["count"] += 1
            stats["last_iteration"] = iteration
            stats["primary_final"] = values[0]
            stats["primary_min"] = min(stats["primary_min"], values[0])
            stats["primary_max"] = max(stats["primary_max"], values[0])
            covered_steps.add(step)
            residual_row_count += 1

    expected_step_sequence = list(range(1, expected_steps + 1))
    if sorted(covered_steps) != expected_step_sequence:
        raise NativeFineResidualContractError(
            "native Fluent residual history steps must be exact and contiguous: "
            f"expected={expected_step_sequence}, actual={sorted(covered_steps)}"
        )
    equations_by_step = {
        step: {equation for group_step, equation in stats_by_group if group_step == step}
        for step in expected_step_sequence
    }
    expected_equations = equations_by_step[1]
    inconsistent_steps = [
        step
        for step, equations in equations_by_step.items()
        if equations != expected_equations
    ]
    if inconsistent_steps:
        raise NativeFineResidualContractError(
            "native Fluent residual equation groups must be identical at every step; "
            f"first_step={sorted(expected_equations)}, inconsistent_steps={inconsistent_steps}"
        )
    if expected_steps == 50 and expected_equations != set(LOCKED_NATIVE_FLUENT_EQUATIONS):
        raise NativeFineResidualContractError(
            "locked native Fluent residual equation groups are incomplete: "
            f"expected={list(LOCKED_NATIVE_FLUENT_EQUATIONS)}, "
            f"actual={sorted(expected_equations)}"
        )

    summary_groups: set[tuple[int, str]] = set()
    with residual_snapshot_summary_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        required = {
            "step",
            "time_s",
            "equation",
            "sample_count",
            "first_iteration",
            "last_iteration",
            "primary_initial",
            "primary_final",
            "primary_min",
            "primary_max",
            "stored_value_column_count",
            "data_path",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise NativeFineResidualContractError(
                f"native Fluent residual snapshot summary is missing columns: {missing}"
            )
        for row_number, row in enumerate(reader, start=2):
            step = _csv_integer(
                row.get("step"),
                f"native Fluent residual summary step at row {row_number}",
                minimum=1,
            )
            _validate_residual_time(
                row.get("time_s"), step=step, dt_s=dt_s, label="summary"
            )
            key = (step, str(row.get("equation") or "").strip())
            if key in summary_groups or key not in stats_by_group:
                raise NativeFineResidualContractError(
                    f"native Fluent residual summary group is invalid: {key!r}"
                )
            stats = stats_by_group[key]
            sample_count = _csv_integer(
                row.get("sample_count"),
                f"native Fluent residual sample_count for {key!r}",
                minimum=1,
            )
            if sample_count != stats["count"]:
                raise NativeFineResidualContractError(
                    "native Fluent residual sample_count mismatch for "
                    f"{key!r}: summary={sample_count}, history={stats['count']}"
                )
            stored_count = _csv_integer(
                row.get("stored_value_column_count"),
                f"native Fluent residual stored value count for {key!r}",
                minimum=1,
            )
            if stored_count != stats["value_column_count"]:
                raise NativeFineResidualContractError(
                    f"native Fluent residual stored column count mismatch for {key!r}"
                )
            for field in (
                "first_iteration",
                "last_iteration",
                "primary_initial",
                "primary_final",
                "primary_min",
                "primary_max",
            ):
                value = _finite_float(row.get(field), f"residual summary {field}")
                if not math.isclose(
                    value, float(stats[field]), rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise NativeFineResidualContractError(
                        f"native Fluent residual {field} mismatch for {key!r}"
                    )
            data_path = str(row.get("data_path") or "").strip().replace("\\", "/")
            if data_path != stats["data_path"]:
                raise NativeFineResidualContractError(
                    f"native Fluent residual data_path mismatch for {key!r}"
                )
            summary_groups.add(key)
    if summary_groups != set(stats_by_group):
        raise NativeFineResidualContractError(
            "native Fluent residual summary groups do not match residual history"
        )
    return {
        "schema": "native_fluent_residual_history_contract_v1",
        "status": "passed",
        "covered_steps": expected_step_sequence,
        "residual_row_count": residual_row_count,
        "snapshot_summary_row_count": len(summary_groups),
        "equations": sorted(expected_equations),
        "equation_group_cross_check": "passed",
        "coverage_cross_check": "passed",
        "snapshot_consistency_cross_check": "passed",
        "streamed_residual_history": True,
    }


def _validate_residual_time(value: Any, *, step: int, dt_s: float, label: str) -> None:
    time_s = _finite_float(value, f"native Fluent residual {label} time at step {step}")
    if not math.isclose(time_s, step * dt_s, rel_tol=0.0, abs_tol=1.0e-12):
        raise NativeFineResidualContractError(
            f"native Fluent residual {label} time mismatch at step {step}"
        )


def _csv_integer(value: Any, label: str, *, minimum: int) -> int:
    number = _finite_float(value, label)
    if not number.is_integer() or number < minimum:
        raise NativeFineResidualContractError(
            f"{label} must be an integer >= {minimum}: {value!r}"
        )
    return int(number)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeFineResidualContractError(
            f"{label} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise NativeFineResidualContractError(f"{label} is not finite: {result!r}")
    return result
