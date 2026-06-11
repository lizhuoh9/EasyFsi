from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FieldDiagnostic:
    step: Any
    field: str
    value: Any
    reason: str


@dataclass(frozen=True)
class BoundaryDriveComplianceReport:
    prescribed_velocity_boundary: bool
    prescribed_pressure_or_flow_boundary: bool
    nonzero_fluid_traction_scale: float
    compliant: bool


def vector_norm(values: Any) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def finite_field_diagnostics(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        for field in fields:
            if field not in row:
                diagnostics.append(
                    asdict(
                        FieldDiagnostic(
                            step=row.get("step"),
                            field=field,
                            value=None,
                            reason="missing",
                        )
                    )
                )
                continue
            value = row[field]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                diagnostics.append(
                    asdict(
                        FieldDiagnostic(
                            step=row.get("step"),
                            field=field,
                            value=value,
                            reason="not_numeric",
                        )
                    )
                )
                continue
            if not math.isfinite(numeric):
                diagnostics.append(
                    asdict(
                        FieldDiagnostic(
                            step=row.get("step"),
                            field=field,
                            value=numeric,
                            reason="nonfinite",
                        )
                    )
                )
    return diagnostics


def force_nonzero_when_loaded(
    *,
    force_components_n: Any,
    load_value: float,
    force_required: bool,
    tolerance_n: float = 0.0,
) -> bool:
    if not force_required:
        return True
    if abs(float(load_value)) <= 0.0:
        return True
    return vector_norm(force_components_n) > float(tolerance_n)


def boundary_drive_compliance_report(
    *,
    prescribed_velocity_boundary: bool,
    prescribed_pressure_or_flow_boundary: bool,
    nonzero_fluid_traction_scale: float,
    force_scale_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    force_scale = float(nonzero_fluid_traction_scale)
    report = BoundaryDriveComplianceReport(
        prescribed_velocity_boundary=bool(prescribed_velocity_boundary),
        prescribed_pressure_or_flow_boundary=bool(prescribed_pressure_or_flow_boundary),
        nonzero_fluid_traction_scale=force_scale,
        compliant=(
            not bool(prescribed_velocity_boundary)
            and not bool(prescribed_pressure_or_flow_boundary)
            and abs(force_scale) <= float(force_scale_tolerance)
        ),
    )
    return asdict(report)


def checks_passed(checks: dict[str, Any]) -> bool:
    return all(bool(value) for value in checks.values())
