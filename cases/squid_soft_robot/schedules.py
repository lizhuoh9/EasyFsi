from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .history import _required_finite_row_number
from .source_config import _mapping_config_float

def pressure_schedule_from_config(
    config: Mapping[str, object],
    analysis: Mapping[str, object],
) -> dict[str, float]:
    defaults = {
        "pressure_t0_s": 0.0,
        "pressure_t1_s": 1.0,
        "pressure_t2_s": 2.0,
        "pressure_p0_pa": 0.0,
        "pressure_p1_pa": 8000.0,
        "pressure_p2_pa": -8000.0,
    }
    sources: list[Mapping[str, object]] = []
    top_schedule = config.get("pressure_schedule", {})
    if isinstance(top_schedule, Mapping):
        sources.append(top_schedule)
    sources.append(config)
    sources.append(analysis)
    analysis_schedule = analysis.get("pressure_schedule", {})
    if isinstance(analysis_schedule, Mapping):
        sources.append(analysis_schedule)

    schedule = dict(defaults)
    aliases = {
        "pressure_t0_s": ("pressure_t0_s", "t0_s"),
        "pressure_t1_s": ("pressure_t1_s", "t1_s"),
        "pressure_t2_s": ("pressure_t2_s", "t2_s"),
        "pressure_p0_pa": ("pressure_p0_pa", "p0_pa"),
        "pressure_p1_pa": ("pressure_p1_pa", "p1_pa"),
        "pressure_p2_pa": ("pressure_p2_pa", "p2_pa"),
    }
    for source in sources:
        schedule = {
            field: _mapping_config_float(
                source,
                aliases[field],
                value,
                field=field,
            )
            for field, value in schedule.items()
        }
    if not (
        schedule["pressure_t0_s"]
        < schedule["pressure_t1_s"]
        < schedule["pressure_t2_s"]
    ):
        raise ValueError("pressure schedule times must satisfy t0 < t1 < t2")
    return schedule

PRESSURE_SCHEDULE_FIELDS = (
    "pressure_t0_s",
    "pressure_t1_s",
    "pressure_t2_s",
    "pressure_p0_pa",
    "pressure_p1_pa",
    "pressure_p2_pa",
)

def pressure_schedule_dict(spec: SquidReducedSpec) -> dict[str, float]:
    return {field: float(getattr(spec, field)) for field in PRESSURE_SCHEDULE_FIELDS}

def spec_with_pressure_schedule_overrides(
    spec: SquidReducedSpec,
    overrides: Mapping[str, object],
) -> tuple[SquidReducedSpec, dict[str, object]]:
    base_schedule = pressure_schedule_dict(spec)
    applied: dict[str, float] = {}
    for field in PRESSURE_SCHEDULE_FIELDS:
        value = overrides.get(field)
        if value is None:
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{field} must be finite")
        applied[field] = parsed
    if not applied:
        return spec, {
            "source": "source_config",
            "cli_override_applied": False,
            "schedule": base_schedule,
            "overrides": {},
            "boundary_condition_input_only": True,
            "computed_response_fields": (
                "tail force, fluid velocity, outlet flow, and jet diagnostics",
            ),
        }
    schedule = {**base_schedule, **applied}
    if not (
        schedule["pressure_t0_s"]
        < schedule["pressure_t1_s"]
        < schedule["pressure_t2_s"]
    ):
        raise ValueError("pressure schedule times must satisfy t0 < t1 < t2")
    return replace(spec, **schedule), {
        "source": "source_config_plus_cli_override",
        "cli_override_applied": True,
        "schedule": schedule,
        "base_source_config_schedule": base_schedule,
        "overrides": applied,
        "boundary_condition_input_only": True,
        "computed_response_fields": (
            "tail force, fluid velocity, outlet flow, and jet diagnostics",
        ),
    }

def pressure_schedule_applied_in_history(rows: Sequence[dict[str, object]]) -> bool:
    for index, row in enumerate(rows):
        pressure_pa = _required_finite_row_number(
            row,
            "pressure_load_pa",
            context=f"history row {index}",
        )
        if abs(pressure_pa) > 0.0:
            return True
    return False

def pressure_schedule_pa(time_s: float, spec: SquidReducedSpec | None = None) -> float:
    if spec is None:
        t0_s, t1_s, t2_s = 0.0, 1.0, 2.0
        p0_pa, p1_pa, p2_pa = 0.0, 8000.0, -8000.0
    else:
        t0_s = float(spec.pressure_t0_s)
        t1_s = float(spec.pressure_t1_s)
        t2_s = float(spec.pressure_t2_s)
        p0_pa = float(spec.pressure_p0_pa)
        p1_pa = float(spec.pressure_p1_pa)
        p2_pa = float(spec.pressure_p2_pa)
    if not (t0_s < t1_s < t2_s):
        raise ValueError("pressure schedule times must satisfy t0 < t1 < t2")
    time = float(time_s)
    if time <= t0_s:
        return p0_pa
    if time <= t1_s:
        alpha = (time - t0_s) / (t1_s - t0_s)
        return p0_pa + (p1_pa - p0_pa) * alpha
    if time <= t2_s:
        alpha = (time - t1_s) / (t2_s - t1_s)
        return p1_pa + (p2_pa - p1_pa) * alpha
    return p2_pa

def pressure_schedule_step_end_pa(
    current_time_s: float,
    dt_s: float,
    spec: SquidReducedSpec | None = None,
) -> float:
    return pressure_schedule_pa(float(current_time_s) + float(dt_s), spec)
