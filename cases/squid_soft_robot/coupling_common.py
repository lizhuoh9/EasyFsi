from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING
from .history import _required_finite_row_number
from .rows import signed_positive_source_flux_ratio

if TYPE_CHECKING:
    from .spec import SquidReducedSpec


def hydraulic_diagnostics(
    spec: SquidReducedSpec,
    main_velocity_z_mps: float,
) -> tuple[float, float, float]:
    q_m3s = -float(spec.main_area_m2) * float(main_velocity_z_mps)
    nozzle_speed_mps = q_m3s / max(float(spec.nozzle_area_m2), 1.0e-12)
    viscous_dp_pa = (
        8.0
        * float(spec.water_viscosity_pa_s)
        * float(spec.nozzle_length_m)
        * q_m3s
        / max(math.pi * float(spec.nozzle_radius_m) ** 4, 1.0e-18)
    )
    inertial_dp_pa = 0.5 * float(spec.water_density_kgm3) * nozzle_speed_mps * abs(nozzle_speed_mps)
    return viscous_dp_pa + inertial_dp_pa, q_m3s, -nozzle_speed_mps


def physical_positive_source_flux_ratio_passes(
    *,
    outlet_negative_z_flux_m3s: float,
    source_flux_m3s: float,
    min_ratio: float,
    min_source_flux_m3s: float = 1.0e-18,
) -> bool:
    outlet_flux = float(outlet_negative_z_flux_m3s)
    source_flux = float(source_flux_m3s)
    ratio = signed_positive_source_flux_ratio(
        outlet_negative_z_flux_m3s=outlet_flux,
        source_flux_m3s=source_flux,
        min_source_flux_m3s=min_source_flux_m3s,
    )
    return (
        math.isfinite(outlet_flux)
        and math.isfinite(source_flux)
        and source_flux > float(min_source_flux_m3s)
        and outlet_flux > 0.0
        and ratio >= float(min_ratio)
    )


def physical_outlet_to_fsi_volume_source_passes(
    *,
    outlet_negative_z_flux_m3s: float,
    fsi_volume_source_m3s: float,
    min_ratio: float,
) -> bool:
    return physical_positive_source_flux_ratio_passes(
        outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
        source_flux_m3s=fsi_volume_source_m3s,
        min_ratio=min_ratio,
    )


def outlet_to_fsi_volume_source_gate_scope(
    *,
    fluid_grid_resolution: dict[str, object],
    validation_scope_complete: bool,
) -> dict[str, object]:
    nozzle_resolved = bool(fluid_grid_resolution.get("nozzle_resolves_diameter_10_cells", False))
    reasons: list[str] = []
    if not nozzle_resolved:
        reasons.append("nozzle_grid_not_resolved")
    if not bool(validation_scope_complete):
        reasons.append("jet_development_scope_incomplete")
    hard_gate = not reasons
    return {
        "gate": "completed_step_check" if hard_gate else "diagnostic_only",
        "hard_gate": hard_gate,
        "nozzle_resolved": nozzle_resolved,
        "jet_development_evaluable": bool(validation_scope_complete),
        "nozzle_diameter_cells_min": int(
            fluid_grid_resolution.get("nozzle_diameter_cells_min", 0) or 0
        ),
        "reasons": reasons,
    }


def pressure_outlet_source_ratio_passes(
    *,
    source_volume_flux_m3s: float,
    velocity_outlet_flux_m3s: float,
    pressure_outlet_flux_m3s: float,
    ratio_tolerance: float,
    min_source_flux_m3s: float = 1.0e-18,
) -> bool:
    source_flux = float(source_volume_flux_m3s)
    velocity_flux = float(velocity_outlet_flux_m3s)
    pressure_flux = float(pressure_outlet_flux_m3s)
    tolerance = float(ratio_tolerance)
    # The physical conservation gate uses the final outlet-face velocity flux.
    # The pressure-correction flux is a diagnostic and may be small for the
    # open z-min projection; require it to be finite without treating it as an
    # independent mass-conservation flux.
    if (
        not math.isfinite(source_flux)
        or not math.isfinite(velocity_flux)
        or not math.isfinite(pressure_flux)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or source_flux <= float(min_source_flux_m3s)
        or velocity_flux <= 0.0
    ):
        return False
    velocity_ratio = velocity_flux / source_flux
    return abs(velocity_ratio - 1.0) <= tolerance


def pressure_flux_trend_report(
    rows: Sequence[dict[str, object]],
    *,
    requested_steps: int,
    min_trend_steps: int = 200,
    near_zero_pressure_ratio: float = 1.0e-3,
    rising_pressure_ratio: float = 1.0e-2,
    growth_factor: float = 5.0,
) -> dict[str, object]:
    required_steps = max(1, int(min_trend_steps))
    requested = int(requested_steps)
    completed = len(rows)
    report: dict[str, object] = {
        "required_steps": required_steps,
        "requested_steps": requested,
        "completed_steps": completed,
        "complete": completed >= required_steps and requested >= required_steps,
    }
    if not report["complete"]:
        report.update(
            {
                "conclusion": "incomplete",
                "reason": "insufficient_completed_steps_for_pressure_flux_trend",
            }
        )
        return report

    pressure_ratio_abs = [
        abs(
            _required_finite_row_number(
                row,
                "pressure_outlet_pressure_to_source_ratio",
                context=f"pressure-flux trend row {index}",
            )
        )
        for index, row in enumerate(rows)
    ]
    velocity_ratio = [
        _required_finite_row_number(
            row,
            "pressure_outlet_velocity_to_source_ratio",
            context=f"pressure-flux trend row {index}",
        )
        for index, row in enumerate(rows)
    ]
    pressure_load = [
        _required_finite_row_number(
            row,
            "pressure_load_pa",
            context=f"pressure-flux trend row {index}",
        )
        for index, row in enumerate(rows)
    ]
    window = max(1, min(20, completed // 10))
    early_pressure_ratio_mean = sum(pressure_ratio_abs[:window]) / float(window)
    late_pressure_ratio_mean = sum(pressure_ratio_abs[-window:]) / float(window)
    max_pressure_ratio = max(pressure_ratio_abs)
    min_pressure_load = min(pressure_load)
    max_pressure_load = max(pressure_load)
    growth_denominator = max(early_pressure_ratio_mean, 1.0e-12)
    pressure_ratio_growth = late_pressure_ratio_mean / growth_denominator
    pressure_ratio_rise = late_pressure_ratio_mean - early_pressure_ratio_mean
    final_velocity_ratio = velocity_ratio[-1]
    report.update(
        {
            "window_steps": window,
            "early_pressure_ratio_mean_abs": early_pressure_ratio_mean,
            "late_pressure_ratio_mean_abs": late_pressure_ratio_mean,
            "max_pressure_ratio_abs": max_pressure_ratio,
            "pressure_ratio_rise_abs": pressure_ratio_rise,
            "pressure_ratio_growth_factor": pressure_ratio_growth,
            "min_pressure_load_pa": min_pressure_load,
            "max_pressure_load_pa": max_pressure_load,
            "pressure_load_range_pa": max_pressure_load - min_pressure_load,
            "final_velocity_to_source_ratio": final_velocity_ratio,
            "final_pressure_to_source_ratio": pressure_ratio_abs[-1],
        }
    )
    if max_pressure_ratio <= float(near_zero_pressure_ratio):
        report.update(
            {
                "conclusion": "pressure_implied_flux_remained_near_zero_kinematic_ibm_dominated",
                "reason": None,
            }
        )
    elif (
        late_pressure_ratio_mean >= float(rising_pressure_ratio)
        and pressure_ratio_growth >= float(growth_factor)
        and pressure_ratio_rise > 0.0
    ):
        report.update(
            {
                "conclusion": "pressure_implied_flux_rose_pressure_driven_component_present",
                "reason": None,
            }
        )
    else:
        report.update(
            {
                "conclusion": "pressure_implied_flux_trend_inconclusive",
                "reason": "pressure_ratio_not_near_zero_but_not_a_clear_late_rise",
            }
        )
    return report
