from __future__ import annotations

import math
from collections.abc import Sequence

from simulation_core import InterfaceReactionRelaxationState, vector_norm

from .cli import INTERFACE_REACTION_ROBIN_TARGET_CHOICES
from .history import _required_finite_row_number
from .rows import signed_positive_source_flux_ratio
from .source_config import _vector3


def _combine_region_pair_vectors(
    primary_vector: Sequence[float],
    secondary_vector: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    primary = _vector3(primary_vector, name="primary_vector")
    secondary = _vector3(secondary_vector, name="secondary_vector")
    return primary + secondary


def robin_previous_velocity_for_step(
    state: InterfaceReactionRelaxationState,
    step_start_velocity_mps: Sequence[float],
) -> tuple[float, ...]:
    step_start_velocity = tuple(float(value) for value in step_start_velocity_mps)
    if not step_start_velocity:
        raise ValueError("step_start_velocity_mps must contain at least one value")
    if any(not math.isfinite(value) for value in step_start_velocity):
        raise ValueError("step_start_velocity_mps must contain only finite values")
    if state.previous_velocity_mps is None:
        return step_start_velocity
    previous_velocity = tuple(float(value) for value in state.previous_velocity_mps)
    if len(previous_velocity) != len(step_start_velocity):
        raise ValueError("previous_velocity_mps and step_start_velocity_mps must match")
    if any(not math.isfinite(value) for value in previous_velocity):
        raise ValueError("previous_velocity_mps must contain only finite values")
    return previous_velocity


def interface_reaction_target_for_mode(
    mode: str,
    *,
    raw_target_force_n: Sequence[float],
    stabilized_target_force_n: Sequence[float],
) -> tuple[float, ...]:
    raw_target = tuple(float(value) for value in raw_target_force_n)
    stabilized_target = tuple(float(value) for value in stabilized_target_force_n)
    if not raw_target:
        raise ValueError("raw_target_force_n must contain at least one value")
    if len(raw_target) != len(stabilized_target):
        raise ValueError("raw_target_force_n and stabilized_target_force_n must match")
    if any(not math.isfinite(value) for value in raw_target + stabilized_target):
        raise ValueError("interface reaction targets must contain only finite values")
    target_mode = str(mode)
    if target_mode == "stabilized":
        return stabilized_target
    if target_mode == "physical":
        return raw_target
    choices = ", ".join(INTERFACE_REACTION_ROBIN_TARGET_CHOICES)
    raise ValueError(f"--interface-reaction-robin-target-mode must be one of: {choices}")


def fsi_same_step_rerun_triggered(
    *,
    current_iterations_requested: int,
    rerun_iterations_max: int,
    residual_norm_n: float,
    residual_threshold_n: float,
    converged: bool,
    safety_rejected: bool = False,
) -> bool:
    """Return whether a projected/reduced FSI step should be rerun in-place."""
    if rerun_iterations_max <= current_iterations_requested:
        return False
    if safety_rejected:
        return True
    if not math.isfinite(residual_threshold_n):
        return False
    residual = float(residual_norm_n)
    if math.isnan(residual):
        return False
    return (not bool(converged)) and residual > residual_threshold_n


def fsi_same_step_rerun_fluid_substeps(
    *,
    current_substeps: int,
    max_substeps: int,
    substep_factor: float,
    safety_rejected: bool,
) -> int:
    """Return same-step fluid substeps after an all-rejected safety attempt."""
    current = int(current_substeps)
    maximum = int(max_substeps)
    factor = float(substep_factor)
    if current < 1 or maximum <= current:
        return current
    if not safety_rejected:
        return current
    if not math.isfinite(factor) or factor <= 1.0:
        return current
    requested = max(current + 1, int(math.ceil(float(current) * factor)))
    return min(maximum, requested)


def _split_region_pair_vector(
    values: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 6:
        raise ValueError("region-pair force vector must contain 6 values")
    return (
        (vector[0], vector[1], vector[2]),
        (vector[3], vector[4], vector[5]),
    )


def _taichi_vector3_to_tuple(value: object) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


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


def fsi_physical_interface_map_stability_report(
    *,
    fsi_coupling_enabled: bool,
    fsi_coupling_iterations: int,
    max_physical_interface_map_amplification: float,
    measurement_sample_count: int,
    raw_interface_map_strict_physical: bool,
    limit: float = 1.0,
) -> dict[str, object]:
    enabled = bool(fsi_coupling_enabled)
    iterations = int(fsi_coupling_iterations)
    samples = int(measurement_sample_count)
    strict_physical = bool(raw_interface_map_strict_physical)
    if not enabled:
        return {
            "applicable": False,
            "measured": False,
            "passes": True,
            "status": "not_applicable_coupling_disabled",
            "reason": "fsi_coupling_disabled",
            "sample_count": samples,
        }
    if iterations <= 1 or samples <= 0:
        return {
            "applicable": True,
            "measured": False,
            "passes": False,
            "status": "unmeasured",
            "reason": "insufficient_distinct_fsi_trials",
            "sample_count": samples,
        }
    if not strict_physical:
        return {
            "applicable": True,
            "measured": True,
            "passes": False,
            "status": "masked_by_stabilizer",
            "reason": "raw_interface_map_not_strict_physical",
            "sample_count": samples,
        }
    amplification = float(max_physical_interface_map_amplification)
    stable_limit = float(limit)
    passes = (
        math.isfinite(amplification)
        and math.isfinite(stable_limit)
        and stable_limit >= 0.0
        and amplification <= stable_limit
    )
    return {
        "applicable": True,
        "measured": True,
        "passes": passes,
        "status": "stable" if passes else "unstable",
        "reason": "measured_raw_interface_map",
        "sample_count": samples,
        "amplification": amplification,
        "limit": stable_limit,
    }


def fsi_physical_interface_map_stability_passes(
    *,
    fsi_coupling_enabled: bool,
    fsi_coupling_iterations: int,
    max_physical_interface_map_amplification: float,
    measurement_sample_count: int,
    raw_interface_map_strict_physical: bool = True,
    limit: float = 1.0,
) -> bool:
    return bool(
        fsi_physical_interface_map_stability_report(
            fsi_coupling_enabled=fsi_coupling_enabled,
            fsi_coupling_iterations=fsi_coupling_iterations,
            max_physical_interface_map_amplification=max_physical_interface_map_amplification,
            measurement_sample_count=measurement_sample_count,
            raw_interface_map_strict_physical=raw_interface_map_strict_physical,
            limit=limit,
        )["passes"]
    )


def solid_response_constraint_force_mobility_ratio(
    *,
    previous_velocity_mps: Sequence[float],
    current_velocity_mps: Sequence[float],
    reaction_force_n: Sequence[float],
    interface_area_m2: float,
    probe_distance_m: float,
    density_kgm3: float,
    dt_s: float,
    axis: int = 2,
    min_abs_reaction_force_n: float = 1.0e-12,
) -> float:
    previous_velocity = _vector3(previous_velocity_mps, name="previous_velocity_mps")
    current_velocity = _vector3(current_velocity_mps, name="current_velocity_mps")
    reaction_force = _vector3(reaction_force_n, name="reaction_force_n")
    if any(not math.isfinite(value) for value in previous_velocity):
        raise ValueError("previous_velocity_mps must contain only finite values")
    if any(not math.isfinite(value) for value in current_velocity):
        raise ValueError("current_velocity_mps must contain only finite values")
    if any(not math.isfinite(value) for value in reaction_force):
        raise ValueError("reaction_force_n must contain only finite values")
    axis_index = int(axis)
    if axis_index not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    area = float(interface_area_m2)
    probe_distance = float(probe_distance_m)
    density = float(density_kgm3)
    dt = float(dt_s)
    min_force = float(min_abs_reaction_force_n)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError("interface_area_m2 must be a finite non-negative number")
    if not math.isfinite(probe_distance) or probe_distance <= 0.0:
        raise ValueError("probe_distance_m must be a finite positive number")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density_kgm3 must be a finite positive number")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be a finite positive number")
    if not math.isfinite(min_force) or min_force < 0.0:
        raise ValueError("min_abs_reaction_force_n must be a finite non-negative number")
    force_component = reaction_force[axis_index]
    if abs(force_component) <= min_force or area <= 0.0:
        return 0.0
    velocity_delta = current_velocity[axis_index] - previous_velocity[axis_index]
    solid_mobility_mps_per_n = abs(velocity_delta / force_component)
    fluid_interface_stiffness_n_per_mps = density * area * probe_distance / dt
    return solid_mobility_mps_per_n * fluid_interface_stiffness_n_per_mps


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
