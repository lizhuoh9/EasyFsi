from __future__ import annotations

from collections.abc import Mapping

from .history import _required_finite_row_number, _row_bool


def sharp_report_fluid_projection_failure_reason(report: object) -> str:
    load_report = getattr(report, "fluid_to_mpm_loads", None)
    projection = getattr(load_report, "fluid_projection", None)
    if not isinstance(projection, Mapping):
        return "missing_fluid_projection_report"

    reasons: list[str] = []
    if bool(projection.get("pressure_solve_failed", False)):
        reasons.append("pressure_solve_failed")
    if bool(projection.get("pressure_projection_physical_failure", False)):
        physical_reason = str(
            projection.get(
                "pressure_projection_physical_failure_reason",
                "",
            )
            or "pressure_projection_physical_failure"
        )
        reasons.append(physical_reason)
    if not bool(projection.get("cg_converged_all", True)):
        reasons.append("cg_converged_all=false")
    cg_breakdown_count = int(projection.get("cg_breakdown_count", 0) or 0)
    if cg_breakdown_count > 0:
        reasons.append(f"cg_breakdown_count={cg_breakdown_count}")
    return "; ".join(reasons)


def _raise_for_step_numerical_guard(
    row: dict[str, object],
    *,
    cfl_limit: float,
    divergence_l2_limit: float,
) -> None:
    step = row.get("step")
    finite_fields = (
        "max_fluid_speed_mps",
        "cfl",
        "divergence_l2",
        "divergence_max_abs",
        "interior_divergence_l2",
        "interior_divergence_max_abs",
        "pressure_correctable_divergence_l2",
        "pressure_correctable_divergence_max_abs",
        "pressure_fixed_divergence_l2",
        "pressure_fixed_divergence_max_abs",
        "interior_pressure_correctable_divergence_l2",
        "interior_pressure_correctable_divergence_max_abs",
        "interior_pressure_fixed_divergence_l2",
        "interior_pressure_fixed_divergence_max_abs",
        "projection_divergence_l2",
        "projection_divergence_max_abs",
        "post_boundary_divergence_l2",
        "post_boundary_divergence_max_abs",
        "post_constraint_divergence_l2",
        "post_constraint_divergence_max_abs",
    )
    values: dict[str, float] = {}
    for field in finite_fields:
        values[field] = _required_finite_row_number(
            row,
            field,
            context=f"step {step} numerical guard",
        )
    if values["cfl"] >= float(cfl_limit):
        raise RuntimeError(
            f"step {step} numerical guard failed: cfl={values['cfl']:.6e} "
            f">= {float(cfl_limit):.6e}"
        )
    if values["interior_divergence_l2"] > float(divergence_l2_limit):
        raise RuntimeError(
            f"step {step} numerical guard failed: interior_divergence_l2="
            f"{values['interior_divergence_l2']:.6e} > {float(divergence_l2_limit):.6e}"
        )
    for converged_field, breakdown_field in (
        (
            "total_pressure_projection_cg_converged_all",
            "total_pressure_projection_cg_breakdown_count",
        ),
        (
            "pressure_projection_cg_converged_all",
            "pressure_projection_cg_breakdown_count",
        ),
    ):
        if converged_field in row and not _row_bool(row[converged_field]):
            raise RuntimeError(
                f"step {step} numerical guard failed: {converged_field}=false"
            )
        if breakdown_field in row:
            breakdown_count = _required_finite_row_number(
                row,
                breakdown_field,
                context=f"step {step} numerical guard",
            )
            if breakdown_count > 0.0:
                raise RuntimeError(
                    f"step {step} numerical guard failed: {breakdown_field}="
                    f"{breakdown_count:.0f}"
                )


def _raise_for_step_solid_out_of_bounds_guard(row: dict[str, object]) -> None:
    step = row.get("step")
    field = "solid_mpm_grid_out_of_bounds_particle_count"
    if field not in row:
        return
    out_of_bounds_count = _required_finite_row_number(
        row,
        field,
        context=f"step {step} numerical guard",
    )
    if out_of_bounds_count > 0.0:
        raise RuntimeError(
            f"step {step} numerical guard failed: {field}="
            f"{out_of_bounds_count:.0f} solid MPM particle(s) outside the solid grid"
        )


def _raise_for_closure_coverage_floor(
    rows: list[dict[str, object]],
    floor: int,
    patience: int,
) -> None:
    """Loud early failure when far-pressure closure coverage stays below a
    floor for `patience` consecutive steps (S2-A11b). The 2s production run
    bled closed markers at 16.2/step for ~110 steps while marching toward an
    unrecoverable state; a healthy run holds the closed count steady. floor=0
    disables the guard (default, bitwise-compatible). A single recovered step
    inside the window resets the streak."""
    if int(floor) <= 0 or int(patience) <= 0:
        return
    if len(rows) < int(patience):
        return
    field = "hibm_full_stress_far_pressure_closed_marker_count"
    recent = rows[-int(patience) :]
    last_value = 0.0
    for row in recent:
        if field not in row:
            return
        value = _required_finite_row_number(
            row,
            field,
            context=f"step {row.get('step')} closure coverage floor guard",
        )
        if value >= float(floor):
            return
        last_value = value
    step = recent[-1].get("step")
    raise RuntimeError(
        f"step {step} closure coverage floor guard failed: "
        f"{field}={last_value:.0f} stayed below floor={int(floor)} for the "
        f"last {int(patience)} consecutive steps"
    )
