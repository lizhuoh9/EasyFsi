"""Classification logic for the R24B oracle-headroom campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .kalman_oracle_headroom_contracts import (
    CG_REDUCTION_MIN,
    FIELD_STATE_NRMSE_MAX,
    MARKER_STATE_NRMSE_MAX,
    NO_SLIP_MAX_MPS,
    TRIAL_REDUCTION_MIN,
    WARM_WALL_REDUCTION_MIN,
    _FIELD_KEYS,
    _FLOW_FIELD_KEYS,
    _as_finite_float,
    _load_run,
    _normalised_rmse,
    _physics_health,
    _reduction,
    _sha256_file,
    _validate_pair,
    _work_metrics,
)

def analyze_oracle_headroom(
    q0_root: Path | str,
    q3_root: Path | str,
) -> dict[str, Any]:
    """Audit a source-matched Q0/Q3 exact8 pair and classify headroom."""

    q0 = _load_run(q0_root, expected_mode="carry_forward")
    q3 = _load_run(q3_root, expected_mode="oracle_replay")
    _validate_pair(q0, q3)
    dt_s = _as_finite_float(q0.config.get("dt_s"), label="dt_s")

    rows: list[dict[str, Any]] = []
    q0_work_rows: list[dict[str, float | int]] = []
    q3_work_rows: list[dict[str, float | int]] = []
    marker_nrmse_max = 0.0
    field_nrmse_max = 0.0
    physics_ok = True
    for q0_step, q3_step in zip(q0.steps, q3.steps):
        q0_work = _work_metrics(q0_step)
        q3_work = _work_metrics(q3_step)
        q0_health = _physics_health(q0_step, dt_s=dt_s)
        q3_health = _physics_health(q3_step, dt_s=dt_s)
        q0_work_rows.append(q0_work)
        q3_work_rows.append(q3_work)
        physics_ok = physics_ok and bool(q0_health["all"]) and bool(q3_health["all"])

        state_errors = {
            key: _normalised_rmse(
                q0_step.arrays[key],
                q3_step.arrays[key],
            )
            for key in (*_FIELD_KEYS, *_FLOW_FIELD_KEYS)
        }
        marker_nrmse_max = max(
            marker_nrmse_max,
            *(state_errors[key] for key in _FIELD_KEYS),
        )
        field_nrmse_max = max(
            field_nrmse_max,
            *(state_errors[key] for key in _FLOW_FIELD_KEYS),
        )
        rows.append(
            {
                "step": q0_step.step,
                "q0_coupling_trials": int(q0_work["coupling_iterations"]),
                "q3_coupling_trials": int(q3_work["coupling_iterations"]),
                "q0_rejected_trials": int(q0_work["rejected_trials"]),
                "q3_rejected_trials": int(q3_work["rejected_trials"]),
                "q0_first_absolute_residual_mps": float(
                    q0_work["first_absolute_residual_mps"]
                ),
                "q3_first_absolute_residual_mps": float(
                    q3_work["first_absolute_residual_mps"]
                ),
                "q0_first_relative_residual": float(
                    q0_work["first_relative_residual"]
                ),
                "q3_first_relative_residual": float(
                    q3_work["first_relative_residual"]
                ),
                "q0_cg_iterations": int(q0_work["cg_iterations_total"]),
                "q3_cg_iterations": int(q3_work["cg_iterations_total"]),
                "q0_fluid_solves": int(q0_work["fluid_solve_count"]),
                "q3_fluid_solves": int(q3_work["fluid_solve_count"]),
                "q0_solid_macro_solves": int(q0_work["solid_macro_solve_count"]),
                "q3_solid_macro_solves": int(q3_work["solid_macro_solve_count"]),
                "q0_flow_momentum_substeps": int(
                    q0_work["flow_momentum_advection_substeps_total"]
                ),
                "q3_flow_momentum_substeps": int(
                    q3_work["flow_momentum_advection_substeps_total"]
                ),
                "q0_flow_sst_substeps": int(
                    q0_work["flow_sst_transport_substeps_total"]
                ),
                "q3_flow_sst_substeps": int(
                    q3_work["flow_sst_transport_substeps_total"]
                ),
                "q0_solid_substeps": int(
                    q0_work["solid_substeps_executed_total"]
                ),
                "q3_solid_substeps": int(
                    q3_work["solid_substeps_executed_total"]
                ),
                "q0_flow_wall_s": float(q0_work["flow_wall_time_s_total"]),
                "q3_flow_wall_s": float(q3_work["flow_wall_time_s_total"]),
                "q0_hibm_wall_s": float(q0_work["hibm_wall_time_s_total"]),
                "q3_hibm_wall_s": float(q3_work["hibm_wall_time_s_total"]),
                "q0_solid_wall_s": float(q0_work["solid_wall_time_s_total"]),
                "q3_solid_wall_s": float(q3_work["solid_wall_time_s_total"]),
                "q0_component_wall_s": float(q0_work["component_wall_s"]),
                "q3_component_wall_s": float(q3_work["component_wall_s"]),
                "marker_velocity_nrmse": state_errors["marker_velocity_mps"],
                "marker_position_nrmse": state_errors["marker_position_m"],
                "solid_position_nrmse": state_errors["solid_position_m"],
                "u_nrmse": state_errors["u"],
                "v_nrmse": state_errors["v"],
                "p_nrmse": state_errors["p"],
                "speed_nrmse": state_errors["speed"],
                "q0_physics_ok": bool(q0_health["all"]),
                "q3_physics_ok": bool(q3_health["all"]),
                "q0_frame_sha256": _sha256_file(q0_step.frame_path),
                "q3_frame_sha256": _sha256_file(q3_step.frame_path),
                "q0_history_sha256": _sha256_file(q0_step.history_path),
                "q3_history_sha256": _sha256_file(q3_step.history_path),
            }
        )

    q0_trials = sum(int(item["coupling_iterations"]) for item in q0_work_rows)
    q3_trials = sum(int(item["coupling_iterations"]) for item in q3_work_rows)
    q0_cg = sum(int(item["cg_iterations_total"]) for item in q0_work_rows)
    q3_cg = sum(int(item["cg_iterations_total"]) for item in q3_work_rows)
    q0_warm_wall = sum(
        float(item["component_wall_s"]) for item in q0_work_rows[1:]
    )
    q3_warm_wall = sum(
        float(item["component_wall_s"]) for item in q3_work_rows[1:]
    )
    trial_reduction = _reduction(float(q0_trials), float(q3_trials))
    cg_reduction = _reduction(float(q0_cg), float(q3_cg))
    warm_wall_reduction = _reduction(q0_warm_wall, q3_warm_wall)
    accepted_state_ok = (
        marker_nrmse_max <= MARKER_STATE_NRMSE_MAX
        and field_nrmse_max <= FIELD_STATE_NRMSE_MAX
    )
    gates = {
        "accepted_state_contract": accepted_state_ok,
        "cg_or_matvec_reduction": cg_reduction >= CG_REDUCTION_MIN,
        "coupling_trial_reduction": trial_reduction >= TRIAL_REDUCTION_MIN,
        "physics_contract": physics_ok,
        "warm_wall_reduction": warm_wall_reduction >= WARM_WALL_REDUCTION_MIN,
    }
    classification = (
        "PASS_ORACLE_HEADROOM"
        if all(gates.values())
        else "STOP_KALMAN_ACCELERATION"
    )
    return {
        "schema_version": 1,
        "campaign": "ansys_vertical_flap_kalman_oracle_headroom_r24b",
        "classification": classification,
        "deployable": False,
        "q0_root": str(q0.root),
        "q3_root": str(q3.root),
        "thresholds": {
            "coupling_trial_reduction_min": TRIAL_REDUCTION_MIN,
            "cg_or_matvec_reduction_min": CG_REDUCTION_MIN,
            "warm_component_wall_reduction_min": WARM_WALL_REDUCTION_MIN,
            "marker_state_nrmse_max": MARKER_STATE_NRMSE_MAX,
            "field_state_nrmse_max": FIELD_STATE_NRMSE_MAX,
            "no_slip_max_mps": NO_SLIP_MAX_MPS,
        },
        "gates": gates,
        "aggregate": {
            "q0_coupling_trials": q0_trials,
            "q3_coupling_trials": q3_trials,
            "coupling_trial_reduction": trial_reduction,
            "q0_rejected_trials": sum(
                int(item["rejected_trials"]) for item in q0_work_rows
            ),
            "q3_rejected_trials": sum(
                int(item["rejected_trials"]) for item in q3_work_rows
            ),
            "q0_first_absolute_residual_mps_mean": float(
                np.mean(
                    [
                        float(item["first_absolute_residual_mps"])
                        for item in q0_work_rows
                    ]
                )
            ),
            "q3_first_absolute_residual_mps_mean": float(
                np.mean(
                    [
                        float(item["first_absolute_residual_mps"])
                        for item in q3_work_rows
                    ]
                )
            ),
            "q0_first_relative_residual_mean": float(
                np.mean(
                    [
                        float(item["first_relative_residual"])
                        for item in q0_work_rows
                    ]
                )
            ),
            "q3_first_relative_residual_mean": float(
                np.mean(
                    [
                        float(item["first_relative_residual"])
                        for item in q3_work_rows
                    ]
                )
            ),
            "q0_cg_iterations": q0_cg,
            "q3_cg_iterations": q3_cg,
            "cg_iteration_reduction": cg_reduction,
            "pressure_matvec_serialized": False,
            "q0_fluid_solves": sum(
                int(item["fluid_solve_count"]) for item in q0_work_rows
            ),
            "q3_fluid_solves": sum(
                int(item["fluid_solve_count"]) for item in q3_work_rows
            ),
            "q0_solid_macro_solves": sum(
                int(item["solid_macro_solve_count"]) for item in q0_work_rows
            ),
            "q3_solid_macro_solves": sum(
                int(item["solid_macro_solve_count"]) for item in q3_work_rows
            ),
            "q0_flow_momentum_substeps": sum(
                int(item["flow_momentum_advection_substeps_total"])
                for item in q0_work_rows
            ),
            "q3_flow_momentum_substeps": sum(
                int(item["flow_momentum_advection_substeps_total"])
                for item in q3_work_rows
            ),
            "q0_flow_sst_substeps": sum(
                int(item["flow_sst_transport_substeps_total"])
                for item in q0_work_rows
            ),
            "q3_flow_sst_substeps": sum(
                int(item["flow_sst_transport_substeps_total"])
                for item in q3_work_rows
            ),
            "q0_solid_substeps": sum(
                int(item["solid_substeps_executed_total"])
                for item in q0_work_rows
            ),
            "q3_solid_substeps": sum(
                int(item["solid_substeps_executed_total"])
                for item in q3_work_rows
            ),
            "q0_warm_component_wall_s": q0_warm_wall,
            "q3_warm_component_wall_s": q3_warm_wall,
            "warm_component_wall_reduction": warm_wall_reduction,
            "marker_state_nrmse_max": marker_nrmse_max,
            "field_state_nrmse_max": field_nrmse_max,
        },
        "steps": rows,
    }
