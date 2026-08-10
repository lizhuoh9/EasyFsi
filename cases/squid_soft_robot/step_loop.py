from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence

import numpy as np

from simulation_core import (
    InterfaceReactionFixedPointResult,
    InterfaceReactionTargetEvaluation,
    action_reaction_balance,
    hibm_mpm_sharp_step_summary,
    robin_neumann_impedance_force,
    solve_and_apply_interface_reaction_step,
    update_interface_reaction_for_next_step,
    vector_norm,
)

from .checkpointing import (
    _sharp_marker_aitken_relaxation,
    _sharp_marker_fixed_point_residual_vector_mps,
    relaxed_sharp_marker_state_arrays,
    relaxed_sharp_pressure_neumann_gradient_state_array,
    restore_sharp_marker_state_arrays,
    restore_sharp_pressure_neumann_gradient_state_array,
    sharp_marker_fixed_point_residual_diagnostics_mps,
    sharp_marker_fixed_point_residual_mps,
    sharp_marker_state_arrays,
    sharp_pressure_neumann_gradient_state_array,
    write_run_checkpoint,
)
from .coupling_common import (
    _combine_region_pair_vectors,
    _split_region_pair_vector,
    _taichi_vector3_to_tuple,
    fsi_same_step_rerun_fluid_substeps,
    fsi_same_step_rerun_triggered,
    interface_reaction_target_for_mode,
    robin_previous_velocity_for_step,
    solid_response_constraint_force_mobility_ratio,
)
from .coupling_legacy import legacy_projected_reduced_fsi_coupling_enabled
from .diagnostics import (
    _raise_for_closure_coverage_floor,
    _raise_for_step_numerical_guard,
    _raise_for_step_solid_out_of_bounds_guard,
    force_decomposition_report,
    fsi_trial_acceptance_rejection_reason,
    sharp_report_fluid_projection_failure_reason,
)
from .fluid_step import z_displacement_vector, z_velocity_vector
from .history import (
    _required_finite_report_number,
    required_fluid_impulse_report,
    required_projected_ibm_force_report,
    solid_force_vector_from_report,
    write_csv,
)
from .rows import build_hibm_mpm_sharp_case_row, signed_positive_source_flux_ratio
from .schedules import pressure_schedule_step_end_pa
from .snapshots import (
    _write_fluid_snapshot_npz,
    _write_hibm_high_residual_cell_dump,
    _write_hibm_pressure_neumann_invalid_row_dump,
    _write_hibm_zero_correctable_cell_dump,
    _write_step_failure_artifacts,
)
from .source_config import _vector3
from .step_context import StepLoopContext, StepLoopResult
from .trial_replay import accepted_trial_replay_reports, coerce_accepted_trial_payload


def run_squid_step_loop(context: StepLoopContext) -> StepLoopResult:
    """Run the squid case time-step loop using an explicit typed context."""
    settings = context.settings
    resources = context.resources
    callbacks = context.callbacks
    state = context.state

    adaptive_fluid_substeps_enabled = settings.adaptive_fluid_substeps_enabled
    effective_fluid_substeps = settings.effective_fluid_substeps
    estimated_solid_particle_spacing_m = settings.estimated_solid_particle_spacing_m
    fluid_probe_distance_m = settings.fluid_probe_distance_m
    fsi_coupling_adaptive_iterations_cfl_threshold = (
        settings.fsi_coupling_adaptive_iterations_cfl_threshold
    )
    fsi_coupling_adaptive_iterations_max = settings.fsi_coupling_adaptive_iterations_max
    fsi_coupling_adaptive_iterations_residual_threshold_n = (
        settings.fsi_coupling_adaptive_iterations_residual_threshold_n
    )
    fsi_coupling_iterations = settings.fsi_coupling_iterations
    fsi_coupling_max_accepted_residual_n = settings.fsi_coupling_max_accepted_residual_n
    fsi_coupling_mode = settings.fsi_coupling_mode
    fsi_coupling_rejected_trial_backtrack = (
        settings.fsi_coupling_rejected_trial_backtrack
    )
    fsi_coupling_residual_continuation_iterations_max = (
        settings.fsi_coupling_residual_continuation_iterations_max
    )
    fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max = settings.fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
    fsi_coupling_residual_continuation_rebound_secant_factor = (
        settings.fsi_coupling_residual_continuation_rebound_secant_factor
    )
    fsi_coupling_residual_continuation_rebound_secant_from_best = (
        settings.fsi_coupling_residual_continuation_rebound_secant_from_best
    )
    fsi_coupling_residual_continuation_threshold_n = (
        settings.fsi_coupling_residual_continuation_threshold_n
    )
    fsi_coupling_residual_growth_rejection_factor = (
        settings.fsi_coupling_residual_growth_rejection_factor
    )
    fsi_coupling_same_step_rerun_fluid_substep_factor = (
        settings.fsi_coupling_same_step_rerun_fluid_substep_factor
    )
    fsi_coupling_same_step_rerun_iterations_max = (
        settings.fsi_coupling_same_step_rerun_iterations_max
    )
    fsi_coupling_same_step_rerun_residual_threshold_n = (
        settings.fsi_coupling_same_step_rerun_residual_threshold_n
    )
    fsi_coupling_solver = settings.fsi_coupling_solver
    fsi_coupling_target_map_relaxation = settings.fsi_coupling_target_map_relaxation
    fsi_coupling_trial_interior_divergence_tolerance = (
        settings.fsi_coupling_trial_interior_divergence_tolerance
    )
    fsi_coupling_trust_region_adaptive = settings.fsi_coupling_trust_region_adaptive
    fsi_coupling_trust_region_force_increment_n = (
        settings.fsi_coupling_trust_region_force_increment_n
    )
    fsi_coupling_trust_region_growth_factor = (
        settings.fsi_coupling_trust_region_growth_factor
    )
    fsi_coupling_trust_region_rebound_backtrack = (
        settings.fsi_coupling_trust_region_rebound_backtrack
    )
    fsi_coupling_trust_region_rebound_factor = (
        settings.fsi_coupling_trust_region_rebound_factor
    )
    fsi_coupling_trust_region_rebound_stop_factor = (
        settings.fsi_coupling_trust_region_rebound_stop_factor
    )
    fsi_coupling_trust_region_rebound_stop_max_residual_n = (
        settings.fsi_coupling_trust_region_rebound_stop_max_residual_n
    )
    fsi_coupling_trust_region_shrink_factor = (
        settings.fsi_coupling_trust_region_shrink_factor
    )
    fsi_marker_coupling_tolerance_mps = settings.fsi_marker_coupling_tolerance_mps
    fsi_solid_response_mobility_coupling = settings.fsi_solid_response_mobility_coupling
    fsi_solid_response_velocity_mobility_coupling = (
        settings.fsi_solid_response_velocity_mobility_coupling
    )
    fsi_velocity_target_solid_mobility_ratio = (
        settings.fsi_velocity_target_solid_mobility_ratio
    )
    full_pressure_waveform_steps = settings.full_pressure_waveform_steps
    interface_reaction_aitken = settings.interface_reaction_aitken
    interface_reaction_aitken_lower_bound = (
        settings.interface_reaction_aitken_lower_bound
    )
    interface_reaction_aitken_upper_bound = (
        settings.interface_reaction_aitken_upper_bound
    )
    interface_reaction_passivity_limit = settings.interface_reaction_passivity_limit
    interface_reaction_relaxation = settings.interface_reaction_relaxation
    interface_reaction_robin_impedance_ns_m = (
        settings.interface_reaction_robin_impedance_ns_m
    )
    interface_reaction_robin_matrix_impedance_ns_m = (
        settings.interface_reaction_robin_matrix_impedance_ns_m
    )
    interface_reaction_robin_target_mode = settings.interface_reaction_robin_target_mode
    max_wall_time_s = settings.max_wall_time_s
    pressure_outlet_zmin_enabled = settings.pressure_outlet_zmin_enabled
    primary_fsi_face_area_m2 = settings.primary_fsi_face_area_m2
    primary_shell_region_id = settings.primary_shell_region_id
    reuse_accepted_fsi_trial_state = settings.reuse_accepted_fsi_trial_state
    secondary_fsi_face_area_m2 = settings.secondary_fsi_face_area_m2
    secondary_shell_region_id = settings.secondary_shell_region_id
    sharp_case_runner_enabled = settings.sharp_case_runner_enabled
    solid_response_dt_s = settings.solid_response_dt_s
    solid_sub_dt_s = settings.solid_sub_dt_s
    solid_substep_velocity_damping = settings.solid_substep_velocity_damping
    step_count = settings.step_count

    args = resources.args
    fluid_substep_controller = resources.fluid_substep_controller
    fsi_coupling_mode_report = resources.fsi_coupling_mode_report
    history_path = resources.history_path
    material = resources.material
    output_dir = resources.output_dir
    process_path = resources.process_path
    run_checkpoint_path = resources.run_checkpoint_path
    frozen_run_fingerprint = resources.frozen_run_fingerprint
    simulator = resources.simulator
    solid_mpm = resources.solid_mpm
    spec = resources.spec
    tri_diagnostics = resources.tri_diagnostics

    advance_fluid_step = callbacks.advance_fluid_step
    advance_physical_solid_step = callbacks.advance_physical_solid_step

    rows = state.rows
    sharp_coupling_state = state.sharp_coupling_state
    interface_reaction_state = state.interface_reaction_state
    partial_run_reason = state.partial_run_reason
    partial_run_stopped = state.partial_run_stopped
    previous_step_cfl = state.previous_step_cfl
    previous_step_fluid_substeps = state.previous_step_fluid_substeps
    previous_step_fsi_coupling_residual_norm_n = (
        state.previous_step_fsi_coupling_residual_norm_n
    )
    for step in range(state.first_step, step_count + 1):
        step_wall_started_at = time.perf_counter()
        step_fluid_substeps = effective_fluid_substeps
        if fluid_substep_controller is not None:
            step_fluid_substeps = fluid_substep_controller.substeps_for_next_step(
                previous_cfl=previous_step_cfl,
                previous_substeps=previous_step_fluid_substeps,
            )
        step_fluid_substep_dt_s = float(spec.dt_s) / float(step_fluid_substeps)
        step_fsi_coupling_iterations = fsi_coupling_iterations
        fsi_coupling_adaptive_iterations_residual_triggered = (
            previous_step_fsi_coupling_residual_norm_n is not None
            and math.isfinite(previous_step_fsi_coupling_residual_norm_n)
            and math.isfinite(fsi_coupling_adaptive_iterations_residual_threshold_n)
            and previous_step_fsi_coupling_residual_norm_n
            > fsi_coupling_adaptive_iterations_residual_threshold_n
        )
        fsi_coupling_adaptive_iterations_cfl_triggered = (
            previous_step_cfl is not None
            and math.isfinite(previous_step_cfl)
            and math.isfinite(fsi_coupling_adaptive_iterations_cfl_threshold)
            and previous_step_cfl > fsi_coupling_adaptive_iterations_cfl_threshold
        )
        fsi_coupling_adaptive_iterations_triggered = (
            fsi_coupling_adaptive_iterations_max > fsi_coupling_iterations
            and (
                fsi_coupling_adaptive_iterations_residual_triggered
                or fsi_coupling_adaptive_iterations_cfl_triggered
            )
        )
        if fsi_coupling_adaptive_iterations_triggered:
            step_fsi_coupling_iterations = fsi_coupling_adaptive_iterations_max
        fsi_coupling_same_step_rerun_triggered = False
        fsi_coupling_same_step_rerun_count = 0
        fsi_coupling_same_step_rerun_initial_iterations_requested = (
            step_fsi_coupling_iterations
        )
        fsi_coupling_same_step_rerun_initial_iterations_used = 0
        fsi_coupling_same_step_rerun_initial_residual_norm_n = math.nan
        fsi_coupling_same_step_rerun_initial_converged = False
        fsi_coupling_same_step_rerun_safety_rejected = False
        fsi_coupling_same_step_rerun_initial_fluid_substeps = step_fluid_substeps
        fsi_coupling_same_step_rerun_final_fluid_substeps = step_fluid_substeps
        fsi_coupling_wall_time_s = 0.0
        solid_advance_wall_time_s = 0.0
        fluid_advance_wall_time_s = 0.0
        sample_wall_time_s = 0.0
        surface_diagnostics_wall_time_s = 0.0
        checkpoint_wall_time_s = 0.0
        solid_mpm_report = None
        velocity_constraint_report = None
        velocity_constraint_spread_report = None
        fsi_coupling_iterations_used = 1
        fsi_coupling_converged = step_fsi_coupling_iterations <= 1
        fsi_coupling_residual_norm_n = 0.0
        fsi_coupling_relaxation_effective = interface_reaction_relaxation
        fsi_coupling_iqn_ils_least_squares_update_count = 0
        fsi_coupling_interface_map_amplification = 0.0
        fsi_coupling_residual_jacobian_amplification = 0.0
        fsi_coupling_physical_interface_map_amplification = 0.0
        fsi_coupling_physical_residual_jacobian_amplification = 0.0
        fsi_coupling_raw_interface_map_amplification = 0.0
        fsi_coupling_raw_residual_jacobian_amplification = 0.0
        fsi_coupling_interface_map_amplification_sample_count = 0
        fsi_coupling_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_physical_interface_map_amplification_sample_count = 0
        fsi_coupling_physical_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_raw_interface_map_amplification_sample_count = 0
        fsi_coupling_raw_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_rejected_trial_count = 0
        fsi_coupling_rejected_trial_backtrack_count = 0
        fsi_coupling_residual_growth_rejected_trial_count = 0
        fsi_coupling_max_residual_rejected_trial_count = 0
        fsi_coupling_trial_cfl_rejected_count = 0
        fsi_coupling_trial_interior_divergence_rejected_count = 0
        fsi_coupling_trust_region_limited_update_count = 0
        fsi_coupling_trust_region_shrink_count = 0
        fsi_coupling_trust_region_growth_count = 0
        fsi_coupling_trust_region_rebound_backtrack_count = 0
        fsi_coupling_trust_region_rebound_stop_count = 0
        fsi_coupling_trust_region_rebound_stop_suppressed_count = 0
        fsi_coupling_residual_continuation_iteration_count = 0
        fsi_coupling_residual_continuation_rebound_secant_count = 0
        fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = 0
        fsi_coupling_trust_region_effective_force_increment_n = (
            fsi_coupling_trust_region_force_increment_n
        )
        fsi_coupling_accepted_trial_cfl = math.nan
        fsi_coupling_accepted_trial_max_fluid_speed_mps = math.nan
        fsi_coupling_accepted_trial_interior_divergence_l2 = math.nan
        fsi_coupling_trial_cfl_max = math.nan
        fsi_coupling_trial_interior_divergence_l2_max = math.nan
        fsi_coupling_trial_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fixed_point_result = None
        accepted_fsi_trial_payload: dict[str, object] | None = None
        accepted_fsi_trial_state_reused = False
        accepted_fsi_trial_state_readvanced = False
        fsi_all_trials_rejected = False
        fsi_zero_force_commit_blocked = False
        fsi_trial_pressure_projection_cg_project_calls = 0
        fsi_trial_pressure_projection_cg_iterations_total = 0
        fsi_trial_pressure_projection_cg_iterations_max = 0
        fsi_trial_pressure_projection_cg_host_residual_checks = 0
        fsi_trial_pressure_projection_cg_mean_projection_count = 0
        fsi_trial_pressure_projection_cg_converged_all = True
        fsi_trial_pressure_projection_cg_max_relative_residual = 0.0
        fsi_trial_pressure_projection_cg_max_initial_relative_residual = 0.0
        fsi_trial_pressure_projection_cg_breakdown_count = 0
        fsi_primary_response_constraint_force_solid_mobility_ratio = 0.0
        fsi_secondary_response_constraint_force_solid_mobility_ratio = 0.0
        fsi_primary_velocity_target_solid_mobility_ratio = (
            fsi_velocity_target_solid_mobility_ratio
        )
        fsi_secondary_velocity_target_solid_mobility_ratio = (
            fsi_velocity_target_solid_mobility_ratio
        )
        fsi_coupling_enabled = legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=args.solid_model,
            fsi_coupling_iterations=step_fsi_coupling_iterations,
        )
        step_start_main_velocity_z_mps = float(simulator.main_v_mps[None])
        step_start_tail_velocity_z_mps = float(simulator.tail_v_mps[None])
        step_start_interface_velocity_mps = _combine_region_pair_vectors(
            z_velocity_vector(step_start_main_velocity_z_mps),
            z_velocity_vector(step_start_tail_velocity_z_mps),
        )
        robin_previous_velocity_mps = robin_previous_velocity_for_step(
            interface_reaction_state,
            step_start_interface_velocity_mps,
        )
        step_start_main_displacement_z_m = float(simulator.main_w_m[None])
        step_start_tail_displacement_z_m = float(simulator.tail_w_m[None])

        def response_constraint_force_solid_mobility_ratios(
            *,
            primary_reaction_n: Sequence[float],
            secondary_reaction_n: Sequence[float],
            solid_report,
        ) -> tuple[float, float]:
            if not fsi_solid_response_mobility_coupling:
                return 0.0, 0.0
            primary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            secondary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            return primary_ratio, secondary_ratio

        def velocity_target_solid_mobility_ratios(
            *,
            primary_reaction_n: Sequence[float],
            secondary_reaction_n: Sequence[float],
            solid_report,
        ) -> tuple[float, float]:
            base_ratio = fsi_velocity_target_solid_mobility_ratio
            if not fsi_solid_response_velocity_mobility_coupling:
                return base_ratio, base_ratio
            primary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            secondary_ratio = (
                base_ratio
                + solid_response_constraint_force_mobility_ratio(
                    previous_velocity_mps=z_velocity_vector(
                        step_start_tail_velocity_z_mps
                    ),
                    current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                    reaction_force_n=secondary_reaction_n,
                    interface_area_m2=secondary_fsi_face_area_m2,
                    probe_distance_m=fluid_probe_distance_m,
                    density_kgm3=spec.water_density_kgm3,
                    dt_s=solid_response_dt_s,
                )
            )
            return primary_ratio, secondary_ratio

        if fsi_coupling_enabled:
            fsi_coupling_wall_started_at = time.perf_counter()
            current_step_time_s = float(simulator.time_s[None])

            def save_fsi_step_state() -> None:
                simulator.save_reduced_state()
                simulator.fluid.save_state()
                solid_mpm.save_state()

            def restore_fsi_trial_state() -> None:
                simulator.restore_reduced_state()
                simulator.fluid.restore_state()
                solid_mpm.restore_state()

            def accumulate_fsi_trial_pressure_projection_stats(fluid_report) -> None:
                nonlocal fsi_trial_pressure_projection_cg_project_calls
                nonlocal fsi_trial_pressure_projection_cg_iterations_total
                nonlocal fsi_trial_pressure_projection_cg_iterations_max
                nonlocal fsi_trial_pressure_projection_cg_host_residual_checks
                nonlocal fsi_trial_pressure_projection_cg_mean_projection_count
                nonlocal fsi_trial_pressure_projection_cg_converged_all
                nonlocal fsi_trial_pressure_projection_cg_max_relative_residual
                nonlocal fsi_trial_pressure_projection_cg_max_initial_relative_residual
                nonlocal fsi_trial_pressure_projection_cg_breakdown_count

                project_calls = int(
                    getattr(fluid_report, "pressure_projection_cg_project_calls", 0)
                    or 0
                )
                if project_calls <= 0:
                    return
                fsi_trial_pressure_projection_cg_project_calls += project_calls
                fsi_trial_pressure_projection_cg_iterations_total += int(
                    getattr(fluid_report, "pressure_projection_cg_iterations_total", 0)
                    or 0
                )
                fsi_trial_pressure_projection_cg_iterations_max = max(
                    fsi_trial_pressure_projection_cg_iterations_max,
                    int(
                        getattr(
                            fluid_report, "pressure_projection_cg_iterations_max", 0
                        )
                        or 0
                    ),
                )
                fsi_trial_pressure_projection_cg_host_residual_checks += int(
                    getattr(
                        fluid_report, "pressure_projection_cg_host_residual_checks", 0
                    )
                    or 0
                )
                fsi_trial_pressure_projection_cg_mean_projection_count += int(
                    getattr(
                        fluid_report,
                        "pressure_projection_cg_mean_projection_count",
                        0,
                    )
                    or 0
                )
                fsi_trial_pressure_projection_cg_converged_all = (
                    fsi_trial_pressure_projection_cg_converged_all
                    and bool(
                        getattr(
                            fluid_report, "pressure_projection_cg_converged_all", True
                        )
                    )
                )
                fsi_trial_pressure_projection_cg_max_relative_residual = max(
                    fsi_trial_pressure_projection_cg_max_relative_residual,
                    float(
                        getattr(
                            fluid_report,
                            "pressure_projection_cg_max_relative_residual",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                fsi_trial_pressure_projection_cg_max_initial_relative_residual = max(
                    fsi_trial_pressure_projection_cg_max_initial_relative_residual,
                    float(
                        getattr(
                            fluid_report,
                            "pressure_projection_cg_max_initial_relative_residual",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                fsi_trial_pressure_projection_cg_breakdown_count += int(
                    getattr(fluid_report, "pressure_projection_cg_breakdown_count", 0)
                    or 0
                )

            def evaluate_fsi_interface_reaction_target(
                reaction_force_n: tuple[float, ...],
            ) -> InterfaceReactionTargetEvaluation:
                nonlocal fsi_primary_response_constraint_force_solid_mobility_ratio
                nonlocal fsi_secondary_response_constraint_force_solid_mobility_ratio
                nonlocal fsi_primary_velocity_target_solid_mobility_ratio
                nonlocal fsi_secondary_velocity_target_solid_mobility_ratio
                nonlocal fsi_coupling_trial_cfl_max
                nonlocal fsi_coupling_trial_interior_divergence_l2_max
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(
                    reaction_force_n
                )
                trial_solid_report = advance_physical_solid_step(
                    current_step_time_s,
                    primary_reaction_n,
                    secondary_reaction_n,
                )
                trial_primary_velocity_mps = z_velocity_vector(
                    0.5
                    * (
                        step_start_main_velocity_z_mps
                        + float(simulator.main_v_mps[None])
                    )
                )
                trial_secondary_velocity_mps = z_velocity_vector(
                    0.5
                    * (
                        step_start_tail_velocity_z_mps
                        + float(simulator.tail_v_mps[None])
                    )
                )
                trial_solid_interface_velocity_mps = _combine_region_pair_vectors(
                    trial_solid_report.primary_mean_velocity_mps,
                    trial_solid_report.secondary_mean_velocity_mps,
                )
                trial_robin_impedance_force_n = robin_neumann_impedance_force(
                    velocity_mps=trial_solid_interface_velocity_mps,
                    previous_velocity_mps=robin_previous_velocity_mps,
                    impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
                )
                (
                    trial_primary_robin_impedance_force_n,
                    trial_secondary_robin_impedance_force_n,
                ) = _split_region_pair_vector(trial_robin_impedance_force_n)
                (
                    primary_response_constraint_force_solid_mobility_ratio,
                    secondary_response_constraint_force_solid_mobility_ratio,
                ) = response_constraint_force_solid_mobility_ratios(
                    primary_reaction_n=primary_reaction_n,
                    secondary_reaction_n=secondary_reaction_n,
                    solid_report=trial_solid_report,
                )
                fsi_primary_response_constraint_force_solid_mobility_ratio = max(
                    fsi_primary_response_constraint_force_solid_mobility_ratio,
                    primary_response_constraint_force_solid_mobility_ratio,
                )
                fsi_secondary_response_constraint_force_solid_mobility_ratio = max(
                    fsi_secondary_response_constraint_force_solid_mobility_ratio,
                    secondary_response_constraint_force_solid_mobility_ratio,
                )
                (
                    primary_velocity_target_solid_mobility_ratio,
                    secondary_velocity_target_solid_mobility_ratio,
                ) = velocity_target_solid_mobility_ratios(
                    primary_reaction_n=primary_reaction_n,
                    secondary_reaction_n=secondary_reaction_n,
                    solid_report=trial_solid_report,
                )
                fsi_primary_velocity_target_solid_mobility_ratio = max(
                    fsi_primary_velocity_target_solid_mobility_ratio,
                    primary_velocity_target_solid_mobility_ratio,
                )
                fsi_secondary_velocity_target_solid_mobility_ratio = max(
                    fsi_secondary_velocity_target_solid_mobility_ratio,
                    secondary_velocity_target_solid_mobility_ratio,
                )
                tri_diagnostics.update_region_offsets(
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    primary_offset_m=z_displacement_vector(
                        0.5
                        * (
                            step_start_main_displacement_z_m
                            + float(simulator.main_w_m[None])
                        )
                    ),
                    secondary_offset_m=z_displacement_vector(
                        0.5
                        * (
                            step_start_tail_displacement_z_m
                            + float(simulator.tail_w_m[None])
                        )
                    ),
                )
                trial_fluid_report = advance_fluid_step(
                    primary_velocity_mps=trial_primary_velocity_mps,
                    secondary_velocity_mps=trial_secondary_velocity_mps,
                    primary_constraint_force_solid_mobility_ratio=(
                        primary_response_constraint_force_solid_mobility_ratio
                    ),
                    secondary_constraint_force_solid_mobility_ratio=(
                        secondary_response_constraint_force_solid_mobility_ratio
                    ),
                    primary_velocity_target_solid_mobility_ratio=(
                        primary_velocity_target_solid_mobility_ratio
                    ),
                    secondary_velocity_target_solid_mobility_ratio=(
                        secondary_velocity_target_solid_mobility_ratio
                    ),
                    primary_interface_impedance_force_n=trial_primary_robin_impedance_force_n,
                    secondary_interface_impedance_force_n=trial_secondary_robin_impedance_force_n,
                    fluid_substeps=step_fluid_substeps,
                    read_full_report=reuse_accepted_fsi_trial_state,
                )
                accumulate_fsi_trial_pressure_projection_stats(trial_fluid_report)
                trial_fluid_substep_dt_s = float(
                    getattr(
                        trial_fluid_report,
                        "fluid_substep_dt_s",
                        step_fluid_substep_dt_s,
                    )
                )
                trial_sample_report = simulator.sample_cfl_report(
                    dt_s=trial_fluid_substep_dt_s,
                )
                trial_cfl = float(trial_sample_report["cfl"])
                if math.isfinite(trial_cfl):
                    fsi_coupling_trial_cfl_max = (
                        trial_cfl
                        if not math.isfinite(fsi_coupling_trial_cfl_max)
                        else max(fsi_coupling_trial_cfl_max, trial_cfl)
                    )
                trial_interior_divergence_l2 = math.nan
                if math.isfinite(fsi_coupling_trial_interior_divergence_tolerance):
                    trial_projection_sample_report = simulator.sample_after_projection(
                        trial_fluid_report.divergence,
                        dt_s=trial_fluid_substep_dt_s,
                    )
                    trial_interior_divergence_l2 = float(
                        trial_projection_sample_report["interior_divergence_l2"]
                    )
                    if math.isfinite(trial_interior_divergence_l2):
                        fsi_coupling_trial_interior_divergence_l2_max = (
                            trial_interior_divergence_l2
                            if not math.isfinite(
                                fsi_coupling_trial_interior_divergence_l2_max
                            )
                            else max(
                                fsi_coupling_trial_interior_divergence_l2_max,
                                trial_interior_divergence_l2,
                            )
                        )
                primary_target_n = (
                    trial_fluid_report.interface_reaction_target.primary_force_n
                )
                secondary_target_n = (
                    trial_fluid_report.interface_reaction_target.secondary_force_n
                )
                stabilized_target_force_n = _combine_region_pair_vectors(
                    primary_target_n,
                    secondary_target_n,
                )
                raw_target_force_n = _combine_region_pair_vectors(
                    tuple(
                        target_value - robin_value
                        for target_value, robin_value in zip(
                            primary_target_n,
                            trial_primary_robin_impedance_force_n,
                        )
                    ),
                    tuple(
                        target_value - robin_value
                        for target_value, robin_value in zip(
                            secondary_target_n,
                            trial_secondary_robin_impedance_force_n,
                        )
                    ),
                )
                selected_target_force_n = interface_reaction_target_for_mode(
                    interface_reaction_robin_target_mode,
                    raw_target_force_n=raw_target_force_n,
                    stabilized_target_force_n=stabilized_target_force_n,
                )
                return InterfaceReactionTargetEvaluation(
                    target_force_n=selected_target_force_n,
                    diagnostic_target_force_n=raw_target_force_n,
                    velocity_mps=trial_solid_interface_velocity_mps,
                    payload={
                        "solid_report": trial_solid_report,
                        "fluid_report": trial_fluid_report,
                        "trial_cfl": trial_cfl,
                        "trial_interior_divergence_l2": trial_interior_divergence_l2,
                        "trial_max_fluid_speed_mps": float(
                            trial_sample_report["max_fluid_speed_mps"]
                        ),
                        "raw_target_force_n": raw_target_force_n,
                        "selected_target_force_n": selected_target_force_n,
                        "robin_impedance_force_n": trial_robin_impedance_force_n,
                        "primary_response_constraint_force_solid_mobility_ratio": (
                            primary_response_constraint_force_solid_mobility_ratio
                        ),
                        "secondary_response_constraint_force_solid_mobility_ratio": (
                            secondary_response_constraint_force_solid_mobility_ratio
                        ),
                        "primary_velocity_target_solid_mobility_ratio": (
                            primary_velocity_target_solid_mobility_ratio
                        ),
                        "secondary_velocity_target_solid_mobility_ratio": (
                            secondary_velocity_target_solid_mobility_ratio
                        ),
                    },
                )

            def accept_fsi_interface_reaction_evaluation(
                evaluation: InterfaceReactionTargetEvaluation,
            ) -> bool:
                nonlocal fsi_coupling_trial_cfl_rejected_count
                nonlocal fsi_coupling_trial_interior_divergence_rejected_count
                payload = evaluation.payload
                if not isinstance(payload, Mapping):
                    fsi_coupling_trial_cfl_rejected_count += 1
                    return False
                rejection_reason = fsi_trial_acceptance_rejection_reason(
                    payload,
                    cfl_limit=0.5,
                    interior_divergence_l2_limit=(
                        fsi_coupling_trial_interior_divergence_tolerance
                    ),
                )
                if rejection_reason == "cfl":
                    fsi_coupling_trial_cfl_rejected_count += 1
                    return False
                if rejection_reason == "interior_divergence_l2":
                    fsi_coupling_trial_interior_divergence_rejected_count += 1
                    return False
                return True

            def apply_accepted_fsi_interface_reaction(
                reaction_force_n: tuple[float, ...],
            ) -> None:
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(
                    reaction_force_n
                )
                simulator.set_interface_reaction(
                    primary_force_n=primary_reaction_n,
                    secondary_force_n=secondary_reaction_n,
                )

            def commit_accepted_fsi_trial_state(payload: object | None) -> None:
                nonlocal accepted_fsi_trial_payload
                accepted_fsi_trial_payload = coerce_accepted_trial_payload(payload)

            initial_fsi_reaction_force_n = _combine_region_pair_vectors(
                simulator.primary_interface_reaction_force_n[None],
                simulator.secondary_interface_reaction_force_n[None],
            )

            def solve_fsi_interface_reaction_attempt(
                iterations_requested: int,
            ) -> InterfaceReactionFixedPointResult:
                return solve_and_apply_interface_reaction_step(
                    initial_force_n=initial_fsi_reaction_force_n,
                    save_state=save_fsi_step_state,
                    evaluate_target=evaluate_fsi_interface_reaction_target,
                    restore_state=restore_fsi_trial_state,
                    apply_force=apply_accepted_fsi_interface_reaction,
                    commit_accepted_state=(
                        commit_accepted_fsi_trial_state
                        if reuse_accepted_fsi_trial_state
                        else None
                    ),
                    max_iterations=iterations_requested,
                    tolerance_n=settings.fsi_coupling_tolerance_n,
                    initial_relaxation=interface_reaction_relaxation,
                    use_aitken=interface_reaction_aitken,
                    passivity_limit=interface_reaction_passivity_limit,
                    solver=fsi_coupling_solver,
                    target_map_relaxation=fsi_coupling_target_map_relaxation,
                    accept_evaluation=accept_fsi_interface_reaction_evaluation,
                    aitken_lower_bound=interface_reaction_aitken_lower_bound,
                    aitken_upper_bound=interface_reaction_aitken_upper_bound,
                    rejected_trial_backtrack=fsi_coupling_rejected_trial_backtrack,
                    residual_growth_rejection_factor=(
                        fsi_coupling_residual_growth_rejection_factor
                    ),
                    max_accepted_residual_n=fsi_coupling_max_accepted_residual_n,
                    trust_region_force_increment_n=(
                        fsi_coupling_trust_region_force_increment_n
                    ),
                    trust_region_adaptive=fsi_coupling_trust_region_adaptive,
                    trust_region_shrink_factor=(
                        fsi_coupling_trust_region_shrink_factor
                    ),
                    trust_region_growth_factor=(
                        fsi_coupling_trust_region_growth_factor
                    ),
                    trust_region_rebound_factor=(
                        fsi_coupling_trust_region_rebound_factor
                    ),
                    trust_region_rebound_backtrack=(
                        fsi_coupling_trust_region_rebound_backtrack
                    ),
                    trust_region_rebound_stop_factor=(
                        fsi_coupling_trust_region_rebound_stop_factor
                    ),
                    trust_region_rebound_stop_max_residual_n=(
                        fsi_coupling_trust_region_rebound_stop_max_residual_n
                    ),
                    residual_continuation_iterations_max=(
                        fsi_coupling_residual_continuation_iterations_max
                    ),
                    residual_continuation_threshold_n=(
                        fsi_coupling_residual_continuation_threshold_n
                    ),
                    residual_continuation_rebound_secant_from_best=(
                        fsi_coupling_residual_continuation_rebound_secant_from_best
                    ),
                    residual_continuation_rebound_secant_factor=(
                        fsi_coupling_residual_continuation_rebound_secant_factor
                    ),
                    residual_continuation_rebound_secant_evaluation_extensions_max=(
                        fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
                    ),
                )

            fixed_point_result = solve_fsi_interface_reaction_attempt(
                step_fsi_coupling_iterations
            )
            fsi_coupling_first_attempt_safety_rejected = (
                fixed_point_result.accepted_trial_index is None
                and fixed_point_result.rejected_trial_count > 0
            )
            fsi_coupling_same_step_rerun_next_fluid_substeps = (
                fsi_same_step_rerun_fluid_substeps(
                    current_substeps=step_fluid_substeps,
                    max_substeps=int(args.adaptive_fluid_substeps_max),
                    substep_factor=(fsi_coupling_same_step_rerun_fluid_substep_factor),
                    safety_rejected=fsi_coupling_first_attempt_safety_rejected,
                )
            )
            fsi_coupling_same_step_iteration_rerun = fsi_same_step_rerun_triggered(
                current_iterations_requested=step_fsi_coupling_iterations,
                rerun_iterations_max=fsi_coupling_same_step_rerun_iterations_max,
                residual_norm_n=fixed_point_result.residual_norm_n,
                residual_threshold_n=(
                    fsi_coupling_same_step_rerun_residual_threshold_n
                ),
                converged=fixed_point_result.converged,
                safety_rejected=fsi_coupling_first_attempt_safety_rejected,
            )
            fsi_coupling_same_step_fluid_substep_rerun = (
                fsi_coupling_same_step_rerun_next_fluid_substeps > step_fluid_substeps
            )
            if (
                fsi_coupling_same_step_iteration_rerun
                or fsi_coupling_same_step_fluid_substep_rerun
            ):
                fsi_coupling_same_step_rerun_triggered = True
                fsi_coupling_same_step_rerun_count = 1
                fsi_coupling_same_step_rerun_initial_iterations_requested = (
                    step_fsi_coupling_iterations
                )
                fsi_coupling_same_step_rerun_initial_iterations_used = (
                    fixed_point_result.iterations_used
                )
                fsi_coupling_same_step_rerun_initial_residual_norm_n = (
                    fixed_point_result.residual_norm_n
                )
                fsi_coupling_same_step_rerun_initial_converged = (
                    fixed_point_result.converged
                )
                fsi_coupling_same_step_rerun_safety_rejected = (
                    fsi_coupling_first_attempt_safety_rejected
                )
                fsi_coupling_same_step_rerun_initial_fluid_substeps = (
                    step_fluid_substeps
                )
                restore_fsi_trial_state()
                accepted_fsi_trial_payload = None
                if fsi_coupling_same_step_iteration_rerun:
                    step_fsi_coupling_iterations = (
                        fsi_coupling_same_step_rerun_iterations_max
                    )
                if fsi_coupling_same_step_fluid_substep_rerun:
                    step_fluid_substeps = (
                        fsi_coupling_same_step_rerun_next_fluid_substeps
                    )
                    step_fluid_substep_dt_s = float(spec.dt_s) / float(
                        step_fluid_substeps
                    )
                fsi_coupling_same_step_rerun_final_fluid_substeps = step_fluid_substeps
                fixed_point_result = solve_fsi_interface_reaction_attempt(
                    step_fsi_coupling_iterations
                )
            fsi_coupling_iterations_used = fixed_point_result.iterations_used
            fsi_coupling_converged = fixed_point_result.converged
            fsi_coupling_residual_norm_n = fixed_point_result.residual_norm_n
            fsi_coupling_relaxation_effective = fixed_point_result.relaxation
            fsi_all_trials_rejected = fixed_point_result.all_trials_rejected
            fsi_zero_force_commit_blocked = fixed_point_result.zero_force_commit_blocked
            fsi_coupling_rejected_trial_count = fixed_point_result.rejected_trial_count
            fsi_coupling_rejected_trial_backtrack_count = (
                fixed_point_result.rejected_trial_backtrack_count
            )
            fsi_coupling_residual_growth_rejected_trial_count = (
                fixed_point_result.residual_growth_rejected_trial_count
            )
            fsi_coupling_max_residual_rejected_trial_count = (
                fixed_point_result.max_residual_rejected_trial_count
            )
            fsi_coupling_trust_region_limited_update_count = (
                fixed_point_result.trust_region_limited_update_count
            )
            fsi_coupling_trust_region_shrink_count = (
                fixed_point_result.trust_region_shrink_count
            )
            fsi_coupling_trust_region_growth_count = (
                fixed_point_result.trust_region_growth_count
            )
            fsi_coupling_trust_region_rebound_backtrack_count = (
                fixed_point_result.trust_region_rebound_backtrack_count
            )
            fsi_coupling_trust_region_rebound_stop_count = (
                fixed_point_result.trust_region_rebound_stop_count
            )
            fsi_coupling_trust_region_rebound_stop_suppressed_count = (
                fixed_point_result.trust_region_rebound_stop_suppressed_count
            )
            fsi_coupling_residual_continuation_iteration_count = (
                fixed_point_result.residual_continuation_iteration_count
            )
            fsi_coupling_residual_continuation_rebound_secant_count = (
                fixed_point_result.residual_continuation_rebound_secant_count
            )
            fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = fixed_point_result.residual_continuation_rebound_secant_evaluation_extension_count
            fsi_coupling_trust_region_effective_force_increment_n = (
                fixed_point_result.trust_region_effective_force_increment_n
            )
            if isinstance(fixed_point_result.accepted_payload, Mapping):
                fsi_coupling_accepted_trial_cfl = float(
                    fixed_point_result.accepted_payload.get("trial_cfl", math.nan)
                )
                fsi_coupling_accepted_trial_max_fluid_speed_mps = float(
                    fixed_point_result.accepted_payload.get(
                        "trial_max_fluid_speed_mps",
                        math.nan,
                    )
                )
                fsi_coupling_accepted_trial_interior_divergence_l2 = float(
                    fixed_point_result.accepted_payload.get(
                        "trial_interior_divergence_l2",
                        math.nan,
                    )
                )
            fsi_coupling_iqn_ils_least_squares_update_count = (
                fixed_point_result.iqn_ils_least_squares_update_count
            )
            fsi_coupling_interface_map_amplification = (
                fixed_point_result.interface_map_amplification_max
            )
            fsi_coupling_residual_jacobian_amplification = (
                fixed_point_result.residual_jacobian_amplification_max
            )
            fsi_coupling_physical_interface_map_amplification = (
                fixed_point_result.physical_interface_map_amplification_max
            )
            fsi_coupling_physical_residual_jacobian_amplification = (
                fixed_point_result.physical_residual_jacobian_amplification_max
            )
            fsi_coupling_raw_interface_map_amplification = (
                fixed_point_result.diagnostic_interface_map_amplification_max
            )
            fsi_coupling_raw_residual_jacobian_amplification = (
                fixed_point_result.diagnostic_residual_jacobian_amplification_max
            )
            fsi_coupling_interface_map_amplification_sample_count = (
                fixed_point_result.interface_map_amplification_sample_count
            )
            fsi_coupling_residual_jacobian_amplification_sample_count = (
                fixed_point_result.residual_jacobian_amplification_sample_count
            )
            fsi_coupling_physical_interface_map_amplification_sample_count = (
                fixed_point_result.physical_interface_map_amplification_sample_count
            )
            fsi_coupling_physical_residual_jacobian_amplification_sample_count = (
                fixed_point_result.physical_residual_jacobian_amplification_sample_count
            )
            fsi_coupling_raw_interface_map_amplification_sample_count = (
                fixed_point_result.diagnostic_interface_map_amplification_sample_count
            )
            fsi_coupling_raw_residual_jacobian_amplification_sample_count = fixed_point_result.diagnostic_residual_jacobian_amplification_sample_count
            fsi_coupling_trial_force_history_n = (
                fixed_point_result.trial_force_history_n
            )
            fsi_coupling_target_force_history_n = (
                fixed_point_result.target_force_history_n
            )
            fsi_coupling_residual_history_n = fixed_point_result.residual_history_n
            fsi_coupling_physical_target_force_history_n = (
                fixed_point_result.physical_target_force_history_n
            )
            fsi_coupling_physical_residual_history_n = (
                fixed_point_result.physical_residual_history_n
            )
            fsi_coupling_raw_target_force_history_n = (
                fixed_point_result.diagnostic_target_force_history_n
            )
            fsi_coupling_raw_residual_history_n = (
                fixed_point_result.diagnostic_residual_history_n
            )
            fsi_coupling_wall_time_s = (
                time.perf_counter() - fsi_coupling_wall_started_at
            )

        if sharp_case_runner_enabled:
            if sharp_coupling_state is None:
                raise RuntimeError("sharp HIBM-MPM coupling state was not initialized")
            current_time_s = float(simulator.time_s[None])
            pressure_pa = pressure_schedule_step_end_pa(
                current_time_s,
                spec.dt_s,
                spec,
            )
            solid_mpm_report = None

            def advance_sharp_solid_substeps():
                nonlocal solid_advance_wall_time_s
                solid_wall_started_at = time.perf_counter()
                # The waveform drive enters through the far-pressure closure in
                # the marker traction sampling (region 7 below), so no direct
                # solid area load is added here: the membrane feels
                # (p_water - p_air) through scattered marker forces, which
                # restores the added-mass back-pressure that a direct area
                # load bypassed.
                report = None
                for _ in range(settings.solid_mpm_substeps):
                    if args.solid_model == "tri_mooney_shell_mpm":
                        report = solid_mpm.advance_with_external_forces(
                            dt_s=solid_sub_dt_s,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                            velocity_damping=solid_substep_velocity_damping,
                            flip_blend=settings.solid_mpm_flip_blend,
                            read_report=False,
                        )
                    elif args.solid_model == "neo_hookean_mpm":
                        report = solid_mpm.step(
                            dt_s=solid_sub_dt_s,
                            mu_pa=material.shear_modulus_pa,
                            lambda_pa=material.lame_lambda_pa,
                            velocity_damping=solid_substep_velocity_damping,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                            fixed_node_lock_policy=settings.neo_fixed_node_lock_policy,
                            read_report=False,
                        )
                    else:
                        raise ValueError(f"Unsupported solid model: {args.solid_model}")
                solid_advance_wall_time_s += time.perf_counter() - solid_wall_started_at
                return solid_mpm.report() if report is None else report

            def advance_sharp_trial_once():
                sharp_report = sharp_coupling_state.advance_mpm_step(
                    fluid=simulator.fluid,
                    mpm_external_force_n=solid_mpm.external_force_n,
                    mpm_particle_position_m=solid_mpm.x,
                    mpm_particle_velocity_mps=solid_mpm.v,
                    mpm_particle_normal=solid_mpm.surface_normal,
                    mpm_particle_area_m2=solid_mpm.area_weight_m2,
                    mpm_particle_count=solid_mpm.particle_count,
                    solid_step=advance_sharp_solid_substeps,
                    search_radius_m=max(
                        2.0 * fluid_probe_distance_m,
                        estimated_solid_particle_spacing_m,
                    ),
                    interior_probe_distance_m=fluid_probe_distance_m,
                    mpm_support_radius_m=max(
                        2.0 * estimated_solid_particle_spacing_m,
                        fluid_probe_distance_m,
                    ),
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    far_pressure_region_id=settings.pressure_load_region_id,
                    far_pressure_barrier_region_id=settings.fixed_rim_region_id,
                    far_pressure_pa=pressure_pa,
                    far_pressure_side_normal_sign=settings.pressure_far_side_normal_sign,
                    far_pressure_inside_probe_max_multiplier=(
                        settings.far_pressure_inside_probe_max_multiplier
                    ),
                    two_sided_probe_max_multiplier=(
                        settings.two_sided_probe_max_multiplier
                    ),
                    one_sided_pressure_region_id=secondary_shell_region_id,
                    one_sided_reference_pressure_pa=0.0,
                    one_sided_probe_max_multiplier=(
                        settings.one_sided_probe_max_multiplier
                    ),
                    far_pressure_air_backed=settings.far_pressure_air_backed,
                    far_pressure_air_backed_probe_normal_sign=(
                        settings.far_pressure_air_backed_probe_normal_sign
                    ),
                    fluid_dt_s=spec.dt_s,
                    fluid_substeps=step_fluid_substeps,
                    projection_iterations=int(args.projection_iterations),
                    run_fluid_predictor=True,
                    pressure_neumann_density_kgm3=spec.water_density_kgm3,
                    pressure_neumann_dt_s=spec.dt_s,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                    pressure_solver=settings.pressure_solver_name,
                    pressure_solve_failure_policy=str(
                        args.pressure_solve_failure_policy
                    ),
                    fluid_advection_scheme=str(args.fluid_advection_scheme),
                    multigrid_cycles=settings.effective_multigrid_cycles,
                    cg_tolerance=settings.cg_tolerance,
                    cg_preconditioner=settings.cg_preconditioner,
                    divergence_cleanup_iterations=(
                        settings.projection_divergence_cleanup_iterations
                    ),
                    divergence_cleanup_relaxation=float(
                        args.divergence_cleanup_relaxation
                    ),
                    convert_internal_nodes_to_obstacles=False,
                    post_dirichlet_consistency_projection_iterations=int(
                        args.hibm_post_dirichlet_consistency_projections
                    ),
                    diagnostic_disable_pressure_neumann_matrix_rows=bool(
                        args.diagnostic_disable_pressure_neumann_matrix_rows
                    ),
                )
                return sharp_report

            def restore_sharp_trial_state(
                marker_state: Mapping[str, object],
                pressure_gradient_state: object,
            ) -> None:
                simulator.restore_reduced_state()
                simulator.fluid.restore_state()
                solid_mpm.restore_state()
                restore_sharp_marker_state_arrays(
                    sharp_coupling_state.markers,
                    marker_state,
                )
                restore_sharp_pressure_neumann_gradient_state_array(
                    sharp_coupling_state,
                    pressure_gradient_state,
                )

            def advance_sharp_marker_fixed_point_step():
                nonlocal fsi_coupling_iterations_used
                nonlocal fsi_coupling_converged
                nonlocal fsi_coupling_residual_norm_n
                nonlocal fsi_coupling_relaxation_effective
                nonlocal fsi_coupling_iqn_ils_least_squares_update_count
                nonlocal fsi_coupling_physical_interface_map_amplification
                nonlocal fsi_coupling_physical_interface_map_amplification_sample_count

                requested_iterations = max(1, int(fsi_coupling_iterations))
                if requested_iterations <= 1:
                    report = advance_sharp_trial_once()
                    fsi_coupling_iterations_used = 1
                    fsi_coupling_converged = False
                    fsi_coupling_residual_norm_n = math.nan
                    return report, {
                        "hibm_coupling_scheme": "explicit_loose",
                        "hibm_added_mass_stability_status": ("unmeasured_single_pass"),
                        "hibm_added_mass_stability_measured": False,
                        "hibm_added_mass_stabilization": "none",
                        "hibm_semi_implicit_coupling_enabled": False,
                        "hibm_semi_implicit_coupling_matrix_active": False,
                        "hibm_fsi_coupling_iterations_used": 1,
                        "hibm_fsi_coupling_converged": False,
                        "hibm_fsi_coupling_explicit_single_pass": True,
                        "hibm_fsi_coupling_residual_source": ("unmeasured_single_pass"),
                    }

                simulator.save_reduced_state()
                simulator.fluid.save_state()
                solid_mpm.save_state()
                marker_guess = sharp_marker_state_arrays(sharp_coupling_state.markers)
                pressure_gradient_state = sharp_pressure_neumann_gradient_state_array(
                    sharp_coupling_state
                )
                previous_velocity_residual_vector: np.ndarray | None = None
                residual_history: list[float] = []
                residual_max_history: list[float] = []
                combined_residual_history: list[float] = []
                combined_residual_max_history: list[float] = []
                residual_position_history: list[float] = []
                residual_velocity_history: list[float] = []
                residual_primary_region_history: list[float] = []
                residual_secondary_region_history: list[float] = []
                residual_other_region_history: list[float] = []
                residual_max_marker_index_history: list[int] = []
                residual_max_marker_region_history: list[int] = []
                relaxation_history: list[float] = []
                relaxation = float(interface_reaction_relaxation)
                converged = False
                iterations_used = 0
                aitken_update_count = 0
                report = None
                residual_norm_mps = math.inf
                residual_max_mps = math.inf
                combined_residual_norm_mps = math.inf
                combined_residual_max_mps = math.inf

                for iteration in range(requested_iterations):
                    restore_sharp_trial_state(marker_guess, pressure_gradient_state)
                    report = advance_sharp_trial_once()
                    marker_candidate = sharp_marker_state_arrays(
                        sharp_coupling_state.markers
                    )
                    candidate_pressure_gradient_state = (
                        sharp_pressure_neumann_gradient_state_array(
                            sharp_coupling_state
                        )
                    )
                    residual = sharp_marker_fixed_point_residual_mps(
                        marker_guess,
                        marker_candidate,
                        dt_s=spec.dt_s,
                    )
                    marker_region_ids = (
                        sharp_coupling_state.markers.region_id.to_numpy()[
                            : int(sharp_coupling_state.markers.marker_count)
                        ]
                    )
                    residual_diagnostics = (
                        sharp_marker_fixed_point_residual_diagnostics_mps(
                            marker_guess,
                            marker_candidate,
                            dt_s=spec.dt_s,
                            marker_region_ids=marker_region_ids,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                        )
                    )
                    residual_vector = _sharp_marker_fixed_point_residual_vector_mps(
                        marker_guess,
                        marker_candidate,
                        dt_s=spec.dt_s,
                    )
                    velocity_residual_vector = residual_vector[:, 3:].reshape(-1)
                    combined_residual_norm_mps = float(residual["l2_mps"])
                    combined_residual_max_mps = float(residual["max_mps"])
                    residual_norm_mps = float(residual_diagnostics["velocity_l2_mps"])
                    residual_max_mps = float(residual_diagnostics["velocity_max_mps"])
                    residual_history.append(residual_norm_mps)
                    residual_max_history.append(residual_max_mps)
                    combined_residual_history.append(combined_residual_norm_mps)
                    combined_residual_max_history.append(combined_residual_max_mps)
                    residual_position_history.append(
                        float(residual_diagnostics["position_l2_mps"])
                    )
                    residual_velocity_history.append(residual_norm_mps)
                    residual_primary_region_history.append(
                        float(residual_diagnostics["primary_region_l2_mps"])
                    )
                    residual_secondary_region_history.append(
                        float(residual_diagnostics["secondary_region_l2_mps"])
                    )
                    residual_other_region_history.append(
                        float(residual_diagnostics["other_region_l2_mps"])
                    )
                    residual_max_marker_index_history.append(
                        int(residual_diagnostics["max_marker_index"])
                    )
                    residual_max_marker_region_history.append(
                        int(residual_diagnostics["max_marker_region_id"])
                    )
                    relaxation_history.append(float(relaxation))
                    iterations_used = iteration + 1
                    velocity_residual_norm_mps = residual_norm_mps
                    trial_projection_failure_reason = (
                        sharp_report_fluid_projection_failure_reason(report)
                    )
                    if trial_projection_failure_reason:
                        raise RuntimeError(
                            "sharp marker fixed point trial fluid projection failed "
                            f"(iteration={int(iterations_used)}, "
                            f"reason={trial_projection_failure_reason}, "
                            f"velocity_residual_l2_mps={float(residual_norm_mps):.6g}, "
                            f"velocity_residual_max_mps={float(residual_max_mps):.6g}, "
                            f"combined_residual_l2_mps={float(combined_residual_norm_mps):.6g}, "
                            f"combined_residual_max_mps={float(combined_residual_max_mps):.6g}, "
                            f"residual_history_mps={residual_history}, "
                            f"residual_max_history_mps={residual_max_history}, "
                            f"combined_residual_history_mps={combined_residual_history}, "
                            f"combined_residual_max_history_mps={combined_residual_max_history}, "
                            f"relaxation_history={relaxation_history})"
                        )
                    if velocity_residual_norm_mps <= fsi_marker_coupling_tolerance_mps:
                        converged = True
                        break
                    if iteration == requested_iterations - 1:
                        break
                    if (
                        interface_reaction_aitken
                        and previous_velocity_residual_vector is not None
                    ):
                        relaxation = _sharp_marker_aitken_relaxation(
                            previous_relaxation=relaxation,
                            previous_residual_mps=previous_velocity_residual_vector,
                            current_residual_mps=velocity_residual_vector,
                        )
                        aitken_update_count += 1
                    previous_velocity_residual_vector = velocity_residual_vector.copy()
                    marker_guess = relaxed_sharp_marker_state_arrays(
                        marker_guess,
                        marker_candidate,
                        relaxation=relaxation,
                    )
                    pressure_gradient_state = (
                        relaxed_sharp_pressure_neumann_gradient_state_array(
                            pressure_gradient_state,
                            candidate_pressure_gradient_state,
                            relaxation=relaxation,
                        )
                    )

                if report is None:
                    raise RuntimeError("sharp marker fixed point produced no trial")
                if not converged:
                    raise RuntimeError(
                        "sharp marker fixed point did not converge "
                        f"(iterations={int(iterations_used)}, "
                        f"velocity_residual_l2_mps={float(residual_norm_mps):.6g}, "
                        f"velocity_residual_max_mps={float(residual_max_mps):.6g}, "
                        f"combined_residual_l2_mps={float(combined_residual_norm_mps):.6g}, "
                        f"combined_residual_max_mps={float(combined_residual_max_mps):.6g}, "
                        f"tolerance_mps={float(fsi_marker_coupling_tolerance_mps):.6g}, "
                        f"residual_history_mps={residual_history}, "
                        f"residual_max_history_mps={residual_max_history}, "
                        f"combined_residual_history_mps={combined_residual_history}, "
                        f"combined_residual_max_history_mps={combined_residual_max_history}, "
                        f"position_residual_history_mps={residual_position_history}, "
                        f"velocity_residual_history_mps={residual_velocity_history}, "
                        f"primary_region_residual_history_mps={residual_primary_region_history}, "
                        f"secondary_region_residual_history_mps={residual_secondary_region_history}, "
                        f"other_region_residual_history_mps={residual_other_region_history}, "
                        f"max_marker_index_history={residual_max_marker_index_history}, "
                        f"max_marker_region_history={residual_max_marker_region_history}, "
                        f"relaxation_history={relaxation_history})"
                    )
                fsi_coupling_iterations_used = iterations_used
                fsi_coupling_converged = converged
                fsi_coupling_residual_norm_n = math.nan
                fsi_coupling_relaxation_effective = relaxation
                fsi_coupling_iqn_ils_least_squares_update_count = aitken_update_count
                if len(residual_history) >= 2 and residual_history[0] > 0.0:
                    amplification = residual_history[-1] / residual_history[0]
                    fsi_coupling_physical_interface_map_amplification = amplification
                    fsi_coupling_physical_interface_map_amplification_sample_count = (
                        len(residual_history) - 1
                    )
                summary = {
                    "hibm_coupling_scheme": "marker_fixed_point",
                    "hibm_added_mass_stability_status": (
                        "converged" if converged else "not_converged"
                    ),
                    "hibm_added_mass_stability_measured": True,
                    "hibm_added_mass_stabilization": (
                        "aitken_marker_state_under_relaxation"
                        if interface_reaction_aitken
                        else "marker_state_under_relaxation"
                    ),
                    "hibm_semi_implicit_coupling_enabled": True,
                    "hibm_semi_implicit_coupling_matrix_active": False,
                    "hibm_fsi_coupling_iterations_used": iterations_used,
                    "hibm_fsi_coupling_converged": converged,
                    "hibm_fsi_coupling_explicit_single_pass": False,
                    "hibm_fsi_coupling_residual_source": (
                        "marker_surface_fixed_point_velocity_residual_l2_mps"
                    ),
                    "hibm_fsi_coupling_residual_l2_mps": residual_norm_mps,
                    "hibm_fsi_coupling_residual_max_mps": residual_max_mps,
                    "hibm_fsi_coupling_residual_history_mps": residual_history,
                    "hibm_fsi_coupling_residual_max_history_mps": residual_max_history,
                    "hibm_fsi_coupling_combined_residual_l2_mps": (
                        combined_residual_norm_mps
                    ),
                    "hibm_fsi_coupling_combined_residual_max_mps": (
                        combined_residual_max_mps
                    ),
                    "hibm_fsi_coupling_combined_residual_history_mps": (
                        combined_residual_history
                    ),
                    "hibm_fsi_coupling_combined_residual_max_history_mps": (
                        combined_residual_max_history
                    ),
                    "hibm_fsi_coupling_position_residual_history_mps": (
                        residual_position_history
                    ),
                    "hibm_fsi_coupling_velocity_residual_history_mps": (
                        residual_velocity_history
                    ),
                    "hibm_fsi_coupling_primary_region_residual_history_mps": (
                        residual_primary_region_history
                    ),
                    "hibm_fsi_coupling_secondary_region_residual_history_mps": (
                        residual_secondary_region_history
                    ),
                    "hibm_fsi_coupling_other_region_residual_history_mps": (
                        residual_other_region_history
                    ),
                    "hibm_fsi_coupling_max_marker_index_history": (
                        residual_max_marker_index_history
                    ),
                    "hibm_fsi_coupling_max_marker_region_history": (
                        residual_max_marker_region_history
                    ),
                    "hibm_fsi_coupling_relaxation_effective": relaxation,
                    "hibm_fsi_coupling_relaxation_history": relaxation_history,
                    "hibm_fsi_coupling_aitken_update_count": aitken_update_count,
                }
                return report, summary

            fluid_wall_started_at = time.perf_counter()
            try:
                sharp_report, sharp_fixed_point_summary = (
                    advance_sharp_marker_fixed_point_step()
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                raise
            sharp_advance_wall_time_s = time.perf_counter() - fluid_wall_started_at
            if fsi_coupling_iterations > 1:
                fsi_coupling_wall_time_s = sharp_advance_wall_time_s
            fluid_advance_wall_time_s = max(
                0.0,
                sharp_advance_wall_time_s - solid_advance_wall_time_s,
            )
            solid_mpm_report = sharp_report.mpm
            if solid_mpm_report is None:
                solid_mpm_report = solid_mpm.report()
            callbacks.publish_solid_report_to_reduced_state(
                current_time_s,
                solid_mpm_report,
            )
            sample_wall_started_at = time.perf_counter()
            fluid_substep_dt_s = step_fluid_substep_dt_s
            latest_fluid_projection_report = (
                sharp_report.post_solid_fluid_projection
                if sharp_report.post_solid_fluid_projection is not None
                else sharp_report.fluid_to_mpm_loads.fluid_projection
            )
            sample_report = simulator.sample_after_projection(
                latest_fluid_projection_report,
                dt_s=fluid_substep_dt_s,
            )
            pressure_outlet_report = simulator.fluid.pressure_outlet_fv_flux_report(
                dt_s=fluid_substep_dt_s,
            )
            sample_wall_time_s = time.perf_counter() - sample_wall_started_at
            sharp_summary = hibm_mpm_sharp_step_summary(sharp_report)
            sharp_summary.update(sharp_fixed_point_summary)
            row = build_hibm_mpm_sharp_case_row(
                step=step,
                sample_report=sample_report,
                sharp_summary=sharp_summary,
                fluid_projection_report=sharp_report.fluid_to_mpm_loads.fluid_projection,
                pressure_outlet_report=pressure_outlet_report,
                fluid_dt_s=spec.dt_s,
                solid_mpm_report=solid_mpm_report,
                solid_model=args.solid_model,
                fsi_coupling_mode_report=fsi_coupling_mode_report,
                fsi_coupling_iterations_requested=fsi_coupling_iterations,
            )
            expected_flux_m3s = float(row["volume_flux_m3s"])
            lip_negative_z_flux_m3s = float(row["lip_flow_negative_z_m3s"])
            outlet_negative_z_flux_m3s = float(row["outlet_flow_negative_z_m3s"])
            downstream_negative_z_flux_m3s = float(
                row["downstream_flow_negative_z_m3s"]
            )
            row["main_volume_flux_to_lip_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=lip_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["main_volume_flux_to_outlet_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["main_volume_flux_to_downstream_ratio"] = (
                signed_positive_source_flux_ratio(
                    outlet_negative_z_flux_m3s=downstream_negative_z_flux_m3s,
                    source_flux_m3s=expected_flux_m3s,
                )
            )
            row["outlet_flux_deficit_m3s"] = (
                expected_flux_m3s - outlet_negative_z_flux_m3s
            )
            row["downstream_flux_deficit_m3s"] = (
                expected_flux_m3s - downstream_negative_z_flux_m3s
            )
            row["accepted_fsi_trial_state_reused"] = False
            row["accepted_fsi_trial_state_readvanced"] = False
            row["fsi_all_trials_rejected"] = False
            row["fsi_zero_force_commit_blocked"] = False
            row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
            row["fsi_coupling_iqn_ils_least_squares_update_count"] = (
                fsi_coupling_iqn_ils_least_squares_update_count
            )
            row["fsi_coupling_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_physical_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_physical_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_raw_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_raw_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_residual_jacobian_amplification_sample_count"] = 0
            row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row[
                "fsi_coupling_physical_residual_jacobian_amplification_sample_count"
            ] = 0
            row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = 0
            row["interface_reaction_relaxation"] = interface_reaction_relaxation
            row["interface_reaction_aitken"] = interface_reaction_aitken
            row["interface_reaction_aitken_lower_bound"] = (
                interface_reaction_aitken_lower_bound
            )
            row["interface_reaction_aitken_upper_bound"] = (
                interface_reaction_aitken_upper_bound
            )
            row["interface_reaction_relaxation_effective"] = (
                fsi_coupling_relaxation_effective
            )
            row["interface_reaction_passivity_limit"] = (
                interface_reaction_passivity_limit
            )
            row["interface_reaction_robin_impedance_ns_m"] = (
                interface_reaction_robin_impedance_ns_m
            )
            row["interface_reaction_robin_matrix_impedance_ns_m"] = (
                interface_reaction_robin_matrix_impedance_ns_m
            )
            row["interface_reaction_robin_target_mode"] = (
                interface_reaction_robin_target_mode
            )
            row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
            row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
            row["sample_wall_time_s"] = sample_wall_time_s
            row["surface_diagnostics_wall_time_s"] = 0.0
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            row["fluid_substeps_base"] = effective_fluid_substeps
            row["adaptive_fluid_substeps_enabled"] = adaptive_fluid_substeps_enabled
            row["adaptive_fluid_substeps_target_cfl"] = float(
                args.adaptive_fluid_substeps_target_cfl
            )
            row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
            row["adaptive_fluid_substeps_previous_substeps"] = (
                previous_step_fluid_substeps
            )
            rows.append(row)
            if args.diagnostic_dump_zero_correctable_cells:
                zero_correctable_summary = _write_hibm_zero_correctable_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                row["diagnostic_zero_correctable_interior_cell_count"] = int(
                    zero_correctable_summary["zero_correctable_interior_cell_count"]
                )
                row["diagnostic_zero_correctable_shell_band_candidate_count"] = int(
                    zero_correctable_summary["shell_band_candidate_cell_count"]
                )
            if args.diagnostic_dump_high_residual_cells:
                high_residual_summary = _write_hibm_high_residual_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                row["diagnostic_high_residual_dumped_cell_count"] = int(
                    high_residual_summary["dumped_cell_count"]
                )
                row["diagnostic_high_residual_max_abs_s"] = float(
                    high_residual_summary["max_abs_residual_s"]
                )
                row["diagnostic_high_residual_velocity_dirichlet_cell_count"] = int(
                    high_residual_summary["dumped_velocity_dirichlet_cell_count"]
                )
            if args.diagnostic_dump_pressure_neumann_invalid_rows:
                load_pressure_neumann_invalid_summary = _write_hibm_pressure_neumann_invalid_row_dump(
                    output_dir=output_dir,
                    step=step,
                    rows=(
                        sharp_report.fluid_to_mpm_loads.pressure_neumann_invalid_diagnostic_rows
                    ),
                    stage="load",
                )
                next_pressure_neumann_invalid_summary = (
                    _write_hibm_pressure_neumann_invalid_row_dump(
                        output_dir=output_dir,
                        step=step,
                        rows=(
                            sharp_report.next_pressure_neumann_invalid_diagnostic_rows
                        ),
                        stage="next",
                    )
                )
                row["diagnostic_pressure_neumann_invalid_load_dumped_row_count"] = int(
                    load_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_load_total_row_count"] = int(
                    load_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_next_dumped_row_count"] = int(
                    next_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_next_total_row_count"] = int(
                    next_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_dumped_row_count"] = int(
                    next_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_total_row_count"] = int(
                    next_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
            snapshot_interval = int(args.fluid_snapshot_interval)
            if snapshot_interval > 0 and (
                step % snapshot_interval == 0 or step == step_count
            ):
                _write_fluid_snapshot_npz(
                    snapshot_dir=output_dir / "snapshots",
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    marker_count=int(sharp_coupling_state.markers.marker_count),
                    time_s=float(row["time_s"]),
                    pressure_pa=float(row["pressure_load_pa"]),
                )
            try:
                _raise_for_step_numerical_guard(
                    row,
                    cfl_limit=0.5,
                    divergence_l2_limit=float(args.projection_divergence_tolerance),
                )
                _raise_for_step_solid_out_of_bounds_guard(row)
                _raise_for_closure_coverage_floor(
                    rows,
                    int(args.closure_coverage_floor),
                    int(args.closure_coverage_floor_patience),
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                raise
            previous_step_cfl = float(row["cfl"])
            previous_step_fluid_substeps = int(
                float(row.get("fluid_substeps", step_fluid_substeps))
            )
            if args.checkpoint_every_step:
                write_csv(history_path, rows)
                checkpoint_wall_started_at = time.perf_counter()
                write_run_checkpoint(
                    run_checkpoint_path,
                    completed_step=step,
                    step_count=step_count,
                    full_pressure_waveform_steps=full_pressure_waveform_steps,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid_mpm,
                    interface_reaction_state=interface_reaction_state,
                    sharp_coupling_state=sharp_coupling_state,
                    frozen_run_fingerprint=frozen_run_fingerprint,
                )
                checkpoint_wall_time_s = (
                    time.perf_counter() - checkpoint_wall_started_at
                )
                row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
                row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
                write_csv(history_path, rows)
            if args.progress and (
                step == 1 or step == step_count or step % args.progress_interval == 0
            ):
                print(
                    "step={step} t={time_s:.6f}s p={pressure_load_pa:.3f}Pa "
                    "main_z={main_displacement_z_m:.6e}m "
                    "outlet_ratio={main_volume_flux_to_outlet_ratio:.6e} "
                    "outlet_neg_z_Q={outlet_flow_negative_z_m3s:.6e}m3/s "
                    "cfl={cfl:.3e} div_l2={divergence_l2:.3e} "
                    "interior_div_l2={interior_divergence_l2:.3e}".format(**row),
                    flush=True,
                )
            if (
                max_wall_time_s > 0.0
                and step < step_count
                and time.perf_counter() - state.run_started_at_perf >= max_wall_time_s
            ):
                partial_run_stopped = True
                partial_run_reason = "max_wall_time_s"
                break
            continue

        reused_fluid_step_report = None
        if accepted_fsi_trial_payload is not None:
            replay_reports = accepted_trial_replay_reports(accepted_fsi_trial_payload)
            solid_mpm_report = replay_reports.solid_report
            reused_fluid_step_report = replay_reports.fluid_report
            accepted_fsi_trial_state_reused = True
        else:
            accepted_fsi_trial_state_readvanced = (
                fixed_point_result is not None
                and fixed_point_result.accepted_trial_index is not None
            )
            solid_wall_started_at = time.perf_counter()
            current_time_s = float(simulator.time_s[None])
            primary_interface_reaction_n = _taichi_vector3_to_tuple(
                simulator.primary_interface_reaction_force_n[None]
            )
            secondary_interface_reaction_n = _taichi_vector3_to_tuple(
                simulator.secondary_interface_reaction_force_n[None]
            )
            solid_mpm_report = advance_physical_solid_step(
                current_time_s,
                primary_interface_reaction_n,
                secondary_interface_reaction_n,
            )
            solid_advance_wall_time_s = time.perf_counter() - solid_wall_started_at
        tri_diagnostics.update_region_offsets(
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            primary_offset_m=z_displacement_vector(
                0.5
                * (step_start_main_displacement_z_m + float(simulator.main_w_m[None]))
            ),
            secondary_offset_m=z_displacement_vector(
                0.5
                * (step_start_tail_displacement_z_m + float(simulator.tail_w_m[None]))
            ),
        )
        if reused_fluid_step_report is None:
            fluid_wall_started_at = time.perf_counter()
            accepted_fluid_step_robin_impedance_force_n = robin_neumann_impedance_force(
                velocity_mps=_combine_region_pair_vectors(
                    solid_mpm_report.primary_mean_velocity_mps,
                    solid_mpm_report.secondary_mean_velocity_mps,
                ),
                previous_velocity_mps=robin_previous_velocity_mps,
                impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
            )
            (
                accepted_primary_fluid_step_robin_impedance_force_n,
                accepted_secondary_fluid_step_robin_impedance_force_n,
            ) = _split_region_pair_vector(accepted_fluid_step_robin_impedance_force_n)
            (
                accepted_primary_response_constraint_force_solid_mobility_ratio,
                accepted_secondary_response_constraint_force_solid_mobility_ratio,
            ) = response_constraint_force_solid_mobility_ratios(
                primary_reaction_n=primary_interface_reaction_n,
                secondary_reaction_n=secondary_interface_reaction_n,
                solid_report=solid_mpm_report,
            )
            fsi_primary_response_constraint_force_solid_mobility_ratio = max(
                fsi_primary_response_constraint_force_solid_mobility_ratio,
                accepted_primary_response_constraint_force_solid_mobility_ratio,
            )
            fsi_secondary_response_constraint_force_solid_mobility_ratio = max(
                fsi_secondary_response_constraint_force_solid_mobility_ratio,
                accepted_secondary_response_constraint_force_solid_mobility_ratio,
            )
            (
                accepted_primary_velocity_target_solid_mobility_ratio,
                accepted_secondary_velocity_target_solid_mobility_ratio,
            ) = velocity_target_solid_mobility_ratios(
                primary_reaction_n=primary_interface_reaction_n,
                secondary_reaction_n=secondary_interface_reaction_n,
                solid_report=solid_mpm_report,
            )
            fsi_primary_velocity_target_solid_mobility_ratio = max(
                fsi_primary_velocity_target_solid_mobility_ratio,
                accepted_primary_velocity_target_solid_mobility_ratio,
            )
            fsi_secondary_velocity_target_solid_mobility_ratio = max(
                fsi_secondary_velocity_target_solid_mobility_ratio,
                accepted_secondary_velocity_target_solid_mobility_ratio,
            )
            fluid_step_report = advance_fluid_step(
                primary_velocity_mps=z_velocity_vector(
                    0.5
                    * (
                        step_start_main_velocity_z_mps
                        + float(simulator.main_v_mps[None])
                    )
                ),
                secondary_velocity_mps=z_velocity_vector(
                    0.5
                    * (
                        step_start_tail_velocity_z_mps
                        + float(simulator.tail_v_mps[None])
                    )
                ),
                primary_constraint_force_solid_mobility_ratio=(
                    accepted_primary_response_constraint_force_solid_mobility_ratio
                ),
                secondary_constraint_force_solid_mobility_ratio=(
                    accepted_secondary_response_constraint_force_solid_mobility_ratio
                ),
                primary_velocity_target_solid_mobility_ratio=(
                    accepted_primary_velocity_target_solid_mobility_ratio
                ),
                secondary_velocity_target_solid_mobility_ratio=(
                    accepted_secondary_velocity_target_solid_mobility_ratio
                ),
                primary_interface_impedance_force_n=(
                    accepted_primary_fluid_step_robin_impedance_force_n
                ),
                secondary_interface_impedance_force_n=(
                    accepted_secondary_fluid_step_robin_impedance_force_n
                ),
                fluid_substeps=step_fluid_substeps,
            )
            fluid_advance_wall_time_s = time.perf_counter() - fluid_wall_started_at
        else:
            fluid_step_report = reused_fluid_step_report
        divergence = fluid_step_report.divergence
        pressure_outlet_report = fluid_step_report.pressure_outlet_report
        force_report = required_projected_ibm_force_report(
            fluid_step_report.force_report
        )
        impulse_report = required_fluid_impulse_report(fluid_step_report.impulse_report)
        velocity_constraint_report = fluid_step_report.velocity_constraint_report
        velocity_constraint_spread_report = (
            fluid_step_report.velocity_constraint_spread_report
        )
        ibm_correction_iterations = fluid_step_report.ibm_correction_iterations
        ibm_correction_dt_s = fluid_step_report.ibm_correction_dt_s
        fluid_substeps = fluid_step_report.fluid_substeps
        fluid_substep_dt_s = fluid_step_report.fluid_substep_dt_s
        sample_wall_started_at = time.perf_counter()
        sample_report = simulator.sample_after_projection(
            divergence,
            dt_s=fluid_substep_dt_s,
        )
        sample_wall_time_s = time.perf_counter() - sample_wall_started_at
        row = {
            "step": step,
            **sample_report,
        }
        primary_fluid_force_n = fluid_step_report.primary_equivalent_fluid_force_n
        secondary_fluid_force_n = fluid_step_report.secondary_equivalent_fluid_force_n
        primary_velocity_constraint_step_impulse_n_s = _vector3(
            getattr(
                fluid_step_report,
                "primary_velocity_constraint_impulse_n_s",
                (0.0, 0.0, 0.0),
            ),
            name="primary_velocity_constraint_impulse_n_s",
        )
        secondary_velocity_constraint_step_impulse_n_s = _vector3(
            getattr(
                fluid_step_report,
                "secondary_velocity_constraint_impulse_n_s",
                (0.0, 0.0, 0.0),
            ),
            name="secondary_velocity_constraint_impulse_n_s",
        )
        primary_velocity_constraint_step_equivalent_fluid_force_n = _vector3(
            getattr(
                fluid_step_report,
                "primary_velocity_constraint_equivalent_fluid_force_n",
                (0.0, 0.0, 0.0),
            ),
            name="primary_velocity_constraint_equivalent_fluid_force_n",
        )
        secondary_velocity_constraint_step_equivalent_fluid_force_n = _vector3(
            getattr(
                fluid_step_report,
                "secondary_velocity_constraint_equivalent_fluid_force_n",
                (0.0, 0.0, 0.0),
            ),
            name="secondary_velocity_constraint_equivalent_fluid_force_n",
        )
        primary_interface_reaction_n = (
            fluid_step_report.interface_reaction_target.primary_force_n
        )
        secondary_interface_reaction_n = (
            fluid_step_report.interface_reaction_target.secondary_force_n
        )
        row["ibm_correction_iterations"] = ibm_correction_iterations
        row["ibm_correction_dt_s"] = ibm_correction_dt_s
        row["fluid_substeps"] = fluid_substeps
        row["fluid_substep_dt_s"] = fluid_substep_dt_s
        row["pressure_outlet_source_volume_flux_m3s"] = pressure_outlet_report[
            "source_volume_flux_m3s"
        ]
        row["pressure_outlet_positive_source_volume_flux_m3s"] = pressure_outlet_report[
            "positive_source_volume_flux_m3s"
        ]
        row["pressure_outlet_abs_source_volume_flux_m3s"] = pressure_outlet_report[
            "abs_source_volume_flux_m3s"
        ]
        row["pressure_outlet_reachable_source_volume_flux_m3s"] = (
            pressure_outlet_report["zmin_reachable_source_volume_flux_m3s"]
        )
        row["pressure_outlet_unreached_source_volume_flux_m3s"] = (
            pressure_outlet_report["zmin_unreached_source_volume_flux_m3s"]
        )
        row["pressure_outlet_reachability_valid"] = bool(
            pressure_outlet_report.get("zmin_reachability_valid", False)
        )
        row["pressure_outlet_reachability_revision"] = pressure_outlet_report[
            "zmin_reachability_revision"
        ]
        row["pressure_outlet_reachable_source_cell_count"] = pressure_outlet_report[
            "zmin_reachable_source_cell_count"
        ]
        row["pressure_outlet_unreached_source_cell_count"] = pressure_outlet_report[
            "zmin_unreached_source_cell_count"
        ]
        row["pressure_outlet_unreached_source_abs_flux_m3s"] = pressure_outlet_report[
            "zmin_unreached_source_abs_flux_m3s"
        ]
        row["pressure_outlet_unreached_source_centroid_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_x_m"
        ]
        row["pressure_outlet_unreached_source_centroid_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_y_m"
        ]
        row["pressure_outlet_unreached_source_centroid_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_z_m"
        ]
        row["pressure_outlet_unreached_source_min_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_x_m"
        ]
        row["pressure_outlet_unreached_source_min_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_y_m"
        ]
        row["pressure_outlet_unreached_source_min_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_z_m"
        ]
        row["pressure_outlet_unreached_source_max_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_x_m"
        ]
        row["pressure_outlet_unreached_source_max_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_y_m"
        ]
        row["pressure_outlet_unreached_source_max_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_z_m"
        ]
        row["pressure_outlet_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_velocity_to_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_source_ratio"
        ]
        row["pressure_outlet_velocity_to_net_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_net_source_ratio"
        ]
        row["pressure_outlet_velocity_to_positive_source_ratio"] = (
            pressure_outlet_report["zmin_velocity_outlet_to_positive_source_ratio"]
        )
        row["pressure_outlet_velocity_to_abs_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_abs_source_ratio"
        ]
        row["pressure_outlet_pressure_flux_m3s"] = pressure_outlet_report[
            "zmin_pressure_outlet_flux_m3s"
        ]
        row["pressure_outlet_pressure_to_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_source_ratio"
        ]
        row["pressure_outlet_pressure_to_net_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_net_source_ratio"
        ]
        row["pressure_outlet_pressure_to_positive_source_ratio"] = (
            pressure_outlet_report["zmin_pressure_outlet_to_positive_source_ratio"]
        )
        row["pressure_outlet_pressure_to_abs_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_abs_source_ratio"
        ]
        row["pressure_outlet_projection_pre_velocity_flux_m3s"] = (
            pressure_outlet_report["zmin_projection_pre_velocity_outlet_flux_m3s"]
        )
        row["pressure_outlet_projection_post_pressure_velocity_flux_m3s"] = (
            pressure_outlet_report[
                "zmin_projection_post_pressure_velocity_outlet_flux_m3s"
            ]
        )
        row["pressure_outlet_projection_post_boundary_velocity_flux_m3s"] = (
            pressure_outlet_report[
                "zmin_projection_post_boundary_velocity_outlet_flux_m3s"
            ]
        )
        if not sharp_case_runner_enabled:
            row["pressure_projection_cg_project_calls"] = getattr(
                fluid_step_report, "pressure_projection_cg_project_calls", 0
            )
            row["pressure_projection_cg_iterations_total"] = getattr(
                fluid_step_report, "pressure_projection_cg_iterations_total", 0
            )
            row["pressure_projection_cg_iterations_max"] = getattr(
                fluid_step_report, "pressure_projection_cg_iterations_max", 0
            )
            row["pressure_projection_cg_host_residual_checks"] = getattr(
                fluid_step_report, "pressure_projection_cg_host_residual_checks", 0
            )
            row["pressure_projection_cg_mean_projection_count"] = getattr(
                fluid_step_report,
                "pressure_projection_cg_mean_projection_count",
                0,
            )
            row["pressure_projection_cg_restart_count"] = getattr(
                fluid_step_report, "pressure_projection_cg_restart_count", 0
            )
            row["pressure_projection_cg_restart_count_measured"] = getattr(
                fluid_step_report,
                "pressure_projection_cg_restart_count_measured",
                False,
            )
            row["pressure_projection_cg_restart_policy"] = getattr(
                fluid_step_report,
                "pressure_projection_cg_restart_policy",
                "",
            )
            row["pressure_projection_cg_converged_all"] = getattr(
                fluid_step_report, "pressure_projection_cg_converged_all", True
            )
            row["pressure_projection_cg_max_relative_residual"] = getattr(
                fluid_step_report, "pressure_projection_cg_max_relative_residual", 0.0
            )
            row["pressure_projection_cg_max_initial_relative_residual"] = getattr(
                fluid_step_report,
                "pressure_projection_cg_max_initial_relative_residual",
                0.0,
            )
            row["pressure_projection_cg_breakdown_count"] = getattr(
                fluid_step_report, "pressure_projection_cg_breakdown_count", 0
            )
            row["pressure_projection_cg_breakdown_code"] = getattr(
                fluid_step_report, "pressure_projection_cg_breakdown_code", 0
            )
            row["pressure_projection_cg_breakdown_dAd"] = getattr(
                fluid_step_report, "pressure_projection_cg_breakdown_dAd", 0.0
            )
            row["pressure_interface_matrix_diagonal_integral"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_diagonal_integral",
                0.0,
            )
            row["pressure_interface_matrix_rhs_integral"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_rhs_integral",
                0.0,
            )
            row["pressure_interface_matrix_max_abs_diagonal"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_max_abs_diagonal",
                0.0,
            )
            row["pressure_interface_matrix_active_cells"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_active_cells",
                0,
            )
        row["fsi_trial_pressure_projection_cg_project_calls"] = (
            fsi_trial_pressure_projection_cg_project_calls
        )
        row["fsi_trial_pressure_projection_cg_iterations_total"] = (
            fsi_trial_pressure_projection_cg_iterations_total
        )
        row["fsi_trial_pressure_projection_cg_iterations_max"] = (
            fsi_trial_pressure_projection_cg_iterations_max
        )
        row["fsi_trial_pressure_projection_cg_host_residual_checks"] = (
            fsi_trial_pressure_projection_cg_host_residual_checks
        )
        row["fsi_trial_pressure_projection_cg_mean_projection_count"] = (
            fsi_trial_pressure_projection_cg_mean_projection_count
        )
        row["fsi_trial_pressure_projection_cg_converged_all"] = (
            fsi_trial_pressure_projection_cg_converged_all
        )
        row["fsi_trial_pressure_projection_cg_max_relative_residual"] = (
            fsi_trial_pressure_projection_cg_max_relative_residual
        )
        row["fsi_trial_pressure_projection_cg_max_initial_relative_residual"] = (
            fsi_trial_pressure_projection_cg_max_initial_relative_residual
        )
        row["fsi_trial_pressure_projection_cg_breakdown_count"] = (
            fsi_trial_pressure_projection_cg_breakdown_count
        )
        accepted_pressure_projection_cg_project_calls_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_project_calls"])
        )
        accepted_pressure_projection_cg_iterations_total_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_iterations_total"])
        )
        accepted_pressure_projection_cg_host_residual_checks_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_host_residual_checks"])
        )
        accepted_pressure_projection_cg_mean_projection_count_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_mean_projection_count"])
        )
        accepted_pressure_projection_cg_breakdown_count_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_breakdown_count"])
        )
        row["total_pressure_projection_cg_project_calls"] = (
            accepted_pressure_projection_cg_project_calls_for_cost
            + fsi_trial_pressure_projection_cg_project_calls
        )
        row["total_pressure_projection_cg_iterations_total"] = (
            accepted_pressure_projection_cg_iterations_total_for_cost
            + fsi_trial_pressure_projection_cg_iterations_total
        )
        row["total_pressure_projection_cg_iterations_max"] = max(
            int(row["pressure_projection_cg_iterations_max"]),
            fsi_trial_pressure_projection_cg_iterations_max,
        )
        row["total_pressure_projection_cg_host_residual_checks"] = (
            accepted_pressure_projection_cg_host_residual_checks_for_cost
            + fsi_trial_pressure_projection_cg_host_residual_checks
        )
        row["total_pressure_projection_cg_mean_projection_count"] = (
            accepted_pressure_projection_cg_mean_projection_count_for_cost
            + fsi_trial_pressure_projection_cg_mean_projection_count
        )
        row["total_pressure_projection_cg_converged_all"] = (
            bool(row["pressure_projection_cg_converged_all"])
            and fsi_trial_pressure_projection_cg_converged_all
        )
        row["total_pressure_projection_cg_max_relative_residual"] = max(
            float(row["pressure_projection_cg_max_relative_residual"]),
            fsi_trial_pressure_projection_cg_max_relative_residual,
        )
        row["total_pressure_projection_cg_max_initial_relative_residual"] = max(
            float(row["pressure_projection_cg_max_initial_relative_residual"]),
            fsi_trial_pressure_projection_cg_max_initial_relative_residual,
        )
        row["total_pressure_projection_cg_breakdown_count"] = (
            accepted_pressure_projection_cg_breakdown_count_for_cost
            + fsi_trial_pressure_projection_cg_breakdown_count
        )
        row["fsi_coupling_iterations_requested"] = step_fsi_coupling_iterations
        row["fsi_coupling_iterations_base"] = fsi_coupling_iterations
        row["fsi_coupling_adaptive_iterations_max"] = (
            fsi_coupling_adaptive_iterations_max
        )
        row["fsi_coupling_adaptive_iterations_residual_threshold_n"] = (
            fsi_coupling_adaptive_iterations_residual_threshold_n
        )
        row["fsi_coupling_adaptive_iterations_cfl_threshold"] = (
            fsi_coupling_adaptive_iterations_cfl_threshold
        )
        row["fsi_coupling_adaptive_iterations_triggered"] = (
            fsi_coupling_adaptive_iterations_triggered
        )
        row["fsi_coupling_adaptive_iterations_residual_triggered"] = (
            fsi_coupling_adaptive_iterations_residual_triggered
        )
        row["fsi_coupling_adaptive_iterations_cfl_triggered"] = (
            fsi_coupling_adaptive_iterations_cfl_triggered
        )
        row["fsi_coupling_same_step_rerun_iterations_max"] = (
            fsi_coupling_same_step_rerun_iterations_max
        )
        row["fsi_coupling_same_step_rerun_residual_threshold_n"] = (
            fsi_coupling_same_step_rerun_residual_threshold_n
        )
        row["fsi_coupling_same_step_rerun_triggered"] = (
            fsi_coupling_same_step_rerun_triggered
        )
        row["fsi_coupling_same_step_rerun_count"] = fsi_coupling_same_step_rerun_count
        row["fsi_coupling_same_step_rerun_initial_iterations_requested"] = (
            fsi_coupling_same_step_rerun_initial_iterations_requested
        )
        row["fsi_coupling_same_step_rerun_initial_iterations_used"] = (
            fsi_coupling_same_step_rerun_initial_iterations_used
        )
        row["fsi_coupling_same_step_rerun_initial_residual_norm_n"] = (
            fsi_coupling_same_step_rerun_initial_residual_norm_n
        )
        row["fsi_coupling_same_step_rerun_initial_converged"] = (
            fsi_coupling_same_step_rerun_initial_converged
        )
        row["fsi_coupling_same_step_rerun_safety_rejected"] = (
            fsi_coupling_same_step_rerun_safety_rejected
        )
        row["fsi_coupling_same_step_rerun_fluid_substep_factor"] = (
            fsi_coupling_same_step_rerun_fluid_substep_factor
        )
        row["fsi_coupling_same_step_rerun_initial_fluid_substeps"] = (
            fsi_coupling_same_step_rerun_initial_fluid_substeps
        )
        row["fsi_coupling_same_step_rerun_final_fluid_substeps"] = (
            fsi_coupling_same_step_rerun_final_fluid_substeps
        )
        row["fsi_coupling_residual_continuation_iterations_max"] = (
            fsi_coupling_residual_continuation_iterations_max
        )
        row["fsi_coupling_residual_continuation_threshold_n"] = (
            fsi_coupling_residual_continuation_threshold_n
        )
        row["fsi_coupling_residual_continuation_rebound_secant_from_best"] = (
            fsi_coupling_residual_continuation_rebound_secant_from_best
        )
        row["fsi_coupling_residual_continuation_rebound_secant_factor"] = (
            fsi_coupling_residual_continuation_rebound_secant_factor
        )
        row[
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max"
        ] = fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
        row["fsi_coupling_residual_continuation_iteration_count"] = (
            fsi_coupling_residual_continuation_iteration_count
        )
        row["fsi_coupling_residual_continuation_rebound_secant_count"] = (
            fsi_coupling_residual_continuation_rebound_secant_count
        )
        row[
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count"
        ] = fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count
        row["fsi_coupling_previous_step_residual_norm_n"] = (
            math.nan
            if previous_step_fsi_coupling_residual_norm_n is None
            else previous_step_fsi_coupling_residual_norm_n
        )
        row["fsi_coupling_previous_step_cfl"] = (
            math.nan if previous_step_cfl is None else previous_step_cfl
        )
        row["fsi_coupling_mode"] = fsi_coupling_mode
        row["fsi_coupling_mode_paper_hibm_mpm"] = bool(
            fsi_coupling_mode_report["paper_hibm_mpm"]
        )
        row["region_pair_reaction_diagnostic_only"] = bool(
            fsi_coupling_mode_report["region_pair_reaction_diagnostic_only"]
        )
        row["fsi_coupling_solver"] = fsi_coupling_solver
        row["fsi_coupling_iterations_used"] = fsi_coupling_iterations_used
        row["fsi_coupling_enabled"] = fsi_coupling_enabled
        row["fsi_coupling_converged"] = fsi_coupling_converged
        row["fsi_all_trials_rejected"] = fsi_all_trials_rejected
        row["fsi_zero_force_commit_blocked"] = fsi_zero_force_commit_blocked
        row["fsi_coupling_residual_norm_n"] = fsi_coupling_residual_norm_n
        row["fsi_coupling_relaxation_effective"] = fsi_coupling_relaxation_effective
        row["fsi_coupling_rejected_trial_count"] = fsi_coupling_rejected_trial_count
        row["fsi_coupling_rejected_trial_backtrack_count"] = (
            fsi_coupling_rejected_trial_backtrack_count
        )
        row["fsi_coupling_residual_growth_rejected_trial_count"] = (
            fsi_coupling_residual_growth_rejected_trial_count
        )
        row["fsi_coupling_max_residual_rejected_trial_count"] = (
            fsi_coupling_max_residual_rejected_trial_count
        )
        row["fsi_coupling_trial_cfl_rejected_count"] = (
            fsi_coupling_trial_cfl_rejected_count
        )
        row["fsi_coupling_trial_interior_divergence_rejected_count"] = (
            fsi_coupling_trial_interior_divergence_rejected_count
        )
        row["fsi_coupling_trust_region_limited_update_count"] = (
            fsi_coupling_trust_region_limited_update_count
        )
        row["fsi_coupling_trust_region_shrink_count"] = (
            fsi_coupling_trust_region_shrink_count
        )
        row["fsi_coupling_trust_region_growth_count"] = (
            fsi_coupling_trust_region_growth_count
        )
        row["fsi_coupling_trust_region_rebound_backtrack_count"] = (
            fsi_coupling_trust_region_rebound_backtrack_count
        )
        row["fsi_coupling_trust_region_rebound_stop_count"] = (
            fsi_coupling_trust_region_rebound_stop_count
        )
        row["fsi_coupling_trust_region_rebound_stop_suppressed_count"] = (
            fsi_coupling_trust_region_rebound_stop_suppressed_count
        )
        row["fsi_coupling_trust_region_effective_force_increment_n"] = (
            fsi_coupling_trust_region_effective_force_increment_n
        )
        row["fsi_coupling_accepted_trial_cfl"] = fsi_coupling_accepted_trial_cfl
        row["fsi_coupling_accepted_trial_max_fluid_speed_mps"] = (
            fsi_coupling_accepted_trial_max_fluid_speed_mps
        )
        row["fsi_coupling_accepted_trial_interior_divergence_l2"] = (
            fsi_coupling_accepted_trial_interior_divergence_l2
        )
        row["fsi_coupling_trial_cfl_max"] = fsi_coupling_trial_cfl_max
        row["fsi_coupling_trial_interior_divergence_tolerance"] = (
            fsi_coupling_trial_interior_divergence_tolerance
        )
        row["fsi_coupling_trial_interior_divergence_l2_max"] = (
            fsi_coupling_trial_interior_divergence_l2_max
        )
        row["fsi_coupling_target_map_relaxation"] = fsi_coupling_target_map_relaxation
        row["fsi_coupling_rejected_trial_backtrack"] = (
            fsi_coupling_rejected_trial_backtrack
        )
        row["fsi_coupling_residual_growth_rejection_factor"] = (
            fsi_coupling_residual_growth_rejection_factor
        )
        row["fsi_coupling_max_accepted_residual_n"] = (
            fsi_coupling_max_accepted_residual_n
        )
        row["fsi_coupling_trust_region_force_increment_n"] = (
            fsi_coupling_trust_region_force_increment_n
        )
        row["fsi_coupling_trust_region_adaptive"] = fsi_coupling_trust_region_adaptive
        row["fsi_coupling_trust_region_shrink_factor"] = (
            fsi_coupling_trust_region_shrink_factor
        )
        row["fsi_coupling_trust_region_growth_factor"] = (
            fsi_coupling_trust_region_growth_factor
        )
        row["fsi_coupling_trust_region_rebound_factor"] = (
            fsi_coupling_trust_region_rebound_factor
        )
        row["fsi_coupling_trust_region_rebound_backtrack"] = (
            fsi_coupling_trust_region_rebound_backtrack
        )
        row["fsi_coupling_trust_region_rebound_stop_factor"] = (
            fsi_coupling_trust_region_rebound_stop_factor
        )
        row["fsi_coupling_trust_region_rebound_stop_max_residual_n"] = (
            fsi_coupling_trust_region_rebound_stop_max_residual_n
        )
        row["fsi_coupling_iqn_ils_least_squares_update_count"] = (
            fsi_coupling_iqn_ils_least_squares_update_count
        )
        row["fsi_coupling_interface_map_amplification"] = (
            fsi_coupling_interface_map_amplification
        )
        row["fsi_coupling_residual_jacobian_amplification"] = (
            fsi_coupling_residual_jacobian_amplification
        )
        row["fsi_coupling_physical_interface_map_amplification"] = (
            fsi_coupling_physical_interface_map_amplification
        )
        row["fsi_coupling_physical_residual_jacobian_amplification"] = (
            fsi_coupling_physical_residual_jacobian_amplification
        )
        row["fsi_coupling_raw_interface_map_amplification"] = (
            fsi_coupling_raw_interface_map_amplification
        )
        row["fsi_coupling_raw_residual_jacobian_amplification"] = (
            fsi_coupling_raw_residual_jacobian_amplification
        )
        row["fsi_coupling_interface_map_amplification_sample_count"] = (
            fsi_coupling_interface_map_amplification_sample_count
        )
        row["fsi_coupling_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_residual_jacobian_amplification_sample_count
        )
        row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
            fsi_coupling_physical_interface_map_amplification_sample_count
        )
        row["fsi_coupling_physical_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_physical_residual_jacobian_amplification_sample_count
        )
        row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
            fsi_coupling_raw_interface_map_amplification_sample_count
        )
        row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_raw_residual_jacobian_amplification_sample_count
        )
        row["accepted_fsi_trial_state_reused"] = accepted_fsi_trial_state_reused
        row["accepted_fsi_trial_state_readvanced"] = accepted_fsi_trial_state_readvanced
        row["fsi_coupling_trial_force_history_n"] = fsi_coupling_trial_force_history_n
        row["fsi_coupling_target_force_history_n"] = fsi_coupling_target_force_history_n
        row["fsi_coupling_residual_history_n"] = fsi_coupling_residual_history_n
        row["fsi_coupling_physical_target_force_history_n"] = (
            fsi_coupling_physical_target_force_history_n
        )
        row["fsi_coupling_physical_residual_history_n"] = (
            fsi_coupling_physical_residual_history_n
        )
        row["fsi_coupling_raw_target_force_history_n"] = (
            fsi_coupling_raw_target_force_history_n
        )
        row["fsi_coupling_raw_residual_history_n"] = fsi_coupling_raw_residual_history_n
        expected_flux_m3s = float(row["volume_flux_m3s"])
        lip_negative_z_flux_m3s = float(row["lip_flow_negative_z_m3s"])
        outlet_negative_z_flux_m3s = float(row["outlet_flow_negative_z_m3s"])
        downstream_negative_z_flux_m3s = float(row["downstream_flow_negative_z_m3s"])
        row["main_volume_flux_to_lip_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=lip_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["main_volume_flux_to_outlet_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["main_volume_flux_to_downstream_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=downstream_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["outlet_flux_deficit_m3s"] = expected_flux_m3s - outlet_negative_z_flux_m3s
        row["downstream_flux_deficit_m3s"] = (
            expected_flux_m3s - downstream_negative_z_flux_m3s
        )
        row["fsi_velocity_constraint_blend"] = settings.fsi_velocity_constraint_blend
        row["fsi_constraint_force_solid_mobility_ratio"] = (
            settings.fsi_constraint_force_solid_mobility_ratio
        )
        row["fsi_solid_response_mobility_coupling"] = (
            fsi_solid_response_mobility_coupling
        )
        row["fsi_solid_response_dt_s"] = settings.fsi_solid_response_dt_s
        row["fsi_velocity_target_solid_mobility_ratio"] = (
            fsi_velocity_target_solid_mobility_ratio
        )
        row["fsi_solid_response_velocity_mobility_coupling"] = (
            fsi_solid_response_velocity_mobility_coupling
        )
        row["fsi_primary_response_constraint_force_solid_mobility_ratio"] = (
            fsi_primary_response_constraint_force_solid_mobility_ratio
        )
        row["fsi_secondary_response_constraint_force_solid_mobility_ratio"] = (
            fsi_secondary_response_constraint_force_solid_mobility_ratio
        )
        row["fsi_primary_velocity_target_solid_mobility_ratio"] = (
            fsi_primary_velocity_target_solid_mobility_ratio
        )
        row["fsi_secondary_velocity_target_solid_mobility_ratio"] = (
            fsi_secondary_velocity_target_solid_mobility_ratio
        )
        row["fsi_velocity_constraint_solid_mobility_ratio"] = (
            settings.fsi_velocity_constraint_solid_mobility_ratio
        )
        row["fsi_velocity_constraint_active_cells"] = (
            0
            if velocity_constraint_report is None
            else velocity_constraint_report.active_cells
        )
        row["fsi_velocity_constraint_max_delta_mps"] = (
            0.0
            if velocity_constraint_report is None
            else velocity_constraint_report.max_delta_mps
        )
        row["fsi_velocity_constraint_mean_delta_mps"] = (
            0.0
            if velocity_constraint_report is None
            else velocity_constraint_report.mean_delta_mps
        )
        velocity_constraint_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        velocity_constraint_primary_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "primary_momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        velocity_constraint_secondary_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "secondary_momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        row["fsi_velocity_constraint_momentum_delta_x_n_s"] = (
            velocity_constraint_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_momentum_delta_y_n_s"] = (
            velocity_constraint_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_momentum_delta_z_n_s"] = (
            velocity_constraint_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_equivalent_force_norm_n"] = vector_norm(
            velocity_constraint_momentum_delta_n_s
        ) / max(float(ibm_correction_dt_s), 1.0e-30)
        row["fsi_velocity_constraint_primary_momentum_delta_x_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_primary_momentum_delta_y_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_primary_momentum_delta_z_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_x_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_y_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_z_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_primary_equivalent_force_norm_n"] = vector_norm(
            velocity_constraint_primary_momentum_delta_n_s
        ) / max(float(ibm_correction_dt_s), 1.0e-30)
        row["fsi_velocity_constraint_secondary_equivalent_force_norm_n"] = vector_norm(
            velocity_constraint_secondary_momentum_delta_n_s
        ) / max(float(ibm_correction_dt_s), 1.0e-30)
        velocity_constraint_step_impulse_n_s = tuple(
            primary_value + secondary_value
            for primary_value, secondary_value in zip(
                primary_velocity_constraint_step_impulse_n_s,
                secondary_velocity_constraint_step_impulse_n_s,
            )
        )
        velocity_constraint_step_equivalent_fluid_force_n = tuple(
            primary_value + secondary_value
            for primary_value, secondary_value in zip(
                primary_velocity_constraint_step_equivalent_fluid_force_n,
                secondary_velocity_constraint_step_equivalent_fluid_force_n,
            )
        )
        row["fsi_velocity_constraint_step_impulse_x_n_s"] = (
            velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_step_impulse_y_n_s"] = (
            velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_step_impulse_z_n_s"] = (
            velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_primary_step_impulse_x_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_primary_step_impulse_y_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_primary_step_impulse_z_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_x_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_y_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_z_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_step_equivalent_force_norm_n"] = vector_norm(
            velocity_constraint_step_equivalent_fluid_force_n
        )
        row["fsi_velocity_constraint_primary_step_equivalent_force_norm_n"] = (
            vector_norm(primary_velocity_constraint_step_equivalent_fluid_force_n)
        )
        row["fsi_velocity_constraint_secondary_step_equivalent_force_norm_n"] = (
            vector_norm(secondary_velocity_constraint_step_equivalent_fluid_force_n)
        )
        row["fsi_velocity_constraint_sample_count"] = (
            0
            if velocity_constraint_spread_report is None
            else velocity_constraint_spread_report.projected_ibm_sample_count
        )
        surface_diagnostics_wall_started_at = time.perf_counter()
        tri_report = tri_diagnostics.diagnose_from_fields(
            simulator.fluid.velocity,
            simulator.fluid.pressure,
            grid_fields=simulator.fluid,
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            primary_velocity_mps=z_velocity_vector(float(row["main_velocity_z_mps"])),
            secondary_velocity_mps=z_velocity_vector(float(row["tail_velocity_z_mps"])),
            probe_distance_m=fluid_probe_distance_m,
            bounds_min_m=spec.fluid_bounds_min_m,
            bounds_max_m=spec.fluid_bounds_max_m,
            spacing_m=settings.fluid_grid_axis_min_spacing_m,
            grid_nodes=spec.grid_nodes,
            viscosity_pa_s=spec.water_viscosity_pa_s,
        )
        surface_diagnostics_wall_time_s = (
            time.perf_counter() - surface_diagnostics_wall_started_at
        )
        row.update(
            {
                "pressure_traction_force_x_n": tri_report.pressure_traction_force_n[0],
                "pressure_traction_force_y_n": tri_report.pressure_traction_force_n[1],
                "pressure_traction_force_z_n": tri_report.pressure_traction_force_n[2],
                "main_pressure_traction_force_z_n": tri_report.primary_pressure_traction_force_n[
                    2
                ],
                "tail_pressure_traction_force_z_n": tri_report.secondary_pressure_traction_force_n[
                    2
                ],
                "viscous_traction_force_x_n": tri_report.viscous_traction_force_n[0],
                "viscous_traction_force_y_n": tri_report.viscous_traction_force_n[1],
                "viscous_traction_force_z_n": tri_report.viscous_traction_force_n[2],
                "main_viscous_traction_force_z_n": tri_report.primary_viscous_traction_force_n[
                    2
                ],
                "tail_viscous_traction_force_z_n": tri_report.secondary_viscous_traction_force_n[
                    2
                ],
                "fluid_stress_traction_force_x_n": tri_report.fluid_stress_traction_force_n[
                    0
                ],
                "fluid_stress_traction_force_y_n": tri_report.fluid_stress_traction_force_n[
                    1
                ],
                "fluid_stress_traction_force_z_n": tri_report.fluid_stress_traction_force_n[
                    2
                ],
                "main_fluid_stress_traction_force_z_n": tri_report.primary_fluid_stress_traction_force_n[
                    2
                ],
                "tail_fluid_stress_traction_force_z_n": tri_report.secondary_fluid_stress_traction_force_n[
                    2
                ],
                "pressure_traction_abs_force_n": tri_report.pressure_traction_abs_force_n,
                "pressure_traction_area_m2": tri_report.pressure_traction_area_m2,
                "pressure_traction_face_count": tri_report.pressure_traction_face_count,
                "projected_ibm_residual_mps": tri_report.projected_ibm_residual_mps,
                "projected_ibm_residual_l2_mps": tri_report.projected_ibm_residual_l2_mps,
                "projected_ibm_sample_count": tri_report.projected_ibm_sample_count,
                "fsi_probe_invalid_count": tri_report.invalid_probe_count,
                "fsi_probe_valid_fraction": tri_report.valid_probe_fraction,
                "fsi_probe_invalid_area_m2": tri_report.invalid_probe_area_m2,
                "fsi_probe_invalid_volume_source_m3s": tri_report.invalid_probe_volume_source_m3s,
                "fsi_force_probe_sample_count": force_report.force_sample_count,
                "fsi_force_probe_invalid_count": force_report.force_invalid_probe_count,
                "fsi_force_probe_valid_fraction": force_report.force_valid_probe_fraction,
                "fsi_force_probe_invalid_area_m2": force_report.invalid_probe_area_m2,
                "fsi_force_probe_invalid_volume_source_m3s": force_report.invalid_probe_volume_source_m3s,
                "fsi_grid_force_x_n": primary_fluid_force_n[0]
                + secondary_fluid_force_n[0],
                "fsi_grid_force_y_n": primary_fluid_force_n[1]
                + secondary_fluid_force_n[1],
                "fsi_grid_force_z_n": primary_fluid_force_n[2]
                + secondary_fluid_force_n[2],
                "fsi_last_correction_grid_force_x_n": force_report.grid_force_n[0],
                "fsi_last_correction_grid_force_y_n": force_report.grid_force_n[1],
                "fsi_last_correction_grid_force_z_n": force_report.grid_force_n[2],
                "main_fsi_fluid_force_x_n": primary_fluid_force_n[0],
                "main_fsi_fluid_force_y_n": primary_fluid_force_n[1],
                "main_fsi_fluid_force_z_n": primary_fluid_force_n[2],
                "tail_fsi_fluid_force_x_n": secondary_fluid_force_n[0],
                "tail_fsi_fluid_force_y_n": secondary_fluid_force_n[1],
                "tail_fsi_fluid_force_z_n": secondary_fluid_force_n[2],
                "main_fsi_fluid_reaction_x_n": primary_interface_reaction_n[0],
                "main_fsi_fluid_reaction_y_n": primary_interface_reaction_n[1],
                "main_fsi_fluid_reaction_z_n": primary_interface_reaction_n[2],
                "tail_fsi_fluid_reaction_x_n": secondary_interface_reaction_n[0],
                "tail_fsi_fluid_reaction_y_n": secondary_interface_reaction_n[1],
                "tail_fsi_fluid_reaction_z_n": secondary_interface_reaction_n[2],
                "fsi_constraint_force_x_n": force_report.constraint_force_n[0],
                "fsi_constraint_force_y_n": force_report.constraint_force_n[1],
                "fsi_constraint_force_z_n": force_report.constraint_force_n[2],
                "main_fsi_constraint_force_z_n": force_report.primary_constraint_force_n[
                    2
                ],
                "tail_fsi_constraint_force_z_n": force_report.secondary_constraint_force_n[
                    2
                ],
                "main_fsi_constraint_reaction_z_n": -force_report.primary_constraint_force_n[
                    2
                ],
                "tail_fsi_constraint_reaction_z_n": -force_report.secondary_constraint_force_n[
                    2
                ],
                "fsi_volume_source_m3s": force_report.volume_source_m3s,
                "main_fsi_volume_source_m3s": force_report.primary_volume_source_m3s,
                "tail_fsi_volume_source_m3s": force_report.secondary_volume_source_m3s,
                "fsi_active_force_cells": force_report.active_force_cells,
                "fluid_impulse_x_ns": impulse_report.grid_impulse_n_s[0],
                "fluid_impulse_y_ns": impulse_report.grid_impulse_n_s[1],
                "fluid_impulse_z_ns": impulse_report.grid_impulse_n_s[2],
                "fluid_impulse_relative_error": impulse_report.impulse_relative_error,
            }
        )
        force_decomposition = force_decomposition_report(
            grid_force_n=force_report.grid_force_n,
            component_forces_n=(
                force_report.primary_fluid_force_n,
                force_report.secondary_fluid_force_n,
            ),
        )
        row.update(
            {
                "fsi_last_correction_grid_decomposition_residual_x_n": force_decomposition[
                    "residual_components_n"
                ][0],
                "fsi_last_correction_grid_decomposition_residual_y_n": force_decomposition[
                    "residual_components_n"
                ][1],
                "fsi_last_correction_grid_decomposition_residual_z_n": force_decomposition[
                    "residual_components_n"
                ][2],
                "fsi_last_correction_grid_decomposition_residual_abs_n": force_decomposition[
                    "residual_norm_n"
                ],
                "fsi_last_correction_grid_decomposition_relative_error": force_decomposition[
                    "relative_error"
                ],
            }
        )
        fsi_interface_reaction_n = (
            primary_interface_reaction_n[0] + secondary_interface_reaction_n[0],
            primary_interface_reaction_n[1] + secondary_interface_reaction_n[1],
            primary_interface_reaction_n[2] + secondary_interface_reaction_n[2],
        )
        fsi_interface_balance = action_reaction_balance(
            (
                row["fsi_grid_force_x_n"],
                row["fsi_grid_force_y_n"],
                row["fsi_grid_force_z_n"],
            ),
            fsi_interface_reaction_n,
        )
        row.update(
            {
                "fsi_action_reaction_residual_x_n": fsi_interface_balance.residual_components_n[
                    0
                ],
                "fsi_action_reaction_residual_y_n": fsi_interface_balance.residual_components_n[
                    1
                ],
                "fsi_action_reaction_residual_z_n": fsi_interface_balance.residual_components_n[
                    2
                ],
                "fsi_action_reaction_residual_abs_n": fsi_interface_balance.residual_norm_n,
                "fsi_action_reaction_relative_error": fsi_interface_balance.relative_error,
            }
        )
        fluid_reaction_balance = action_reaction_balance(
            (
                row["main_fsi_fluid_force_x_n"],
                row["main_fsi_fluid_force_y_n"],
                row["main_fsi_fluid_force_z_n"],
                row["tail_fsi_fluid_force_x_n"],
                row["tail_fsi_fluid_force_y_n"],
                row["tail_fsi_fluid_force_z_n"],
            ),
            (
                row["main_fsi_fluid_reaction_x_n"],
                row["main_fsi_fluid_reaction_y_n"],
                row["main_fsi_fluid_reaction_z_n"],
                row["tail_fsi_fluid_reaction_x_n"],
                row["tail_fsi_fluid_reaction_y_n"],
                row["tail_fsi_fluid_reaction_z_n"],
            ),
        )
        row["fsi_fluid_reaction_action_reaction_residual_x_n"] = (
            fluid_reaction_balance.residual_components_n[0]
            + fluid_reaction_balance.residual_components_n[3]
        )
        row["fsi_fluid_reaction_action_reaction_residual_y_n"] = (
            fluid_reaction_balance.residual_components_n[1]
            + fluid_reaction_balance.residual_components_n[4]
        )
        row["fsi_fluid_reaction_action_reaction_residual_z_n"] = (
            fluid_reaction_balance.residual_components_n[2]
            + fluid_reaction_balance.residual_components_n[5]
        )
        row["fsi_fluid_reaction_action_reaction_residual_abs_n"] = (
            fluid_reaction_balance.residual_norm_n
        )
        row["fsi_fluid_reaction_action_reaction_relative_error"] = (
            fluid_reaction_balance.relative_error
        )
        main_full_reaction_balance = (
            fluid_step_report.primary_interface_reaction_balance
        )
        tail_full_reaction_balance = (
            fluid_step_report.secondary_interface_reaction_balance
        )
        row["main_fsi_fluid_reaction_full_residual_n"] = (
            main_full_reaction_balance.residual_norm_n
        )
        row["main_fsi_fluid_reaction_full_relative_error"] = (
            main_full_reaction_balance.relative_error
        )
        row["tail_fsi_fluid_reaction_full_residual_n"] = (
            tail_full_reaction_balance.residual_norm_n
        )
        row["tail_fsi_fluid_reaction_full_relative_error"] = (
            tail_full_reaction_balance.relative_error
        )
        if solid_mpm_report is None:
            solid_mpm_report = solid_mpm.report()
        solid_report_context = f"solid model {args.solid_model!r} report"
        solid_mpm_total_force_n = solid_force_vector_from_report(
            solid_mpm_report,
            solid_model=args.solid_model,
        )
        solid_mpm_row = {
            "solid_mpm_particle_count": solid_mpm_report.particle_count,
            "solid_mpm_active_grid_nodes": solid_mpm_report.active_grid_nodes,
            "solid_mpm_particle_spacing_m": solid_mpm_report.particle_spacing_m,
            "solid_mpm_grid_dx_m": solid_mpm_report.grid_spacing_m[0],
            "solid_mpm_grid_dy_m": solid_mpm_report.grid_spacing_m[1],
            "solid_mpm_grid_dz_m": solid_mpm_report.grid_spacing_m[2],
            "solid_mpm_total_mass_kg": solid_mpm_report.total_mass_kg,
            "solid_mpm_particle_momentum_x_kg_mps": solid_mpm_report.particle_momentum_kg_mps[
                0
            ],
            "solid_mpm_particle_momentum_y_kg_mps": solid_mpm_report.particle_momentum_kg_mps[
                1
            ],
            "solid_mpm_particle_momentum_z_kg_mps": solid_mpm_report.particle_momentum_kg_mps[
                2
            ],
            "solid_mpm_grid_momentum_x_kg_mps": solid_mpm_report.grid_momentum_kg_mps[
                0
            ],
            "solid_mpm_grid_momentum_y_kg_mps": solid_mpm_report.grid_momentum_kg_mps[
                1
            ],
            "solid_mpm_grid_momentum_z_kg_mps": solid_mpm_report.grid_momentum_kg_mps[
                2
            ],
            "solid_mpm_transfer_relative_error": solid_mpm_report.transfer_relative_error,
            "solid_mpm_max_speed_mps": _required_finite_report_number(
                solid_mpm_report,
                field="max_speed_mps",
                context=solid_report_context,
            ),
            "solid_mpm_total_force_x_n": solid_mpm_total_force_n[0],
            "solid_mpm_total_force_y_n": solid_mpm_total_force_n[1],
            "solid_mpm_total_force_z_n": solid_mpm_total_force_n[2],
        }
        if args.solid_model == "neo_hookean_mpm":
            solid_mpm_row["solid_mpm_max_abs_j"] = _required_finite_report_number(
                solid_mpm_report,
                field="max_abs_j",
                context=solid_report_context,
            )
        row.update(solid_mpm_row)
        previous_primary_reaction_n = _taichi_vector3_to_tuple(
            simulator.primary_interface_reaction_force_n[None]
        )
        previous_secondary_reaction_n = _taichi_vector3_to_tuple(
            simulator.secondary_interface_reaction_force_n[None]
        )
        stabilized_primary_reaction_target_n = _vector3(
            fluid_step_report.interface_reaction_target.primary_force_n,
            name="stabilized_primary_reaction_target_n",
        )
        stabilized_secondary_reaction_target_n = _vector3(
            fluid_step_report.interface_reaction_target.secondary_force_n,
            name="stabilized_secondary_reaction_target_n",
        )
        accepted_interface_velocity_mps = _combine_region_pair_vectors(
            solid_mpm_report.primary_mean_velocity_mps,
            solid_mpm_report.secondary_mean_velocity_mps,
        )
        accepted_robin_impedance_force_n = robin_neumann_impedance_force(
            velocity_mps=accepted_interface_velocity_mps,
            previous_velocity_mps=robin_previous_velocity_mps,
            impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
        )
        (
            accepted_primary_robin_impedance_force_n,
            accepted_secondary_robin_impedance_force_n,
        ) = _split_region_pair_vector(accepted_robin_impedance_force_n)
        raw_primary_reaction_target_n = tuple(
            target_value - robin_value
            for target_value, robin_value in zip(
                stabilized_primary_reaction_target_n,
                accepted_primary_robin_impedance_force_n,
            )
        )
        raw_secondary_reaction_target_n = tuple(
            target_value - robin_value
            for target_value, robin_value in zip(
                stabilized_secondary_reaction_target_n,
                accepted_secondary_robin_impedance_force_n,
            )
        )
        selected_reaction_target_n = interface_reaction_target_for_mode(
            interface_reaction_robin_target_mode,
            raw_target_force_n=_combine_region_pair_vectors(
                raw_primary_reaction_target_n,
                raw_secondary_reaction_target_n,
            ),
            stabilized_target_force_n=_combine_region_pair_vectors(
                stabilized_primary_reaction_target_n,
                stabilized_secondary_reaction_target_n,
            ),
        )
        (
            selected_primary_reaction_target_n,
            selected_secondary_reaction_target_n,
        ) = _split_region_pair_vector(selected_reaction_target_n)
        raw_main_reaction_target_z_n = raw_primary_reaction_target_n[2]
        raw_tail_reaction_target_z_n = raw_secondary_reaction_target_n[2]
        main_velocity_z_mps = float(row["main_velocity_z_mps"])
        tail_velocity_z_mps = float(row["tail_velocity_z_mps"])
        reaction_step_update = update_interface_reaction_for_next_step(
            previous_force_n=_combine_region_pair_vectors(
                previous_primary_reaction_n,
                previous_secondary_reaction_n,
            ),
            target_force_n=selected_reaction_target_n,
            velocity_mps=accepted_interface_velocity_mps,
            state=interface_reaction_state,
            initial_relaxation=interface_reaction_relaxation,
            use_aitken=interface_reaction_aitken,
            passivity_limit=interface_reaction_passivity_limit,
            robin_impedance_ns_per_m=0.0,
            aitken_lower_bound=interface_reaction_aitken_lower_bound,
            aitken_upper_bound=interface_reaction_aitken_upper_bound,
        )
        interface_reaction_state = reaction_step_update.next_state
        reaction_update = reaction_step_update.update
        interface_reaction_relaxation_used = reaction_step_update.relaxation
        relaxed_primary_reaction_n, relaxed_secondary_reaction_n = (
            _split_region_pair_vector(reaction_update.force_n)
        )
        relaxed_main_reaction_z_n = relaxed_primary_reaction_n[2]
        relaxed_tail_reaction_z_n = relaxed_secondary_reaction_n[2]
        row["interface_reaction_relaxation"] = interface_reaction_relaxation
        row["interface_reaction_aitken"] = interface_reaction_aitken
        row["interface_reaction_aitken_lower_bound"] = (
            interface_reaction_aitken_lower_bound
        )
        row["interface_reaction_aitken_upper_bound"] = (
            interface_reaction_aitken_upper_bound
        )
        row["interface_reaction_relaxation_effective"] = (
            interface_reaction_relaxation_used
        )
        row["interface_reaction_passivity_limit"] = interface_reaction_passivity_limit
        row["interface_reaction_robin_impedance_ns_m"] = (
            interface_reaction_robin_impedance_ns_m
        )
        row["interface_reaction_robin_matrix_impedance_ns_m"] = (
            interface_reaction_robin_matrix_impedance_ns_m
        )
        row["interface_reaction_robin_target_mode"] = (
            interface_reaction_robin_target_mode
        )
        row["raw_main_pressure_traction_z_n"] = (
            tri_report.primary_pressure_traction_force_n[2]
        )
        row["raw_tail_pressure_traction_z_n"] = (
            tri_report.secondary_pressure_traction_force_n[2]
        )
        row["raw_main_interface_reaction_z_n"] = raw_main_reaction_target_z_n
        row["raw_tail_interface_reaction_z_n"] = raw_tail_reaction_target_z_n
        row["main_interface_reaction_robin_impedance_force_z_n"] = (
            accepted_robin_impedance_force_n[2]
        )
        row["tail_interface_reaction_robin_impedance_force_z_n"] = (
            accepted_robin_impedance_force_n[5]
        )
        row["main_interface_reaction_stabilized_target_z_n"] = (
            stabilized_primary_reaction_target_n[2]
        )
        row["tail_interface_reaction_stabilized_target_z_n"] = (
            stabilized_secondary_reaction_target_n[2]
        )
        row["main_interface_reaction_selected_target_z_n"] = (
            selected_primary_reaction_target_n[2]
        )
        row["tail_interface_reaction_selected_target_z_n"] = (
            selected_secondary_reaction_target_n[2]
        )
        row["main_interface_reaction_residual_z_n"] = reaction_update.residual_n[2]
        row["tail_interface_reaction_residual_z_n"] = reaction_update.residual_n[5]
        row["relaxed_main_interface_reaction_z_n_next"] = relaxed_main_reaction_z_n
        row["relaxed_tail_interface_reaction_z_n_next"] = relaxed_tail_reaction_z_n
        row["raw_main_pressure_traction_power_w"] = (
            tri_report.primary_pressure_traction_force_n[2] * main_velocity_z_mps
        )
        row["raw_main_interface_reaction_power_w"] = (
            raw_main_reaction_target_z_n * main_velocity_z_mps
        )
        row["relaxed_main_interface_reaction_power_w_next"] = reaction_update.power_w[2]
        row["raw_tail_pressure_traction_power_w"] = (
            tri_report.secondary_pressure_traction_force_n[2] * tail_velocity_z_mps
        )
        row["raw_tail_interface_reaction_power_w"] = (
            raw_tail_reaction_target_z_n * tail_velocity_z_mps
        )
        row["relaxed_tail_interface_reaction_power_w_next"] = reaction_update.power_w[5]
        row["main_interface_reaction_passivity_limited"] = (
            reaction_update.passivity_limited[2]
        )
        row["tail_interface_reaction_passivity_limited"] = (
            reaction_update.passivity_limited[5]
        )
        simulator.set_interface_reaction(
            primary_force_n=relaxed_primary_reaction_n,
            secondary_force_n=relaxed_secondary_reaction_n,
        )
        row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
        row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
        row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
        row["sample_wall_time_s"] = sample_wall_time_s
        row["surface_diagnostics_wall_time_s"] = surface_diagnostics_wall_time_s
        row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
        row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
        row["fluid_substeps_base"] = effective_fluid_substeps
        row["adaptive_fluid_substeps_enabled"] = adaptive_fluid_substeps_enabled
        row["adaptive_fluid_substeps_target_cfl"] = float(
            args.adaptive_fluid_substeps_target_cfl
        )
        row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
        row["adaptive_fluid_substeps_previous_substeps"] = previous_step_fluid_substeps
        rows.append(row)
        try:
            _raise_for_step_numerical_guard(
                row,
                cfl_limit=0.5,
                divergence_l2_limit=float(args.projection_divergence_tolerance),
            )
            _raise_for_step_solid_out_of_bounds_guard(row)
        except Exception as exc:
            _write_step_failure_artifacts(
                process_path=process_path,
                output_dir=output_dir,
                rows=rows,
                step=step,
                exc=exc,
                fluid=simulator.fluid,
            )
            raise
        previous_step_cfl = float(row["cfl"])
        previous_step_fsi_coupling_residual_norm_n = float(
            row["fsi_coupling_residual_norm_n"]
        )
        previous_step_fluid_substeps = int(
            float(row.get("fluid_substeps", step_fluid_substeps))
        )
        if args.checkpoint_every_step:
            write_csv(history_path, rows)
            checkpoint_wall_started_at = time.perf_counter()
            write_run_checkpoint(
                run_checkpoint_path,
                completed_step=step,
                step_count=step_count,
                full_pressure_waveform_steps=full_pressure_waveform_steps,
                args=args,
                simulator=simulator,
                solid_mpm=solid_mpm,
                interface_reaction_state=interface_reaction_state,
                sharp_coupling_state=sharp_coupling_state,
                frozen_run_fingerprint=frozen_run_fingerprint,
            )
            checkpoint_wall_time_s = time.perf_counter() - checkpoint_wall_started_at
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            write_csv(history_path, rows)
        if args.progress and (
            step == 1 or step == step_count or step % args.progress_interval == 0
        ):
            print(
                "step={step} t={time_s:.6f}s p={pressure_load_pa:.3f}Pa "
                "main_z={main_displacement_z_m:.6e}m "
                "outlet_ratio={main_volume_flux_to_outlet_ratio:.6e} "
                "outlet_neg_z_Q={outlet_flow_negative_z_m3s:.6e}m3/s "
                "cfl={cfl:.3e} div_l2={divergence_l2:.3e} "
                "interior_div_l2={interior_divergence_l2:.3e}".format(**row),
                flush=True,
            )
        if (
            max_wall_time_s > 0.0
            and step < step_count
            and time.perf_counter() - state.run_started_at_perf >= max_wall_time_s
        ):
            partial_run_stopped = True
            partial_run_reason = "max_wall_time_s"
            break
    return StepLoopResult(
        rows=rows,
        interface_reaction_state=interface_reaction_state,
        sharp_coupling_state=sharp_coupling_state,
        partial_run_stopped=partial_run_stopped,
        partial_run_reason=partial_run_reason,
    )
