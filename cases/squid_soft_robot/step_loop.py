from __future__ import annotations

import math
from collections.abc import Mapping


def _required_context_value(context: Mapping[str, object], name: str) -> object:
    try:
        return context[name]
    except KeyError as exc:
        raise KeyError(f"squid step loop context is missing {name!r}") from exc

def run_squid_step_loop(context: Mapping[str, object]) -> dict[str, object]:
    """Run the squid case time-step loop using runner-built context."""
    _raise_for_closure_coverage_floor = _required_context_value(context, '_raise_for_closure_coverage_floor')
    _raise_for_step_numerical_guard = _required_context_value(context, '_raise_for_step_numerical_guard')
    _raise_for_step_solid_out_of_bounds_guard = _required_context_value(context, '_raise_for_step_solid_out_of_bounds_guard')
    _sharp_marker_aitken_relaxation = _required_context_value(context, '_sharp_marker_aitken_relaxation')
    _sharp_marker_fixed_point_residual_vector_mps = _required_context_value(context, '_sharp_marker_fixed_point_residual_vector_mps')
    _write_fluid_snapshot_npz = _required_context_value(context, '_write_fluid_snapshot_npz')
    _write_hibm_high_residual_cell_dump = _required_context_value(context, '_write_hibm_high_residual_cell_dump')
    _write_hibm_pressure_neumann_invalid_row_dump = _required_context_value(context, '_write_hibm_pressure_neumann_invalid_row_dump')
    _write_hibm_zero_correctable_cell_dump = _required_context_value(context, '_write_hibm_zero_correctable_cell_dump')
    _write_step_failure_artifacts = _required_context_value(context, '_write_step_failure_artifacts')
    adaptive_fluid_substeps_enabled = _required_context_value(context, 'adaptive_fluid_substeps_enabled')
    args = _required_context_value(context, 'args')
    build_hibm_mpm_sharp_case_row = _required_context_value(context, 'build_hibm_mpm_sharp_case_row')
    cg_preconditioner = _required_context_value(context, 'cg_preconditioner')
    cg_tolerance = _required_context_value(context, 'cg_tolerance')
    effective_fluid_substeps = _required_context_value(context, 'effective_fluid_substeps')
    effective_multigrid_cycles = _required_context_value(context, 'effective_multigrid_cycles')
    estimated_solid_particle_spacing_m = _required_context_value(context, 'estimated_solid_particle_spacing_m')
    first_step = _required_context_value(context, 'first_step')
    fluid_probe_distance_m = _required_context_value(context, 'fluid_probe_distance_m')
    fluid_substep_controller = _required_context_value(context, 'fluid_substep_controller')
    fsi_coupling_iterations = _required_context_value(context, 'fsi_coupling_iterations')
    fsi_coupling_mode_report = _required_context_value(context, 'fsi_coupling_mode_report')
    fsi_marker_coupling_tolerance_mps = _required_context_value(context, 'fsi_marker_coupling_tolerance_mps')
    full_pressure_waveform_steps = _required_context_value(context, 'full_pressure_waveform_steps')
    hibm_mpm_sharp_step_summary = _required_context_value(context, 'hibm_mpm_sharp_step_summary')
    history_path = _required_context_value(context, 'history_path')
    interface_reaction_aitken = _required_context_value(context, 'interface_reaction_aitken')
    interface_reaction_relaxation = _required_context_value(context, 'interface_reaction_relaxation')
    material = _required_context_value(context, 'material')
    max_wall_time_s = _required_context_value(context, 'max_wall_time_s')
    output_dir = _required_context_value(context, 'output_dir')
    partial_run_reason = _required_context_value(context, 'partial_run_reason')
    partial_run_stopped = _required_context_value(context, 'partial_run_stopped')
    pressure_far_side_normal_sign = _required_context_value(context, 'pressure_far_side_normal_sign')
    pressure_load_region_id = _required_context_value(context, 'pressure_load_region_id')
    pressure_outlet_zmin_enabled = _required_context_value(context, 'pressure_outlet_zmin_enabled')
    pressure_schedule_step_end_pa = _required_context_value(context, 'pressure_schedule_step_end_pa')
    pressure_solver_name = _required_context_value(context, 'pressure_solver_name')
    previous_step_cfl = _required_context_value(context, 'previous_step_cfl')
    previous_step_fluid_substeps = _required_context_value(context, 'previous_step_fluid_substeps')
    primary_shell_region_id = _required_context_value(context, 'primary_shell_region_id')
    process_path = _required_context_value(context, 'process_path')
    projection_divergence_cleanup_iterations = _required_context_value(context, 'projection_divergence_cleanup_iterations')
    publish_solid_report_to_reduced_state = _required_context_value(context, 'publish_solid_report_to_reduced_state')
    relaxed_sharp_marker_state_arrays = _required_context_value(context, 'relaxed_sharp_marker_state_arrays')
    relaxed_sharp_pressure_neumann_gradient_state_array = _required_context_value(context, 'relaxed_sharp_pressure_neumann_gradient_state_array')
    restore_sharp_marker_state_arrays = _required_context_value(context, 'restore_sharp_marker_state_arrays')
    restore_sharp_pressure_neumann_gradient_state_array = _required_context_value(context, 'restore_sharp_pressure_neumann_gradient_state_array')
    rows = _required_context_value(context, 'rows')
    run_checkpoint_path = _required_context_value(context, 'run_checkpoint_path')
    run_started_at_perf = _required_context_value(context, 'run_started_at_perf')
    secondary_shell_region_id = _required_context_value(context, 'secondary_shell_region_id')
    sharp_coupling_state = _required_context_value(context, 'sharp_coupling_state')
    sharp_marker_fixed_point_residual_diagnostics_mps = _required_context_value(context, 'sharp_marker_fixed_point_residual_diagnostics_mps')
    sharp_marker_fixed_point_residual_mps = _required_context_value(context, 'sharp_marker_fixed_point_residual_mps')
    sharp_marker_state_arrays = _required_context_value(context, 'sharp_marker_state_arrays')
    sharp_pressure_neumann_gradient_state_array = _required_context_value(context, 'sharp_pressure_neumann_gradient_state_array')
    sharp_report_fluid_projection_failure_reason = _required_context_value(context, 'sharp_report_fluid_projection_failure_reason')
    signed_positive_source_flux_ratio = _required_context_value(context, 'signed_positive_source_flux_ratio')
    simulator = _required_context_value(context, 'simulator')
    solid_mpm = _required_context_value(context, 'solid_mpm')
    solid_mpm_flip_blend = _required_context_value(context, 'solid_mpm_flip_blend')
    solid_mpm_substeps = _required_context_value(context, 'solid_mpm_substeps')
    solid_sub_dt_s = _required_context_value(context, 'solid_sub_dt_s')
    solid_substep_velocity_damping = _required_context_value(context, 'solid_substep_velocity_damping')
    spec = _required_context_value(context, 'spec')
    step_count = _required_context_value(context, 'step_count')
    time = _required_context_value(context, 'time')
    write_csv = _required_context_value(context, 'write_csv')
    write_run_checkpoint = _required_context_value(context, 'write_run_checkpoint')

    for step in range(first_step, step_count + 1):
        step_wall_started_at = time.perf_counter()
        step_fluid_substeps = effective_fluid_substeps
        if fluid_substep_controller is not None:
            step_fluid_substeps = fluid_substep_controller.substeps_for_next_step(
                previous_cfl=previous_step_cfl,
                previous_substeps=previous_step_fluid_substeps,
            )
        step_fluid_substep_dt_s = float(spec.dt_s) / float(step_fluid_substeps)
        fsi_coupling_wall_time_s = 0.0
        solid_advance_wall_time_s = 0.0
        fluid_advance_wall_time_s = 0.0
        sample_wall_time_s = 0.0
        checkpoint_wall_time_s = 0.0
        fsi_coupling_relaxation_effective = interface_reaction_relaxation
        fsi_coupling_aitken_update_count = 0
        fsi_coupling_physical_interface_map_amplification = 0.0
        fsi_coupling_physical_interface_map_amplification_sample_count = 0
        if sharp_coupling_state is None:
            raise RuntimeError("sharp HIBM-MPM coupling state was not initialized")
        current_time_s = float(simulator.time_s[None])
        pressure_pa = pressure_schedule_step_end_pa(
            current_time_s,
            spec.dt_s,
            spec,
        )

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
            for _ in range(solid_mpm_substeps):
                if args.solid_model == "tri_mooney_shell_mpm":
                    report = solid_mpm.advance_with_external_forces(
                        dt_s=solid_sub_dt_s,
                        primary_region_id=primary_shell_region_id,
                        secondary_region_id=secondary_shell_region_id,
                        velocity_damping=solid_substep_velocity_damping,
                        flip_blend=solid_mpm_flip_blend,
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
                        read_report=False,
                    )
                else:
                    raise ValueError(f"Unsupported solid model: {args.solid_model}")
            solid_advance_wall_time_s += (
                time.perf_counter() - solid_wall_started_at
            )
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
                far_pressure_region_id=pressure_load_region_id,
                far_pressure_barrier_region_id=5,
                far_pressure_pa=pressure_pa,
                far_pressure_side_normal_sign=pressure_far_side_normal_sign,
                far_pressure_inside_probe_max_multiplier=12.0,
                two_sided_probe_max_multiplier=12.0,
                one_sided_pressure_region_id=secondary_shell_region_id,
                one_sided_reference_pressure_pa=0.0,
                one_sided_probe_max_multiplier=12.0,
                far_pressure_air_backed=True,
                far_pressure_air_backed_probe_normal_sign=pressure_far_side_normal_sign,
                fluid_dt_s=spec.dt_s,
                fluid_substeps=step_fluid_substeps,
                projection_iterations=int(args.projection_iterations),
                run_fluid_predictor=True,
                pressure_neumann_density_kgm3=spec.water_density_kgm3,
                pressure_neumann_dt_s=spec.dt_s,
                pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                pressure_solver=pressure_solver_name,
                pressure_solve_failure_policy=str(args.pressure_solve_failure_policy),
                fluid_advection_scheme=str(args.fluid_advection_scheme),
                multigrid_cycles=effective_multigrid_cycles,
                cg_tolerance=cg_tolerance,
                cg_preconditioner=cg_preconditioner,
                divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
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
            nonlocal fsi_coupling_relaxation_effective
            nonlocal fsi_coupling_aitken_update_count
            nonlocal fsi_coupling_physical_interface_map_amplification
            nonlocal fsi_coupling_physical_interface_map_amplification_sample_count

            requested_iterations = max(1, int(fsi_coupling_iterations))
            if requested_iterations <= 1:
                report = advance_sharp_trial_once()
                return report, {
                    "hibm_coupling_scheme": "explicit_loose",
                    "hibm_added_mass_stability_status": (
                        "unmeasured_single_pass"
                    ),
                    "hibm_added_mass_stability_measured": False,
                    "hibm_added_mass_stabilization": "none",
                    "hibm_semi_implicit_coupling_enabled": False,
                    "hibm_semi_implicit_coupling_matrix_active": False,
                    "hibm_fsi_coupling_iterations_used": 1,
                    "hibm_fsi_coupling_converged": False,
                    "hibm_fsi_coupling_explicit_single_pass": True,
                    "hibm_fsi_coupling_residual_source": (
                        "unmeasured_single_pass"
                    ),
                }

            simulator.save_reduced_state()
            simulator.fluid.save_state()
            solid_mpm.save_state()
            marker_guess = sharp_marker_state_arrays(sharp_coupling_state.markers)
            pressure_gradient_state = (
                sharp_pressure_neumann_gradient_state_array(sharp_coupling_state)
            )
            previous_velocity_residual_vector = None
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
                    sharp_coupling_state.markers.region_id.to_numpy()
                    [: int(sharp_coupling_state.markers.marker_count)]
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
                previous_velocity_residual_vector = (
                    velocity_residual_vector.copy()
                )
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
            fsi_coupling_relaxation_effective = relaxation
            fsi_coupling_aitken_update_count = aitken_update_count
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
        publish_solid_report_to_reduced_state(current_time_s, solid_mpm_report)
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
        row["outlet_flux_deficit_m3s"] = (
            expected_flux_m3s - outlet_negative_z_flux_m3s
        )
        row["downstream_flux_deficit_m3s"] = (
            expected_flux_m3s - downstream_negative_z_flux_m3s
        )
        row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
        row["fsi_coupling_aitken_update_count"] = (
            fsi_coupling_aitken_update_count
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
        row["fsi_coupling_physical_residual_jacobian_amplification_sample_count"] = 0
        row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
            fsi_coupling_physical_interface_map_amplification_sample_count
        )
        row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = 0
        row["interface_reaction_relaxation"] = interface_reaction_relaxation
        row["interface_reaction_aitken"] = interface_reaction_aitken
        row["interface_reaction_relaxation_effective"] = (
            fsi_coupling_relaxation_effective
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
            load_pressure_neumann_invalid_summary = (
                _write_hibm_pressure_neumann_invalid_row_dump(
                    output_dir=output_dir,
                    step=step,
                    rows=(
                        sharp_report.fluid_to_mpm_loads
                        .pressure_neumann_invalid_diagnostic_rows
                    ),
                    stage="load",
                )
            )
            next_pressure_neumann_invalid_summary = (
                _write_hibm_pressure_neumann_invalid_row_dump(
                    output_dir=output_dir,
                    step=step,
                    rows=(
                        sharp_report
                        .next_pressure_neumann_invalid_diagnostic_rows
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
                sharp_coupling_state=sharp_coupling_state,
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
                "interior_div_l2={interior_divergence_l2:.3e}".format(
                    **row
                ),
                flush=True,
            )
        if (
            max_wall_time_s > 0.0
            and step < step_count
            and time.perf_counter() - run_started_at_perf >= max_wall_time_s
        ):
            partial_run_stopped = True
            partial_run_reason = "max_wall_time_s"
            break
    return {
        "rows": rows,
        "partial_run_stopped": partial_run_stopped,
        "partial_run_reason": partial_run_reason,
    }
