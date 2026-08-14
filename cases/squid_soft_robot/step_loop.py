from __future__ import annotations

import time
from collections.abc import Mapping

from simulation_core import hibm_mpm_sharp_step_summary
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiCouplingReport,
    FsiSolverConfig,
    FsiStepContext,
    solve_fsi_runtime,
)

from .checkpointing import write_run_checkpoint
from .coupling_sharp import (
    SquidSharpFsiRuntime,
    squid_sharp_coupling_summary,
)
from .diagnostics import (
    _raise_for_closure_coverage_floor,
    _raise_for_step_numerical_guard,
    _raise_for_step_solid_out_of_bounds_guard,
    sharp_report_fluid_projection_failure_reason,
)
from .history import write_csv
from .rows import build_hibm_mpm_sharp_case_row, signed_positive_source_flux_ratio
from .schedules import pressure_schedule_step_end_pa
from .snapshots import (
    _write_fluid_snapshot_npz,
    _write_hibm_high_residual_cell_dump,
    _write_hibm_pressure_neumann_invalid_row_dump,
    _write_hibm_zero_correctable_cell_dump,
    _write_step_failure_artifacts,
)
from .step_context import StepLoopContext, StepLoopResult


def run_squid_step_loop(context: StepLoopContext) -> StepLoopResult:
    """Run Squid sharp FSI through the sole generic physical-step loop."""

    return _run_squid_sharp_runtime(context)


def _run_squid_sharp_runtime(context: StepLoopContext) -> StepLoopResult:
    """Prepare Squid callbacks and delegate every physical step to the core."""

    settings = context.settings
    resources = context.resources
    callbacks = context.callbacks
    state = context.state
    args = resources.args
    simulator = resources.simulator
    solid_mpm = resources.solid_mpm
    sharp_coupling_state = state.sharp_coupling_state

    if sharp_coupling_state is None:
        raise RuntimeError("sharp HIBM-MPM coupling state was not initialized")
    if solid_mpm is None:
        raise RuntimeError("sharp HIBM-MPM coupling requires an MPM solid")
    if float(settings.max_wall_time_s) > 0.0:
        raise ValueError(
            "Squid sharp max-wall-time stopping is unavailable because "
            "solve_fsi_runtime has no early-stop contract"
        )

    remaining_step_count = int(settings.step_count) - int(state.first_step) + 1
    if remaining_step_count <= 0:
        return StepLoopResult(
            rows=state.rows,
            sharp_coupling_state=sharp_coupling_state,
            partial_run_stopped=state.partial_run_stopped,
            partial_run_reason=state.partial_run_reason,
        )

    spec = resources.spec
    material = resources.material
    step_state: dict[str, object] = {}
    previous_step_cfl = state.previous_step_cfl
    previous_step_fluid_substeps = state.previous_step_fluid_substeps
    pending_sharp_report = None

    def prepare_step(runtime_context: FsiStepContext) -> None:
        logical_step = int(runtime_context.step)
        fluid_substeps = int(settings.effective_fluid_substeps)
        if resources.fluid_substep_controller is not None:
            fluid_substeps = int(
                resources.fluid_substep_controller.substeps_for_next_step(
                    previous_cfl=previous_step_cfl,
                    previous_substeps=previous_step_fluid_substeps,
                )
            )
        current_time_s = float(simulator.time_s[None])
        step_state.clear()
        step_state.update(
            {
                "step": logical_step,
                "step_wall_started_at": time.perf_counter(),
                "fluid_substeps": fluid_substeps,
                "fluid_substep_dt_s": float(spec.dt_s) / float(fluid_substeps),
                "current_time_s": current_time_s,
                "pressure_pa": pressure_schedule_step_end_pa(
                    current_time_s,
                    spec.dt_s,
                    spec,
                ),
                "trial_wall_started_at": time.perf_counter(),
                "solid_advance_wall_time_s": 0.0,
            }
        )

    def evaluate_trial_once(runtime_context: FsiStepContext):
        logical_step = int(step_state["step"])
        expected_step = int(runtime_context.step)
        if logical_step != expected_step:
            raise RuntimeError(
                "Squid sharp runtime context changed during a physical step"
            )

        def advance_sharp_solid_substeps():
            solid_wall_started_at = time.perf_counter()
            report = None
            for _ in range(int(settings.solid_mpm_substeps)):
                if args.solid_model == "tri_mooney_shell_mpm":
                    report = solid_mpm.advance_with_external_forces(
                        dt_s=settings.solid_sub_dt_s,
                        primary_region_id=settings.primary_shell_region_id,
                        secondary_region_id=settings.secondary_shell_region_id,
                        velocity_damping=settings.solid_substep_velocity_damping,
                        flip_blend=settings.solid_mpm_flip_blend,
                        read_report=False,
                    )
                elif args.solid_model == "neo_hookean_mpm":
                    report = solid_mpm.step(
                        dt_s=settings.solid_sub_dt_s,
                        mu_pa=material.shear_modulus_pa,
                        lambda_pa=material.lame_lambda_pa,
                        velocity_damping=settings.solid_substep_velocity_damping,
                        primary_region_id=settings.primary_shell_region_id,
                        secondary_region_id=settings.secondary_shell_region_id,
                        fixed_node_lock_policy=settings.neo_fixed_node_lock_policy,
                        read_report=False,
                    )
                else:
                    raise ValueError(f"Unsupported solid model: {args.solid_model}")
            step_state["solid_advance_wall_time_s"] = float(
                step_state["solid_advance_wall_time_s"]
            ) + (time.perf_counter() - solid_wall_started_at)
            return solid_mpm.report() if report is None else report

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
                2.0 * settings.fluid_probe_distance_m,
                settings.estimated_solid_particle_spacing_m,
            ),
            interior_probe_distance_m=settings.fluid_probe_distance_m,
            mpm_support_radius_m=max(
                2.0 * settings.estimated_solid_particle_spacing_m,
                settings.fluid_probe_distance_m,
            ),
            primary_region_id=settings.primary_shell_region_id,
            secondary_region_id=settings.secondary_shell_region_id,
            far_pressure_region_id=settings.pressure_load_region_id,
            far_pressure_barrier_region_id=settings.fixed_rim_region_id,
            far_pressure_pa=float(step_state["pressure_pa"]),
            far_pressure_side_normal_sign=settings.pressure_far_side_normal_sign,
            far_pressure_inside_probe_max_multiplier=(
                settings.far_pressure_inside_probe_max_multiplier
            ),
            two_sided_probe_max_multiplier=(
                settings.two_sided_probe_max_multiplier
            ),
            one_sided_pressure_region_id=settings.secondary_shell_region_id,
            one_sided_reference_pressure_pa=0.0,
            one_sided_probe_max_multiplier=(
                settings.one_sided_probe_max_multiplier
            ),
            far_pressure_air_backed=settings.far_pressure_air_backed,
            far_pressure_air_backed_probe_normal_sign=(
                settings.far_pressure_air_backed_probe_normal_sign
            ),
            fluid_dt_s=spec.dt_s,
            fluid_substeps=int(step_state["fluid_substeps"]),
            projection_iterations=int(args.projection_iterations),
            run_fluid_predictor=True,
            pressure_neumann_density_kgm3=spec.water_density_kgm3,
            pressure_neumann_dt_s=spec.dt_s,
            pressure_outlet_zmin=settings.pressure_outlet_zmin_enabled,
            pressure_solver=settings.pressure_solver_name,
            pressure_solve_failure_policy=str(args.pressure_solve_failure_policy),
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
        projection_failure_reason = sharp_report_fluid_projection_failure_reason(
            sharp_report
        )
        if projection_failure_reason:
            raise RuntimeError(
                "sharp marker-velocity trial fluid projection failed "
                f"(reason={projection_failure_reason})"
            )
        return sharp_report

    def commit_trial(
        runtime_context: FsiStepContext,
        sharp_report,
        coupling_report: FsiCouplingReport,
    ) -> Mapping[str, object]:
        nonlocal previous_step_cfl
        nonlocal previous_step_fluid_substeps
        nonlocal pending_sharp_report

        step = int(step_state["step"])
        expected_step = int(runtime_context.step)
        if step != expected_step:
            raise RuntimeError(
                "Squid sharp runtime context changed before step commit"
            )
        coupling_wall_time_s = (
            time.perf_counter() - float(step_state["trial_wall_started_at"])
        )
        solid_advance_wall_time_s = float(
            step_state["solid_advance_wall_time_s"]
        )
        fluid_advance_wall_time_s = max(
            0.0,
            coupling_wall_time_s - solid_advance_wall_time_s,
        )
        solid_mpm_report = sharp_report.mpm
        if solid_mpm_report is None:
            solid_mpm_report = solid_mpm.report()
        callbacks.publish_solid_report_to_reduced_state(
            float(step_state["current_time_s"]),
            solid_mpm_report,
        )

        sample_wall_started_at = time.perf_counter()
        latest_fluid_projection_report = (
            sharp_report.post_solid_fluid_projection
            if sharp_report.post_solid_fluid_projection is not None
            else sharp_report.fluid_to_mpm_loads.fluid_projection
        )
        sample_report = simulator.sample_after_projection(
            latest_fluid_projection_report,
            dt_s=float(step_state["fluid_substep_dt_s"]),
        )
        pressure_outlet_report = simulator.fluid.pressure_outlet_fv_flux_report(
            dt_s=float(step_state["fluid_substep_dt_s"]),
        )
        sample_wall_time_s = time.perf_counter() - sample_wall_started_at
        sharp_summary = hibm_mpm_sharp_step_summary(sharp_report)
        sharp_summary.update(squid_sharp_coupling_summary(coupling_report))
        row = build_hibm_mpm_sharp_case_row(
            step=step,
            sample_report=sample_report,
            sharp_summary=sharp_summary,
            fluid_projection_report=(
                sharp_report.fluid_to_mpm_loads.fluid_projection
            ),
            pressure_outlet_report=pressure_outlet_report,
            fluid_dt_s=spec.dt_s,
            solid_mpm_report=solid_mpm_report,
            solid_model=args.solid_model,
            fsi_coupling_iterations_requested=settings.fsi_coupling_iterations,
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
        row["main_volume_flux_to_outlet_ratio"] = (
            signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
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
        residual_history = coupling_report.absolute_residual_history_mps
        interface_map_amplification = 0.0
        interface_map_sample_count = 0
        if len(residual_history) >= 2 and residual_history[0] > 0.0:
            interface_map_amplification = (
                residual_history[-1] / residual_history[0]
            )
            interface_map_sample_count = len(residual_history) - 1
        row["fsi_coupling_wall_time_s"] = coupling_wall_time_s
        row["fsi_coupling_iqn_ils_least_squares_update_count"] = sum(
            mode == "iqn_ils" for mode in coupling_report.update_modes
        )
        row["fsi_coupling_interface_map_amplification"] = (
            interface_map_amplification
        )
        row["fsi_coupling_residual_jacobian_amplification"] = 0.0
        row["fsi_coupling_physical_interface_map_amplification"] = (
            interface_map_amplification
        )
        row["fsi_coupling_physical_residual_jacobian_amplification"] = 0.0
        row["fsi_coupling_raw_interface_map_amplification"] = (
            interface_map_amplification
        )
        row["fsi_coupling_raw_residual_jacobian_amplification"] = 0.0
        row["fsi_coupling_interface_map_amplification_sample_count"] = (
            interface_map_sample_count
        )
        row["fsi_coupling_residual_jacobian_amplification_sample_count"] = 0
        row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
            interface_map_sample_count
        )
        row[
            "fsi_coupling_physical_residual_jacobian_amplification_sample_count"
        ] = 0
        row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
            interface_map_sample_count
        )
        row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = 0
        row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
        row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
        row["sample_wall_time_s"] = sample_wall_time_s
        row["surface_diagnostics_wall_time_s"] = 0.0
        row["checkpoint_wall_time_s"] = 0.0
        row["step_wall_time_s"] = (
            time.perf_counter() - float(step_state["step_wall_started_at"])
        )
        row["fluid_substeps_base"] = settings.effective_fluid_substeps
        row["adaptive_fluid_substeps_enabled"] = (
            settings.adaptive_fluid_substeps_enabled
        )
        row["adaptive_fluid_substeps_target_cfl"] = float(
            args.adaptive_fluid_substeps_target_cfl
        )
        row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
        row["adaptive_fluid_substeps_previous_substeps"] = (
            previous_step_fluid_substeps
        )
        candidate_rows = [*state.rows, row]
        try:
            _raise_for_step_numerical_guard(
                row,
                cfl_limit=0.5,
                divergence_l2_limit=float(args.projection_divergence_tolerance),
            )
            _raise_for_step_solid_out_of_bounds_guard(row)
            _raise_for_closure_coverage_floor(
                candidate_rows,
                int(args.closure_coverage_floor),
                int(args.closure_coverage_floor_patience),
            )
        except Exception as exc:
            _write_step_failure_artifacts(
                process_path=resources.process_path,
                output_dir=resources.output_dir,
                rows=candidate_rows,
                step=step,
                exc=exc,
                fluid=simulator.fluid,
                markers=sharp_coupling_state.markers,
                pressure_outlet_zmin=settings.pressure_outlet_zmin_enabled,
            )
            raise

        state.rows.append(row)
        previous_step_cfl = float(row["cfl"])
        previous_step_fluid_substeps = int(
            float(row.get("fluid_substeps", step_state["fluid_substeps"]))
        )
        pending_sharp_report = sharp_report
        return row

    def publish_trial(
        runtime_context: FsiStepContext,
        committed_row: Mapping[str, object],
    ) -> None:
        nonlocal pending_sharp_report

        step = int(step_state["step"])
        expected_step = int(runtime_context.step)
        if step != expected_step or pending_sharp_report is None:
            raise RuntimeError("Squid committed step has no publication payload")
        state.rows[-1] = dict(committed_row)
        row = state.rows[-1]
        sharp_report = pending_sharp_report

        if args.diagnostic_dump_zero_correctable_cells:
            zero_correctable_summary = _write_hibm_zero_correctable_cell_dump(
                output_dir=resources.output_dir,
                step=step,
                fluid=simulator.fluid,
                markers=sharp_coupling_state.markers,
                pressure_outlet_zmin=settings.pressure_outlet_zmin_enabled,
            )
            row["diagnostic_zero_correctable_interior_cell_count"] = int(
                zero_correctable_summary["zero_correctable_interior_cell_count"]
            )
            row["diagnostic_zero_correctable_shell_band_candidate_count"] = int(
                zero_correctable_summary["shell_band_candidate_cell_count"]
            )
        if args.diagnostic_dump_high_residual_cells:
            high_residual_summary = _write_hibm_high_residual_cell_dump(
                output_dir=resources.output_dir,
                step=step,
                fluid=simulator.fluid,
                markers=sharp_coupling_state.markers,
                pressure_outlet_zmin=settings.pressure_outlet_zmin_enabled,
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
            load_invalid_summary = _write_hibm_pressure_neumann_invalid_row_dump(
                output_dir=resources.output_dir,
                step=step,
                rows=(
                    sharp_report.fluid_to_mpm_loads.pressure_neumann_invalid_diagnostic_rows
                ),
                stage="load",
            )
            next_invalid_summary = _write_hibm_pressure_neumann_invalid_row_dump(
                output_dir=resources.output_dir,
                step=step,
                rows=sharp_report.next_pressure_neumann_invalid_diagnostic_rows,
                stage="next",
            )
            row["diagnostic_pressure_neumann_invalid_load_dumped_row_count"] = int(
                load_invalid_summary["captured_invalid_row_count"]
            )
            row["diagnostic_pressure_neumann_invalid_load_total_row_count"] = int(
                load_invalid_summary["total_invalid_row_count"]
            )
            row["diagnostic_pressure_neumann_invalid_next_dumped_row_count"] = int(
                next_invalid_summary["captured_invalid_row_count"]
            )
            row["diagnostic_pressure_neumann_invalid_next_total_row_count"] = int(
                next_invalid_summary["total_invalid_row_count"]
            )
            row["diagnostic_pressure_neumann_invalid_dumped_row_count"] = int(
                next_invalid_summary["captured_invalid_row_count"]
            )
            row["diagnostic_pressure_neumann_invalid_total_row_count"] = int(
                next_invalid_summary["total_invalid_row_count"]
            )
        snapshot_interval = int(args.fluid_snapshot_interval)
        if snapshot_interval > 0 and (
            step % snapshot_interval == 0 or step == int(settings.step_count)
        ):
            _write_fluid_snapshot_npz(
                snapshot_dir=resources.output_dir / "snapshots",
                step=step,
                fluid=simulator.fluid,
                markers=sharp_coupling_state.markers,
                marker_count=int(sharp_coupling_state.markers.marker_count),
                time_s=float(row["time_s"]),
                pressure_pa=float(row["pressure_load_pa"]),
            )
        if args.checkpoint_every_step:
            write_csv(resources.history_path, state.rows)
            checkpoint_wall_started_at = time.perf_counter()
            write_run_checkpoint(
                resources.run_checkpoint_path,
                completed_step=step,
                step_count=settings.step_count,
                full_pressure_waveform_steps=settings.full_pressure_waveform_steps,
                args=args,
                simulator=simulator,
                solid_mpm=solid_mpm,
                sharp_coupling_state=sharp_coupling_state,
                frozen_run_fingerprint=resources.frozen_run_fingerprint,
            )
            row["checkpoint_wall_time_s"] = (
                time.perf_counter() - checkpoint_wall_started_at
            )
            row["step_wall_time_s"] = (
                time.perf_counter() - float(step_state["step_wall_started_at"])
            )
            write_csv(resources.history_path, state.rows)
        if args.progress and (
            step == 1
            or step == int(settings.step_count)
            or step % int(args.progress_interval) == 0
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
        pending_sharp_report = None

    def finalize_run() -> Mapping[str, object]:
        return {
            "report": {
                "first_step": int(state.first_step),
                "completed_step_count": remaining_step_count,
            }
        }

    runtime = SquidSharpFsiRuntime(
        simulator=simulator,
        solid_mpm=solid_mpm,
        sharp_coupling_state=sharp_coupling_state,
        prepare_step=prepare_step,
        evaluate_trial_once=evaluate_trial_once,
        commit_trial=commit_trial,
        publish_trial=publish_trial,
        finalize=finalize_run,
    )
    solver_config = FsiSolverConfig(
        step_count=remaining_step_count,
        time_step_s=float(spec.dt_s),
        completed_step_offset=int(state.first_step) - 1,
        coupling=FsiCouplingConfig(
            max_iterations=int(settings.fsi_coupling_iterations),
            absolute_tolerance_mps=float(
                settings.fsi_marker_coupling_tolerance_mps
            ),
        ),
    )
    try:
        solve_fsi_runtime(runtime, solver_config)
    except Exception as exc:
        if "step" in step_state:
            _write_step_failure_artifacts(
                process_path=resources.process_path,
                output_dir=resources.output_dir,
                rows=state.rows,
                step=int(step_state["step"]),
                exc=exc,
                fluid=simulator.fluid,
                markers=sharp_coupling_state.markers,
                pressure_outlet_zmin=settings.pressure_outlet_zmin_enabled,
            )
        raise
    return StepLoopResult(
        rows=state.rows,
        sharp_coupling_state=sharp_coupling_state,
        partial_run_stopped=state.partial_run_stopped,
        partial_run_reason=state.partial_run_reason,
    )
