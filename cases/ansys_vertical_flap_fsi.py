from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from benchmarks.official.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)
from benchmarks.official.solid_mpm_fsi_runner import (
    run_hibm_mpm_fsi,
)
from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
)
from simulation_core.drivers.case_spec import FsiCaseSpec


ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS: dict[str, dict[str, object]] = {
    "inlet": {"type": "velocity-inlet", "velocity_mps": 10.0},
    "outlet": {"type": "pressure-outlet", "gauge_pressure_pa": 0.0},
    "symmetry": {"type": "symmetry"},
    "stationary_walls": {"type": "wall", "motion": "stationary"},
    "flap_root": {
        "structure": "fixed-displacement",
        "x_displacement_m": 0.0,
        "y_displacement_m": 0.0,
    },
    "flap_wall": {"type": "fluid-solid-interface", "coupling": "intrinsic-two-way-fsi"},
}


ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS: dict[str, float | int | tuple[float, float]] = {
    # Official Fluent 2025 R1 intrinsic-FSI tutorial rerun, structure monitor
    # `official_fsi_50step_monitor_timeseries.csv` column
    # `solid_max_total_col0_col6_m` at step 50 (t = 0.025 s): 5.829606e-5 m.
    # This is a same-time snapshot on the downswing of a lightly damped
    # ringing response (period ~= 8.5 ms), NOT a steady value: the run peaks
    # at 4.316392e-4 m (step 9, t = 0.0045 s).
    "max_displacement_m": 5.8296e-5,
    # Peak over the 50-step run (same monitor column, step 9, t = 0.0045 s);
    # use this for amplitude comparisons that must not depend on ringing phase.
    "max_displacement_peak_over_run_m": 4.3164e-4,
    "max_displacement_peak_time_s": 4.5e-3,
    "displacement_ringing_period_s": 8.5e-3,
    "local_velocity_peak_mps": 28.1,
    "local_velocity_peak_range_mps": (20.0, 29.0),
    "time_step_s": 5.0e-4,
    "step_count": 50,
}


ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING: dict[str, float | str] = {
    "model": "two-sided-fluid-pressure",
    "probe_max_multiplier": 12.0,
}


ANSYS_VERTICAL_FLAP_DEFAULT_CLI_PRESET: dict[str, str] = {
    "name": "smoke",
    "flow_driver_mode": "projection_only",
    "flow_advection_scheme": "euler",
    "flow_turbulence_model": "laminar",
    "flow_sst_near_wall_treatment": "resolved",
}


ANSYS_VERTICAL_FLAP_CASE_METADATA: dict[str, Any] = {
    "source": {
        "name": "ANSYS Fluent v251 two-way intrinsic FSI vertical-flap tutorial",
        "url": "https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/flu_tg/flu_tg_fsi_2way.html",
    },
    "geometry": {
        "duct_length_m": 0.10,
        "duct_height_m": 0.04,
        "modeled_domain": "lower-symmetry-half",
        "modeled_height_m": 0.02,
        "flap_height_m": 0.01,
        "flap_thickness_m": 0.003,
        "flap_streamwise_min_m": 0.050,
        "flap_streamwise_max_m": 0.053,
    },
    "fluid": {
        "material": "air",
        "density_kgm3": 1.2,
        "viscosity_pa_s": 1.8e-5,
        "inlet_velocity_mps": 10.0,
        "outlet": "pressure-outlet",
        "symmetry_plane": "upper boundary of lower half-domain",
    },
    "solid": {
        "material": "silicone rubber",
        "density_kgm3": 1600.0,
        "young_modulus_pa": 1.0e6,
        "poisson_ratio": 0.47,
        "constitutive_model": "linear-elastic",
        "stress_state": "plane-stress",
    },
    "structure_damping": {
        "native_fluent_structure_damping_enabled": False,
        "solver_net_velocity_damping_per_physical_step": 0.995,
    },
    "solid_boundary": {
        "flap_attach": "fixed x/y displacement",
    },
    "fsi_interface": {
        "flap_wall": "two-way intrinsic FSI",
        "flap_wall_shadow": "two-way intrinsic FSI",
        "thin_wall_pressure_sampling": ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
    },
    "coupling_time_layer": {
        "scheme": "direct_explicit_partitioned",
        "physical_step_owner": (
            "benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi"
        ),
        "step_end_flow_stage": "pre_solid_projection",
        "step_end_structure_geometry_stage": "post_solid_observer",
        "transport_advanced_by_step_end_projection": False,
        "fail_closed_on_solver_health": True,
    },
    "time_integration": {
        "dt_s": 5.0e-4,
        "step_count": 50,
        "total_time_s": 0.025,
    },
    "default_cli_preset": ANSYS_VERTICAL_FLAP_DEFAULT_CLI_PRESET,
    "reference_results": ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS,
    "boundary_conditions": ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
}


CASE_SPEC = FsiCaseSpec(
    case_id="ansys-vertical-flap-fsi",
    source_url=ANSYS_VERTICAL_FLAP_CASE_METADATA["source"]["url"],
    coordinate_model="cartesian-2d",
    geometry=ANSYS_VERTICAL_FLAP_CASE_METADATA["geometry"],
    fluid=ANSYS_VERTICAL_FLAP_CASE_METADATA["fluid"],
    solid=ANSYS_VERTICAL_FLAP_CASE_METADATA["solid"],
    boundary_conditions=ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
    reference_results={
        "max_displacement_m": float(
            ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS["max_displacement_m"]
        ),
        "local_velocity_peak_mps": float(
            ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS["local_velocity_peak_mps"]
        ),
    },
    acceptance_tolerance=0.05,
)


@dataclass(frozen=True)
class VerticalFlapFsiConfig:
    duct_length_m: float = 0.10
    duct_height_m: float = 0.04
    span_m: float = 0.003
    flap_height_m: float = 0.01
    flap_thickness_m: float = 0.003
    flap_streamwise_min_m: float = 0.050
    flap_streamwise_max_m: float = 0.053
    inlet_velocity_mps: float = 10.0
    air_density_kgm3: float = 1.2
    air_viscosity_pa_s: float = 1.8e-5
    solid_density_kgm3: float = 1600.0
    young_modulus_pa: float = 1.0e6
    poisson_ratio: float = 0.47
    solid_constitutive_model: str = "plane_stress_linear_elastic"
    dt_s: float = 5.0e-4
    step_count: int = 50
    coupling_mode: str = "direct_explicit"
    fsi_coupling_max_iterations: int = 16
    fsi_coupling_absolute_tolerance_mps: float = 0.0
    fsi_coupling_relative_tolerance: float = 1.0e-3
    iqn_history_limit: int = 8
    iqn_initial_picard_relaxation: float = 0.5
    iqn_svd_relative_cutoff: float = 1.0e-10
    iqn_reuse_previous_step_history: bool = False
    # Initial-guess routing is independent of modified-physics writeback.
    initial_guess_mode: str = "carry_forward"
    initial_guess_kalman_config: InterfaceKalmanConfig | None = None
    initial_guess_oracle_path: str | None = None
    # Offline research probe: solve one target macro step repeatedly from the
    # same accepted in-memory base, without committing any probe trial.
    iqn_kalman_oracle_interpolation_target_step: int | None = None
    iqn_kalman_oracle_interpolation_oracle_path: str | None = None
    iqn_kalman_oracle_interpolation_alphas: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.975,
        0.99,
        1.0,
    )
    # Explicitly modified-physics experiment.  ``off`` preserves the legacy
    # solver path and constructs no predictor; active modes write posterior
    # values only to their uniquely owned feedback/state field.
    kalman_writeback_mode: str = "off"
    kalman_interface_config: InterfaceKalmanConfig | None = None
    kalman_fluid_config: InterfaceKalmanConfig | None = None
    kalman_solid_config: InterfaceKalmanConfig | None = None
    grid_nodes: tuple[int, int, int] = (4, 32, 64)
    solid_particle_counts: tuple[int, int, int] = (1, 12, 4)
    marker_count: int = 12
    flow_projection_iterations: int = 1080
    flow_pressure_solver: str = "fv_jacobi"
    flow_cg_tolerance: float = 1.0e-6
    flow_cg_preconditioner: str = "fv_multigrid_light"
    # Re-imposing reconstructed sharp-interface Dirichlet rows after a
    # pressure solve reintroduces a small divergence.  Reassemble the rows and
    # solve that residual as a pressure *increment* before sampling tractions.
    flow_post_dirichlet_consistency_projection_iterations: int = 1
    flow_reprojection_iterations: int | None = None
    flow_reprojection_cg_tolerance: float | None = None
    flow_pressure_solve_failure_policy: str = "raise"
    flow_divergence_cleanup_iterations: int = 0
    # Preserve the direct sharp pipeline that completed the validated 50 steps.
    velocity_damping: float = 0.995
    solid_velocity_transfer_flip_blend: float = 0.0
    # None is production auto mode; a positive count is an explicit A/B override.
    solid_substeps: int | None = None
    solid_max_substep_retries: int = 3
    solid_max_automatic_substeps: int = 65536
    solid_max_deformation_clamp_count_per_macro_step: int | None = 0
    # A/B-calibrated Courant target: 0.5 was stable but under-resolved the
    # vertical-flap tip trajectory; 0.14 keeps the selector state-adaptive.
    solid_cfl_target: float = 0.14
    preflow_steps: int = 0
    preflow_convergence_tolerance: float = 0.0
    preflow_convergence_mode: str = "single_step_legacy"
    detailed_preflow_stage_progress: bool = False
    preflow_stationary_min_steps: int = 20
    preflow_stationary_window_steps: int = 10
    preflow_stationary_consecutive_windows: int = 3
    preflow_stationary_tolerance: float = 0.05
    preflow_stationary_divergence_tolerance: float = 0.05
    preflow_stationary_no_slip_tolerance_fraction: float = 0.05
    preflow_traction_readiness_mode: str = "flow_only"
    preflow_snapshot_input_path: str | None = None
    preflow_snapshot_output_path: str | None = None
    fsi_checkpoint_input_path: str | None = None
    fsi_checkpoint_output_path: str | None = None
    fsi_checkpoint_expected_generation: str | None = None
    apply_marker_feedback_to_fluid: bool = True
    flow_reset_pressure_each_step: bool = False
    flow_reinitialize_inlet_each_step: bool = False
    flow_driver_mode: str = "projection_only"
    preflow_flow_driver_mode: str | None = None
    flow_inlet_source_strength: float = 1.0
    flow_inlet_source_ramp_steps: int = 0
    flow_inlet_source_profile: str = "constant"
    flow_inlet_source_schedule_scope: str = "global"
    flow_advection_scheme: str = "euler"
    flow_predictor_substeps: int = 8
    flow_predictor_kinematic_viscosity_multiplier: float = 1.0
    # The selected Fluent-comparison formulation enables SST explicitly.
    flow_turbulence_model: str = "laminar"
    flow_turbulence_intensity: float = 0.05
    flow_turbulent_viscosity_ratio: float = 10.0
    flow_backflow_turbulence_intensity: float = 0.05
    flow_backflow_turbulent_viscosity_ratio: float = 10.0
    flow_turbulence_inlet_face: str = "zmax"
    flow_turbulence_outlet_face: str = "zmin"
    flow_sst_near_wall_treatment: str = "resolved"
    flow_sst_max_automatic_substeps: int = 4096
    flow_report_include_percentiles: bool = False
    flow_predictor_no_slip_domain_walls: tuple[str, ...] = ("ymin",)
    flow_symmetry_domain_walls: tuple[str, ...] = ("ymax",)
    flow_ymin_no_slip_rows: int = 0
    flow_solid_boundary_mode: str = field(
        default="hibm_sharp_marker_rows",
        init=False,
    )
    flow_hibm_sharp_search_radius_m: float | None = None
    flow_hibm_sharp_search_radius_xyz_m: tuple[float, float, float] | None = None
    flow_hibm_sharp_interior_probe_distance_m: float | None = None
    flow_hibm_sharp_interior_probe_distance_xyz_m: (
        tuple[float, float, float] | None
    ) = None
    flow_hibm_sharp_interpolate_velocity_rows: bool = False
    flow_hibm_marker_mac_constraint_iterations: int = 64
    # Strict validation default.  Research campaigns may explicitly relax
    # this fail-closed closure gate without changing the MAC solve tolerance.
    flow_hibm_marker_compatibility_closure_tolerance_mps: float = 1.0e-6
    # Keep the moving physical flap volume independent of the narrow HIBM
    # interface-row search.  Validation launchers enable this together with
    # update_fluid_obstacle_from_solid after selecting a mesh-scaled envelope.
    flow_hibm_dynamic_solid_volume_enabled: bool = False
    flow_hibm_tiny_unreached_cleanup_component_cells: int = 0
    flow_pressure_outlet_enabled: bool = True
    flow_pressure_outlet_backflow_policy: str = "allow"
    flow_outlet_balance_policy: str = "report_only"
    update_fluid_obstacle_from_solid: bool = False
    # Root-clamp integrity (2026-07-03 static-cantilever audit): with
    # "pure_fixed_mass", mixed fixed/free grid nodes stay mobile and fixed
    # particles contribute mass but no stress, so the root clamp has no
    # elastic restoring path - the flap creeps monotonically past static
    # equilibrium (pure-solid probe: 8.57e-4 m and rising at t=0.025 s vs
    # Euler-Bernoulli 2.14e-4 m; on refined grids the clamp fails entirely
    # and the flap free-falls, ejecting particles). "any_fixed_particle"
    # locks every node touched by a fixed particle, reproducing the
    # tutorial's rigid flap_attach constraint (probe: 2.23e-4 m, settled).
    fixed_node_lock_policy: str = "any_fixed_particle"
    preserve_marker_velocity_constraints: bool = True
    marker_velocity_constraint_blend: float = 1.0
    marker_velocity_constraint_solid_mobility_ratio: float = 0.0
    traction_marker_layout: str = "dual_physical_faces"
    traction_pressure_sampling_mode: str = "two_sided_pressure_jump"
    traction_include_viscous: bool = False
    traction_tip_cap_pressure_enabled: bool = False
    # Material markers belong on the physical solid. Only pressure probes
    # retain the outward sampling offset; they are not material degrees of freedom.
    traction_marker_face_offset_cells: float = 0.0
    traction_pressure_probe_origin_mode: str = "physical_face_offset"
    traction_pressure_probe_origin_offset_cells: float | None = 0.51
    traction_pressure_probe_start_offset_cells: float | None = None
    traction_pressure_probe_ladder_spacing_cells: float = 0.5
    traction_pressure_probe_ladder_rung_count: int = 5
    traction_pressure_probe_ladder_mode: str = "current_normal_cell_ladder"
    traction_pressure_pair_policy: str = "independent_ladder"
    traction_pressure_pair_max_cell_delta: int = 1
    traction_pressure_pair_require_opposite_sides: bool = True
    traction_one_sided_pressure_policy: str = "disabled"
    traction_one_sided_primary_fluid_side_normal_sign: float | None = None
    traction_one_sided_secondary_fluid_side_normal_sign: float | None = None
    traction_one_sided_primary_reference_pressure_pa: float = 0.0
    traction_one_sided_secondary_reference_pressure_pa: float = 0.0
    traction_one_sided_pressure_pair_policy: str = "baseline_anchored_cell_pair"
    traction_pressure_pair_anchor_markers_json: str | None = None
    traction_pressure_pair_runtime_provider_mode: str = "disabled"
    traction_viscosity_pa_s: float = 0.0
    allow_selected_traction_formulation_coupled_smoke: bool = False
    allow_selected_traction_formulation_coupled_long_validation: bool = False
    allow_selected_traction_formulation_coupled_research_250: bool = False
    export_final_flow_snapshot: bool = False
    enforce_plane_strain_x: bool = False
    # Opt-in guard (2026-07-03 fine-flap ejection audit): refuse to run when
    # solid particle spacing exceeds solid_seeding_max_spacing_cells background
    # cells on the y/z axes. Under-seeded MPM solids numerically fracture at
    # the root clamp and free-fall (grid 4x256x320 with counts (1, 64, 12) put
    # ~2 cells between y layers and ejected particles by step 30).
    enforce_solid_seeding_limit: bool = False
    solid_seeding_max_spacing_cells: float = 1.5
    preserve_marker_area_during_surface_feedback: bool = True
    surface_transfer_method: str = "cartesian_reference_adjoint_v1"
    mpm_support_radius_m: float = 0.006
    displacement_tolerance: float = 0.05
    velocity_peak_tolerance: float = 0.05


def thin_wall_pressure_probe_max_multiplier(config: VerticalFlapFsiConfig) -> float:
    base_multiplier = float(
        ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING["probe_max_multiplier"]
    )
    streamwise_spacing_m = float(config.duct_length_m) / float(config.grid_nodes[2])
    max_spacing_m = max(
        float(config.span_m) / float(config.grid_nodes[0]),
        float(config.duct_height_m) / float(config.grid_nodes[1]),
        streamwise_spacing_m,
    )
    hibm_search_envelope_m = 3.0 * max_spacing_m
    # The probe must remain long enough to cross the physical wall thickness
    # and the classified HIBM row-cloud envelope after mesh refinement; a
    # fixed cell-count reach shrinks with dz and can stop inside dead row-cloud
    # pressure instead of reaching the opposite water side.
    thickness_multiplier = (
        float(config.flap_thickness_m) + hibm_search_envelope_m
    ) / streamwise_spacing_m + 2.5
    return max(base_multiplier, thickness_multiplier)


def surface_force_support_radius_m(config: VerticalFlapFsiConfig) -> float:
    solid_dy = float(config.flap_height_m) / float(config.solid_particle_counts[1])
    solid_dz = float(config.flap_thickness_m) / float(config.solid_particle_counts[2])
    grid_dy = float(config.duct_height_m) / float(config.grid_nodes[1])
    grid_dz = float(config.duct_length_m) / float(config.grid_nodes[2])
    local_radius = max(2.5 * solid_dy, 2.5 * solid_dz, 2.0 * grid_dy, 2.0 * grid_dz)
    thickness_limited_radius = min(local_radius, 0.5 * float(config.flap_thickness_m))
    return max(thickness_limited_radius, 1.25 * max(solid_dy, solid_dz))


def with_local_surface_force_support(
    config: VerticalFlapFsiConfig,
) -> VerticalFlapFsiConfig:
    return replace(
        config,
        mpm_support_radius_m=surface_force_support_radius_m(config),
    )


ANSYS_VERTICAL_FLAP_SELECTED_TRACTION_PRESET: dict[str, Any] = {
    "reference_formulation_candidate": (
        "anchored_dual_face_pressure_pair_with_per_face_one_sided"
    ),
    "pressure_pair_policy": "baseline_anchored_cell_pair",
    "generic_pressure_pair_mode": "runtime_anchored_cell_pair",
    "pressure_pair_source_status": "runtime_generated",
    "one_sided_pressure_policy": "per_face_mirrored",
    "primary_fluid_side_normal_sign": 1.0,
    "secondary_fluid_side_normal_sign": 1.0,
}


def selected_formulation_solver_config(
    *,
    step_count: int,
) -> VerticalFlapFsiConfig:
    config = VerticalFlapFsiConfig(
        step_count=step_count,
        flow_driver_mode="sustained_boundary_predictor",
        preflow_flow_driver_mode="sustained_boundary_predictor",
        flow_turbulence_model="sst_2003",
        flow_turbulence_intensity=0.05,
        flow_turbulent_viscosity_ratio=10.0,
        flow_backflow_turbulence_intensity=0.05,
        flow_backflow_turbulent_viscosity_ratio=10.0,
        # Production uses one physical outer step; the core conservative
        # transport owns its CFL-sized SSP-RK2 slices.  Repeating a fixed
        # semi-Lagrangian remap here was the dominant long-run diffusion source.
        flow_advection_scheme="muscl_tvd",
        flow_predictor_substeps=1,
        traction_pressure_sampling_mode="one_sided_surface_pressure",
        # Sharp HIBM velocity rows represent the physical no-slip surface.
        # Keep their markers synchronized with the dynamic MPM volume; only
        # the pressure sampling origin belongs outside the physical face.
        traction_marker_face_offset_cells=0.0,
        traction_pressure_probe_origin_mode="physical_face_offset",
        traction_pressure_probe_origin_offset_cells=0.51,
        traction_pressure_pair_policy=str(
            ANSYS_VERTICAL_FLAP_SELECTED_TRACTION_PRESET["pressure_pair_policy"]
        ),
        traction_one_sided_pressure_policy=str(
            ANSYS_VERTICAL_FLAP_SELECTED_TRACTION_PRESET[
                "one_sided_pressure_policy"
            ]
        ),
        traction_one_sided_primary_fluid_side_normal_sign=float(
            ANSYS_VERTICAL_FLAP_SELECTED_TRACTION_PRESET[
                "primary_fluid_side_normal_sign"
            ]
        ),
        traction_one_sided_secondary_fluid_side_normal_sign=float(
            ANSYS_VERTICAL_FLAP_SELECTED_TRACTION_PRESET[
                "secondary_fluid_side_normal_sign"
            ]
        ),
        traction_pressure_pair_anchor_markers_json=None,
        traction_pressure_pair_runtime_provider_mode=(
            "runtime_anchored_cell_pair"
        ),
        allow_selected_traction_formulation_coupled_smoke=True,
        allow_selected_traction_formulation_coupled_long_validation=True,
        allow_selected_traction_formulation_coupled_research_250=True,
    )
    return with_local_surface_force_support(config)


def run_ansys_vertical_flap_benchmark(
    config: VerticalFlapFsiConfig | None = None,
    *,
    step_observer: Callable[..., None] | None = None,
    progress_observer: Callable[[dict[str, object]], None] | None = None,
    profile_wall_time: bool = False,
) -> dict[str, object]:
    cfg = with_local_surface_force_support(config or VerticalFlapFsiConfig())

    def runner(active_config: VerticalFlapFsiConfig) -> dict[str, object]:
        return run_hibm_mpm_fsi(
            case_id=CASE_SPEC.case_id,
            case_metadata=ANSYS_VERTICAL_FLAP_CASE_METADATA,
            boundary_conditions=ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
            reference_results=CASE_SPEC.reference_results,
            config=active_config,
            step_observer=step_observer,
            progress_observer=progress_observer,
            profile_wall_time=profile_wall_time,
        )

    return run_official_fsi_benchmark(
        OfficialBenchmarkRunSpec(
            case_spec=CASE_SPEC,
            solver_family="rectangular-solid-marker-mpm",
            case_metadata=ANSYS_VERTICAL_FLAP_CASE_METADATA,
            boundary_conditions=ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
            config=cfg,
            runner=runner,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ANSYS vertical-flap two-way FSI smoke benchmark."
    )
    parser.add_argument("--steps", type=int, default=VerticalFlapFsiConfig.step_count)
    parser.add_argument(
        "--fsi-checkpoint-in",
        help="Resume from a complete accepted FSI checkpoint prefix.",
    )
    parser.add_argument(
        "--fsi-checkpoint-out",
        help="Save every accepted complete FSI state under this prefix.",
    )
    parser.add_argument(
        "--preflow-steps",
        type=int,
        default=VerticalFlapFsiConfig.preflow_steps,
        help="Project flow around a fixed flap before FSI steps.",
    )
    parser.add_argument(
        "--preflow-convergence-tolerance",
        type=float,
        default=VerticalFlapFsiConfig.preflow_convergence_tolerance,
        help="Relative p/velocity tolerance for early preflow stop; 0 disables.",
    )
    parser.add_argument(
        "--disable-marker-feedback",
        action="store_true",
        help="Diagnostic mode: do not impose marker velocity feedback on fluid.",
    )
    parser.add_argument(
        "--flow-reset-pressure-each-step",
        action="store_true",
        help="Diagnostic mode: reset pressure before every flow projection.",
    )
    parser.add_argument(
        "--flow-reinitialize-inlet-each-step",
        action="store_true",
        help="Diagnostic mode: reinitialize inlet flow before every projection.",
    )
    parser.add_argument(
        "--flow-driver-mode",
        default=VerticalFlapFsiConfig.flow_driver_mode,
        choices=(
            "projection_only",
            "reinitialize_inlet_each_step_diagnostic",
            "sustained_boundary_inlet",
            "sustained_boundary_predictor",
            "sustained_volume_source_inlet",
            "sustained_inlet_predictor",
            "sharp_hibm_mpm_reference",
        ),
        help="Explicit flow driver path for ANSYS vertical-flap diagnostics.",
    )
    parser.add_argument(
        "--preflow-flow-driver-mode",
        default=VerticalFlapFsiConfig.preflow_flow_driver_mode,
        choices=(
            "projection_only",
            "reinitialize_inlet_each_step_diagnostic",
            "sustained_boundary_inlet",
            "sustained_boundary_predictor",
            "sustained_volume_source_inlet",
            "sustained_inlet_predictor",
            "sharp_hibm_mpm_reference",
        ),
        help="Optional fixed-solid preflow driver override.",
    )
    parser.add_argument(
        "--flow-inlet-source-strength",
        type=float,
        default=VerticalFlapFsiConfig.flow_inlet_source_strength,
        help="Sustained inlet source strength multiplier.",
    )
    parser.add_argument(
        "--flow-inlet-source-ramp-steps",
        type=int,
        default=VerticalFlapFsiConfig.flow_inlet_source_ramp_steps,
        help="Ramp sustained inlet source over this many steps; 0 disables ramp.",
    )
    parser.add_argument(
        "--flow-inlet-source-profile",
        default=VerticalFlapFsiConfig.flow_inlet_source_profile,
        choices=("constant", "linear_ramp"),
        help="Sustained inlet source temporal profile.",
    )
    parser.add_argument(
        "--flow-inlet-source-schedule-scope",
        default=VerticalFlapFsiConfig.flow_inlet_source_schedule_scope,
        choices=("global", "phase_local"),
        help="Whether source ramps continue across preflow/FSI phases.",
    )
    parser.add_argument(
        "--flow-advection-scheme",
        default=VerticalFlapFsiConfig.flow_advection_scheme,
        choices=("euler", "rk2", "muscl_tvd"),
        help="Advection scheme used by predictor-style flow drivers.",
    )
    parser.add_argument(
        "--flow-predictor-substeps",
        type=int,
        default=VerticalFlapFsiConfig.flow_predictor_substeps,
        help="Number of core predictor substeps inside each FSI flow step.",
    )
    parser.add_argument(
        "--flow-predictor-kinematic-viscosity-multiplier",
        type=float,
        default=VerticalFlapFsiConfig.flow_predictor_kinematic_viscosity_multiplier,
        help="Multiplier on material nu used by predictor-style flow drivers.",
    )
    parser.add_argument(
        "--flow-turbulence-model",
        default=VerticalFlapFsiConfig.flow_turbulence_model,
        choices=("laminar", "sst_2003"),
        help="Core momentum/turbulence closure used by predictor-style flow drivers.",
    )
    parser.add_argument(
        "--flow-turbulence-intensity",
        type=float,
        default=VerticalFlapFsiConfig.flow_turbulence_intensity,
        help="Velocity-inlet turbulence intensity fraction for SST-2003.",
    )
    parser.add_argument(
        "--flow-turbulent-viscosity-ratio",
        type=float,
        default=VerticalFlapFsiConfig.flow_turbulent_viscosity_ratio,
        help="Velocity-inlet turbulent-to-molecular viscosity ratio for SST-2003.",
    )
    parser.add_argument(
        "--flow-backflow-turbulence-intensity",
        type=float,
        default=VerticalFlapFsiConfig.flow_backflow_turbulence_intensity,
        help="Pressure-outlet reverse-flow turbulence intensity fraction.",
    )
    parser.add_argument(
        "--flow-backflow-turbulent-viscosity-ratio",
        type=float,
        default=VerticalFlapFsiConfig.flow_backflow_turbulent_viscosity_ratio,
        help="Pressure-outlet reverse-flow turbulent viscosity ratio.",
    )
    parser.add_argument(
        "--flow-turbulence-inlet-face",
        default=VerticalFlapFsiConfig.flow_turbulence_inlet_face,
        choices=("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
        help="Physical velocity-inlet face supplying SST k/omega.",
    )
    parser.add_argument(
        "--flow-turbulence-outlet-face",
        default=VerticalFlapFsiConfig.flow_turbulence_outlet_face,
        choices=("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
        help="Physical pressure-outlet face supplying SST reverse-flow state.",
    )
    parser.add_argument(
        "--flow-sst-max-automatic-substeps",
        type=int,
        default=VerticalFlapFsiConfig.flow_sst_max_automatic_substeps,
        help="Fail-closed ceiling for automatically required SST stability substeps.",
    )
    parser.add_argument(
        "--flow-predictor-no-slip-domain-walls",
        default=",".join(VerticalFlapFsiConfig.flow_predictor_no_slip_domain_walls),
        help="Comma-separated domain faces with no-slip predictor wall diffusion.",
    )
    parser.add_argument(
        "--flow-symmetry-domain-walls",
        default=",".join(VerticalFlapFsiConfig.flow_symmetry_domain_walls),
        help="Comma-separated domain faces using symmetry wall velocity constraints.",
    )
    parser.add_argument(
        "--flow-divergence-cleanup-iterations",
        type=int,
        default=VerticalFlapFsiConfig.flow_divergence_cleanup_iterations,
        help="Post-projection local divergence cleanup iterations.",
    )
    parser.add_argument(
        "--flow-ymin-no-slip-rows",
        type=int,
        default=VerticalFlapFsiConfig.flow_ymin_no_slip_rows,
        help="Number of near-ymin fluid rows constrained to zero velocity.",
    )
    parser.add_argument(
        "--disable-pressure-outlet",
        action="store_true",
        help="Diagnostic mode: disable zmin pressure outlet during projection.",
    )
    parser.add_argument(
        "--flow-pressure-outlet-backflow-policy",
        default=VerticalFlapFsiConfig.flow_pressure_outlet_backflow_policy,
        choices=("clamp", "allow"),
        help="Generic pressure-outlet backflow handling policy.",
    )
    parser.add_argument(
        "--flow-outlet-balance-policy",
        default=VerticalFlapFsiConfig.flow_outlet_balance_policy,
        choices=("report_only",),
        help="Outlet balance policy; this diagnostic step is report-only.",
    )
    parser.add_argument(
        "--disable-marker-velocity-constraints",
        action="store_true",
        help="Diagnostic mode: do not preserve marker velocity constraints during projection.",
    )
    parser.add_argument(
        "--marker-velocity-constraint-blend",
        type=float,
        default=VerticalFlapFsiConfig.marker_velocity_constraint_blend,
        help="Blend factor used when marker velocity constraints are preserved.",
    )
    parser.add_argument(
        "--marker-velocity-constraint-solid-mobility-ratio",
        type=float,
        default=VerticalFlapFsiConfig.marker_velocity_constraint_solid_mobility_ratio,
        help="Solid mobility ratio used when marker velocity constraints are preserved.",
    )
    parser.add_argument(
        "--coupling-mode",
        default=VerticalFlapFsiConfig.coupling_mode,
        choices=("direct_explicit", "iqn_ils"),
        help="Direct one-pass reference or generic IQN-ILS coupling.",
    )
    parser.add_argument(
        "--fsi-max-iterations",
        type=int,
        default=VerticalFlapFsiConfig.fsi_coupling_max_iterations,
        help="Maximum same-time FSI coupling trials per physical macro step.",
    )
    parser.add_argument(
        "--fsi-absolute-tolerance-mps",
        type=float,
        default=VerticalFlapFsiConfig.fsi_coupling_absolute_tolerance_mps,
        help="Absolute marker-velocity coupling tolerance in m/s.",
    )
    parser.add_argument(
        "--fsi-relative-tolerance",
        type=float,
        default=VerticalFlapFsiConfig.fsi_coupling_relative_tolerance,
        help="Relative marker-velocity coupling tolerance.",
    )
    parser.add_argument(
        "--iqn-history-limit",
        type=int,
        default=VerticalFlapFsiConfig.iqn_history_limit,
        help="Maximum same-step IQN-ILS history pairs.",
    )
    parser.add_argument(
        "--iqn-initial-picard-relaxation",
        type=float,
        default=VerticalFlapFsiConfig.iqn_initial_picard_relaxation,
        help="Fixed Picard startup and IQN fallback relaxation.",
    )
    parser.add_argument(
        "--iqn-svd-relative-cutoff",
        type=float,
        default=VerticalFlapFsiConfig.iqn_svd_relative_cutoff,
        help="Relative singular-value cutoff for IQN-ILS history.",
    )
    parser.add_argument(
        "--initial-guess-mode",
        default=VerticalFlapFsiConfig.initial_guess_mode,
        choices=("carry_forward", "linear_extrapolation"),
        help=(
            "First marker-velocity guess used by generic IQN-ILS. Kalman and "
            "oracle replay experiments use the formal validation runner."
        ),
    )
    parser.add_argument(
        "--experimental-modified-physics-kalman-writeback",
        default=VerticalFlapFsiConfig.kalman_writeback_mode,
        choices=("off",),
        help=(
            "The public smoke runner keeps posterior-state writeback off. Use "
            "the formal validation runner for modified-physics experiments."
        ),
    )
    parser.add_argument(
        "--export-final-flow-snapshot",
        action="store_true",
        help="Include the final structured pressure/velocity field in the report.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _parse_wall_names(value: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in str(value).split(",") if part.strip())


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _build_parser().parse_args(argv)
    report = run_ansys_vertical_flap_benchmark(
        VerticalFlapFsiConfig(
            step_count=args.steps,
            fsi_checkpoint_input_path=args.fsi_checkpoint_in,
            fsi_checkpoint_output_path=args.fsi_checkpoint_out,
            coupling_mode=args.coupling_mode,
            fsi_coupling_max_iterations=args.fsi_max_iterations,
            fsi_coupling_absolute_tolerance_mps=(
                args.fsi_absolute_tolerance_mps
            ),
            fsi_coupling_relative_tolerance=args.fsi_relative_tolerance,
            iqn_history_limit=args.iqn_history_limit,
            iqn_initial_picard_relaxation=(
                args.iqn_initial_picard_relaxation
            ),
            iqn_svd_relative_cutoff=args.iqn_svd_relative_cutoff,
            initial_guess_mode=args.initial_guess_mode,
            preflow_steps=args.preflow_steps,
            preflow_convergence_tolerance=args.preflow_convergence_tolerance,
            apply_marker_feedback_to_fluid=not args.disable_marker_feedback,
            flow_reset_pressure_each_step=args.flow_reset_pressure_each_step,
            flow_reinitialize_inlet_each_step=args.flow_reinitialize_inlet_each_step,
            flow_driver_mode=args.flow_driver_mode,
            preflow_flow_driver_mode=args.preflow_flow_driver_mode,
            flow_inlet_source_strength=args.flow_inlet_source_strength,
            flow_inlet_source_ramp_steps=args.flow_inlet_source_ramp_steps,
            flow_inlet_source_profile=args.flow_inlet_source_profile,
            flow_inlet_source_schedule_scope=args.flow_inlet_source_schedule_scope,
            flow_advection_scheme=args.flow_advection_scheme,
            flow_predictor_substeps=args.flow_predictor_substeps,
            flow_predictor_kinematic_viscosity_multiplier=(
                args.flow_predictor_kinematic_viscosity_multiplier
            ),
            flow_turbulence_model=args.flow_turbulence_model,
            flow_turbulence_intensity=args.flow_turbulence_intensity,
            flow_turbulent_viscosity_ratio=(
                args.flow_turbulent_viscosity_ratio
            ),
            flow_backflow_turbulence_intensity=(
                args.flow_backflow_turbulence_intensity
            ),
            flow_backflow_turbulent_viscosity_ratio=(
                args.flow_backflow_turbulent_viscosity_ratio
            ),
            flow_turbulence_inlet_face=args.flow_turbulence_inlet_face,
            flow_turbulence_outlet_face=args.flow_turbulence_outlet_face,
            flow_sst_max_automatic_substeps=(
                args.flow_sst_max_automatic_substeps
            ),
            flow_predictor_no_slip_domain_walls=_parse_wall_names(
                args.flow_predictor_no_slip_domain_walls
            ),
            flow_symmetry_domain_walls=_parse_wall_names(
                args.flow_symmetry_domain_walls
            ),
            flow_divergence_cleanup_iterations=(
                args.flow_divergence_cleanup_iterations
            ),
            flow_ymin_no_slip_rows=args.flow_ymin_no_slip_rows,
            flow_pressure_outlet_enabled=not args.disable_pressure_outlet,
            flow_pressure_outlet_backflow_policy=(
                args.flow_pressure_outlet_backflow_policy
            ),
            flow_outlet_balance_policy=args.flow_outlet_balance_policy,
            preserve_marker_velocity_constraints=(
                not args.disable_marker_velocity_constraints
            ),
            marker_velocity_constraint_blend=args.marker_velocity_constraint_blend,
            marker_velocity_constraint_solid_mobility_ratio=(
                args.marker_velocity_constraint_solid_mobility_ratio
            ),
            kalman_writeback_mode=(
                args.experimental_modified_physics_kalman_writeback
            ),
            export_final_flow_snapshot=args.export_final_flow_snapshot,
        )
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "ANSYS vertical flap smoke: "
            f"max displacement={report['max_displacement_m']:.6e} m, "
            f"relative error={report['max_displacement_relative_error']:.3f}"
        )
    return report


if __name__ == "__main__":
    main()
