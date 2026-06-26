from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from typing import Any

from simulation_core.fsi_driver import FsiCaseSpec
from benchmarks.official.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)
from benchmarks.official.solid_mpm_fsi_runner import (
    run_rectangular_solid_marker_mpm_fsi_smoke,
)


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
    "max_displacement_m": 5.1e-5,
    "local_velocity_peak_mps": 28.1,
    "local_velocity_peak_range_mps": (20.0, 29.0),
    "time_step_s": 5.0e-4,
    "step_count": 50,
}


ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING: dict[str, float | str] = {
    "model": "two-sided-fluid-pressure",
    "probe_max_multiplier": 12.0,
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
        "inlet_velocity_mps": 10.0,
        "outlet": "pressure-outlet",
        "symmetry_plane": "upper boundary of lower half-domain",
    },
    "solid": {
        "material": "silicone rubber",
        "density_kgm3": 1600.0,
        "young_modulus_pa": 1.0e6,
        "poisson_ratio": 0.47,
    },
    "solid_boundary": {
        "flap_attach": "fixed x/y displacement",
    },
    "fsi_interface": {
        "flap_wall": "two-way intrinsic FSI",
        "flap_wall_shadow": "two-way intrinsic FSI",
        "thin_wall_pressure_sampling": ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
    },
    "time_integration": {
        "dt_s": 5.0e-4,
        "step_count": 50,
        "total_time_s": 0.025,
    },
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
    air_density_kgm3: float = 1.225
    air_viscosity_pa_s: float = 1.8e-5
    solid_density_kgm3: float = 1600.0
    young_modulus_pa: float = 1.0e6
    poisson_ratio: float = 0.47
    dt_s: float = 5.0e-4
    step_count: int = 50
    grid_nodes: tuple[int, int, int] = (4, 32, 64)
    solid_particle_counts: tuple[int, int, int] = (1, 12, 4)
    marker_count: int = 12
    flow_projection_iterations: int = 1080
    flow_pressure_solver: str = "fv_jacobi"
    flow_cg_tolerance: float = 1.0e-6
    flow_divergence_cleanup_iterations: int = 0
    velocity_damping: float = 0.995
    solid_substeps: int = 1600
    solid_cfl_target: float = 0.5
    preflow_steps: int = 0
    preflow_convergence_tolerance: float = 0.0
    apply_marker_feedback_to_fluid: bool = True
    flow_reset_pressure_each_step: bool = False
    flow_reinitialize_inlet_each_step: bool = False
    flow_driver_mode: str = "projection_only"
    flow_inlet_source_strength: float = 1.0
    flow_inlet_source_ramp_steps: int = 0
    flow_inlet_source_profile: str = "constant"
    flow_inlet_source_schedule_scope: str = "global"
    flow_pressure_outlet_enabled: bool = True
    flow_outlet_balance_policy: str = "report_only"
    traction_marker_layout: str = "dual_physical_faces"
    traction_pressure_sampling_mode: str = "two_sided_pressure_jump"
    traction_include_viscous: bool = False
    traction_marker_face_offset_cells: float = 0.51
    traction_viscosity_pa_s: float = 0.0
    enforce_plane_strain_x: bool = True
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


def run_vertical_flap_fsi_smoke(config: VerticalFlapFsiConfig | None = None) -> dict[str, object]:
    cfg = with_local_surface_force_support(config or VerticalFlapFsiConfig())
    return run_official_fsi_benchmark(
        OfficialBenchmarkRunSpec(
            case_spec=CASE_SPEC,
            solver_family="rectangular-solid-marker-mpm",
            case_metadata=ANSYS_VERTICAL_FLAP_CASE_METADATA,
            boundary_conditions=ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
            config=cfg,
            runner=_run_vertical_flap_fsi_core,
        )
    )


def _run_vertical_flap_fsi_core(config: VerticalFlapFsiConfig) -> dict[str, object]:
    return run_rectangular_solid_marker_mpm_fsi_smoke(
        case_id=CASE_SPEC.case_id,
        case_metadata=ANSYS_VERTICAL_FLAP_CASE_METADATA,
        boundary_conditions=ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
        reference_results=ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS,
        config=config,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ANSYS vertical-flap two-way FSI smoke benchmark."
    )
    parser.add_argument("--steps", type=int, default=VerticalFlapFsiConfig.step_count)
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
            "sustained_volume_source_inlet",
            "sustained_inlet_predictor",
            "sharp_hibm_mpm_reference",
        ),
        help="Explicit flow driver path for ANSYS vertical-flap diagnostics.",
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
        "--disable-pressure-outlet",
        action="store_true",
        help="Diagnostic mode: disable zmin pressure outlet during projection.",
    )
    parser.add_argument(
        "--flow-outlet-balance-policy",
        default=VerticalFlapFsiConfig.flow_outlet_balance_policy,
        choices=("report_only",),
        help="Outlet balance policy; this diagnostic step is report-only.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _build_parser().parse_args(argv)
    report = run_vertical_flap_fsi_smoke(
        VerticalFlapFsiConfig(
            step_count=args.steps,
            preflow_steps=args.preflow_steps,
            preflow_convergence_tolerance=args.preflow_convergence_tolerance,
            apply_marker_feedback_to_fluid=not args.disable_marker_feedback,
            flow_reset_pressure_each_step=args.flow_reset_pressure_each_step,
            flow_reinitialize_inlet_each_step=args.flow_reinitialize_inlet_each_step,
            flow_driver_mode=args.flow_driver_mode,
            flow_inlet_source_strength=args.flow_inlet_source_strength,
            flow_inlet_source_ramp_steps=args.flow_inlet_source_ramp_steps,
            flow_inlet_source_profile=args.flow_inlet_source_profile,
            flow_inlet_source_schedule_scope=args.flow_inlet_source_schedule_scope,
            flow_pressure_outlet_enabled=not args.disable_pressure_outlet,
            flow_outlet_balance_policy=args.flow_outlet_balance_policy,
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
