from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from simulation_core.fsi_driver import FsiCaseSpec
from simulation_core.benchmarking.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)
from simulation_core.benchmarking.solid_mpm_fsi_runner import (
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


ANSYS_VERTICAL_FLAP_CASE_METADATA: dict[str, Any] = {
    "source": {
        "name": "ANSYS Fluent v242 two-way intrinsic FSI vertical-flap tutorial",
        "url": "https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_tg/flu_tg_fsi_2way.html",
    },
    "geometry": {
        "duct_length_m": 0.10,
        "duct_height_m": 0.04,
        "modeled_domain": "lower-symmetry-half",
        "modeled_height_m": 0.02,
        "flap_height_m": 0.01,
        "flap_thickness_m": 0.003,
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
    enforce_plane_strain_x: bool = True
    mpm_support_radius_m: float = 0.006
    displacement_tolerance: float = 0.05
    velocity_peak_tolerance: float = 0.05


def run_vertical_flap_fsi_smoke(config: VerticalFlapFsiConfig | None = None) -> dict[str, object]:
    cfg = config or VerticalFlapFsiConfig()
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
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _build_parser().parse_args(argv)
    report = run_vertical_flap_fsi_smoke(
        VerticalFlapFsiConfig(
            step_count=args.steps,
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
