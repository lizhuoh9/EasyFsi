from __future__ import annotations

import math
from dataclasses import dataclass

from simulation_core.benchmarking.axisymmetric_geometry import (
    AxisymmetricNeckedEllipseProfile,
)
from simulation_core.benchmarking.axisymmetric_membrane import (
    SmoothAxisymmetricMembraneStressConfig,
    smooth_axisymmetric_membrane_stress_report,
)
from simulation_core.benchmarking.inlet_flow import TimeWindowedInletFlow
from simulation_core.benchmarking.membrane_inflation_fsi import (
    MembraneInflationConfig,
    run_uv_membrane_inflation_smoke,
)
from simulation_core.benchmarking.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)
from simulation_core.benchmarking.ogden_membrane import (
    OgdenMembraneMaterial,
    stretch_from_volume_ratio,
)
from simulation_core.fsi_driver import FsiCaseSpec
from simulation_core.validation import ReferenceCurve


COMSOL_WATER_BALLOON_OGDEN_ALPHA = (1.3, 5.0, -2.0)
COMSOL_WATER_BALLOON_OGDEN_SHEAR_MODULUS_PA = (6.3e5, 0.012e5, -0.1e5)
COMSOL_WATER_BALLOON_MAX_MISES_STRESS_PA = 4_675_129.483782177


COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML = ReferenceCurve(
    name="dom1 water content",
    units="ml",
    source="COMSOL water_balloon_solved.mph Probe Table 1, dom1 water content",
    points=(
        (0.0, 47.115599468997),
        (0.2, 47.299419059471),
        (1.0, 56.556955658354),
        (3.0, 80.118900464556),
        (6.0, 115.473653591449),
        (9.2, 153.061380044839),
        (15.0, 153.260483578228),
    ),
)


COMSOL_WATER_BALLOON_BOUNDARY_CONDITIONS: dict[str, dict[str, object]] = {
    "inlet": {
        "type": "time-windowed-velocity-inlet",
        "peak_velocity_mps": 0.15,
        "nominal_flow_lpm": 1.4,
        "control": "rectangle-function",
        "window_start_s": 0.2,
        "window_end_s": 9.2,
        "smooth_s": 0.2,
    },
    "axis": {
        "type": "axisymmetry",
        "mesh_constraint": "prescribed-normal-mesh-displacement",
    },
    "fluid_structure_interface": {
        "type": "two-way-fsi",
        "mesh": "deforming-domain",
    },
    "gravity": {"enabled": True},
}


COMSOL_WATER_BALLOON_CASE_METADATA = {
    "source": {
        "name": "COMSOL water balloon FSI blog tutorial",
        "url": "https://www.comsol.com/blogs/how-to-model-fluid-structure-interaction-in-a-water-balloon",
    },
    "geometry": {
        "coordinate_model": "axisymmetric-2d",
        "initial_shapes": "parameterized rectangle and ellipse with rubber thickness",
        "sweep_parameter": "initial balloon size factor",
        "validated_size_factor": 2.0,
        "neck_radius_m": 0.005,
        "body_radius_m": 0.02,
        "height_m": 0.08,
        "fillet_radius_m": 0.013333333333333334,
    },
    "fluid": {
        "material": "water",
        "flow": "laminar",
        "inlet_velocity_mps": 0.15,
    },
    "solid": {
        "material": "rubber",
        "model": "Ogden hyperelastic membrane",
        "ogden_alpha": COMSOL_WATER_BALLOON_OGDEN_ALPHA,
        "ogden_shear_modulus_pa": COMSOL_WATER_BALLOON_OGDEN_SHEAR_MODULUS_PA,
        "thickness_m": 0.0002,
    },
    "study": {
        "parametric_sweep": "initial-size-factor",
        "case_count": 3,
        "same_water_volume_for_all_sizes": True,
    },
    "reference_results": {
        "source": "official water_balloon_solved.mph Probe Table 1",
        "extraction_required": False,
        "probe": "dom1 water content",
        "validated_size_factor": 2.0,
        "max_mises_stress_pa": COMSOL_WATER_BALLOON_MAX_MISES_STRESS_PA,
        "stress_source": (
            "COMSOL water_balloon_solved.mph Stress plot group, "
            "mbrn.mises rangecolormax at t=15 s, fact=2"
        ),
    },
}


CASE_SPEC = FsiCaseSpec(
    case_id="comsol-water-balloon-fsi",
    source_url=COMSOL_WATER_BALLOON_CASE_METADATA["source"]["url"],
    coordinate_model="axisymmetric-2d",
    geometry=COMSOL_WATER_BALLOON_CASE_METADATA["geometry"],
    fluid=COMSOL_WATER_BALLOON_CASE_METADATA["fluid"],
    solid=COMSOL_WATER_BALLOON_CASE_METADATA["solid"],
    boundary_conditions=COMSOL_WATER_BALLOON_BOUNDARY_CONDITIONS,
    reference_results={
        "initial_water_content_ml": (
            COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.value_at(0.0)
        ),
        "final_water_content_ml": (
            COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.value_at(15.0)
        ),
        "max_mises_stress_pa": COMSOL_WATER_BALLOON_MAX_MISES_STRESS_PA,
    },
    acceptance_tolerance=0.05,
)


@dataclass(frozen=True)
class WaterBalloonFsiConfig:
    initial_size_factor: float = 2.0
    neck_radius_m: float = 0.005
    body_radius_base_m: float = 0.01
    height_base_m: float = 0.04
    radius_m: float | None = None
    thickness_m: float = 0.0002
    density_kgm3: float = 1000.0
    c1_pa: float = 2.1125e5
    c2_pa: float = 0.0
    ogden_alpha: tuple[float, ...] = COMSOL_WATER_BALLOON_OGDEN_ALPHA
    ogden_shear_modulus_pa: tuple[float, ...] = (
        COMSOL_WATER_BALLOON_OGDEN_SHEAR_MODULUS_PA
    )
    inlet_velocity_mps: float = 0.15
    inlet_flow_m3s: float | None = None
    fill_start_s: float = 0.2
    fill_duration_s: float = 9.0
    observation_time_s: float = 15.0
    dt_s: float = 2.0e-4
    step_count: int = 50
    pressure_bulk_modulus_pa: float = 2.0e5
    latitude_bands: int = 4
    longitude_segments: int = 8
    grid_nodes: tuple[int, int, int] = (12, 12, 12)
    velocity_damping: float = 0.995
    gravity_mps2: float = 9.81
    runtime_arch: str = "cuda"
    local_stress_sample_count: int = 4096
    local_profile_fillet_radius_m: float | None = None


def run_water_balloon_fsi_smoke(
    config: WaterBalloonFsiConfig | None = None,
) -> dict[str, object]:
    cfg = config or WaterBalloonFsiConfig()
    return run_official_fsi_benchmark(
        OfficialBenchmarkRunSpec(
            case_spec=CASE_SPEC,
            solver_family="axisymmetric-membrane-inflation",
            case_metadata=COMSOL_WATER_BALLOON_CASE_METADATA,
            boundary_conditions=COMSOL_WATER_BALLOON_BOUNDARY_CONDITIONS,
            config=cfg,
            runner=_run_water_balloon_fsi_core,
        )
    )


def _run_water_balloon_fsi_core(
    config: WaterBalloonFsiConfig,
) -> dict[str, object]:
    cfg = config or WaterBalloonFsiConfig()
    initial_volume_m3 = _initial_water_content_m3(cfg)
    inlet_flow_m3s = _inlet_flow_m3s(cfg)
    report = run_uv_membrane_inflation_smoke(
        MembraneInflationConfig(
            radius_m=_membrane_radius_m(cfg),
            thickness_m=cfg.thickness_m,
            density_kgm3=cfg.density_kgm3,
            c1_pa=cfg.c1_pa,
            c2_pa=cfg.c2_pa,
            inlet_flow_m3s=inlet_flow_m3s,
            fill_duration_s=cfg.fill_duration_s,
            dt_s=cfg.dt_s,
            step_count=cfg.step_count,
            pressure_bulk_modulus_pa=cfg.pressure_bulk_modulus_pa,
            latitude_bands=cfg.latitude_bands,
            longitude_segments=cfg.longitude_segments,
            grid_nodes=cfg.grid_nodes,
            velocity_damping=cfg.velocity_damping,
            gravity_mps2=cfg.gravity_mps2,
            fill_start_s=cfg.fill_start_s,
            initial_volume_m3=initial_volume_m3,
            observation_time_s=cfg.observation_time_s,
        )
    )
    water_content = _water_content_report(cfg, initial_volume_m3, inlet_flow_m3s)
    structure_stress = _structure_stress_report(
        cfg,
        initial_volume_m3,
        float(water_content["final_water_content_ml"]),
    )
    computed_result_sources = dict(report["computed_result_sources"])
    computed_result_sources.update(
        {
            "initial_water_content_ml": "Taichi axisymmetric profile volume integral",
            "final_water_content_ml": (
                "Taichi integral of inlet_velocity * inlet_area over time"
            ),
            "global_equibiaxial_mises_stress_pa": (
                "Ogden membrane stress from computed volume stretch"
            ),
            "local_axisymmetric_mises_stress_pa": (
                "Taichi local Ogden stress from smooth axisymmetric neck/ellipse profile"
            ),
        }
    )
    return {
        **report,
        "case": CASE_SPEC.case_id,
        "case_metadata": COMSOL_WATER_BALLOON_CASE_METADATA,
        "boundary_conditions": COMSOL_WATER_BALLOON_BOUNDARY_CONDITIONS,
        "acceptance_tolerance": CASE_SPEC.acceptance_tolerance,
        "computed_result_sources": computed_result_sources,
        **water_content,
        **structure_stress,
        "official_reference_passed": (
            bool(water_content["official_volume_reference_passed"])
            and bool(structure_stress["official_structure_reference_passed"])
        ),
    }


def _structure_stress_report(
    cfg: WaterBalloonFsiConfig,
    initial_volume_m3: float,
    final_water_content_ml: float,
) -> dict[str, object]:
    final_volume_m3 = final_water_content_ml * 1.0e-6
    stretch = stretch_from_volume_ratio(
        current_volume_m3=final_volume_m3,
        rest_volume_m3=initial_volume_m3,
    )
    stress_pa = OgdenMembraneMaterial.from_sequences(
        alpha=cfg.ogden_alpha,
        shear_modulus_pa=cfg.ogden_shear_modulus_pa,
    ).equibiaxial_cauchy_stress_pa(stretch)
    local_report = smooth_axisymmetric_membrane_stress_report(
        SmoothAxisymmetricMembraneStressConfig(
            neck_radius_m=cfg.neck_radius_m,
            initial_height_m=cfg.height_base_m * cfg.initial_size_factor,
            final_height_m=cfg.height_base_m * cfg.initial_size_factor,
            initial_bulb_radius_m=_membrane_radius_m(cfg),
            target_volume_m3=final_volume_m3,
            ogden_alpha=cfg.ogden_alpha,
            ogden_shear_modulus_pa=cfg.ogden_shear_modulus_pa,
            blend_exponent=_local_profile_blend_exponent(cfg),
            sample_count=cfg.local_stress_sample_count,
            runtime_arch=cfg.runtime_arch,
        )
    )
    reference = CASE_SPEC.reference_results["max_mises_stress_pa"]
    global_relative_error = _relative_error(stress_pa, reference)
    relative_error = _relative_error(local_report.max_mises_stress_pa, reference)
    return {
        "global_equibiaxial_stretch": stretch,
        "global_equibiaxial_mises_stress_pa": stress_pa,
        "global_equibiaxial_mises_stress_relative_error": global_relative_error,
        "local_axisymmetric_final_bulb_radius_m": local_report.final_bulb_radius_m,
        "local_axisymmetric_blend_exponent": _local_profile_blend_exponent(cfg),
        "local_axisymmetric_fillet_radius_m": _local_profile_fillet_radius_m(cfg),
        "local_axisymmetric_mises_stress_pa": local_report.max_mises_stress_pa,
        "local_axisymmetric_max_circumferential_stretch": (
            local_report.max_circumferential_stretch
        ),
        "local_axisymmetric_max_meridional_stretch": (
            local_report.max_meridional_stretch
        ),
        "local_axisymmetric_max_stress_z_m": local_report.max_stress_z_m,
        "official_max_mises_stress_pa": reference,
        "structure_reference_errors": {
            "max_mises_stress_pa": relative_error,
        },
        "official_structure_reference_passed": (
            relative_error <= CASE_SPEC.acceptance_tolerance
        ),
        "structure_reference_gap": (
            "smooth axisymmetric local Ogden membrane stress is computed "
            "from the official fillet geometry and target water volume, "
            "but a full local membrane equilibrium solve is still required"
        ),
    }


def _water_content_report(
    cfg: WaterBalloonFsiConfig,
    initial_volume_m3: float,
    inlet_flow_m3s: float,
) -> dict[str, object]:
    inlet = TimeWindowedInletFlow(
        volumetric_flow_m3s=inlet_flow_m3s,
        start_s=cfg.fill_start_s,
        end_s=cfg.fill_start_s + cfg.fill_duration_s,
    )
    computed_curve_ml = {
        time_s: _water_content_ml_at(
            inlet=inlet,
            initial_volume_m3=initial_volume_m3,
            time_s=time_s,
            runtime_arch=cfg.runtime_arch,
        )
        for time_s, _ in COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.points
    }
    curve_errors = {
        f"{time_s:g}": _relative_error(
            computed_curve_ml[time_s],
            COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.value_at(time_s),
        )
        for time_s in computed_curve_ml
    }
    initial_ml = computed_curve_ml[0.0]
    final_ml = _water_content_ml_at(
        inlet=inlet,
        initial_volume_m3=initial_volume_m3,
        time_s=cfg.observation_time_s,
        runtime_arch=cfg.runtime_arch,
    )
    reference_errors = {
        "initial_water_content_ml": _relative_error(
            initial_ml,
            CASE_SPEC.reference_results["initial_water_content_ml"],
        ),
        "final_water_content_ml": _relative_error(
            final_ml,
            CASE_SPEC.reference_results["final_water_content_ml"],
        ),
    }
    return {
        "initial_water_content_ml": initial_ml,
        "final_water_content_ml": final_ml,
        "water_content_curve_ml": computed_curve_ml,
        "water_content_curve_relative_errors": curve_errors,
        "volume_reference_errors": reference_errors,
        "official_volume_reference_passed": all(
            error <= CASE_SPEC.acceptance_tolerance
            for error in (*reference_errors.values(), *curve_errors.values())
        ),
    }


def _water_content_ml_at(
    *,
    inlet: TimeWindowedInletFlow,
    initial_volume_m3: float,
    time_s: float,
    runtime_arch: str,
) -> float:
    return 1.0e6 * (
        initial_volume_m3
        + inlet.volume_between_m3(
            0.0,
            time_s,
            runtime_arch=runtime_arch,
        )
    )


def _initial_water_content_m3(cfg: WaterBalloonFsiConfig) -> float:
    height_m = cfg.height_base_m * cfg.initial_size_factor
    return AxisymmetricNeckedEllipseProfile(
        neck_radius_m=cfg.neck_radius_m,
        neck_z_min_m=-0.5 * height_m,
        neck_z_max_m=0.0,
        bulb_radius_m=_membrane_radius_m(cfg),
        bulb_center_z_m=-2.0 * height_m / 3.0,
        bulb_half_height_m=height_m / 3.0,
    ).volume_m3(runtime_arch=cfg.runtime_arch)


def _membrane_radius_m(cfg: WaterBalloonFsiConfig) -> float:
    if cfg.radius_m is not None:
        return float(cfg.radius_m)
    return cfg.body_radius_base_m * cfg.initial_size_factor


def _inlet_flow_m3s(cfg: WaterBalloonFsiConfig) -> float:
    if cfg.inlet_flow_m3s is not None:
        return float(cfg.inlet_flow_m3s)
    return math.pi * cfg.neck_radius_m * cfg.neck_radius_m * cfg.inlet_velocity_mps


def _local_profile_fillet_radius_m(cfg: WaterBalloonFsiConfig) -> float:
    if cfg.local_profile_fillet_radius_m is not None:
        return float(cfg.local_profile_fillet_radius_m)
    return (cfg.height_base_m / cfg.body_radius_base_m) * 0.01 / 3.0


def _local_profile_blend_exponent(cfg: WaterBalloonFsiConfig) -> float:
    fillet_radius_m = _local_profile_fillet_radius_m(cfg)
    bulb_radius_m = _membrane_radius_m(cfg)
    return 2.0 + bulb_radius_m / (bulb_radius_m + fillet_radius_m)


def _relative_error(computed_value: float, reference_value: float) -> float:
    if abs(reference_value) <= 1.0e-30:
        return 0.0
    return abs(float(computed_value) - float(reference_value)) / abs(reference_value)
