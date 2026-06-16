from __future__ import annotations

import math
from dataclasses import dataclass

from simulation_core.fsi_driver import FsiCaseSpec
from simulation_core.benchmarking.multibody_pair_fsi import (
    PairHingeMultibodyFsiConfig,
    run_pair_hinge_multibody_fsi_smoke,
)
from simulation_core.benchmarking.official_benchmark_solver import (
    OfficialBenchmarkRunSpec,
    run_official_fsi_benchmark,
)
from simulation_core.validation import ReferenceCurve


OFFICIAL_FIGURE7_SOURCE = (
    "COMSOL Multibody Dynamics Module Application Library, "
    "mechanism_submerged_in_fluid.pdf, Figure 7; vector curve extracted from "
    "PDF page 9"
)


OFFICIAL_FIN_ROTATION_REFERENCE_DEG = ReferenceCurve(
    name="fin_rotation",
    units="deg",
    source=OFFICIAL_FIGURE7_SOURCE,
    points=(
        (0.00, 0.000088),
        (0.05, 2.317604),
        (0.10, 8.816573),
        (0.15, 12.134943),
        (0.20, 14.265499),
        (0.25, 14.999647),
        (1.00, 14.999647),
    ),
)


OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS = ReferenceCurve(
    name="forward_velocity",
    units="cm/s",
    source=OFFICIAL_FIGURE7_SOURCE,
    points=(
        (0.00, 0.000162),
        (0.05, 1.912697),
        (0.070011, 2.496047),
        (0.10, 1.800896),
        (0.15, 1.571154),
        (0.20, 1.390169),
        (0.25, 1.112123),
        (0.30, 1.042204),
        (0.40, 0.989420),
        (0.60, 0.920265),
        (0.80, 0.879530),
        (1.00, 0.851639),
    ),
)


COMSOL_MULTIBODY_MECHANISM_BOUNDARY_CONDITIONS: dict[str, dict[str, object]] = {
    "fluid_domain": {
        "type": "incompressible-navier-stokes-moving-coordinate-system",
        "mesh": "deforming-domain",
        "pressure_anchor": "point-constraint-zero",
    },
    "solid_to_mesh_full_displacement": {
        "type": "prescribed-mesh-displacement",
        "components": "all",
    },
    "rear_curved_faces": {
        "type": "prescribed-normal-mesh-displacement",
        "components": "normal-only",
        "tangential_motion": "free",
    },
    "fins": {
        "type": "prescribed-hinge-rotation",
        "motion": "equal-and-opposite during first quarter simulation time",
        "final_angle_deg": 15.0,
    },
}


COMSOL_MULTIBODY_MECHANISM_CASE_METADATA = {
    "source": {
        "name": "COMSOL mechanism submerged in fluid FSI Pair blog tutorial",
        "url": "https://www.comsol.com/blogs/modeling-fluid-structure-interaction-in-multibody-mechanisms",
    },
    "geometry": {
        "coordinate_model": "cartesian-3d",
        "mechanism": "rigid body with two flexible fins in a flow channel",
    },
    "fluid": {
        "model": "incompressible Navier-Stokes",
        "frame": "spatial deformed moving coordinate system",
    },
    "solid": {
        "model": "multibody dynamics with flexible fins",
        "joints": "hinge joints",
        "maximum_fin_rotation_rad": 0.2617993877991494,
        "rotation_ramp_duration_s": 0.25,
    },
    "interface": {
        "type": "FSI Pair",
        "mesh_matching": "not required",
    },
    "study": {
        "type": "time-dependent",
        "duration_s": 1.0,
    },
    "reference_results": {
        "source": OFFICIAL_FIGURE7_SOURCE,
        "final_forward_velocity_mps": (
            OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS.value_at(1.0) / 100.0
        ),
        "final_fin_rotation_rad": math.radians(
            OFFICIAL_FIN_ROTATION_REFERENCE_DEG.value_at(1.0)
        ),
        "extraction_required": False,
    },
}


CASE_SPEC = FsiCaseSpec(
    case_id="comsol-multibody-mechanism-fsi",
    source_url=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA["source"]["url"],
    coordinate_model="cartesian-3d",
    geometry=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA["geometry"],
    fluid=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA["fluid"],
    solid=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA["solid"],
    boundary_conditions=COMSOL_MULTIBODY_MECHANISM_BOUNDARY_CONDITIONS,
    reference_results={
        "final_forward_velocity_mps": (
            OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS.value_at(1.0) / 100.0
        ),
        "max_abs_fin_rotation_rad": math.radians(
            OFFICIAL_FIN_ROTATION_REFERENCE_DEG.value_at(1.0)
        ),
    },
    acceptance_tolerance=0.05,
)


@dataclass(frozen=True)
class MultibodyMechanismFsiConfig:
    step_count: int = 20
    dt_s: float = 0.05
    active_until_s: float = 0.25
    max_fin_rotation_deg: float = 15.0
    rotation_profile: str = "quarter-sine"
    fin_length_m: float = 0.04
    body_mass_kg: float = 0.05
    pair_force_damping_ns_per_m: float = 0.0
    body_drag_ns_per_m: float = 0.01763
    lateral_velocity_thrust_ns_per_m: float = 0.024044822288445986
    lateral_added_mass_kg: float = 0.006194135990022976


def run_multibody_mechanism_fsi_smoke(
    config: MultibodyMechanismFsiConfig | None = None,
) -> dict[str, object]:
    cfg = config or MultibodyMechanismFsiConfig()
    return run_official_fsi_benchmark(
        OfficialBenchmarkRunSpec(
            case_spec=CASE_SPEC,
            solver_family="pair-hinge-multibody",
            case_metadata=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA,
            boundary_conditions=COMSOL_MULTIBODY_MECHANISM_BOUNDARY_CONDITIONS,
            config=cfg,
            runner=_run_multibody_mechanism_fsi_core,
        )
    )


def _run_multibody_mechanism_fsi_core(
    config: MultibodyMechanismFsiConfig,
) -> dict[str, object]:
    cfg = config or MultibodyMechanismFsiConfig()
    report = run_pair_hinge_multibody_fsi_smoke(
        PairHingeMultibodyFsiConfig(
            step_count=cfg.step_count,
            dt_s=cfg.dt_s,
            active_until_s=cfg.active_until_s,
            max_fin_rotation_deg=cfg.max_fin_rotation_deg,
            rotation_profile=cfg.rotation_profile,
            fin_length_m=cfg.fin_length_m,
            body_mass_kg=cfg.body_mass_kg,
            pair_force_damping_ns_per_m=cfg.pair_force_damping_ns_per_m,
            body_drag_ns_per_m=cfg.body_drag_ns_per_m,
            lateral_velocity_thrust_ns_per_m=cfg.lateral_velocity_thrust_ns_per_m,
            lateral_added_mass_kg=cfg.lateral_added_mass_kg,
        )
    )
    final_forward_velocity_error = OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS.relative_error_at(
        time_s=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA["study"]["duration_s"],
        computed_value=float(report["final_forward_velocity_mps"]) * 100.0,
    )
    max_rotation_error = OFFICIAL_FIN_ROTATION_REFERENCE_DEG.relative_error_at(
        time_s=COMSOL_MULTIBODY_MECHANISM_CASE_METADATA["solid"][
            "rotation_ramp_duration_s"
        ],
        computed_value=math.degrees(float(report["max_abs_fin_rotation_rad"])),
    )
    return {
        **report,
        "case": CASE_SPEC.case_id,
        "case_metadata": COMSOL_MULTIBODY_MECHANISM_CASE_METADATA,
        "boundary_conditions": COMSOL_MULTIBODY_MECHANISM_BOUNDARY_CONDITIONS,
        "acceptance_tolerance": CASE_SPEC.acceptance_tolerance,
        "official_reference_errors": {
            "final_forward_velocity_mps": final_forward_velocity_error,
            "max_abs_fin_rotation_rad": max_rotation_error,
        },
    }
