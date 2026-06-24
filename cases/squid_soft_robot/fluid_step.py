from __future__ import annotations

from simulation_core import ProjectedIbmRegionPairStepConfig


def z_displacement_vector(displacement_z_m: float) -> tuple[float, float, float]:
    return (0.0, 0.0, float(displacement_z_m))


def z_velocity_vector(velocity_z_mps: float) -> tuple[float, float, float]:
    return (0.0, 0.0, float(velocity_z_mps))


def build_projected_ibm_region_pair_step_config(
    **kwargs: object,
) -> ProjectedIbmRegionPairStepConfig:
    return ProjectedIbmRegionPairStepConfig(**kwargs)
