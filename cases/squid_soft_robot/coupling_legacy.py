from __future__ import annotations

from dataclasses import dataclass

from simulation_core import FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED


@dataclass(frozen=True)
class LegacyProjectedReducedCouplingControl:
    enabled: bool
    requested_iterations: int


def legacy_projected_reduced_fsi_coupling_enabled(
    *,
    fsi_coupling_mode: str,
    solid_model: str,
    fsi_coupling_iterations: int,
) -> bool:
    if str(fsi_coupling_mode) != FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED:
        return False
    return (
        str(solid_model) in ("tri_mooney_shell_mpm", "neo_hookean_mpm")
        and int(fsi_coupling_iterations) > 1
    )


def legacy_projected_reduced_coupling_control(
    *,
    fsi_coupling_mode: str,
    solid_model: str,
    fsi_coupling_iterations: int,
) -> LegacyProjectedReducedCouplingControl:
    return LegacyProjectedReducedCouplingControl(
        enabled=legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=solid_model,
            fsi_coupling_iterations=fsi_coupling_iterations,
        ),
        requested_iterations=int(fsi_coupling_iterations),
    )
