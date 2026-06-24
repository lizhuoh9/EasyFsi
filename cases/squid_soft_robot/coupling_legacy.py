from __future__ import annotations

from simulation_core import FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED


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
