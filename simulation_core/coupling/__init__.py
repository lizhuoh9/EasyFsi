from __future__ import annotations

from simulation_core.coupling.interface_forces import (
    ForceBalanceReport,
    RegionPairInterfaceReactionTarget,
    action_reaction_balance,
    region_pair_interface_reaction_forces,
)
from simulation_core.coupling.hibm_mpm import (
    HibmMpmExternalForceClearReport,
    HibmMpmFluidStressSampleReport,
    HibmMpmIbBoundaryConditionReport,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmIbNodeSearchReport,
    HibmMpmMpmForceScatterReport,
    HibmMpmNoSlipResidualReport,
    HibmMpmPressureNeumannGradientReport,
    HibmMpmPressureNeumannMatrixReport,
    HibmMpmSharpCouplingState,
    HibmMpmSharpFluidToMpmLoadReport,
    HibmMpmSharpMpmStepReport,
    HibmMpmSharpNeoHookeanStepReport,
    HibmMpmSurfaceMarkerForceReport,
    HibmMpmSurfaceMarkers,
    HibmMpmSurfaceUpdateReport,
    advance_hibm_mpm_sharp_mpm_step,
    advance_hibm_mpm_sharp_neo_hookean_step,
    assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
    hibm_mpm_paper_requirements,
    hibm_mpm_sharp_step_summary,
)
from simulation_core.coupling.pressure_interface import (
    PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS,
    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
    far_pressure_side_normal_sign_from_direction,
)
from simulation_core.coupling.tri_surface import (
    TriSurfaceDiagnosticReport,
    TriSurfaceForcePairReport,
    TriSurfaceRegionDiagnostics,
)

_PROJECTED_IBM_EXPORTS = {
    "ProjectedIbmRegionPairStepConfig",
    "ProjectedIbmRegionPairStepReport",
    "advance_projected_ibm_region_pair_fluid_step",
}


def __getattr__(name: str):
    if name in _PROJECTED_IBM_EXPORTS:
        from simulation_core.coupling import projected_ibm

        value = getattr(projected_ibm, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ForceBalanceReport",
    "HibmMpmExternalForceClearReport",
    "HibmMpmFluidStressSampleReport",
    "HibmMpmIbBoundaryConditionReport",
    "HibmMpmIbBoundaryConditions",
    "HibmMpmIbNodeSearch",
    "HibmMpmIbNodeSearchReport",
    "HibmMpmMpmForceScatterReport",
    "HibmMpmNoSlipResidualReport",
    "HibmMpmPressureNeumannGradientReport",
    "HibmMpmPressureNeumannMatrixReport",
    "HibmMpmSharpCouplingState",
    "HibmMpmSharpFluidToMpmLoadReport",
    "HibmMpmSharpMpmStepReport",
    "HibmMpmSharpNeoHookeanStepReport",
    "HibmMpmSurfaceMarkerForceReport",
    "HibmMpmSurfaceMarkers",
    "HibmMpmSurfaceUpdateReport",
    "PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS",
    "PRESSURE_INTERFACE_COUPLING_SLOT_COUNT",
    "ProjectedIbmRegionPairStepConfig",
    "ProjectedIbmRegionPairStepReport",
    "RegionPairInterfaceReactionTarget",
    "TriSurfaceDiagnosticReport",
    "TriSurfaceForcePairReport",
    "TriSurfaceRegionDiagnostics",
    "action_reaction_balance",
    "advance_hibm_mpm_sharp_mpm_step",
    "advance_hibm_mpm_sharp_neo_hookean_step",
    "advance_projected_ibm_region_pair_fluid_step",
    "assemble_hibm_mpm_sharp_fluid_to_mpm_loads",
    "far_pressure_side_normal_sign_from_direction",
    "hibm_mpm_paper_requirements",
    "hibm_mpm_sharp_step_summary",
    "region_pair_interface_reaction_forces",
]
