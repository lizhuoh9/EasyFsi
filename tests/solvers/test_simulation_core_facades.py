from __future__ import annotations

import importlib
import unittest


class SimulationCoreFacadeTests(unittest.TestCase):
    def test_fluid_facade_exports_existing_fluid_api(self) -> None:
        fluids = importlib.import_module("simulation_core.fluids")
        legacy = importlib.import_module("simulation_core.fluid")

        self.assertIs(fluids.CartesianFluidSolver, legacy.CartesianFluidSolver)
        self.assertIs(fluids.FluidDomainSpec, legacy.FluidDomainSpec)
        self.assertIs(fluids.CartesianGrid, legacy.CartesianGrid)
        self.assertIs(fluids.ForceSpreadingReport, legacy.ForceSpreadingReport)

    def test_solids_facade_exports_existing_solid_api(self) -> None:
        solids = importlib.import_module("simulation_core.solids")
        neo = importlib.import_module("simulation_core.neo_hookean_mpm")
        mooney = importlib.import_module("simulation_core.mooney_shell_mpm")

        self.assertIs(solids.NeoHookeanMpmState, neo.NeoHookeanMpmState)
        self.assertIs(solids.NeoHookeanMpmReport, neo.NeoHookeanMpmReport)
        self.assertIs(solids.TriMooneyShellMpmState, mooney.TriMooneyShellMpmState)
        self.assertIs(solids.UvMooneyShellMpmState, mooney.UvMooneyShellMpmState)

    def test_coupling_facade_exports_existing_coupling_api(self) -> None:
        coupling = importlib.import_module("simulation_core.coupling")
        fsi = importlib.import_module("simulation_core.fsi_coupling")
        projected = importlib.import_module("simulation_core.projected_ibm")
        hibm = importlib.import_module("simulation_core.hibm_mpm")
        tri_surface = importlib.import_module("simulation_core.tri_surface")

        self.assertIs(
            coupling.InterfaceReactionFixedPointResult,
            fsi.InterfaceReactionFixedPointResult,
        )
        self.assertIs(
            coupling.ProjectedIbmRegionPairStepConfig,
            projected.ProjectedIbmRegionPairStepConfig,
        )
        self.assertIs(
            coupling.HibmMpmSharpCouplingState,
            hibm.HibmMpmSharpCouplingState,
        )
        self.assertIs(
            coupling.TriSurfaceForcePairReport,
            tri_surface.TriSurfaceForcePairReport,
        )

    def test_geometry_materials_diagnostics_facades_import(self) -> None:
        geometry_tools = importlib.import_module("simulation_core.geometry_tools")
        materials = importlib.import_module("simulation_core.materials")
        diagnostics = importlib.import_module("simulation_core.diagnostics")

        self.assertTrue(hasattr(geometry_tools, "SurfaceMesh"))
        self.assertTrue(hasattr(geometry_tools, "StepTessellationSettings"))
        self.assertTrue(hasattr(materials, "NeoHookeanMaterial"))
        self.assertTrue(hasattr(diagnostics, "ReferenceCurve"))
        self.assertTrue(hasattr(diagnostics, "CflSubstepController"))


if __name__ == "__main__":
    unittest.main()
