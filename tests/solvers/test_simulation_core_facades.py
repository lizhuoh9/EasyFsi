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

    def test_fluid_implementation_lives_under_fluids_package(self) -> None:
        solver = importlib.import_module("simulation_core.fluids.solver")

        self.assertTrue(hasattr(solver, "CartesianFluidSolver"))

    def test_legacy_fluid_module_is_compatibility_shim(self) -> None:
        legacy = importlib.import_module("simulation_core.fluid")
        fluids = importlib.import_module("simulation_core.fluids")

        self.assertIs(legacy.CartesianFluidSolver, fluids.CartesianFluidSolver)
        self.assertIs(legacy.FluidDomainSpec, fluids.FluidDomainSpec)

    def test_solids_facade_exports_existing_solid_api(self) -> None:
        solids = importlib.import_module("simulation_core.solids")
        neo = importlib.import_module("simulation_core.neo_hookean_mpm")
        mooney = importlib.import_module("simulation_core.mooney_shell_mpm")

        self.assertIs(solids.NeoHookeanMpmState, neo.NeoHookeanMpmState)
        self.assertIs(solids.NeoHookeanMpmReport, neo.NeoHookeanMpmReport)
        self.assertIs(solids.TriMooneyShellMpmState, mooney.TriMooneyShellMpmState)
        self.assertIs(solids.UvMooneyShellMpmState, mooney.UvMooneyShellMpmState)

    def test_solver_support_package_exports_match_legacy_shims(self) -> None:
        solids = importlib.import_module("simulation_core.solids")
        neo_impl = importlib.import_module("simulation_core.solids.neo_hookean_mpm")
        mooney_pkg = importlib.import_module("simulation_core.solids.mooney_shell")
        geometry_tools = importlib.import_module("simulation_core.geometry_tools")
        surface_mesh = importlib.import_module("simulation_core.geometry_tools.surface_mesh")
        coordinate_models = importlib.import_module(
            "simulation_core.geometry_tools.coordinate_models"
        )
        fluid_domain = importlib.import_module("simulation_core.geometry_tools.fluid_domain")
        materials = importlib.import_module("simulation_core.materials")
        hyperelastic = importlib.import_module("simulation_core.materials.hyperelastic")
        diagnostics = importlib.import_module("simulation_core.diagnostics")
        validation = importlib.import_module("simulation_core.diagnostics.validation")
        time_stepping = importlib.import_module("simulation_core.diagnostics.time_stepping")

        legacy_neo = importlib.import_module("simulation_core.neo_hookean_mpm")
        legacy_mooney = importlib.import_module("simulation_core.mooney_shell_mpm")
        legacy_geometry = importlib.import_module("simulation_core.geometry")
        legacy_coordinates = importlib.import_module("simulation_core.coordinate_models")
        legacy_domain = importlib.import_module("simulation_core.fluid_domain")
        legacy_hyperelastic = importlib.import_module("simulation_core.hyperelastic")
        legacy_validation = importlib.import_module("simulation_core.validation")
        legacy_time_stepping = importlib.import_module("simulation_core.time_stepping")

        self.assertIs(solids.NeoHookeanMpmState, neo_impl.NeoHookeanMpmState)
        self.assertIs(legacy_neo.NeoHookeanMpmState, neo_impl.NeoHookeanMpmState)
        self.assertIs(legacy_mooney.TriMooneyShellMpmState, mooney_pkg.TriMooneyShellMpmState)
        self.assertIs(legacy_mooney.TriMooneyShellMpmReport, mooney_pkg.TriMooneyShellMpmReport)
        self.assertIs(geometry_tools.SurfaceMesh, surface_mesh.SurfaceMesh)
        self.assertIs(legacy_geometry.SurfaceMesh, surface_mesh.SurfaceMesh)
        self.assertIs(
            legacy_coordinates.Cartesian3DCoordinateModel,
            coordinate_models.Cartesian3DCoordinateModel,
        )
        self.assertIs(legacy_domain.FluidDomain, fluid_domain.FluidDomain)
        self.assertIs(materials.NeoHookeanMaterial, hyperelastic.NeoHookeanMaterial)
        self.assertIs(legacy_hyperelastic.NeoHookeanMaterial, hyperelastic.NeoHookeanMaterial)
        self.assertIs(diagnostics.ReferenceCurve, validation.ReferenceCurve)
        self.assertIs(legacy_validation.ReferenceCurve, validation.ReferenceCurve)
        self.assertIs(
            legacy_time_stepping.CflSubstepController,
            time_stepping.CflSubstepController,
        )

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

    def test_hibm_mpm_package_exports_legacy_api(self) -> None:
        package = importlib.import_module("simulation_core.coupling.hibm_mpm")
        legacy = importlib.import_module("simulation_core.hibm_mpm")

        self.assertIs(
            package.HibmMpmSharpCouplingState,
            legacy.HibmMpmSharpCouplingState,
        )
        self.assertIs(package.HibmMpmSurfaceMarkers, legacy.HibmMpmSurfaceMarkers)
        self.assertIs(package.HibmMpmIbNodeSearch, legacy.HibmMpmIbNodeSearch)
        self.assertIs(
            package.advance_hibm_mpm_sharp_mpm_step,
            legacy.advance_hibm_mpm_sharp_mpm_step,
        )

    def test_coupling_facade_uses_hibm_mpm_package_exports(self) -> None:
        coupling = importlib.import_module("simulation_core.coupling")
        package = importlib.import_module("simulation_core.coupling.hibm_mpm")

        self.assertIs(
            coupling.HibmMpmSharpCouplingState,
            package.HibmMpmSharpCouplingState,
        )
        self.assertIs(coupling.HibmMpmSurfaceMarkers, package.HibmMpmSurfaceMarkers)

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
