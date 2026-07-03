from __future__ import annotations

import importlib
import unittest


LEGACY_TOP_LEVEL_MODULES = (
    "fluid",
    "fsi_coupling",
    "fsi_driver",
    "generic_fsi_solver",
    "hibm",
    "hibm_mpm",
    "interface_pair",
    "moving_boundary",
    "pressure_interface",
    "pressure_sample_pairs",
    "projected_ibm",
    "runtime",
    "tri_surface",
    "neo_hookean_mpm",
    "mooney_shell_mpm",
    "geometry",
    "coordinate_models",
    "fluid_domain",
    "cad_import",
    "cad_tessellation",
    "hyperelastic",
    "validation",
    "time_stepping",
)


class SimulationCoreFacadeTests(unittest.TestCase):
    def test_fluid_facade_exports_existing_fluid_api(self) -> None:
        fluids = importlib.import_module("simulation_core.fluids")

        self.assertTrue(hasattr(fluids, "CartesianFluidSolver"))
        self.assertTrue(hasattr(fluids, "FluidDomainSpec"))
        self.assertTrue(hasattr(fluids, "CartesianGrid"))
        self.assertTrue(hasattr(fluids, "ForceSpreadingReport"))

    def test_fluid_implementation_lives_under_fluids_package(self) -> None:
        solver = importlib.import_module("simulation_core.fluids.solver")

        self.assertTrue(hasattr(solver, "CartesianFluidSolver"))

    def test_solids_facade_exports_existing_solid_api(self) -> None:
        solids = importlib.import_module("simulation_core.solids")
        neo = importlib.import_module("simulation_core.solids.neo_hookean_mpm")
        mooney = importlib.import_module("simulation_core.solids.mooney_shell")

        self.assertIs(solids.NeoHookeanMpmState, neo.NeoHookeanMpmState)
        self.assertIs(solids.NeoHookeanMpmReport, neo.NeoHookeanMpmReport)
        self.assertIs(solids.TriMooneyShellMpmState, mooney.TriMooneyShellMpmState)
        self.assertIs(solids.UvMooneyShellMpmState, mooney.UvMooneyShellMpmState)

    def test_support_facades_export_layered_modules(self) -> None:
        solids = importlib.import_module("simulation_core.solids")
        neo_impl = importlib.import_module("simulation_core.solids.neo_hookean_mpm")
        mooney_pkg = importlib.import_module("simulation_core.solids.mooney_shell")
        geometry_tools = importlib.import_module("simulation_core.geometry_tools")
        surface_mesh = importlib.import_module("simulation_core.geometry_tools.surface_mesh")
        coordinate_models = importlib.import_module(
            "simulation_core.geometry_tools.coordinate_models"
        )
        fluid_domain = importlib.import_module(
            "simulation_core.geometry_tools.fluid_domain"
        )
        materials = importlib.import_module("simulation_core.materials")
        hyperelastic = importlib.import_module("simulation_core.materials.hyperelastic")
        diagnostics = importlib.import_module("simulation_core.diagnostics")
        validation = importlib.import_module("simulation_core.diagnostics.validation")
        time_stepping = importlib.import_module("simulation_core.diagnostics.time_stepping")

        self.assertIs(solids.NeoHookeanMpmState, neo_impl.NeoHookeanMpmState)
        self.assertIs(solids.TriMooneyShellMpmState, mooney_pkg.TriMooneyShellMpmState)
        self.assertIs(geometry_tools.SurfaceMesh, surface_mesh.SurfaceMesh)
        self.assertIs(
            geometry_tools.Cartesian3DCoordinateModel,
            coordinate_models.Cartesian3DCoordinateModel,
        )
        self.assertIs(geometry_tools.FluidDomain, fluid_domain.FluidDomain)
        self.assertIs(materials.NeoHookeanMaterial, hyperelastic.NeoHookeanMaterial)
        self.assertIs(diagnostics.ReferenceCurve, validation.ReferenceCurve)
        self.assertIs(
            diagnostics.CflSubstepController,
            time_stepping.CflSubstepController,
        )

    def test_root_public_api_uses_package_backed_objects(self) -> None:
        import simulation_core

        fluids = importlib.import_module("simulation_core.fluids")
        coupling = importlib.import_module("simulation_core.coupling")
        solids = importlib.import_module("simulation_core.solids")
        geometry_tools = importlib.import_module("simulation_core.geometry_tools")
        materials = importlib.import_module("simulation_core.materials")
        diagnostics = importlib.import_module("simulation_core.diagnostics")

        self.assertIs(simulation_core.CartesianFluidSolver, fluids.CartesianFluidSolver)
        self.assertIs(
            simulation_core.HibmMpmSharpCouplingState,
            coupling.HibmMpmSharpCouplingState,
        )
        self.assertIs(simulation_core.NeoHookeanMpmState, solids.NeoHookeanMpmState)
        self.assertIs(simulation_core.SurfaceMesh, geometry_tools.SurfaceMesh)
        self.assertIs(simulation_core.NeoHookeanMaterial, materials.NeoHookeanMaterial)
        self.assertIs(simulation_core.vector_norm, diagnostics.vector_norm)

    def test_coupling_facade_exports_existing_coupling_api(self) -> None:
        coupling = importlib.import_module("simulation_core.coupling")
        fsi = importlib.import_module("simulation_core.coupling.fsi_coupling")
        projected = importlib.import_module("simulation_core.coupling.projected_ibm")
        hibm = importlib.import_module("simulation_core.coupling.hibm_mpm")
        tri_surface = importlib.import_module("simulation_core.coupling.tri_surface")

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

    def test_driver_facade_exports_existing_driver_api(self) -> None:
        drivers = importlib.import_module("simulation_core.drivers")
        fsi_driver = importlib.import_module("simulation_core.drivers.fsi_driver")
        generic = importlib.import_module("simulation_core.drivers.generic_fsi_solver")

        self.assertIs(drivers.FsiCaseSpec, fsi_driver.FsiCaseSpec)
        self.assertIs(drivers.FsiDriver, fsi_driver.FsiDriver)
        self.assertIs(drivers.FsiProblem, generic.FsiProblem)
        self.assertIs(drivers.solve_fsi, generic.solve_fsi)

    def test_coupling_support_packages_export_support_api(self) -> None:
        interface_pair = importlib.import_module("simulation_core.coupling.interface_pair")
        pressure_pairs = importlib.import_module(
            "simulation_core.coupling.pressure_sample_pairs"
        )
        runtime = importlib.import_module("simulation_core.diagnostics.runtime")

        self.assertTrue(hasattr(interface_pair, "InterfacePairMap"))
        self.assertTrue(hasattr(pressure_pairs, "RuntimeAnchoredCellPairProvider"))
        self.assertTrue(hasattr(runtime, "TaichiRuntimeConfig"))

    def test_hibm_mpm_package_exports_api(self) -> None:
        package = importlib.import_module("simulation_core.coupling.hibm_mpm")

        self.assertTrue(hasattr(package, "HibmMpmSharpCouplingState"))
        self.assertTrue(hasattr(package, "HibmMpmSurfaceMarkers"))
        self.assertTrue(hasattr(package, "HibmMpmIbNodeSearch"))
        self.assertTrue(hasattr(package, "advance_hibm_mpm_sharp_mpm_step"))

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

    def test_legacy_top_level_modules_are_not_installed(self) -> None:
        import simulation_core

        for module_name in LEGACY_TOP_LEVEL_MODULES:
            with self.subTest(module_name=module_name):
                self.assertFalse(hasattr(simulation_core, module_name))
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(f"simulation_core.{module_name}")


if __name__ == "__main__":
    unittest.main()
