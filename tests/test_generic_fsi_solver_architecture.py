from __future__ import annotations

import inspect
from pathlib import Path
import unittest


class GenericFsiSolverArchitectureTests(unittest.TestCase):
    def test_core_exposes_case_agnostic_fsi_driver_contract(self) -> None:
        from simulation_core.fsi_driver import FsiCaseSpec, FsiDriver

        spec = FsiCaseSpec(
            case_id="toy-fluid-solid",
            source_url="https://example.invalid/toy-fluid-solid",
            coordinate_model="cartesian-2d",
            geometry={"domain": "toy"},
            fluid={"model": "toy-fluid"},
            solid={"model": "toy-solid"},
            boundary_conditions={"interface": {"type": "two-way-fsi"}},
            reference_results={"max_displacement_m": 2.0},
            acceptance_tolerance=0.05,
        )

        class ToyFluid:
            def __init__(self) -> None:
                self.force_n = (0.0, -2.0)

            def advance(self, context):
                return {"fluid_advanced": context.step_index}

            def interface_force_n(self):
                return self.force_n

            def apply_interface_displacement(self, displacement_m):
                self.displacement_m = displacement_m

        class ToySolid:
            def __init__(self) -> None:
                self.displacement_m = (0.0, 0.0)

            def apply_interface_force(self, force_n):
                self.force_n = force_n

            def advance(self, context):
                self.displacement_m = (0.0, -float(context.step_index + 1))
                return {"solid_advanced": context.step_index}

            def interface_displacement_m(self):
                return self.displacement_m

        driver = FsiDriver(
            case_spec=spec,
            fluid_model=ToyFluid(),
            solid_model=ToySolid(),
        )

        report = driver.run(step_count=2)

        self.assertEqual(report.case_id, "toy-fluid-solid")
        self.assertEqual(report.step_count, 2)
        self.assertEqual(report.final_results["max_displacement_m"], 2.0)
        self.assertLessEqual(
            report.relative_errors["max_displacement_m"],
            spec.acceptance_tolerance,
        )
        self.assertEqual(report.step_reports[-1].solid_displacement_m, (0.0, -2.0))
        self.assertLessEqual(
            report.step_reports[-1].action_reaction.relative_error,
            1.0e-12,
        )

    def test_core_fluid_domain_is_not_axisymmetric_by_default(self) -> None:
        from simulation_core.coordinate_models import (
            Axisymmetric2DCoordinateModel,
            Cartesian2DCoordinateModel,
            Cartesian3DCoordinateModel,
        )
        from simulation_core.fluid_domain import (
            AxisAlignedBoundary,
            BoundaryRegion,
            FluidDomain,
        )

        domain = FluidDomain(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 2.0, 3.0),
            grid_nodes=(4, 5, 6),
            coordinate_model=Cartesian3DCoordinateModel(),
            boundary_regions=(
                BoundaryRegion(name="inlet", kind="velocity-inlet", selector="zmax"),
                BoundaryRegion(name="outlet", kind="pressure-outlet", selector="zmin"),
            ),
        )

        self.assertEqual(domain.dimension, 3)
        self.assertEqual(domain.coordinate_model.name, "cartesian-3d")
        self.assertNotEqual(domain.coordinate_model.name, "axisymmetric-2d")
        self.assertEqual(Cartesian2DCoordinateModel().dimension, 2)
        self.assertEqual(Cartesian3DCoordinateModel().dimension, 3)
        self.assertEqual(Axisymmetric2DCoordinateModel(radial_axis="x").dimension, 2)

    def test_pressure_outlet_boundary_is_axis_aligned_not_case_named(self) -> None:
        from simulation_core.fluid_domain import AxisAlignedBoundary

        outlet = AxisAlignedBoundary.pressure_outlet(axis="z", side="min")

        self.assertEqual(outlet.selector, "z_min")
        self.assertEqual(outlet.axis_index, 2)
        self.assertEqual(outlet.side_index, 0)
        self.assertTrue(outlet.legacy_zmin_outlet)
        self.assertEqual(outlet.as_boundary_region().selector, "z_min")
        self.assertNotIn("squid", repr(outlet).lower())
        self.assertNotIn("nozzle", repr(outlet).lower())

        inlet = AxisAlignedBoundary.from_selector(
            name="inlet",
            kind="velocity-inlet",
            selector="x-max",
        )
        self.assertEqual(inlet.selector, "x_max")
        self.assertFalse(inlet.legacy_zmin_outlet)

    def test_benchmark_case_files_only_define_specs_and_entrypoints(self) -> None:
        import cases.ansys_vertical_flap_fsi as ansys_case
        import cases.comsol_multibody_mechanism_fsi as multibody_case
        import cases.comsol_water_balloon_fsi as water_balloon_case

        for case_module in (ansys_case, multibody_case, water_balloon_case):
            source = inspect.getsource(case_module)
            self.assertIn("CASE_SPEC", source)
            self.assertIn("run_official_fsi_benchmark(", source)
            self.assertIn("OfficialBenchmarkRunSpec(", source)
            self.assertIn("simulation_core.benchmarking", source)
            self.assertNotIn("cases.official_benchmarks", source)
            self.assertNotIn("CartesianFluidSolver(", source)
            self.assertNotIn("NeoHookeanMpmState(", source)
            self.assertNotIn("HibmMpmSurfaceMarkers(", source)
            self.assertNotIn("UvMooneyShellMpmState(", source)
            self.assertNotIn("for step_index in range", source)
            self.assertNotIn(".to_numpy()", source)

    def test_generic_benchmark_helpers_live_in_simulation_core_not_cases(self) -> None:
        core_benchmarking = Path("simulation_core") / "benchmarking"
        expected_modules = {
            "axisymmetric_geometry.py",
            "axisymmetric_membrane.py",
            "inlet_flow.py",
            "membrane_inflation_fsi.py",
            "multibody_pair_fsi.py",
            "official_benchmark_solver.py",
            "ogden_membrane.py",
            "rigid_multibody.py",
            "solid_mpm_fsi_runner.py",
        }

        self.assertTrue(core_benchmarking.is_dir())
        self.assertTrue(
            expected_modules.issubset(
                path.name for path in core_benchmarking.glob("*.py")
            )
        )
        case_benchmarking = Path("cases") / "official_benchmarks"
        self.assertFalse(
            any(path.suffix == ".py" for path in case_benchmarking.glob("*.py"))
        )

    def test_three_official_benchmark_cases_are_registered_as_specs(self) -> None:
        from cases import CASE_MODULES

        self.assertEqual(
            CASE_MODULES["comsol-water-balloon-fsi"],
            "cases.comsol_water_balloon_fsi",
        )
        self.assertEqual(
            CASE_MODULES["comsol-multibody-mechanism-fsi"],
            "cases.comsol_multibody_mechanism_fsi",
        )
        self.assertEqual(
            CASE_MODULES["ansys-vertical-flap-fsi"],
            "cases.ansys_vertical_flap_fsi",
        )

        benchmark_modules = (
            CASE_MODULES["comsol-water-balloon-fsi"],
            CASE_MODULES["comsol-multibody-mechanism-fsi"],
            CASE_MODULES["ansys-vertical-flap-fsi"],
        )
        for module_name in benchmark_modules:
            module = __import__(module_name, fromlist=["CASE_SPEC"])
            self.assertEqual(module.CASE_SPEC.acceptance_tolerance, 0.05)
