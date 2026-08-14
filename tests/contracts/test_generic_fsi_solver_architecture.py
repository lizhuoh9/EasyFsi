from __future__ import annotations
import inspect
from pathlib import Path
import unittest


class GenericFsiSolverArchitectureTests(unittest.TestCase):
    def test_generic_solver_boundary_is_case_agnostic_and_injected(self) -> None:
        import numpy as np

        from simulation_core.drivers.generic_fsi_solver import (
            DiagnosticsConfig,
            FluidDomain,
            FsiProblem,
            FsiSolverConfig,
            FsiTrialResult,
            InterfaceSurface,
            OneSidedPressurePolicy,
            PressureSamplePairProvider,
            PressureSamplingConfig,
            SolidBody,
            SurfaceRegion,
            TractionConfig,
            solve_fsi,
        )

        class ToyRuntime:
            def begin_step(self, context):
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                target = np.full((1, 3), float(context.step), dtype=np.float64)
                return FsiTrialResult(
                    marker_velocity_mps=(
                        0.5 * (np.asarray(marker_velocity_guess_mps) + target)
                    )
                )

            def commit_step(self, context, trial, coupling):
                return {"max_displacement_m": float(context.step) * 0.1}

            def rollback_step(self, context):
                raise AssertionError("converged toy step must not roll back")

            def finalize_run(self):
                return {
                    "diagnostics": {"runtime": "toy"},
                    "artifacts": {"matrix": "toy-matrix.json"},
                }

        provider = PressureSamplePairProvider(
            mode="runtime_anchored_cell_pair",
            pair_source_status="runtime_generated",
        )
        sampling = PressureSamplingConfig(pair_provider=provider)
        traction = TractionConfig(
            pressure_sampling=sampling,
            one_sided_pressure=OneSidedPressurePolicy(),
        )
        problem = FsiProblem(
            problem_id="toy-fsi",
            fluid_domain=FluidDomain(
                domain_id="toy-fluid",
                coordinate_model="cartesian-3d",
                grid_nodes=(2, 3, 4),
                bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                boundary_conditions={"interface": {"type": "two-way"}},
            ),
            solid_bodies=(
                SolidBody(
                    body_id="toy-solid",
                    material={"density": 1.0},
                    initial_state={"displacement_m": 0.0},
                ),
            ),
            interface_surfaces=(
                InterfaceSurface(
                    surface_id="toy-interface",
                    regions=(SurfaceRegion(region_id="face-a"),),
                ),
            ),
            traction_config=traction,
            runtime_factory=lambda problem, solver_config, diagnostics_config: ToyRuntime(),
        )

        result = solve_fsi(
            problem,
            FsiSolverConfig(step_count=2, time_step_s=0.5),
            DiagnosticsConfig(output_root="outputs/toy"),
        )

        self.assertEqual(result.problem_id, "toy-fsi")
        self.assertEqual(result.run_status, "completed")
        self.assertEqual(result.completed_step_count, 2)
        self.assertTrue(result.diagnostics["generic_api_invoked"])
        self.assertEqual(
            result.diagnostics["pressure_pair_policy"]["mode"],
            "runtime_anchored_cell_pair",
        )
        self.assertFalse(
            result.diagnostics["pressure_pair_policy"]["transition_backed"]
        )
        self.assertEqual(result.artifacts["matrix"], "toy-matrix.json")

        source = (
            Path("simulation_core") / "drivers" / "generic_fsi_solver.py"
        ).read_text(encoding="utf-8")
        forbidden_terms = ("ansys", "fluent", "vertical_flap", "vertical flap")
        for term in forbidden_terms:
            self.assertNotIn(term, source.lower())

    def test_pressure_pair_provider_reports_transition_replay_explicitly(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import PressureSamplePairProvider

        provider = PressureSamplePairProvider(
            mode="runtime_anchored_cell_pair",
            pair_source_status="transition_seeded_from_anchor_artifact",
            source="validation/input.json",
        )

        self.assertTrue(provider.transition_backed)
        self.assertEqual(
            provider.as_diagnostics(),
            {
                "mode": "runtime_anchored_cell_pair",
                "pair_source_status": "transition_seeded_from_anchor_artifact",
                "source": "validation/input.json",
                "transition_backed": True,
            },
        )

    def test_core_fluid_domain_is_not_axisymmetric_by_default(self) -> None:
        from simulation_core.geometry_tools.coordinate_models import (
            Axisymmetric2DCoordinateModel,
            Cartesian2DCoordinateModel,
            Cartesian3DCoordinateModel,
        )
        from simulation_core.geometry_tools.fluid_domain import (
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
        from simulation_core.geometry_tools.fluid_domain import AxisAlignedBoundary

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
            self.assertIn("benchmarks.official", source)
            self.assertNotIn("cases.official_benchmarks", source)
            self.assertNotIn("CartesianFluidSolver(", source)
            self.assertNotIn("NeoHookeanMpmState(", source)
            self.assertNotIn("HibmMpmSurfaceMarkers(", source)
            self.assertNotIn("UvMooneyShellMpmState(", source)
            self.assertNotIn("for step_index in range", source)
            self.assertNotIn(".to_numpy()", source)

    def test_generic_benchmark_helpers_live_in_benchmarks_not_solver_core(self) -> None:
        official_benchmarking = Path("benchmarks") / "official"
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

        self.assertTrue(official_benchmarking.is_dir())
        self.assertTrue(
            expected_modules.issubset(
                path.name for path in official_benchmarking.glob("*.py")
            )
        )
        self.assertFalse((Path("simulation_core") / "benchmarking").exists())
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

class FsiSolverConfigValidationTests(unittest.TestCase):
    """FsiSolverConfig.__post_init__ must reject non-finite numeric fields.

    ``nan <= 0.0`` and ``inf <= 0.0`` are both False under IEEE-754, so a
    bare ``time_step_s <= 0.0`` check silently let NaN/inf time steps
    through instead of rejecting them at config-construction time.
    """

    def test_step_count_must_be_positive(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        with self.assertRaisesRegex(ValueError, "step_count"):
            FsiSolverConfig(step_count=0, time_step_s=0.1)

    def test_step_count_rejects_bool(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        for value in (False, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "step_count.*integer"):
                    FsiSolverConfig(step_count=value, time_step_s=0.1)

    def test_step_count_rejects_fractional_float(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        with self.assertRaisesRegex(ValueError, "step_count.*integer"):
            FsiSolverConfig(step_count=1.5, time_step_s=0.1)

    def test_time_step_rejects_bool(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        for value in (False, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "time_step_s"):
                    FsiSolverConfig(step_count=5, time_step_s=value)

    def test_time_step_rejects_nan(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        with self.assertRaisesRegex(ValueError, "time_step_s.*finite"):
            FsiSolverConfig(step_count=5, time_step_s=float("nan"))

    def test_time_step_rejects_infinite(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        with self.assertRaisesRegex(ValueError, "time_step_s.*finite"):
            FsiSolverConfig(step_count=5, time_step_s=float("inf"))
        with self.assertRaisesRegex(ValueError, "time_step_s.*finite"):
            FsiSolverConfig(step_count=5, time_step_s=float("-inf"))

    def test_time_step_rejects_non_positive_finite_value(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        with self.assertRaisesRegex(ValueError, "time_step_s.*finite"):
            FsiSolverConfig(step_count=5, time_step_s=0.0)
        with self.assertRaisesRegex(ValueError, "time_step_s.*finite"):
            FsiSolverConfig(step_count=5, time_step_s=-0.1)

    def test_finite_positive_time_step_is_accepted(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FsiSolverConfig

        config = FsiSolverConfig(step_count=5, time_step_s=0.01)
        self.assertEqual(config.step_count, 5)
        self.assertEqual(config.time_step_s, 0.01)


class GenericFsiGeometryAndPressureValidationTests(unittest.TestCase):
    def test_required_names_reject_none_and_blank_text(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "domain_id.*non-empty"):
                    FluidDomain(
                        domain_id=value,
                        coordinate_model="cartesian-3d",
                        grid_nodes=(2, 2, 2),
                        bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    )

    def test_grid_nodes_require_exactly_three_entries(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        for grid_nodes in ((2, 2), (2, 2, 2, 2)):
            with self.subTest(grid_nodes=grid_nodes):
                with self.assertRaisesRegex(ValueError, "grid_nodes"):
                    FluidDomain(
                        domain_id="toy-fluid",
                        coordinate_model="cartesian-3d",
                        grid_nodes=grid_nodes,
                        bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    )

    def test_grid_nodes_reject_bool_float_and_string_entries(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        for value in (False, True, 2.0, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "grid_nodes.*integer"):
                    FluidDomain(
                        domain_id="toy-fluid",
                        coordinate_model="cartesian-3d",
                        grid_nodes=(2, value, 2),
                        bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    )

    def test_grid_nodes_reject_nonpositive_integers(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "grid_nodes.*positive"):
                    FluidDomain(
                        domain_id="toy-fluid",
                        coordinate_model="cartesian-3d",
                        grid_nodes=(2, value, 2),
                        bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    )

    def test_grid_nodes_accept_three_positive_integers(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        domain = FluidDomain(
            domain_id="toy-fluid",
            coordinate_model="cartesian-3d",
            grid_nodes=(2, 3, 4),
            bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        self.assertEqual(domain.grid_nodes, (2, 3, 4))

    def test_fluid_bounds_reject_nonfinite_components(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "bounds_m.*finite"):
                    FluidDomain(
                        domain_id="toy-fluid",
                        coordinate_model="cartesian-3d",
                        grid_nodes=(2, 2, 2),
                        bounds_m=((value, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    )

    def test_fluid_bounds_require_strictly_positive_extent_on_every_axis(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        invalid_bounds = (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
            ((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0)),
        )
        for bounds_m in invalid_bounds:
            with self.subTest(bounds_m=bounds_m):
                with self.assertRaisesRegex(ValueError, "bounds_m.*min.*max"):
                    FluidDomain(
                        domain_id="toy-fluid",
                        coordinate_model="cartesian-3d",
                        grid_nodes=(2, 2, 2),
                        bounds_m=bounds_m,
                    )

    def test_fluid_bounds_accept_finite_strict_extents(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import FluidDomain

        domain = FluidDomain(
            domain_id="toy-fluid",
            coordinate_model="cartesian-3d",
            grid_nodes=(2, 2, 2),
            bounds_m=((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)),
        )

        self.assertEqual(domain.bounds_m[0], (-1.0, -2.0, -3.0))
        self.assertEqual(domain.bounds_m[1], (1.0, 2.0, 3.0))

    def test_surface_region_reference_pressure_rejects_nonfinite_values(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import SurfaceRegion

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reference_pressure_pa.*finite"):
                    SurfaceRegion(region_id="face-a", reference_pressure_pa=value)

    def test_surface_region_policy_reference_pressure_rejects_nonfinite_values(
        self,
    ) -> None:
        from simulation_core.drivers.generic_fsi_solver import SurfaceRegionPolicy

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reference_pressure_pa.*finite"):
                    SurfaceRegionPolicy(
                        region_id="face-a",
                        fluid_side_normal_sign=1.0,
                        reference_pressure_pa=value,
                    )

    def test_finite_reference_pressures_are_accepted(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            SurfaceRegion,
            SurfaceRegionPolicy,
        )

        region = SurfaceRegion(region_id="face-a", reference_pressure_pa=-12.5)
        policy = SurfaceRegionPolicy(
            region_id="face-a",
            fluid_side_normal_sign=-1.0,
            reference_pressure_pa=42.0,
        )

        self.assertEqual(region.reference_pressure_pa, -12.5)
        self.assertEqual(policy.reference_pressure_pa, 42.0)


class GenericFsiDiscreteCountValidationTests(unittest.TestCase):
    def test_surface_region_marker_count_rejects_bool_float_and_string(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import SurfaceRegion

        for value in (False, True, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "marker_count.*integer"):
                    SurfaceRegion(region_id="face-a", marker_count=value)

    def test_surface_region_marker_count_rejects_negative_integer(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import SurfaceRegion

        with self.assertRaisesRegex(ValueError, "marker_count.*non-negative"):
            SurfaceRegion(region_id="face-a", marker_count=-1)

    def test_surface_region_marker_count_accepts_nonnegative_integers(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import SurfaceRegion

        self.assertEqual(SurfaceRegion(region_id="empty", marker_count=0).marker_count, 0)
        self.assertEqual(SurfaceRegion(region_id="seeded", marker_count=7).marker_count, 7)

    def test_pressure_fallback_count_rejects_bool_float_and_string(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            PressureSamplePairProvider,
            PressureSamplingConfig,
        )

        provider = PressureSamplePairProvider(mode="runtime_anchored_cell_pair")
        for value in (False, True, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "sample_pair_fallback_count_max.*integer"
                ):
                    PressureSamplingConfig(
                        pair_provider=provider,
                        sample_pair_fallback_count_max=value,
                    )

    def test_pressure_fallback_count_rejects_negative_integer(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            PressureSamplePairProvider,
            PressureSamplingConfig,
        )

        provider = PressureSamplePairProvider(mode="runtime_anchored_cell_pair")
        with self.assertRaisesRegex(
            ValueError, "sample_pair_fallback_count_max.*non-negative"
        ):
            PressureSamplingConfig(
                pair_provider=provider,
                sample_pair_fallback_count_max=-1,
            )

    def test_pressure_fallback_count_accepts_nonnegative_integers(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            PressureSamplePairProvider,
            PressureSamplingConfig,
        )

        provider = PressureSamplePairProvider(mode="runtime_anchored_cell_pair")
        zero = PressureSamplingConfig(
            pair_provider=provider,
            sample_pair_fallback_count_max=0,
        )
        positive = PressureSamplingConfig(
            pair_provider=provider,
            sample_pair_fallback_count_max=3,
        )

        self.assertEqual(zero.sample_pair_fallback_count_max, 0)
        self.assertEqual(positive.sample_pair_fallback_count_max, 3)
