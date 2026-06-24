from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    TaichiRuntimeConfig,
    TriSurfaceRegionDiagnostics,
)
from simulation_core.projected_ibm import (
    ProjectedIbmRegionPairStepConfig,
    ProjectedIbmRegionPairStepReport,
    advance_projected_ibm_region_pair_fluid_step,
)


class _FakeFluid:
    dx = 0.01
    dy = 0.02
    dz = 0.03
    velocity = object()
    pressure = object()
    force = object()
    volume_source_s = object()
    pressure_interface_matrix_diagonal = object()
    pressure_interface_matrix_rhs = object()
    obstacle = object()
    velocity_constraint_sum = object()
    velocity_constraint_weight = object()

    def __init__(self) -> None:
        self.grid = SimpleNamespace(is_uniform=True)
        self.spec = SimpleNamespace(cell_volume_m3=6.0e-6)
        self.saved_pressure = SimpleNamespace(snapshot_index=0)
        self.fsi_pressure = SimpleNamespace(snapshot_index=0)
        self.predict_calls: list[float] = []
        self.predict_advection_schemes: list[str] = []
        self.clear_force_calls = 0
        self.clear_volume_source_calls = 0
        self.clear_pressure_interface_matrix_terms_calls = 0
        self.pressure_interface_matrix_terms_report_calls = 0
        self.clear_pressure_calls = 0
        self.snapshot_pressure_calls = 0
        self.apply_body_force_calls: list[float] = []
        self.project_calls: list[dict[str, object]] = []
        self.pressure_outlet_report_calls: list[float] = []
        self.clear_velocity_constraints_calls = 0
        self.apply_velocity_constraints_calls: list[float] = []
        self.apply_velocity_constraints_solid_mobility_ratio_calls: list[float] = []
        self.apply_velocity_constraints_read_report_calls: list[bool] = []
        self.apply_body_force_read_report_calls: list[bool] = []
        self.velocity_constraint_report_calls = 0

    def predict(self, *, dt_s: float, advection_scheme: str = "euler") -> None:
        self.predict_calls.append(dt_s)
        self.predict_advection_schemes.append(str(advection_scheme))

    def clear_force(self) -> None:
        self.clear_force_calls += 1

    def clear_volume_source(self) -> None:
        self.clear_volume_source_calls += 1

    def clear_pressure_interface_matrix_terms(self) -> None:
        self.clear_pressure_interface_matrix_terms_calls += 1

    def pressure_interface_matrix_terms_report(self) -> dict[str, float | int]:
        self.pressure_interface_matrix_terms_report_calls += 1
        return {
            "diagonal_integral": 5.5,
            "rhs_integral": -2.25,
            "max_abs_diagonal": 7.75,
            "active_cells": 9,
        }

    def clear_pressure(self) -> None:
        self.clear_pressure_calls += 1

    def snapshot_pressure(self) -> None:
        self.snapshot_pressure_calls += 1
        self.fsi_pressure = SimpleNamespace(snapshot_index=self.snapshot_pressure_calls)

    def apply_body_force(self, *, dt_s: float, read_report: bool = True) -> dict[str, float] | None:
        self.apply_body_force_calls.append(dt_s)
        self.apply_body_force_read_report_calls.append(bool(read_report))
        if not read_report:
            return None
        return {"dt_s": dt_s, "read_report": True}

    def clear_velocity_constraints(self) -> None:
        self.clear_velocity_constraints_calls += 1

    def apply_velocity_constraints(
        self,
        *,
        blend: float,
        solid_mobility_ratio: float = 0.0,
        read_report: bool = True,
    ) -> dict[str, float] | None:
        self.apply_velocity_constraints_calls.append(blend)
        self.apply_velocity_constraints_solid_mobility_ratio_calls.append(
            solid_mobility_ratio
        )
        self.apply_velocity_constraints_read_report_calls.append(bool(read_report))
        if not read_report:
            return None
        return {"blend": blend}

    def velocity_constraint_report(self) -> dict[str, bool]:
        self.velocity_constraint_report_calls += 1
        return {"after_project": True}

    def project(self, **kwargs: object) -> dict[str, float]:
        self.project_calls.append(kwargs)
        if kwargs.get("pressure_solver") == "fv_cg":
            self.last_project_cg_project_calls = 1
            self.last_project_cg_iterations_total = 7
            self.last_project_cg_iterations_max = 7
            self.last_project_cg_host_residual_checks = 2
            self.last_project_cg_converged_all = True
            self.last_project_cg_relative_residual_max = 2.5e-7
            self.last_project_cg_initial_relative_residual_max = 0.75
            self.last_project_cg_breakdown_count = 0
            self.last_project_cg_restart_count = 0
            self.last_project_cg_restart_count_measured = False
            self.last_project_cg_restart_policy = "not_implemented"
        else:
            self.last_project_cg_project_calls = 0
            self.last_project_cg_iterations_total = 0
            self.last_project_cg_iterations_max = 0
            self.last_project_cg_host_residual_checks = 0
            self.last_project_cg_converged_all = True
            self.last_project_cg_relative_residual_max = 0.0
            self.last_project_cg_initial_relative_residual_max = 0.0
            self.last_project_cg_breakdown_count = 0
            self.last_project_cg_restart_count = 0
            self.last_project_cg_restart_count_measured = False
            self.last_project_cg_restart_policy = "not_applicable_non_cg"
        if not bool(kwargs.get("read_report", True)):
            return {}
        report = {"l2": 0.25, "max_abs": 0.5}
        if kwargs.get("pressure_solver") == "fv_cg":
            report["cg_iterations_total"] = self.last_project_cg_iterations_total
        return report

    def pressure_outlet_fv_flux_report(self, *, dt_s: float) -> dict[str, float]:
        self.pressure_outlet_report_calls.append(dt_s)
        return {
            "source_volume_flux_m3s": 3.0e-6,
            "zmin_pressure_outlet_flux_m3s": 2.9e-6,
            "zmin_velocity_outlet_flux_m3s": 3.1e-6,
            "zmin_pressure_outlet_to_source_ratio": 0.9666666667,
            "zmin_velocity_outlet_to_source_ratio": 1.0333333333,
        }


class _FakeSurfaceDiagnostics:
    def __init__(self) -> None:
        self.force_args: list[tuple[object, ...]] = []
        self.force_calls: list[dict[str, object]] = []
        self.pressure_matrix_args: list[tuple[object, ...]] = []
        self.pressure_matrix_calls: list[dict[str, object]] = []
        self.post_projection_force_calls: list[dict[str, object]] = []
        self.velocity_constraint_calls: list[dict[str, object]] = []

    def spread_fsi_forces(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.force_args.append(args)
        self.force_calls.append(kwargs)
        return SimpleNamespace(
            kind="force",
            dt_s=kwargs["dt_s"],
            primary_fluid_force_n=(1.0, -2.0, 3.0),
            secondary_fluid_force_n=(-4.0, 5.0, -6.0),
        )

    def spread_pressure_interface_matrix_terms(self, *args: object, **kwargs: object) -> None:
        self.pressure_matrix_args.append(args)
        self.pressure_matrix_calls.append(kwargs)

    def diagnose_fsi_forces_from_fields(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.post_projection_force_calls.append(kwargs)
        return SimpleNamespace(
            kind="post_projection_force",
            dt_s=kwargs["dt_s"],
            primary_fluid_force_n=(7.0, -8.0, 9.0),
            secondary_fluid_force_n=(-10.0, 11.0, -12.0),
        )

    def spread_fsi_velocity_constraints(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.velocity_constraint_calls.append(kwargs)
        return {"kind": "constraint"}


def _config(**overrides: object) -> ProjectedIbmRegionPairStepConfig:
    values = {
        "primary_region_id": 7,
        "secondary_region_id": 8,
        "primary_velocity_mps": (0.0, 0.0, -0.02),
        "secondary_velocity_mps": (0.0, 0.0, 0.01),
        "dt_s": 0.12,
        "ibm_correction_iterations": 3,
        "projection_iterations": 5,
        "pressure_outlet_zmin": True,
        "velocity_constraint_blend": 0.0,
        "constraint_force_scale": 0.002,
        "density_kgm3": 1025.0,
        "viscosity_pa_s": 1.05e-3,
        "bounds_min_m": (0.0, 0.0, 0.0),
        "bounds_max_m": (1.0, 1.0, 1.0),
        "grid_nodes": (8, 8, 8),
        "fluid_substeps": 1,
    }
    values.update(overrides)
    return ProjectedIbmRegionPairStepConfig(**values)


class ProjectedIbmRegionPairStepTests(unittest.TestCase):
    def test_step_config_has_no_z_only_velocity_fields(self) -> None:
        field_names = {field.name for field in fields(ProjectedIbmRegionPairStepConfig)}

        self.assertIn("primary_velocity_mps", field_names)
        self.assertIn("secondary_velocity_mps", field_names)
        self.assertNotIn("primary_velocity_z_mps", field_names)
        self.assertNotIn("secondary_velocity_z_mps", field_names)
        self.assertNotIn("pressure_force_scale", field_names)
        self.assertNotIn("surface_pressure_pa", field_names)
        self.assertIn("viscosity_pa_s", field_names)
        self.assertIn("fluid_substeps", field_names)
        self.assertIn("fluid_advection_scheme", field_names)
        self.assertIn("divergence_cleanup_iterations", field_names)
        self.assertIn("divergence_cleanup_relaxation", field_names)
        self.assertIn("pressure_solver", field_names)
        self.assertIn("multigrid_cycles", field_names)
        self.assertIn("cg_tolerance", field_names)
        self.assertIn("cg_preconditioner", field_names)
        self.assertIn("read_full_report", field_names)
        self.assertIn("primary_interface_impedance_force_n", field_names)
        self.assertIn("secondary_interface_impedance_force_n", field_names)
        self.assertIn("primary_pressure_robin_impedance_ns_m", field_names)
        self.assertIn("secondary_pressure_robin_impedance_ns_m", field_names)
        self.assertIn("primary_pressure_robin_reference_pa", field_names)
        self.assertIn("secondary_pressure_robin_reference_pa", field_names)
        self.assertIn("primary_interface_area_m2", field_names)
        self.assertIn("secondary_interface_area_m2", field_names)
        self.assertIn("constraint_force_solid_mobility_ratio", field_names)
        self.assertIn("primary_constraint_force_solid_mobility_ratio", field_names)
        self.assertIn("secondary_constraint_force_solid_mobility_ratio", field_names)
        self.assertIn("velocity_target_solid_mobility_ratio", field_names)
        self.assertIn("primary_velocity_target_solid_mobility_ratio", field_names)
        self.assertIn("secondary_velocity_target_solid_mobility_ratio", field_names)
        self.assertIn("velocity_constraint_solid_mobility_ratio", field_names)

    def test_step_report_exposes_pressure_outlet_face_flux_diagnostics(self) -> None:
        field_names = {field.name for field in fields(ProjectedIbmRegionPairStepReport)}

        self.assertIn("pressure_outlet_report", field_names)
        self.assertIn("pressure_projection_cg_iterations_total", field_names)
        self.assertIn("pressure_projection_cg_restart_count_measured", field_names)
        self.assertIn("pressure_projection_cg_restart_policy", field_names)
        self.assertIn("fluid_advection_scheme", field_names)

    def test_step_config_rejects_non_3d_target_velocities_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_velocity_mps"):
            _config(primary_velocity_mps=(0.0, -0.02))

        with self.assertRaisesRegex(ValueError, "secondary_velocity_mps"):
            _config(secondary_velocity_mps=(0.01,))

    def test_step_config_rejects_nonfinite_vector_inputs_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_velocity_mps"):
            _config(primary_velocity_mps=(float("nan"), 0.0, -0.02))

        with self.assertRaisesRegex(ValueError, "secondary_interface_impedance_force_n"):
            _config(secondary_interface_impedance_force_n=(0.0, float("inf"), 0.0))

    def test_step_config_rejects_invalid_step_scalar_inputs_at_construction(self) -> None:
        invalid_overrides = (
            ("dt_s", 0.0, "dt_s"),
            ("dt_s", float("nan"), "dt_s"),
            ("fluid_substeps", 0, "fluid_substeps"),
            ("ibm_correction_iterations", 0, "ibm_correction_iterations"),
            ("projection_iterations", 0, "projection_iterations"),
            ("velocity_constraint_blend", -0.01, "velocity_constraint_blend"),
            ("velocity_constraint_blend", float("nan"), "velocity_constraint_blend"),
            (
                "constraint_force_solid_mobility_ratio",
                -0.01,
                "constraint_force_solid_mobility_ratio",
            ),
            (
                "constraint_force_solid_mobility_ratio",
                float("nan"),
                "constraint_force_solid_mobility_ratio",
            ),
            (
                "primary_constraint_force_solid_mobility_ratio",
                -0.01,
                "primary_constraint_force_solid_mobility_ratio",
            ),
            (
                "primary_constraint_force_solid_mobility_ratio",
                float("nan"),
                "primary_constraint_force_solid_mobility_ratio",
            ),
            (
                "secondary_constraint_force_solid_mobility_ratio",
                -0.01,
                "secondary_constraint_force_solid_mobility_ratio",
            ),
            (
                "secondary_constraint_force_solid_mobility_ratio",
                float("nan"),
                "secondary_constraint_force_solid_mobility_ratio",
            ),
            (
                "velocity_target_solid_mobility_ratio",
                -0.01,
                "velocity_target_solid_mobility_ratio",
            ),
            (
                "velocity_target_solid_mobility_ratio",
                float("nan"),
                "velocity_target_solid_mobility_ratio",
            ),
            (
                "primary_velocity_target_solid_mobility_ratio",
                -0.01,
                "primary_velocity_target_solid_mobility_ratio",
            ),
            (
                "primary_velocity_target_solid_mobility_ratio",
                float("nan"),
                "primary_velocity_target_solid_mobility_ratio",
            ),
            (
                "secondary_velocity_target_solid_mobility_ratio",
                -0.01,
                "secondary_velocity_target_solid_mobility_ratio",
            ),
            (
                "secondary_velocity_target_solid_mobility_ratio",
                float("nan"),
                "secondary_velocity_target_solid_mobility_ratio",
            ),
            (
                "velocity_constraint_solid_mobility_ratio",
                -0.01,
                "velocity_constraint_solid_mobility_ratio",
            ),
            (
                "velocity_constraint_solid_mobility_ratio",
                float("nan"),
                "velocity_constraint_solid_mobility_ratio",
            ),
            ("constraint_force_scale", float("inf"), "constraint_force_scale"),
            ("density_kgm3", 0.0, "density_kgm3"),
            ("viscosity_pa_s", -1.0e-3, "viscosity_pa_s"),
            ("divergence_cleanup_iterations", -1, "divergence_cleanup_iterations"),
            ("divergence_cleanup_relaxation", 1.01, "divergence_cleanup_relaxation"),
        )
        for field, value, pattern in invalid_overrides:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, pattern):
                    _config(**{field: value})

    def test_step_config_rejects_invalid_domain_inputs_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounds_max_m"):
            _config(bounds_min_m=(0.0, 0.0, 0.0), bounds_max_m=(1.0, 0.0, 1.0))

        with self.assertRaisesRegex(ValueError, "grid_nodes"):
            _config(grid_nodes=(16, 16, 0))

    def test_step_config_rejects_nonfinite_cg_tolerance_at_construction(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "cg_tolerance"):
                    _config(cg_tolerance=value)

    def test_step_config_rejects_invalid_interface_impedance_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_interface_impedance_force_n"):
            _config(primary_interface_impedance_force_n=(1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "primary_interface_area_m2"):
            _config(
                primary_interface_impedance_force_n=(0.0, 0.0, 1.0),
                primary_interface_area_m2=0.0,
            )

    def test_predict_runs_once_while_ibm_correction_repeats_projection(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(),
        )

        self.assertEqual(fluid.predict_calls, [0.12])
        self.assertEqual(fluid.predict_advection_schemes, ["euler"])
        self.assertEqual(fluid.clear_pressure_calls, 1)
        self.assertEqual(fluid.snapshot_pressure_calls, 1)
        self.assertEqual(fluid.clear_force_calls, 3)
        self.assertEqual(fluid.clear_volume_source_calls, 3)
        self.assertEqual(fluid.apply_body_force_calls, [0.04, 0.04, 0.04])
        self.assertEqual(len(fluid.project_calls), 3)
        self.assertEqual(fluid.apply_body_force_calls, [0.04, 0.04, 0.04])
        self.assertEqual(fluid.apply_body_force_read_report_calls, [False, False, True])
        self.assertEqual(
            [call["read_report"] for call in fluid.project_calls],
            [False, False, True],
        )
        self.assertEqual(fluid.pressure_outlet_report_calls, [0.04])
        self.assertEqual(len(diagnostics.force_calls), 3)
        self.assertEqual(
            [call.get("read_full_report", True) for call in diagnostics.force_calls],
            [False, False, True],
        )
        self.assertTrue(all(args[1] is fluid.fsi_pressure for args in diagnostics.force_args))
        self.assertTrue(all(call["grid_fields"] is fluid for call in diagnostics.force_calls))
        self.assertTrue(all(call["viscosity_pa_s"] == 1.05e-3 for call in diagnostics.force_calls))
        self.assertEqual(report.ibm_correction_iterations, 3)
        self.assertAlmostEqual(report.ibm_correction_dt_s, 0.04)
        self.assertEqual(report.fluid_substeps, 1)
        self.assertAlmostEqual(report.fluid_substep_dt_s, 0.12)
        self.assertEqual(report.fluid_advection_scheme, "euler")
        self.assertEqual(report.impulse_report, {"dt_s": 0.04, "read_report": True})
        self.assertEqual(report.divergence, {"l2": 0.25, "max_abs": 0.5})
        self.assertEqual(report.pressure_projection_cg_project_calls, 0)
        self.assertEqual(report.pressure_projection_cg_iterations_total, 0)
        self.assertEqual(
            report.pressure_outlet_report["zmin_velocity_outlet_to_source_ratio"],
            1.0333333333,
        )
        self.assertEqual(report.interface_reaction_target.primary_force_n, (-1.0, 2.0, -3.0))
        np.testing.assert_allclose(
            report.interface_reaction_target.secondary_force_n,
            (4.0, -5.0, 6.0),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertEqual(report.primary_interface_reaction_balance.residual_components_n, (0.0, 0.0, 0.0))
        self.assertEqual(report.secondary_interface_reaction_balance.residual_components_n, (0.0, 0.0, 0.0))
        self.assertEqual(report.primary_interface_reaction_balance.relative_error, 0.0)
        self.assertEqual(report.secondary_interface_reaction_balance.relative_error, 0.0)
        self.assertTrue(all(call["divergence_cleanup_iterations"] == 0 for call in fluid.project_calls))

    def test_fluid_advection_scheme_is_forwarded_to_predictor(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(fluid_advection_scheme="rk2", fluid_substeps=2),
        )

        self.assertEqual(fluid.predict_calls, [0.06, 0.06])
        self.assertEqual(fluid.predict_advection_schemes, ["rk2", "rk2"])
        self.assertEqual(report.fluid_advection_scheme, "rk2")

    def test_interface_impedance_force_is_forwarded_to_surface_force_pass(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                primary_interface_impedance_force_n=(0.0, 0.0, -2.0),
                secondary_interface_impedance_force_n=(0.0, 0.0, 3.0),
                primary_interface_area_m2=0.25,
                secondary_interface_area_m2=0.5,
            ),
        )

        self.assertTrue(
            all(
                call["primary_interface_impedance_force_n"] == (0.0, 0.0, -2.0)
                for call in diagnostics.force_calls
            )
        )
        self.assertTrue(
            all(
                call["secondary_interface_impedance_force_n"] == (0.0, 0.0, 3.0)
                for call in diagnostics.force_calls
            )
        )
        self.assertTrue(all(call["primary_interface_area_m2"] == 0.25 for call in diagnostics.force_calls))
        self.assertTrue(all(call["secondary_interface_area_m2"] == 0.5 for call in diagnostics.force_calls))

    def test_pressure_robin_matrix_terms_are_forwarded_to_surface_matrix_pass(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                primary_pressure_robin_impedance_ns_m=11.0,
                secondary_pressure_robin_impedance_ns_m=17.0,
                primary_pressure_robin_reference_pa=3.0,
                secondary_pressure_robin_reference_pa=-2.0,
                primary_interface_area_m2=0.25,
                secondary_interface_area_m2=0.5,
            ),
        )

        self.assertEqual(fluid.clear_pressure_interface_matrix_terms_calls, 3)
        self.assertEqual(fluid.pressure_interface_matrix_terms_report_calls, 3)
        self.assertEqual(len(diagnostics.pressure_matrix_calls), 3)
        self.assertTrue(
            all(args[0] is fluid.pressure_interface_matrix_diagonal for args in diagnostics.pressure_matrix_args)
        )
        self.assertTrue(
            all(args[1] is fluid.pressure_interface_matrix_rhs for args in diagnostics.pressure_matrix_args)
        )
        self.assertTrue(
            all(call["primary_pressure_robin_impedance_ns_m"] == 11.0 for call in diagnostics.pressure_matrix_calls)
        )
        self.assertTrue(
            all(call["secondary_pressure_robin_impedance_ns_m"] == 17.0 for call in diagnostics.pressure_matrix_calls)
        )
        self.assertTrue(
            all(call["primary_pressure_robin_reference_pa"] == 3.0 for call in diagnostics.pressure_matrix_calls)
        )
        self.assertTrue(
            all(call["secondary_pressure_robin_reference_pa"] == -2.0 for call in diagnostics.pressure_matrix_calls)
        )
        self.assertTrue(all(call["primary_interface_area_m2"] == 0.25 for call in diagnostics.pressure_matrix_calls))
        self.assertTrue(all(call["secondary_interface_area_m2"] == 0.5 for call in diagnostics.pressure_matrix_calls))
        self.assertAlmostEqual(report.pressure_interface_matrix_diagonal_integral, 5.5)
        self.assertAlmostEqual(report.pressure_interface_matrix_rhs_integral, -2.25)
        self.assertAlmostEqual(report.pressure_interface_matrix_max_abs_diagonal, 7.75)
        self.assertEqual(report.pressure_interface_matrix_active_cells, 9)

    def test_pressure_robin_matrix_target_uses_post_projection_pressure_forces(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                primary_pressure_robin_impedance_ns_m=11.0,
                secondary_pressure_robin_impedance_ns_m=17.0,
                primary_interface_area_m2=0.25,
                secondary_interface_area_m2=0.5,
            ),
        )

        self.assertEqual(len(diagnostics.force_calls), 3)
        self.assertEqual(len(diagnostics.post_projection_force_calls), 3)
        np.testing.assert_allclose(
            report.interface_reaction_target.primary_force_n,
            (-7.0, 8.0, -9.0),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            report.interface_reaction_target.secondary_force_n,
            (10.0, -11.0, 12.0),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(
            report.primary_interface_reaction_balance.residual_norm_n,
            0.0,
        )

    def test_constraint_force_solid_mobility_ratio_is_forwarded_to_surface_force_pass(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                constraint_force_solid_mobility_ratio=2.5,
                ibm_correction_iterations=2,
            ),
        )

        self.assertEqual(len(diagnostics.force_calls), 2)
        self.assertTrue(
            all(
                call["constraint_force_solid_mobility_ratio"] == 2.5
                for call in diagnostics.force_calls
            )
        )

    def test_region_constraint_force_solid_mobility_ratios_override_global_ratio(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                constraint_force_solid_mobility_ratio=0.25,
                primary_constraint_force_solid_mobility_ratio=2.0,
                secondary_constraint_force_solid_mobility_ratio=4.0,
                ibm_correction_iterations=2,
            ),
        )

        self.assertEqual(len(diagnostics.force_calls), 2)
        self.assertTrue(
            all(
                call["primary_constraint_force_solid_mobility_ratio"] == 2.0
                for call in diagnostics.force_calls
            )
        )
        self.assertTrue(
            all(
                call["secondary_constraint_force_solid_mobility_ratio"] == 4.0
                for call in diagnostics.force_calls
            )
        )

    def test_region_velocity_target_solid_mobility_ratios_override_global_ratio(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                velocity_target_solid_mobility_ratio=0.25,
                primary_velocity_target_solid_mobility_ratio=2.0,
                secondary_velocity_target_solid_mobility_ratio=4.0,
                ibm_correction_iterations=2,
            ),
        )

        self.assertEqual(len(diagnostics.force_calls), 2)
        self.assertTrue(
            all(
                call["primary_velocity_target_solid_mobility_ratio"] == 2.0
                for call in diagnostics.force_calls
            )
        )
        self.assertTrue(
            all(
                call["secondary_velocity_target_solid_mobility_ratio"] == 4.0
                for call in diagnostics.force_calls
            )
        )

    def test_trial_step_can_skip_full_reports_without_skipping_physics(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(read_full_report=False),
        )

        self.assertEqual(fluid.predict_calls, [0.12])
        self.assertEqual(fluid.clear_force_calls, 3)
        self.assertEqual(fluid.clear_volume_source_calls, 3)
        self.assertEqual(fluid.apply_body_force_calls, [0.04, 0.04, 0.04])
        self.assertEqual(fluid.apply_body_force_read_report_calls, [False, False, False])
        self.assertEqual(
            [call["read_report"] for call in fluid.project_calls],
            [False, False, False],
        )
        self.assertEqual(fluid.pressure_outlet_report_calls, [])
        self.assertEqual(
            [call.get("read_full_report", True) for call in diagnostics.force_calls],
            [False, False, False],
        )
        self.assertEqual(report.divergence, {})
        self.assertEqual(report.pressure_outlet_report, {})
        self.assertEqual(report.pressure_projection_cg_project_calls, 0)
        self.assertEqual(report.pressure_projection_cg_iterations_total, 0)
        self.assertIsNone(report.impulse_report)
        self.assertEqual(report.interface_reaction_target.primary_force_n, (-1.0, 2.0, -3.0))
        self.assertEqual(report.primary_equivalent_fluid_force_n, (1.0, -2.0, 3.0))

    def test_fv_cg_iteration_stats_accumulate_without_full_reports(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(pressure_solver="fv_cg", read_full_report=False),
        )

        self.assertEqual(
            [call["read_report"] for call in fluid.project_calls],
            [False, False, False],
        )
        self.assertEqual(report.divergence, {})
        self.assertEqual(report.pressure_projection_cg_project_calls, 3)
        self.assertEqual(report.pressure_projection_cg_iterations_total, 21)
        self.assertEqual(report.pressure_projection_cg_iterations_max, 7)
        self.assertEqual(report.pressure_projection_cg_host_residual_checks, 6)
        self.assertEqual(report.pressure_projection_cg_restart_count, 0)
        self.assertFalse(report.pressure_projection_cg_restart_count_measured)
        self.assertEqual(report.pressure_projection_cg_restart_policy, "not_implemented")
        self.assertTrue(report.pressure_projection_cg_converged_all)
        self.assertAlmostEqual(report.pressure_projection_cg_max_relative_residual, 2.5e-7)
        self.assertAlmostEqual(
            report.pressure_projection_cg_max_initial_relative_residual,
            0.75,
        )

    def test_trial_step_surfaces_lightweight_invalid_probe_diagnostics(self) -> None:
        class InvalidProbeDiagnostics(_FakeSurfaceDiagnostics):
            def spread_fsi_forces(self, *args: object, **kwargs: object) -> dict[str, object]:
                self.force_args.append(args)
                self.force_calls.append(kwargs)
                return SimpleNamespace(
                    primary_fluid_force_n=(1.0, -2.0, 3.0),
                    secondary_fluid_force_n=(-4.0, 5.0, -6.0),
                    force_sample_count=0,
                    force_invalid_probe_count=2,
                    force_valid_probe_count=0,
                    force_valid_probe_fraction=0.0,
                    invalid_probe_count=2,
                    valid_probe_fraction=0.0,
                    invalid_probe_area_m2=0.75,
                    invalid_probe_volume_source_m3s=-0.02,
                )

        fluid = _FakeFluid()
        diagnostics = InvalidProbeDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(read_full_report=False),
        )

        self.assertEqual(
            [call.get("read_full_report", True) for call in diagnostics.force_calls],
            [False, False, False],
        )
        self.assertEqual(report.force_report.force_sample_count, 0)
        self.assertEqual(report.force_report.force_invalid_probe_count, 2)
        self.assertEqual(report.force_report.force_valid_probe_count, 0)
        self.assertAlmostEqual(report.force_report.force_valid_probe_fraction, 0.0)
        self.assertEqual(report.force_report.invalid_probe_count, 2)
        self.assertAlmostEqual(report.force_report.valid_probe_fraction, 0.0)
        self.assertAlmostEqual(report.force_report.invalid_probe_area_m2, 0.75)
        self.assertAlmostEqual(report.force_report.invalid_probe_volume_source_m3s, -0.02)

    def test_interface_reaction_uses_time_averaged_force_over_ibm_corrections(self) -> None:
        class VaryingForceDiagnostics(_FakeSurfaceDiagnostics):
            def spread_fsi_forces(self, *args: object, **kwargs: object) -> dict[str, object]:
                self.force_calls.append(kwargs)
                force_index = len(self.force_calls)
                primary_force = (float(force_index), 0.0, 0.0)
                secondary_force = (0.0, -2.0 * float(force_index), 0.0)
                return SimpleNamespace(
                    kind="force",
                    dt_s=kwargs["dt_s"],
                    primary_fluid_force_n=primary_force,
                    secondary_fluid_force_n=secondary_force,
                )

        fluid = _FakeFluid()
        diagnostics = VaryingForceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(ibm_correction_iterations=3),
        )

        self.assertEqual(report.force_report.primary_fluid_force_n, (3.0, 0.0, 0.0))
        self.assertEqual(report.primary_equivalent_fluid_force_n, (2.0, 0.0, 0.0))
        self.assertEqual(report.secondary_equivalent_fluid_force_n, (0.0, -4.0, 0.0))
        self.assertEqual(report.interface_reaction_target.primary_force_n, (-2.0, -0.0, -0.0))
        self.assertEqual(report.interface_reaction_target.secondary_force_n, (-0.0, 4.0, -0.0))
        self.assertEqual(report.primary_interface_reaction_balance.residual_components_n, (0.0, 0.0, 0.0))
        self.assertEqual(report.secondary_interface_reaction_balance.residual_components_n, (0.0, 0.0, 0.0))

    def test_velocity_constraint_impulse_contributes_to_interface_reaction_target(self) -> None:
        class VelocityImpulseFluid(_FakeFluid):
            def __init__(self) -> None:
                super().__init__()
                self.primary_velocity_impulse = [0.0, 0.0, 0.0]
                self.secondary_velocity_impulse = [0.0, 0.0, 0.0]
                self.velocity_constraint_impulse_report_calls = 0

            def reset_velocity_constraint_impulse_accumulator(self) -> None:
                self.primary_velocity_impulse = [0.0, 0.0, 0.0]
                self.secondary_velocity_impulse = [0.0, 0.0, 0.0]

            def apply_velocity_constraints(
                self,
                *,
                blend: float,
                solid_mobility_ratio: float = 0.0,
                read_report: bool = True,
            ) -> dict[str, float] | None:
                result = super().apply_velocity_constraints(
                    blend=blend,
                    solid_mobility_ratio=solid_mobility_ratio,
                    read_report=read_report,
                )
                pass_index = len(self.apply_velocity_constraints_calls)
                self.primary_velocity_impulse[2] += 0.01 * float(pass_index)
                self.secondary_velocity_impulse[2] += -0.02 * float(pass_index)
                return result

            def velocity_constraint_impulse_report(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
                self.velocity_constraint_impulse_report_calls += 1
                return (
                    tuple(self.primary_velocity_impulse),
                    tuple(self.secondary_velocity_impulse),
                )

        fluid = VelocityImpulseFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                velocity_constraint_blend=0.5,
                ibm_correction_iterations=3,
                read_full_report=False,
            ),
        )

        self.assertEqual(fluid.velocity_constraint_impulse_report_calls, 1)
        self.assertEqual(report.primary_velocity_constraint_impulse_n_s, (0.0, 0.0, 0.06))
        self.assertEqual(report.secondary_velocity_constraint_impulse_n_s, (0.0, 0.0, -0.12))
        self.assertEqual(
            report.primary_velocity_constraint_equivalent_fluid_force_n,
            (0.0, 0.0, 0.5),
        )
        self.assertEqual(
            report.secondary_velocity_constraint_equivalent_fluid_force_n,
            (0.0, 0.0, -1.0),
        )
        self.assertEqual(
            report.interface_reaction_target.primary_force_n,
            (-1.0, 2.0, -3.5),
        )
        np.testing.assert_allclose(
            report.interface_reaction_target.secondary_force_n,
            (4.0, -5.0, 7.0),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_device_impulse_accumulator_skips_nonfinal_force_pair_reads(self) -> None:
        class AccumulatingDiagnostics(_FakeSurfaceDiagnostics):
            def __init__(self) -> None:
                super().__init__()
                self.primary_impulse = [0.0, 0.0, 0.0]
                self.secondary_impulse = [0.0, 0.0, 0.0]
                self.last_primary_force = (0.0, 0.0, 0.0)
                self.last_secondary_force = (0.0, 0.0, 0.0)
                self.force_impulse_report_calls = 0

            def reset_force_impulse_accumulator(self) -> None:
                self.primary_impulse = [0.0, 0.0, 0.0]
                self.secondary_impulse = [0.0, 0.0, 0.0]

            def accumulate_force_impulse(self, dt_s: float) -> None:
                for component in range(3):
                    self.primary_impulse[component] += self.last_primary_force[component] * dt_s
                    self.secondary_impulse[component] += self.last_secondary_force[component] * dt_s

            def force_impulse_report(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
                self.force_impulse_report_calls += 1
                return tuple(self.primary_impulse), tuple(self.secondary_impulse)

            def spread_fsi_forces(self, *args: object, **kwargs: object) -> dict[str, object] | None:
                self.force_args.append(args)
                self.force_calls.append(kwargs)
                force_index = float(len(self.force_calls))
                self.last_primary_force = (force_index, 0.0, 0.0)
                self.last_secondary_force = (0.0, -2.0 * force_index, 0.0)
                if not bool(kwargs.get("read_force_pair_report", True)) and not bool(
                    kwargs.get("read_full_report", True)
                ):
                    return None
                return SimpleNamespace(
                    primary_fluid_force_n=self.last_primary_force,
                    secondary_fluid_force_n=self.last_secondary_force,
                )

        fluid = _FakeFluid()
        diagnostics = AccumulatingDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(ibm_correction_iterations=3, read_full_report=False),
        )

        self.assertEqual(
            [call["read_force_pair_report"] for call in diagnostics.force_calls],
            [False, False, True],
        )
        self.assertEqual(diagnostics.force_impulse_report_calls, 1)
        self.assertEqual(report.force_report.primary_fluid_force_n, (3.0, 0.0, 0.0))
        self.assertEqual(report.primary_equivalent_fluid_force_n, (2.0, 0.0, 0.0))
        self.assertEqual(report.secondary_equivalent_fluid_force_n, (0.0, -4.0, 0.0))

    def test_divergence_cleanup_options_are_forwarded_to_projection(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                ibm_correction_iterations=2,
                divergence_cleanup_iterations=5,
                divergence_cleanup_relaxation=0.4,
            ),
        )

        self.assertEqual(len(fluid.project_calls), 2)
        self.assertTrue(all(call["divergence_cleanup_iterations"] == 5 for call in fluid.project_calls))
        self.assertTrue(all(call["divergence_cleanup_relaxation"] == 0.4 for call in fluid.project_calls))

    def test_pressure_solver_options_are_forwarded_to_projection(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                ibm_correction_iterations=2,
                pressure_solver="fv_cg",
                cg_tolerance=1.0e-8,
                cg_preconditioner="jacobi",
            ),
        )

        self.assertEqual(len(fluid.project_calls), 2)
        self.assertTrue(all(call["pressure_solver"] == "fv_cg" for call in fluid.project_calls))
        self.assertTrue(all(call["cg_tolerance"] == 1.0e-8 for call in fluid.project_calls))
        self.assertTrue(all(call["cg_preconditioner"] == "jacobi" for call in fluid.project_calls))

    def test_step_config_rejects_bad_pressure_solver_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported pressure_solver"):
            _config(pressure_solver="bad")
        with self.assertRaisesRegex(ValueError, "multigrid_cycles must be positive"):
            _config(pressure_solver="fv_multigrid", multigrid_cycles=0)
        with self.assertRaisesRegex(ValueError, "unsupported cg_preconditioner"):
            _config(pressure_solver="fv_cg", cg_preconditioner="bad")
        with self.assertRaisesRegex(ValueError, "fluid_advection_scheme"):
            _config(fluid_advection_scheme="bad")

    def test_fluid_substeps_repeat_full_fluid_advance_without_scaling_reaction(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(fluid_substeps=2, ibm_correction_iterations=3),
        )

        self.assertEqual(fluid.predict_calls, [0.06, 0.06])
        self.assertEqual(fluid.snapshot_pressure_calls, 1)
        self.assertEqual(fluid.clear_pressure_calls, 1)
        self.assertEqual(fluid.clear_force_calls, 6)
        self.assertEqual(fluid.clear_volume_source_calls, 6)
        self.assertEqual(fluid.apply_body_force_calls, [0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        self.assertEqual(len(fluid.project_calls), 6)
        self.assertTrue(all(call["reset_pressure"] for call in fluid.project_calls))
        self.assertTrue(all(call["dt_s"] == 0.02 for call in diagnostics.force_calls))
        self.assertEqual(
            [getattr(args[1], "snapshot_index", None) for args in diagnostics.force_args],
            [1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(report.fluid_substeps, 2)
        self.assertAlmostEqual(report.fluid_substep_dt_s, 0.06)
        self.assertAlmostEqual(report.ibm_correction_dt_s, 0.02)
        np.testing.assert_allclose(
            report.primary_equivalent_fluid_force_n,
            (1.0, -2.0, 3.0),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            report.secondary_equivalent_fluid_force_n,
            (-4.0, 5.0, -6.0),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_fluid_substeps_refresh_pressure_only_for_continuous_pressure_projection(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                fluid_substeps=2,
                ibm_correction_iterations=3,
                reset_pressure_each_projection=False,
            ),
        )

        self.assertEqual(fluid.predict_calls, [0.06, 0.06])
        self.assertEqual(fluid.snapshot_pressure_calls, 2)
        self.assertEqual(fluid.clear_pressure_calls, 0)
        self.assertEqual(len(fluid.project_calls), 6)
        self.assertTrue(all(not call["reset_pressure"] for call in fluid.project_calls))
        self.assertEqual(
            [getattr(args[1], "snapshot_index", None) for args in diagnostics.force_args],
            [1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(report.fluid_substeps, 2)
        self.assertAlmostEqual(report.fluid_substep_dt_s, 0.06)
        self.assertAlmostEqual(report.ibm_correction_dt_s, 0.02)

    def test_velocity_constraints_are_preserved_through_projection_when_enabled(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(velocity_constraint_blend=0.6, ibm_correction_iterations=2),
        )

        self.assertEqual(fluid.clear_velocity_constraints_calls, 2)
        self.assertEqual(fluid.apply_velocity_constraints_calls, [0.6, 0.6])
        self.assertEqual(fluid.apply_velocity_constraints_read_report_calls, [False, False])
        self.assertEqual(fluid.velocity_constraint_report_calls, 1)
        self.assertEqual(len(diagnostics.velocity_constraint_calls), 2)
        self.assertEqual(
            [call.get("read_full_report", True) for call in diagnostics.velocity_constraint_calls],
            [False, True],
        )
        self.assertTrue(
            all(call["grid_fields"] is fluid for call in diagnostics.velocity_constraint_calls)
        )
        self.assertTrue(all(call["preserve_velocity_constraints"] for call in fluid.project_calls))
        self.assertEqual(report.velocity_constraint_report, {"after_project": True})

    def test_velocity_constraint_solid_mobility_ratio_reaches_constraint_and_projection(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                velocity_constraint_blend=0.6,
                velocity_constraint_solid_mobility_ratio=2.5,
                ibm_correction_iterations=2,
            ),
        )

        self.assertEqual(
            fluid.apply_velocity_constraints_solid_mobility_ratio_calls,
            [2.5, 2.5],
        )
        self.assertTrue(
            all(
                call["velocity_constraint_solid_mobility_ratio"] == 2.5
                for call in fluid.project_calls
            )
        )

    def test_nonuniform_grid_uses_one_probe_distance_for_force_and_constraints(self) -> None:
        fluid = _FakeFluid()
        fluid.grid = SimpleNamespace(is_uniform=False)
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(velocity_constraint_blend=0.6, ibm_correction_iterations=2),
        )

        self.assertTrue(all(call["probe_distance_m"] == 0.01 for call in diagnostics.force_calls))
        self.assertTrue(
            all("control_volume_thickness_m" not in call for call in diagnostics.force_calls)
        )
        self.assertTrue(
            all(call["probe_distance_m"] == 0.01 for call in diagnostics.velocity_constraint_calls)
        )

    def test_full_3d_target_velocities_are_forwarded_to_surface_diagnostics(self) -> None:
        fluid = _FakeFluid()
        diagnostics = _FakeSurfaceDiagnostics()

        advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                primary_velocity_mps=(0.11, -0.04, 0.03),
                secondary_velocity_mps=(-0.02, 0.05, -0.07),
                velocity_constraint_blend=0.5,
                ibm_correction_iterations=1,
            ),
        )

        self.assertEqual(
            diagnostics.force_calls[0]["primary_velocity_mps"],
            (0.11, -0.04, 0.03),
        )
        self.assertEqual(
            diagnostics.force_calls[0]["secondary_velocity_mps"],
            (-0.02, 0.05, -0.07),
        )
        self.assertEqual(
            diagnostics.velocity_constraint_calls[0]["primary_velocity_mps"],
            (0.11, -0.04, 0.03),
        )
        self.assertEqual(
            diagnostics.velocity_constraint_calls[0]["secondary_velocity_mps"],
            (-0.02, 0.05, -0.07),
        )
        self.assertNotIn("main_velocity_z_mps", diagnostics.force_calls[0])
        self.assertNotIn("tail_velocity_z_mps", diagnostics.force_calls[0])
        self.assertNotIn("primary_velocity_z_mps", diagnostics.force_calls[0])
        self.assertNotIn("secondary_velocity_z_mps", diagnostics.force_calls[0])
        self.assertNotIn("pressure_force_scale", diagnostics.force_calls[0])

    def test_reaction_target_balances_actual_spread_fluid_force_not_pressure_diagnostic(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        fluid.pressure.from_numpy(np.full(fluid.spec.grid_nodes, 10.0, dtype=np.float32))
        diagnostics.load_faces(
            centroid_m=np.asarray(
                [
                    [0.4, 0.5, 0.5],
                    [0.6, 0.5, 0.5],
                ],
                dtype=np.float32,
            ),
            normal=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.asarray([2.0, 3.0], dtype=np.float32),
            region_id=np.asarray([7, 8], dtype=np.int32),
        )

        report = advance_projected_ibm_region_pair_fluid_step(
            fluid,
            diagnostics,
            _config(
                primary_velocity_mps=(0.0, 0.0, 0.2),
                secondary_velocity_mps=(0.0, 0.0, -0.1),
                ibm_correction_iterations=1,
                velocity_constraint_blend=0.0,
                constraint_force_scale=0.5,
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(16, 16, 16),
            ),
        )
        force_report = report.force_report

        np.testing.assert_allclose(
            report.interface_reaction_target.primary_force_n,
            -np.asarray(force_report.primary_fluid_force_n),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            report.interface_reaction_target.secondary_force_n,
            -np.asarray(force_report.secondary_fluid_force_n),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        self.assertAlmostEqual(report.primary_interface_reaction_balance.relative_error, 0.0)
        self.assertAlmostEqual(report.secondary_interface_reaction_balance.relative_error, 0.0)
        self.assertGreater(float(np.linalg.norm(force_report.primary_constraint_force_n)), 0.0)
        self.assertGreater(float(np.linalg.norm(force_report.secondary_constraint_force_n)), 0.0)
        self.assertGreater(
            float(
                np.linalg.norm(
                    np.asarray(report.interface_reaction_target.primary_force_n)
                    - np.asarray(force_report.primary_pressure_traction_force_n)
                )
            ),
            1.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
