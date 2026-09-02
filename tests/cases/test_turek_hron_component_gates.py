from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from benchmarks.official.solid_mpm_fsi_runner import _lame_parameters
from cases import turek_hron_fsi as turek


class _ArrayField:
    def __init__(self, value: np.ndarray | None = None) -> None:
        self.value = None if value is None else np.asarray(value).copy()

    def from_numpy(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()

    def to_numpy(self) -> np.ndarray:
        if self.value is None:
            raise AssertionError("test field was read before initialization")
        return self.value.copy()


class _SolidMacroSpy:
    def __init__(self, *, fail_at_step: int | None = None) -> None:
        self.fail_at_step = fail_at_step
        self.begin_calls = 0
        self.end_calls = 0
        self.abort_calls = 0
        self.step_calls: list[dict[str, object]] = []
        self.projection_calls = 0

    def begin_out_of_bounds_guard_batch(self) -> None:
        self.begin_calls += 1

    def step(self, **kwargs: object) -> None:
        self.step_calls.append(kwargs)
        if self.fail_at_step == len(self.step_calls):
            raise RuntimeError("synthetic solid step failure")

    def enforce_rest_x_plane(self) -> None:
        self.projection_calls += 1

    def end_out_of_bounds_guard_batch(self) -> object:
        self.end_calls += 1
        return {"guard": "closed"}

    def abort_out_of_bounds_guard_batch(self) -> None:
        self.abort_calls += 1


class TurekHronForceReportingTests(unittest.TestCase):
    def test_force_components_have_one_span_normalization_and_exact_total(self) -> None:
        fields = turek._force_reporting_per_span_fields(
            beam_force_n=(7.0, 3.0, -2.0),
            cylinder_pressure_force_n=(5.0, 3.0, -4.0),
            cylinder_viscous_force_n=(11.0, 0.5, -1.0),
            span_m=2.0,
        )

        self.assertEqual(fields["beam_drag_per_span_n_per_m"], 1.0)
        self.assertEqual(fields["beam_lift_per_span_n_per_m"], 1.5)
        self.assertEqual(fields["cylinder_form_drag_per_span_n_per_m"], 2.0)
        self.assertEqual(fields["cylinder_friction_drag_per_span_n_per_m"], 0.5)
        self.assertEqual(fields["cylinder_drag_per_span_n_per_m"], 2.5)
        self.assertEqual(fields["cylinder_lift_per_span_n_per_m"], 1.75)
        self.assertEqual(fields["total_drag_per_span_n_per_m"], 3.5)
        self.assertEqual(fields["total_lift_per_span_n_per_m"], 3.25)
        self.assertEqual(
            fields["total_drag_per_span_n_per_m"],
            fields["beam_drag_per_span_n_per_m"]
            + fields["cylinder_drag_per_span_n_per_m"],
        )
        self.assertEqual(
            fields["total_lift_per_span_n_per_m"],
            fields["beam_lift_per_span_n_per_m"]
            + fields["cylinder_lift_per_span_n_per_m"],
        )

    def test_force_reporting_rejects_nonphysical_internal_inputs(self) -> None:
        valid = ((0.0, 0.0, 0.0),) * 3
        for span_m in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(span_m=span_m):
                with self.assertRaises(ValueError):
                    turek._force_reporting_per_span_fields(
                        beam_force_n=valid[0],
                        cylinder_pressure_force_n=valid[1],
                        cylinder_viscous_force_n=valid[2],
                        span_m=span_m,
                    )
        with self.assertRaises(ValueError):
            turek._force_reporting_per_span_fields(
                beam_force_n=(0.0, math.nan, 0.0),
                cylinder_pressure_force_n=valid[1],
                cylinder_viscous_force_n=valid[2],
                span_m=1.0,
            )


class TurekHronSolidMacroStepTests(unittest.TestCase):
    def test_macro_step_consumes_full_time_and_preserves_substep_order(self) -> None:
        solid = _SolidMacroSpy()
        writes: list[int] = []
        report = turek._advance_turek_hron_solid_macro_step(
            solid=solid,
            dt_s=0.8,
            solid_substeps=4,
            mu_pa=0.5e6,
            lambda_pa=2.0e6,
            velocity_damping=1.0,
            constitutive_model="saint_venant_kirchhoff",
            enforce_plane_strain_x=True,
            particle_position_write_observer=lambda: writes.append(1),
        )

        self.assertEqual(report, {"guard": "closed"})
        self.assertEqual(solid.begin_calls, 1)
        self.assertEqual(solid.end_calls, 1)
        self.assertEqual(solid.abort_calls, 0)
        self.assertEqual(len(solid.step_calls), 4)
        self.assertEqual(solid.projection_calls, 4)
        self.assertEqual(len(writes), 8)
        self.assertAlmostEqual(
            sum(float(call["dt_s"]) for call in solid.step_calls), 0.8
        )
        for call in solid.step_calls:
            self.assertAlmostEqual(float(call["dt_s"]), 0.2)
            self.assertAlmostEqual(float(call["mu_pa"]), 0.5e6)
            self.assertAlmostEqual(float(call["lambda_pa"]), 2.0e6)
            self.assertEqual(call["velocity_damping"], 1.0)
            self.assertEqual(call["constitutive_model"], "saint_venant_kirchhoff")
            self.assertFalse(bool(call["read_report"]))

    def test_macro_step_aborts_guard_when_a_substep_fails(self) -> None:
        solid = _SolidMacroSpy(fail_at_step=2)

        with self.assertRaisesRegex(RuntimeError, "synthetic solid step failure"):
            turek._advance_turek_hron_solid_macro_step(
                solid=solid,
                dt_s=0.2,
                solid_substeps=4,
                mu_pa=1.0,
                lambda_pa=2.0,
                velocity_damping=1.0,
                constitutive_model="saint_venant_kirchhoff",
                enforce_plane_strain_x=False,
                particle_position_write_observer=lambda: None,
            )

        self.assertEqual(solid.begin_calls, 1)
        self.assertEqual(solid.end_calls, 0)
        self.assertEqual(solid.abort_calls, 1)


class TurekHronPresetComponentTests(unittest.TestCase):
    def test_preset_lame_and_plane_strain_contracts(self) -> None:
        expected = {
            "fsi1": (0.5e6, 2.0e6),
            "fsi2": (0.5e6, 2.0e6),
            "fsi3": (2.0e6, 8.0e6),
        }
        for name, builder in turek.PRESET_BUILDERS.items():
            with self.subTest(preset=name):
                config = builder()
                mu_pa, lambda_pa = _lame_parameters(config)
                self.assertAlmostEqual(mu_pa, expected[name][0])
                self.assertAlmostEqual(lambda_pa, expected[name][1])
                self.assertEqual(
                    config.solid_constitutive_model, "saint_venant_kirchhoff"
                )
                self.assertTrue(config.enforce_plane_strain_x)
                self.assertEqual(config.velocity_damping, 1.0)

    def test_builders_forward_fluid_and_solid_component_specs_without_taichi(
        self,
    ) -> None:
        fluid_specs: list[object] = []
        solid_constructors: list[dict[str, object]] = []
        solid_initializations: list[dict[str, object]] = []

        class FakeFluid:
            def __init__(self, spec: object, *, runtime: object) -> None:
                fluid_specs.append(spec)
                self.velocity = _ArrayField()
                self.velocity_prev = _ArrayField()
                self.pressure = _ArrayField()
                self.obstacle = _ArrayField()

            def clear_volume_source(self) -> None:
                pass

            def set_velocity_dirichlet_boundary_authority(self, _value: str) -> None:
                pass

        class FakeSolid:
            def __init__(self, **kwargs: object) -> None:
                solid_constructors.append(kwargs)
                self.particle_count = int(kwargs["particle_capacity"])
                self.rest_x = _ArrayField(
                    np.zeros((self.particle_count, 3), dtype=np.float32)
                )
                self.fixed_particle = _ArrayField()
                self.region_id = _ArrayField()
                self.surface_normal = _ArrayField()
                self.rest_surface_normal = _ArrayField()
                self.area_weight_m2 = _ArrayField()
                self.rest_area_weight_m2 = _ArrayField()
                self.external_force_n = _ArrayField()

            def initialize_box(self, **kwargs: object) -> None:
                solid_initializations.append(kwargs)

        with (
            mock.patch.object(turek, "CartesianFluidSolver", FakeFluid),
            mock.patch.object(turek, "NeoHookeanMpmState", FakeSolid),
        ):
            for builder in turek.PRESET_BUILDERS.values():
                config = builder()
                turek._build_fluid(config, runtime=object())
                turek._build_solid(config, runtime=object())

        for config, fluid_spec, solid_ctor, solid_init in zip(
            (turek.fsi1_config(), turek.fsi2_config(), turek.fsi3_config()),
            fluid_specs,
            solid_constructors,
            solid_initializations,
            strict=True,
        ):
            with self.subTest(grid_nodes=config.grid_nodes):
                bounds_min, bounds_max = turek._full_bounds(config)
                self.assertEqual(fluid_spec.bounds_min_m, bounds_min)
                self.assertEqual(fluid_spec.bounds_max_m, bounds_max)
                self.assertEqual(fluid_spec.grid_nodes, config.grid_nodes)
                self.assertEqual(fluid_spec.density_kgm3, config.fluid_density_kgm3)
                self.assertEqual(fluid_spec.viscosity_pa_s, config.fluid_viscosity_pa_s)
                self.assertEqual(fluid_spec.dt_s, config.dt_s)
                self.assertEqual(solid_ctor["bounds_min_m"], bounds_min)
                self.assertEqual(solid_ctor["bounds_max_m"], bounds_max)
                self.assertEqual(solid_ctor["grid_nodes"], config.grid_nodes)
                self.assertEqual(
                    solid_ctor["particle_capacity"],
                    math.prod(config.solid_particle_counts),
                )
                self.assertEqual(
                    solid_init["particle_counts"], config.solid_particle_counts
                )
                self.assertEqual(solid_init["density_kgm3"], config.solid_density_kgm3)

    def test_quasi_2d_grid_contract_changes_only_spanwise_resolution(self) -> None:
        nx4 = turek.fsi1_config(
            grid_nodes=(4, 48, 288),
            markers_per_side="auto",
            markers_per_tip="auto",
        )
        nx8 = turek.fsi1_config(
            grid_nodes=(8, 48, 288),
            markers_per_side="auto",
            markers_per_tip="auto",
        )

        self.assertEqual(turek._full_bounds(nx4), turek._full_bounds(nx8))
        self.assertEqual(turek.beam_box_solver_m(nx4), turek.beam_box_solver_m(nx8))
        self.assertEqual(nx4.solid_particle_counts[0], 1)
        self.assertEqual(nx8.solid_particle_counts[0], 1)
        dx4, dy4, dz4 = turek.fluid_cell_spacing_m(nx4)
        dx8, dy8, dz8 = turek.fluid_cell_spacing_m(nx8)
        self.assertAlmostEqual(dx8, 0.5 * dx4)
        self.assertEqual(dy8, dy4)
        self.assertEqual(dz8, dz4)
        mask4 = turek.build_cylinder_obstacle_mask(nx4)
        mask8 = turek.build_cylinder_obstacle_mask(nx8)
        np.testing.assert_array_equal(mask4[0], mask8[0])
        np.testing.assert_array_equal(mask4, np.broadcast_to(mask4[0], mask4.shape))
        np.testing.assert_array_equal(mask8, np.broadcast_to(mask8[0], mask8.shape))
        marker_layout4 = turek.build_marker_layout(nx4)
        marker_layout8 = turek.build_marker_layout(nx8)
        for component4, component8 in zip(marker_layout4, marker_layout8, strict=True):
            np.testing.assert_array_equal(
                np.asarray(component4), np.asarray(component8)
            )


if __name__ == "__main__":
    unittest.main()
