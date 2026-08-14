from __future__ import annotations

import ast
import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from cases import turek_hron_fsi as turek
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiCouplingReport,
    FsiStepContext,
    solve_fsi_step,
)
from simulation_core.coupling.hibm_mpm import capture_marker_interface_state


class _FakeField:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()

    def to_numpy(self) -> np.ndarray:
        return self.value.copy()

    def from_numpy(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()


class _FakeRestorable:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.saved_value: float | None = None
        self.save_calls = 0
        self.restore_calls = 0

    def save_state(self) -> None:
        self.save_calls += 1
        self.saved_value = self.value

    def restore_state(self) -> None:
        if self.saved_value is None:
            raise AssertionError("restore before save")
        self.restore_calls += 1
        self.value = self.saved_value


class TurekGenericFsiStepArchitectureTests(unittest.TestCase):
    def test_case_has_no_local_coupling_state_machine_or_single_pass(self) -> None:
        module_source = inspect.getsource(turek)
        run_source = inspect.getsource(turek.run_turek_hron_fsi)
        tree = ast.parse(module_source)
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertIn("solve_fsi_runtime(", run_source)
        self.assertIn("FsiSolverConfig(", run_source)
        self.assertNotIn("solve_fsi_step(", run_source)
        self.assertNotIn("for step_index in", run_source)
        self.assertNotIn("for coupling_iteration", run_source)
        self.assertNotIn("strong_coupling_enabled", run_source)
        self.assertNotIn("unmeasured_single_pass", module_source)
        self.assertNotIn("legacy explicit loose single pass", module_source.lower())
        self.assertNotIn("fsi_aitken", module_source.lower())
        self.assertNotIn("--fsi-coupling-accelerator", module_source)
        self.assertNotIn("_aitken_relaxation_factor", function_names)
        self.assertNotIn("_globalized_iqn_velocity_guess", function_names)
        self.assertNotIn("_physical_context", function_names)
        self.assertNotIn(
            "_run_fsi_coupling_state_machine_operation",
            function_names,
        )
        self.assertFalse(
            any(name.startswith("_iqn_ils_") for name in function_names),
            sorted(name for name in function_names if name.startswith("_iqn_ils_")),
        )

    def test_all_presets_require_the_generic_iterative_path(self) -> None:
        for builder in (turek.fsi1_config, turek.fsi2_config, turek.fsi3_config):
            with self.subTest(builder=builder.__name__):
                config = builder()
                self.assertGreaterEqual(config.fsi_coupling_iterations, 2)
                self.assertFalse(hasattr(config, "fsi_coupling_accelerator"))
                self.assertGreater(config.fsi_coupling_initial_relaxation, 0.0)

    def test_shared_core_owns_substep_time_scaling_and_resume_offset(self) -> None:
        from simulation_core.coupling.hibm_mpm import core as hibm_core
        from simulation_core.drivers import generic_fsi_solver

        run_source = inspect.getsource(turek.run_turek_hron_fsi)
        hibm_source = inspect.getsource(
            hibm_core.assemble_hibm_mpm_sharp_fluid_to_mpm_loads
        )
        generic_source = inspect.getsource(generic_fsi_solver.solve_fsi_runtime)

        self.assertIn("pressure_neumann_dt_s=float(config.dt_s)", run_source)
        self.assertIn(
            "pressure_neumann_dt = pressure_neumann_dt / float(substeps)",
            hibm_source,
        )
        self.assertIn("completed_step_offset=int(completed_step_offset)", run_source)
        self.assertIn("completed_step_offset + local_step_index", generic_source)


class TurekGenericFsiStepRuntimeTests(unittest.TestCase):
    def test_every_trial_restores_one_fluid_solid_marker_base(self) -> None:
        fluid = _FakeRestorable(10.0)
        solid = _FakeRestorable(20.0)
        marker_base = {
            "x_gamma_m": np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
            ),
            "v_gamma_mps": np.asarray(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
            ),
            "pressure_probe_origin_m": np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
            ),
            "n_gamma": np.asarray(
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32
            ),
            "A_gamma_m2": np.asarray([0.5, 0.75], dtype=np.float32),
        }
        markers = SimpleNamespace(
            marker_count=2,
            projection_vertex_count=2,
            projection_triangle_count=0,
            projection_segment_count=0,
            _open_ribbon_tip_cap_binding=None,
            _begin_marker_geometry_write=lambda: None,
            **{name: _FakeField(value) for name, value in marker_base.items()},
        )
        gradient_base = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        boundary = SimpleNamespace(
            marker_pressure_neumann_gradient_field=_FakeField(gradient_base)
        )
        prepared_marker = {
            name: value.copy() for name, value in marker_base.items()
        }
        prepared_marker["x_gamma_m"] += 0.5
        prepared_marker["pressure_probe_origin_m"] += 0.75
        prepared_marker["v_gamma_mps"] += 0.125
        prepared_marker["n_gamma"] = -prepared_marker["n_gamma"]
        prepared_marker["A_gamma_m2"] *= 1.5
        prepared_gradient = gradient_base + 10.0
        trial_inputs: list[dict[str, object]] = []
        reports = [object(), object()]

        def prepare_step(_context: FsiStepContext) -> None:
            for name, value in prepared_marker.items():
                getattr(markers, name).from_numpy(value)
            boundary.marker_pressure_neumann_gradient_field.from_numpy(
                prepared_gradient
            )

        def advance_trial(_context: FsiStepContext, _trial_index: int) -> object:
            trial_inputs.append(
                {
                    "fluid": fluid.value,
                    "solid": solid.value,
                    "marker": capture_marker_interface_state(markers),
                    "gradient": boundary.marker_pressure_neumann_gradient_field.to_numpy(),
                }
            )
            guess = markers.v_gamma_mps.to_numpy()
            candidate = 0.25 * guess + 0.75
            fluid.value = -100.0
            solid.value = -200.0
            markers.v_gamma_mps.from_numpy(candidate)
            markers.x_gamma_m.from_numpy(np.full_like(marker_base["x_gamma_m"], 99.0))
            markers.n_gamma.from_numpy(np.full_like(marker_base["n_gamma"], 7.0))
            markers.A_gamma_m2.from_numpy(
                np.full_like(marker_base["A_gamma_m2"], 8.0)
            )
            boundary.marker_pressure_neumann_gradient_field.from_numpy(
                np.full_like(gradient_base, 9.0)
            )
            return reports[len(trial_inputs) - 1]

        boundary_refreshes: list[int] = []
        runtime = turek._TurekHronFsiRuntime(
            fluid=fluid,
            solid=solid,
            markers=markers,
            boundary=boundary,
            advance_trial=advance_trial,
            prepare_step=prepare_step,
            restore_case_boundaries=lambda _context: boundary_refreshes.append(1),
            commit_case_step=lambda _context, _trial, _coupling: {},
            finalize_case_run=lambda: {},
        )
        context = FsiStepContext(step=1, step_index=0, time_s=0.25, dt_s=0.25)

        initial = runtime.begin_step(context)
        first_guess = initial + 1.0
        first = runtime.evaluate_trial(context, first_guess)
        fluid.value = 1234.0
        solid.value = 5678.0
        second_guess = initial - 0.5
        second = runtime.evaluate_trial(context, second_guess)

        self.assertEqual(fluid.save_calls, 1)
        self.assertEqual(solid.save_calls, 1)
        self.assertEqual(fluid.restore_calls, 2)
        self.assertEqual(solid.restore_calls, 2)
        self.assertEqual(len(boundary_refreshes), 2)
        for recorded, expected_guess in zip(
            trial_inputs,
            (first_guess, second_guess),
            strict=True,
        ):
            self.assertEqual(recorded["fluid"], 10.0)
            self.assertEqual(recorded["solid"], 20.0)
            np.testing.assert_allclose(recorded["gradient"], prepared_gradient)
            marker = recorded["marker"]
            np.testing.assert_allclose(
                marker["x_gamma_m"], prepared_marker["x_gamma_m"]
            )
            np.testing.assert_allclose(
                marker["pressure_probe_origin_m"],
                prepared_marker["pressure_probe_origin_m"],
            )
            np.testing.assert_allclose(
                marker["n_gamma"], prepared_marker["n_gamma"]
            )
            np.testing.assert_allclose(
                marker["A_gamma_m2"], prepared_marker["A_gamma_m2"]
            )
            np.testing.assert_allclose(marker["v_gamma_mps"], expected_guess)

        np.testing.assert_allclose(first.marker_velocity_mps, 0.25 * first_guess + 0.75)
        np.testing.assert_allclose(second.marker_velocity_mps, 0.25 * second_guess + 0.75)
        self.assertIs(first.payload["latest_report"], reports[0])
        self.assertIs(second.payload["latest_report"], reports[1])

        runtime.rollback_step(context)
        self.assertEqual(fluid.value, 10.0)
        self.assertEqual(solid.value, 20.0)
        self.assertEqual(fluid.restore_calls, 3)
        self.assertEqual(solid.restore_calls, 3)
        self.assertEqual(len(boundary_refreshes), 2)
        rolled_back = capture_marker_interface_state(markers)
        for name, expected in marker_base.items():
            np.testing.assert_allclose(rolled_back[name], expected)
        np.testing.assert_allclose(
            boundary.marker_pressure_neumann_gradient_field.to_numpy(),
            gradient_base,
        )

    def test_prepare_failure_restores_pre_step_turek_state(self) -> None:
        fluid = _FakeRestorable(10.0)
        solid = _FakeRestorable(20.0)
        marker_base = {
            "x_gamma_m": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            "v_gamma_mps": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
            "pressure_probe_origin_m": np.asarray(
                [[1.0, 2.0, 3.25]], dtype=np.float32
            ),
            "n_gamma": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "A_gamma_m2": np.asarray([0.5], dtype=np.float32),
        }
        markers = SimpleNamespace(
            marker_count=1,
            projection_vertex_count=1,
            projection_triangle_count=0,
            projection_segment_count=0,
            _open_ribbon_tip_cap_binding=None,
            _begin_marker_geometry_write=lambda: None,
            **{name: _FakeField(value) for name, value in marker_base.items()},
        )
        gradient_base = np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32)
        boundary = SimpleNamespace(
            marker_pressure_neumann_gradient_field=_FakeField(gradient_base)
        )

        def fail_prepare(_context: FsiStepContext) -> None:
            fluid.value = -10.0
            solid.value = -20.0
            markers.x_gamma_m.from_numpy(
                np.asarray([[9.0, 9.0, 9.0]], dtype=np.float32)
            )
            markers.pressure_probe_origin_m.from_numpy(
                np.asarray([[8.0, 8.0, 8.0]], dtype=np.float32)
            )
            markers.marker_count = 0
            markers.projection_vertex_count = 0
            raise RuntimeError("synthetic Turek prepare failure")

        runtime = turek._TurekHronFsiRuntime(
            fluid=fluid,
            solid=solid,
            markers=markers,
            boundary=boundary,
            advance_trial=lambda _context, _trial_index: None,
            prepare_step=fail_prepare,
            restore_case_boundaries=lambda _context: None,
            commit_case_step=lambda _context, _trial, _coupling: {},
            finalize_case_run=lambda: {},
        )
        context = FsiStepContext(step=1, step_index=0, time_s=0.25, dt_s=0.25)

        with self.assertRaisesRegex(RuntimeError, "synthetic Turek prepare failure"):
            solve_fsi_step(
                runtime,
                context,
                FsiCouplingConfig(max_iterations=2),
            )

        self.assertEqual(fluid.value, 10.0)
        self.assertEqual(solid.value, 20.0)
        self.assertEqual(markers.marker_count, 1)
        self.assertEqual(markers.projection_vertex_count, 1)
        restored = capture_marker_interface_state(markers)
        for name, expected in marker_base.items():
            np.testing.assert_allclose(restored[name], expected)
        np.testing.assert_allclose(
            boundary.marker_pressure_neumann_gradient_field.to_numpy(),
            gradient_base,
        )

    def test_generic_report_fields_use_only_generic_names(self) -> None:
        report = FsiCouplingReport(
            iterations=3,
            converged=True,
            relative_residual=2.0e-4,
            absolute_residual_mps=3.0e-6,
            max_marker_residual_mps=5.0e-6,
            relative_residual_history=(0.4, 0.02, 2.0e-4),
            absolute_residual_history_mps=(0.04, 2.0e-4, 3.0e-6),
            update_modes=("picard", "iqn_ils"),
        )

        fields = turek._turek_hron_coupling_report_fields(
            report,
            relative_tolerance=1.0e-3,
            absolute_tolerance_mps=1.0e-5,
            initial_relaxation=0.5,
        )

        self.assertEqual(fields["fsi_coupling_iterations_used"], 3)
        self.assertEqual(fields["fsi_coupling_residual"], 2.0e-4)
        self.assertEqual(fields["fsi_coupling_initial_relaxation"], 0.5)
        self.assertNotIn("fsi_aitken_relaxation", fields)
        self.assertTrue(fields["fsi_coupling_residual_measured"])
        self.assertTrue(fields["fsi_coupling_converged"])
        self.assertEqual(
            fields["fsi_coupling_convergence_reason"],
            "relative_tolerance",
        )
        self.assertEqual(
            fields["fsi_coupling_residual_history"],
            [0.4, 0.02, 2.0e-4],
        )
        self.assertEqual(
            fields["fsi_coupling_update_diagnostics"],
            [
                {"iteration": 1, "update_mode": "picard"},
                {"iteration": 2, "update_mode": "iqn_ils"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
