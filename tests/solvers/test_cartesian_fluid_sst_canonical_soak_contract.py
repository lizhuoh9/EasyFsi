from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "solver_soaks"
    / "run_cartesian_fluid_sst_canonical_5000.py"
)


def _source_tree() -> tuple[str, ast.Module]:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(RUNNER_PATH))


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"runner is missing function {name!r}")


def _top_level_constants(tree: ast.Module) -> dict[str, object]:
    constants: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    return constants


def _load_runner_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "cartesian_fluid_sst_canonical_5000_runner",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise AssertionError("could not load soak runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class CartesianFluidSstCanonicalSoakHostContracts(unittest.TestCase):
    def test_runner_has_fixed_5000_step_checkpoint_and_diagnostic_contract(self) -> None:
        _source, tree = _source_tree()
        constants = _top_level_constants(tree)

        self.assertEqual(constants["TOTAL_STEPS"], 5000)
        self.assertEqual(constants["CHECKPOINT_STEP"], 2500)
        self.assertEqual(constants["DIAGNOSTIC_INTERVAL"], 100)
        self.assertEqual(constants["MAX_TRANSPORT_SUBSTEPS"], 64)
        self.assertEqual(constants["MOVING_WALL_SEGMENT_SCALES"], (1.0, 0.5, -1.0, -0.5, 1.0))

    def test_primary_loop_is_single_uninterrupted_solver_and_only_saves_at_2500(
        self,
    ) -> None:
        _source, tree = _source_tree()
        run_function = _top_level_function(tree, "_run_impl")
        loops = [node for node in ast.walk(run_function) if isinstance(node, ast.For)]
        primary_loops = [
            node
            for node in loops
            if ast.unparse(node.iter) == "range(1, TOTAL_STEPS + 1)"
        ]
        self.assertEqual(len(primary_loops), 1)
        primary_loop = primary_loops[0]
        primary_source = ast.unparse(primary_loop)

        self.assertIn("_advance_one_step(primary_solver, step=step)", primary_source)
        self.assertIn("step == CHECKPOINT_STEP", primary_source)
        self.assertIn("_save_schema_v8_checkpoint", primary_source)
        self.assertNotIn("load_preflow_snapshot", primary_source)
        self.assertFalse(
            any(isinstance(node, (ast.Break, ast.Return)) for node in ast.walk(primary_loop))
        )
        solver_builds = [
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_solver"
        ]
        self.assertEqual(len(solver_builds), 2)
        self.assertNotIn("_build_solver", primary_source)

    def test_fresh_restore_and_replay_begin_only_after_primary_loop(self) -> None:
        source, tree = _source_tree()
        run_function = _top_level_function(tree, "_run_impl")
        run_source = ast.get_source_segment(source, run_function)
        self.assertIsNotNone(run_source)
        assert run_source is not None

        primary_end = run_source.index("primary_final_fields =")
        fresh_build = run_source.index("fresh_solver = _build_solver")
        restore = run_source.index("loaded = load_preflow_snapshot")
        replay = run_source.index(
            "for step in range(CHECKPOINT_STEP + 1, TOTAL_STEPS + 1):"
        )
        self.assertLess(primary_end, fresh_build)
        self.assertLess(fresh_build, restore)
        self.assertLess(restore, replay)
        self.assertIn("_assert_replay_diagnostic_equivalent", run_source[replay:])
        self.assertIn("_assert_replay_fields_physically_equivalent", run_source[replay:])
        self.assertNotIn("replay_final_sha256 != primary_final_sha256", run_source[replay:])

    def test_replay_progress_is_persisted_before_equivalence_can_fail(self) -> None:
        source, tree = _source_tree()
        run_function = _top_level_function(tree, "_run_impl")
        run_source = ast.get_source_segment(source, run_function)
        self.assertIsNotNone(run_source)
        assert run_source is not None

        replay_source = run_source[
            run_source.index(
                "for step in range(CHECKPOINT_STEP + 1, TOTAL_STEPS + 1):"
            ) :
        ]
        progress = replay_source.index('report["replay_completed_steps"] =')
        persist = replay_source.index("_atomic_write_json(report_path, report)")
        compare = replay_source.index("_assert_replay_fields_physically_equivalent")
        self.assertLess(progress, persist)
        self.assertLess(persist, compare)

    def test_runner_uses_schema_v8_full_field_capture_and_canonical_ab_walls(
        self,
    ) -> None:
        source, tree = _source_tree()
        capture = ast.unparse(_top_level_function(tree, "_capture_snapshot_fields"))
        checkpoint = ast.unparse(
            _top_level_function(tree, "_save_schema_v8_checkpoint")
        )
        ledger = ast.unparse(_top_level_function(tree, "_canonical_ledger_arrays"))

        self.assertIn("PREFLOW_SNAPSHOT_FIELD_NAMES", capture)
        self.assertIn("validate_preflow_snapshot_fields", capture)
        self.assertIn("PreflowSnapshot", checkpoint)
        self.assertIn("save_preflow_snapshot", checkpoint)
        self.assertIn("PREFLOW_SNAPSHOT_SCHEMA_VERSION", checkpoint)
        self.assertIn("active[:, 1, :]", ledger)
        self.assertIn("active[:, 4, :]", ledger)
        self.assertIn("target[patch[0], 1, patch[1], 0] = speed", ledger)
        self.assertIn("target[patch[0], 3, patch[1], 0] = -speed", ledger)
        self.assertIn('"canonical"', source)

    def test_canonical_ab_ledger_is_schema_v8_host_valid(self) -> None:
        import numpy as np

        from simulation_core.fluids.preflow_snapshot import (
            validate_preflow_snapshot_fields,
        )
        from tests.solvers.test_preflow_snapshot import _valid_fields

        runner = _load_runner_module()
        fields = _valid_fields(runner.GRID_NODES)
        ledger = runner._canonical_ledger_arrays(wall_scale=-1.0)
        fields.update(ledger)
        obstacle = np.zeros(runner.GRID_NODES, dtype=np.int32)
        obstacle[:, 0, :] = 1
        obstacle[:, 4, :] = 1
        fields["obstacle"] = obstacle
        fields["hibm_base_obstacle"] = obstacle.copy()
        fields["velocity_dirichlet_boundary_active"] = np.zeros(
            runner.GRID_NODES, dtype=np.int32
        )
        fields["velocity_dirichlet_boundary_projection_weight"] = np.zeros(
            runner.GRID_NODES, dtype=np.float32
        )
        fields["velocity_dirichlet_boundary_enforcement_weight"] = np.zeros(
            runner.GRID_NODES, dtype=np.float32
        )
        fields["velocity_dirichlet_boundary_marker_region_id"] = np.full(
            runner.GRID_NODES, -1, dtype=np.int32
        )
        fields["velocity_dirichlet_boundary_external_exact_component_mask"] = (
            np.zeros(runner.GRID_NODES, dtype=np.int32)
        )
        fields["velocity_dirichlet_boundary_owned_row"] = np.zeros(
            runner.GRID_NODES, dtype=np.int32
        )

        validated = validate_preflow_snapshot_fields(
            fields,
            velocity_dirichlet_boundary_authority="canonical",
        )
        self.assertEqual(
            int(validated["velocity_dirichlet_boundary_active_component_mask"][0, 1, 0])
            & 0b010,
            0b010,
        )
        self.assertEqual(
            int(validated["velocity_dirichlet_boundary_active_component_mask"][0, 4, 0])
            & 0b010,
            0b010,
        )

    def test_cli_is_gpu_only_and_does_not_offer_a_short_success_path(self) -> None:
        source, _tree = _source_tree()
        self.assertIn('"--output-dir"', source)
        self.assertIn('choices=("cuda", "gpu")', source)
        self.assertNotIn('"--steps"', source)
        self.assertNotIn('"--skip-replay"', source)

    def test_replay_field_contract_accepts_only_f64_pressure_roundoff(self) -> None:
        import numpy as np

        runner = _load_runner_module()
        expected = {
            "pressure": np.array([[[1.0e-3, -1.0e-3]]], dtype=np.float64),
            "fsi_pressure": np.array([[[2.0e-3, -2.0e-3]]], dtype=np.float64),
            "velocity": np.array([[[[0.05, 0.0, 0.0]]]], dtype=np.float32),
            "obstacle": np.array([[[0]]], dtype=np.int32),
        }
        actual = {name: values.copy() for name, values in expected.items()}
        actual["pressure"][0, 0, 0] = np.nextafter(
            actual["pressure"][0, 0, 0], np.inf
        )
        actual["fsi_pressure"][0, 0, 1] = np.nextafter(
            actual["fsi_pressure"][0, 0, 1], -np.inf
        )

        comparison = runner._assert_replay_fields_physically_equivalent(
            expected=expected,
            actual=actual,
            step=2600,
        )

        self.assertFalse(comparison["bitwise_identical"])
        self.assertEqual(comparison["non_bitwise_fields"], ["fsi_pressure", "pressure"])
        self.assertGreater(comparison["pressure"]["linf_difference_pa"], 0.0)
        self.assertGreater(comparison["fsi_pressure"]["linf_difference_pa"], 0.0)

    def test_replay_field_contract_rejects_physical_pressure_or_other_field_drift(
        self,
    ) -> None:
        import numpy as np

        runner = _load_runner_module()
        expected = {
            "pressure": np.array([[[1.0e-3, -1.0e-3]]], dtype=np.float64),
            "fsi_pressure": np.array([[[2.0e-3, -2.0e-3]]], dtype=np.float64),
            "velocity": np.array([[[[0.05, 0.0, 0.0]]]], dtype=np.float32),
        }
        pressure_drift = {name: values.copy() for name, values in expected.items()}
        pressure_drift["pressure"][0, 0, 0] += 1.0e-9
        with self.assertRaisesRegex(RuntimeError, "pressure"):
            runner._assert_replay_fields_physically_equivalent(
                expected=expected,
                actual=pressure_drift,
                step=2600,
            )

        velocity_drift = {name: values.copy() for name, values in expected.items()}
        velocity_drift["velocity"][0, 0, 0, 0] = np.nextafter(
            velocity_drift["velocity"][0, 0, 0, 0], np.float32(np.inf)
        )
        with self.assertRaisesRegex(RuntimeError, "velocity"):
            runner._assert_replay_fields_physically_equivalent(
                expected=expected,
                actual=velocity_drift,
                step=2600,
            )

    def test_replay_nonpressure_exact_contract_rejects_signed_zero_bit_drift(
        self,
    ) -> None:
        import numpy as np

        runner = _load_runner_module()
        expected = {
            "pressure": np.array([1.0], dtype=np.float64),
            "velocity": np.array([0.0], dtype=np.float32),
        }
        actual = {name: values.copy() for name, values in expected.items()}
        actual["velocity"][0] = np.float32(-0.0)
        self.assertTrue(np.array_equal(expected["velocity"], actual["velocity"]))
        self.assertNotEqual(
            expected["velocity"].tobytes(), actual["velocity"].tobytes()
        )

        with self.assertRaisesRegex(RuntimeError, "velocity"):
            runner._assert_replay_fields_physically_equivalent(
                expected=expected,
                actual=actual,
                step=2600,
            )

    def test_pressure_replay_tolerance_is_locked_at_64_eps_times_scale(self) -> None:
        import numpy as np

        runner = _load_runner_module()
        eps = np.finfo(np.float64).eps
        expected = {"pressure": np.array([1.0], dtype=np.float64)}
        below = {"pressure": np.array([1.0 + 63.0 * eps], dtype=np.float64)}
        above = {"pressure": np.array([1.0 + 65.0 * eps], dtype=np.float64)}

        runner._assert_replay_fields_physically_equivalent(
            expected=expected,
            actual=below,
            step=2600,
        )
        with self.assertRaisesRegex(RuntimeError, "pressure"):
            runner._assert_replay_fields_physically_equivalent(
                expected=expected,
                actual=above,
                step=2600,
            )

        near_zero = {"pressure": np.array([1.0e-300], dtype=np.float64)}
        near_zero_next = {
            "pressure": np.array(
                [np.nextafter(np.float64(1.0e-300), np.float64(np.inf))],
                dtype=np.float64,
            )
        }
        runner._assert_replay_fields_physically_equivalent(
            expected=near_zero,
            actual=near_zero_next,
            step=2600,
        )

    def test_replay_diagnostic_contract_tolerates_hash_and_pressure_ulp_only(
        self,
    ) -> None:
        import numpy as np

        runner = _load_runner_module()
        expected = {
            "step": 2600,
            "state_sha256": "a" * 64,
            "pressure_min_pa": -1.0e-3,
            "pressure_max_pa": 1.0e-3,
            "pressure_range_pa": 2.0e-3,
            "velocity_max_abs_mps": 0.05,
            "cg_converged_all": True,
        }
        actual = dict(expected)
        actual["state_sha256"] = "b" * 64
        actual["pressure_max_pa"] = np.nextafter(
            actual["pressure_max_pa"], np.inf
        )

        comparison = runner._assert_replay_diagnostic_equivalent(
            expected=expected,
            actual=actual,
        )
        self.assertFalse(comparison["state_sha256_identical"])
        self.assertEqual(comparison["tolerated_pressure_diagnostics"], ["pressure_max_pa"])

        actual["pressure_max_pa"] += 1.0e-9
        with self.assertRaisesRegex(RuntimeError, "pressure_max_pa"):
            runner._assert_replay_diagnostic_equivalent(
                expected=expected,
                actual=actual,
            )

    def test_pressure_diagnostics_share_the_full_diagnostic_pressure_scale(
        self,
    ) -> None:
        runner = _load_runner_module()
        expected = {
            "step": 2600,
            "state_sha256": "a" * 64,
            "pressure_min_pa": -1.0,
            "pressure_max_pa": 0.0,
            "pressure_range_pa": 1.0,
        }
        actual = dict(expected)
        actual["state_sha256"] = "b" * 64
        actual["pressure_max_pa"] = 1.0e-15

        comparison = runner._assert_replay_diagnostic_equivalent(
            expected=expected,
            actual=actual,
        )
        self.assertEqual(
            comparison["pressure_diagnostics"]["pressure_max_pa"]["scale_pa"],
            1.0,
        )

    def test_run_records_failed_state_in_both_live_and_failure_reports(self) -> None:
        runner = _load_runner_module()
        with TemporaryDirectory() as root:
            output_dir = Path(root)
            live_report = {
                "status": "running_fresh_replay",
                "primary_completed_steps": 5000,
                "replay_completed_steps": 100,
            }

            def failing_impl(_config):
                runner._atomic_write_json(
                    output_dir / "soak_report.json", live_report
                )
                raise RuntimeError("synthetic replay failure")

            with patch.object(
                runner,
                "_run_impl",
                side_effect=failing_impl,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    runner.run(runner.SoakConfig(output_dir=output_dir, arch="cuda"))

            report = __import__("json").loads(
                (output_dir / "soak_report.json").read_text(encoding="utf-8")
            )
            failure = __import__("json").loads(
                (output_dir / "soak_failure.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_phase"], "running_fresh_replay")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["primary_completed_steps"], 5000)
        self.assertEqual(failure["replay_completed_steps"], 100)

    def test_run_clears_stale_failure_and_publishes_current_initializing_state(
        self,
    ) -> None:
        import json

        runner = _load_runner_module()
        with TemporaryDirectory() as root:
            output_dir = Path(root)
            failure_path = output_dir / "soak_failure.json"
            failure_path.write_text(
                json.dumps({"status": "failed", "error": "stale"}),
                encoding="utf-8",
            )

            def successful_impl(_config):
                self.assertFalse(failure_path.exists())
                live = json.loads(
                    (output_dir / "soak_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(live["status"], "initializing")
                return {"status": "completed"}

            with patch.object(runner, "_run_impl", side_effect=successful_impl):
                report = runner.run(
                    runner.SoakConfig(output_dir=output_dir, arch="cuda")
                )

            self.assertEqual(report["status"], "completed")
            self.assertFalse(failure_path.exists())


@unittest.skipUnless(
    os.environ.get("RUN_CUDA_SOAK") == "1",
    "set RUN_CUDA_SOAK=1 to execute the full 5000+2500-step CUDA soak",
)
class CartesianFluidSstCanonicalSoakCudaIntegration(unittest.TestCase):
    def test_full_cuda_soak_and_checkpoint_replay(self) -> None:
        runner = _load_runner_module()
        with TemporaryDirectory() as root:
            report = runner.run(
                runner.SoakConfig(output_dir=Path(root), arch="cuda")
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["primary_completed_steps"], 5000)
        self.assertEqual(report["replay_completed_steps"], 2500)
        self.assertTrue(
            report["checkpoint_replay_sampled_and_final_physically_equivalent"]
        )
        self.assertTrue(
            report[
                "checkpoint_replay_nonpressure_bitwise_identical_at_sampled_and_final_states"
            ]
        )
        self.assertIn("checkpoint_replay_bitwise_identical", report)


if __name__ == "__main__":
    unittest.main()
