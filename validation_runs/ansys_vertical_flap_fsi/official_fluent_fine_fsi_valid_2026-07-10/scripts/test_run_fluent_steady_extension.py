from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


MODULE_PATH = Path(__file__).with_name("run_fluent_steady_extension.py")
SPEC = importlib.util.spec_from_file_location("run_fluent_steady_extension", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_data(path: Path, *, multiplier: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        cells = handle.create_group("results/1/phase-1/cells")
        fields = {
            "SV_U": np.asarray([3.0, 0.0, 4.0]) * multiplier,
            "SV_V": np.asarray([4.0, 0.0, 3.0]) * multiplier,
            "SV_P": np.asarray([-2.0, 1.0, 6.0]) * multiplier,
            "SV_K": np.asarray([1.0, 2.0, 3.0]) * multiplier,
            "SV_O": np.asarray([10.0, 20.0, 30.0]) * multiplier,
            "SV_MU_T": np.asarray([0.1, 0.2, 0.3]) * multiplier,
        }
        for name, values in fields.items():
            group = cells.create_group(name)
            group.create_dataset("1", data=values)


def touch_pair(root: Path, name: str = "source") -> tuple[Path, Path]:
    case_path = root / f"{name}.cas.h5"
    data_path = root / f"{name}.dat.h5"
    case_path.write_bytes(b"case")
    write_data(data_path)
    return case_path, data_path


def monitor(scale: float) -> dict[str, float]:
    return {
        name: float((index + 1) * scale)
        for index, name in enumerate(MODULE.MONITOR_FIELDS)
    }


def surface_monitor(scale: float = 1.0) -> dict[str, float]:
    return {
        "flap_fluid_force_x_n": 1.0 * scale,
        "flap_fluid_force_y_n": 2.0 * scale,
        "flap_fluid_force_z_n": 3.0 * scale,
        "inlet_mass_flow_kg_s": -4.0 * scale,
        "outlet_mass_flow_kg_s": 3.99 * scale,
        "net_mass_flow_kg_s": -0.01 * scale,
        "relative_mass_imbalance": 0.0025 * scale,
    }


class FakeIterate:
    def __init__(self, owner: "FakeSession") -> None:
        self.owner = owner

    def __call__(self, *, iter_count: int) -> None:
        self.owner.iteration_calls.append(iter_count)
        if self.owner.fail_on_call == len(self.owner.iteration_calls):
            raise RuntimeError("synthetic Fluent iteration failure")


class FakeFile:
    def __init__(self, owner: "FakeSession") -> None:
        self.owner = owner

    def read_case_data(self, *, file_name: str) -> None:
        self.owner.read_paths.append(Path(file_name))

    def write_case_data(self, *, file_name: str) -> None:
        case_path = Path(file_name)
        data_path = MODULE.paired_data_path(case_path)
        case_path.write_bytes(b"checkpoint")
        write_data(data_path, multiplier=float(len(self.owner.iteration_calls) + 1))


class FakeSession:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.iteration_calls: list[int] = []
        self.read_paths: list[Path] = []
        self.fail_on_call = fail_on_call
        self.file = FakeFile(self)
        calculation = type("Calculation", (), {})()
        calculation.iterate = FakeIterate(self)
        solution = type("Solution", (), {})()
        solution.run_calculation = calculation
        self.solution = solution
        self.exited = False

    def exit(self) -> None:
        self.exited = True


class FluentSteadyExtensionTests(unittest.TestCase):
    def make_config(
        self,
        root: Path,
        *,
        max_iterations: int = 30,
        block_iterations: int = 10,
        window_blocks: int = 2,
        min_windows: int = 2,
        consecutive_windows: int = 2,
        tolerance: float = 0.01,
        resume: bool = False,
        dry_run: bool = False,
        recover_stale_lock: bool = False,
        force_full_budget: bool = False,
    ):
        case_path, data_path = touch_pair(root)
        return MODULE.SteadyExtensionConfig(
            run_dir=root / "extension",
            source_case=case_path,
            source_data=data_path,
            block_iterations=block_iterations,
            max_additional_iterations=max_iterations,
            window_blocks=window_blocks,
            minimum_windows=min_windows,
            consecutive_windows=consecutive_windows,
            relative_tolerance=tolerance,
            resume=resume,
            dry_run=dry_run,
            recover_stale_lock=recover_stale_lock,
            force_full_budget=force_full_budget,
        )

    def test_hdf5_monitor_contains_all_pressure_velocity_and_sst_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.dat.h5"
            write_data(path)

            result = MODULE.read_hdf5_monitor(path)

        self.assertEqual(set(result), set(MODULE.HDF5_MONITOR_FIELDS))
        self.assertEqual(result["pressure_min_pa"], -2.0)
        self.assertEqual(result["pressure_max_pa"], 6.0)
        self.assertEqual(result["pressure_mean_pa"], 5.0 / 3.0)
        self.assertEqual(result["pressure_range_pa"], 8.0)
        self.assertEqual(result["speed_max_mps"], 5.0)
        self.assertEqual(result["speed_mean_mps"], 10.0 / 3.0)
        self.assertAlmostEqual(result["k_p90_m2_s2"], 2.8)
        self.assertAlmostEqual(result["omega_p90_s_inv"], 28.0)
        self.assertAlmostEqual(result["mu_t_p90_pa_s"], 0.28)

    def test_stationarity_requires_minimum_windows_and_consecutive_passes(self) -> None:
        stable = [monitor(1.0), monitor(1.001), monitor(1.0015)]

        before = MODULE.evaluate_stationarity(
            stable[:2],
            window_blocks=2,
            minimum_windows=2,
            consecutive_windows=2,
            relative_tolerance=0.01,
        )
        after = MODULE.evaluate_stationarity(
            stable,
            window_blocks=2,
            minimum_windows=2,
            consecutive_windows=2,
            relative_tolerance=0.01,
        )

        self.assertFalse(before["stationary"])
        self.assertEqual(before["reason"], "insufficient_windows")
        self.assertTrue(after["stationary"])
        self.assertEqual(after["consecutive_windows_passed"], 2)
        self.assertEqual(after["evaluated_window_count"], 2)

    def test_stationarity_fails_if_any_physical_monitor_keeps_changing(self) -> None:
        rows = [monitor(1.0), monitor(1.001), monitor(1.0015)]
        rows[-1] = {**rows[-1], "mu_t_max_pa_s": 2.0 * rows[-2]["mu_t_max_pa_s"]}

        report = MODULE.evaluate_stationarity(
            rows,
            window_blocks=2,
            minimum_windows=2,
            consecutive_windows=2,
            relative_tolerance=0.01,
        )

        self.assertFalse(report["stationary"])
        self.assertIn("mu_t_max_pa_s", report["latest_failed_metrics"])

    def test_stationarity_rejects_slow_monotonic_drift_across_accepted_interval(self) -> None:
        # Every overlapping two-block window changes by < 1%, but the union of
        # the three required accepted windows drifts by > 1%.
        rows = [monitor(scale) for scale in (1.000, 1.004, 1.008, 1.012)]

        report = MODULE.evaluate_stationarity(
            rows,
            window_blocks=2,
            minimum_windows=3,
            consecutive_windows=3,
            relative_tolerance=0.01,
        )

        self.assertFalse(report["stationary"])
        self.assertEqual(report["consecutive_windows_passed"], 3)
        self.assertFalse(report["accepted_interval"]["passed"])
        self.assertIn(
            "pressure_min_pa",
            report["accepted_interval"]["failed_span_metrics"],
        )
        self.assertGreater(
            report["accepted_interval"]["relative_trends"]["pressure_min_pa"],
            0.01,
        )

    def test_dry_run_reads_source_but_never_launches_or_creates_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp), dry_run=True)
            with mock.patch.object(MODULE.guarded, "launch_fluent") as launch:
                report = MODULE.run_extension(config)

            self.assertEqual(report["status"], "dry_run")
            self.assertFalse(config.run_dir.exists())
            launch.assert_not_called()

    def test_stable_blocks_stop_only_after_stationary_gate_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root, max_iterations=40)
            session = FakeSession()
            stable_rows = [monitor(1.0), monitor(1.001), monitor(1.0015)]
            lock = root / "global.lock"
            with (
                mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", lock),
                mock.patch.object(MODULE.guarded, "launch_fluent", return_value=session) as launch,
                mock.patch.object(MODULE.guarded, "transcript_cursor", return_value=(None, 0)),
                mock.patch.object(
                    MODULE.guarded,
                    "transcript_delta",
                    return_value=("clean", (Path("fluent.trn"), 5)),
                ),
                mock.patch.object(MODULE.guarded, "require_clean_transcript"),
                mock.patch.object(MODULE.guarded, "copy_latest_transcript", return_value=None),
                mock.patch.object(
                    MODULE.guarded,
                    "read_surface_integrals",
                    side_effect=[surface_monitor(), surface_monitor(1.001), surface_monitor(1.0015)],
                ),
                mock.patch.object(MODULE, "read_hdf5_monitor", side_effect=[monitor(0.5), *stable_rows]),
            ):
                report = MODULE.run_extension(config)

            self.assertEqual(report["status"], "stationary")
            self.assertEqual(report["completed_iterations"], 30)
            self.assertEqual(session.iteration_calls, [10, 10, 10])
            self.assertTrue(session.exited)
            launch.assert_called_once()
            self.assertFalse(lock.exists())
            progress = json.loads((config.run_dir / "progress.json").read_text())
            persisted = json.loads((config.run_dir / "report.json").read_text())
            self.assertEqual(progress["status"], "stationary")
            self.assertEqual(persisted["status"], "stationary")
            self.assertTrue(Path(report["final_case"]).is_file())
            self.assertTrue(Path(report["final_data"]).is_file())

    def test_maximum_budget_is_honest_not_stationary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root, max_iterations=20)
            session = FakeSession()
            changing = [monitor(1.0), monitor(1.5)]
            with (
                mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", root / "global.lock"),
                mock.patch.object(MODULE.guarded, "launch_fluent", return_value=session),
                mock.patch.object(MODULE.guarded, "transcript_cursor", return_value=(None, 0)),
                mock.patch.object(
                    MODULE.guarded,
                    "transcript_delta",
                    return_value=("clean", (Path("fluent.trn"), 5)),
                ),
                mock.patch.object(MODULE.guarded, "require_clean_transcript"),
                mock.patch.object(MODULE.guarded, "copy_latest_transcript", return_value=None),
                mock.patch.object(
                    MODULE.guarded,
                    "read_surface_integrals",
                    side_effect=[surface_monitor(), surface_monitor(1.5)],
                ),
                mock.patch.object(MODULE, "read_hdf5_monitor", side_effect=[monitor(0.5), *changing]),
            ):
                report = MODULE.run_extension(config)

            self.assertEqual(report["status"], "not_stationary")
            self.assertFalse(report["stationarity"]["stationary"])
            self.assertEqual(report["completed_iterations"], 20)

    def test_force_full_budget_never_stops_or_claims_stationary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(
                root,
                max_iterations=40,
                force_full_budget=True,
            )
            session = FakeSession()
            stable_rows = [
                monitor(1.0),
                monitor(1.001),
                monitor(1.0015),
                monitor(1.0017),
            ]
            with (
                mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", root / "global.lock"),
                mock.patch.object(MODULE.guarded, "launch_fluent", return_value=session),
                mock.patch.object(MODULE.guarded, "transcript_cursor", return_value=(None, 0)),
                mock.patch.object(
                    MODULE.guarded,
                    "transcript_delta",
                    return_value=("clean", (Path("fluent.trn"), 5)),
                ),
                mock.patch.object(MODULE.guarded, "require_clean_transcript"),
                mock.patch.object(MODULE.guarded, "copy_latest_transcript", return_value=None),
                mock.patch.object(
                    MODULE.guarded,
                    "read_surface_integrals",
                    side_effect=[surface_monitor()] * 4,
                ),
                mock.patch.object(
                    MODULE,
                    "read_hdf5_monitor",
                    side_effect=[monitor(0.5), *stable_rows],
                ),
            ):
                report = MODULE.run_extension(config)

            self.assertEqual(report["status"], "fixed_budget_complete")
            self.assertEqual(report["completed_iterations"], 40)
            self.assertEqual(session.iteration_calls, [10, 10, 10, 10])
            self.assertFalse(report["stationarity"]["stationary"])
            self.assertTrue(report["stationarity"]["diagnostic_stationary"])
            self.assertEqual(
                report["stationarity"]["reason"],
                "force_full_budget_diagnostic_only",
            )

    def test_surface_integrals_are_required_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root, max_iterations=10)
            session = FakeSession()
            with (
                mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", root / "global.lock"),
                mock.patch.object(MODULE.guarded, "launch_fluent", return_value=session),
                mock.patch.object(MODULE.guarded, "transcript_cursor", return_value=(None, 0)),
                mock.patch.object(
                    MODULE.guarded,
                    "transcript_delta",
                    return_value=("clean", (Path("fluent.trn"), 5)),
                ),
                mock.patch.object(MODULE.guarded, "require_clean_transcript"),
                mock.patch.object(MODULE.guarded, "copy_latest_transcript", return_value=None),
                mock.patch.object(MODULE, "read_hdf5_monitor", side_effect=[monitor(0.5), monitor(1.0)]),
                mock.patch.object(
                    MODULE.guarded,
                    "read_surface_integrals",
                    side_effect=RuntimeError("surface API unavailable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "surface API unavailable"):
                    MODULE.run_extension(config)

            failure = json.loads((config.run_dir / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["completed_iterations"], 0)

    def test_failure_is_atomic_and_resume_uses_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root, max_iterations=30)
            failed_session = FakeSession(fail_on_call=2)
            def enter_common_patches(stack: ExitStack) -> None:
                stack.enter_context(
                    mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", root / "global.lock")
                )
                stack.enter_context(
                    mock.patch.object(
                        MODULE.guarded, "transcript_cursor", return_value=(None, 0)
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        MODULE.guarded,
                        "transcript_delta",
                        return_value=("clean", (Path("fluent.trn"), 5)),
                    )
                )
                stack.enter_context(
                    mock.patch.object(MODULE.guarded, "require_clean_transcript")
                )
                stack.enter_context(
                    mock.patch.object(
                        MODULE.guarded, "copy_latest_transcript", return_value=None
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        MODULE.guarded,
                        "read_surface_integrals",
                        side_effect=[surface_monitor(), surface_monitor(1.001), surface_monitor(1.0015)],
                    )
                )

            with ExitStack() as stack:
                enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        MODULE.guarded, "launch_fluent", return_value=failed_session
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        MODULE,
                        "read_hdf5_monitor",
                        side_effect=[monitor(0.5), monitor(1.0)],
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "synthetic Fluent"):
                    MODULE.run_extension(config)

            failure = json.loads((config.run_dir / "failure.json").read_text())
            progress = json.loads((config.run_dir / "progress.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed_iterations"], 10)

            resumed_session = FakeSession()
            resume_config = MODULE.SteadyExtensionConfig(**{**vars(config), "resume": True})
            with ExitStack() as stack:
                enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        MODULE.guarded, "launch_fluent", return_value=resumed_session
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        MODULE,
                        "read_hdf5_monitor",
                        side_effect=[monitor(1.001), monitor(1.0015)],
                    )
                )
                report = MODULE.run_extension(resume_config)

            self.assertEqual(report["status"], "stationary")
            self.assertEqual(resumed_session.iteration_calls, [10, 10])
            self.assertTrue(
                resumed_session.read_paths[0].name.endswith("block_0001.cas.h5")
            )
            self.assertFalse((config.run_dir / "failure.json").exists())

    def test_lock_conflict_precedes_all_run_directory_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            lock = root / "global.lock"
            lock.write_text(
                json.dumps({"pid": 12345, "run_dir": "other", "lock_id": "owner"}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", lock),
                mock.patch.object(MODULE, "_process_is_alive", return_value=True),
                mock.patch.object(MODULE.guarded, "launch_fluent") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "owns the fail-closed lock"):
                    MODULE.run_extension(config)

            self.assertFalse(config.run_dir.exists())
            self.assertFalse((config.run_dir / "failure.json").exists())
            launch.assert_not_called()

    def test_explicit_stale_lock_recovery_requires_confirmed_dead_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(
                root,
                dry_run=True,
                recover_stale_lock=True,
            )
            lock = root / "global.lock"
            lock.write_text(
                json.dumps({"pid": 54321, "run_dir": "old", "lock_id": "old"}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(MODULE, "GLOBAL_FLUENT_LOCK", lock),
                mock.patch.object(MODULE, "_process_is_alive", return_value=False) as alive,
            ):
                report = MODULE.run_extension(config)

            self.assertEqual(report["status"], "dry_run")
            self.assertFalse(lock.exists())
            self.assertFalse(config.run_dir.exists())
            alive.assert_called_once_with(54321)

    def test_windows_pid_probe_uses_non_destructive_windows_api(self) -> None:
        with (
            mock.patch.object(MODULE.os, "name", "nt"),
            mock.patch.object(
                MODULE,
                "_windows_process_is_alive",
                return_value=False,
            ) as windows_probe,
            mock.patch.object(MODULE.os, "kill") as posix_kill,
        ):
            result = MODULE._process_is_alive(32123)

        self.assertFalse(result)
        windows_probe.assert_called_once_with(32123)
        posix_kill.assert_not_called()

    def test_serial_and_new_directory_contracts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            with self.assertRaisesRegex(ValueError, "one Fluent process|serial"):
                MODULE.validate_config(
                    MODULE.SteadyExtensionConfig(**{**vars(config), "processor_count": 2})
                )
            config.run_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                MODULE.validate_config(config)


if __name__ == "__main__":
    unittest.main()
