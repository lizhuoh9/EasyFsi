from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cases.squid_soft_robot import checkpointing as checkpointing_module
from cases.squid_soft_robot.checkpointing import (
    RUN_CHECKPOINT_VERSION,
    checkpoint_run_fingerprint,
    load_run_checkpoint,
    write_run_checkpoint,
)


class _ScalarField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self.value

    def __setitem__(self, _index, value):
        self.value = value


class _ArrayField:
    def __init__(self, values):
        self.values = np.asarray(values).copy()

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()

    def from_numpy(self, values) -> None:
        self.values = np.asarray(values).copy()


@dataclass(frozen=True)
class _Spec:
    source_config_path: str = "missing-test-config.json"
    grid_nodes: tuple[int, int, int] = (2, 2, 2)


def _fixture():
    scalar_names = (
        "time_s",
        "pressure_load_pa",
        "hydraulic_pressure_pa",
        "main_w_m",
        "main_v_mps",
        "tail_w_m",
        "tail_v_mps",
        "volume_flux_m3s",
        "nozzle_velocity_z_mps",
        "max_speed_mps",
        "lip_flow_z_m3s",
        "outlet_flow_z_m3s",
        "downstream_flow_z_m3s",
    )
    simulator = SimpleNamespace(spec=_Spec())
    for index, name in enumerate(scalar_names, start=1):
        setattr(simulator, name, _ScalarField(float(index)))
    for name in ("lip_sample_count", "outlet_sample_count", "downstream_sample_count"):
        setattr(simulator, name, _ScalarField(3))
    for name in (
        "primary_interface_reaction_force_n",
        "secondary_interface_reaction_force_n",
    ):
        setattr(simulator, name, _ScalarField(np.asarray([1.0, 2.0, 3.0])))
    velocity = np.ones((2, 2, 2, 3), dtype=np.float32)
    pressure = np.ones((2, 2, 2), dtype=np.float32)
    simulator.fluid = SimpleNamespace(
        velocity=_ArrayField(velocity),
        velocity_prev=_ArrayField(velocity * 2.0),
        pressure=_ArrayField(pressure),
        pressure_tmp=_ArrayField(pressure),
        pressure_accum=_ArrayField(pressure),
    )
    solid = SimpleNamespace(
        particle_count=1,
        x=_ArrayField(np.ones((1, 3), dtype=np.float32)),
        v=_ArrayField(np.ones((1, 3), dtype=np.float32) * 2.0),
        C=_ArrayField(np.ones((1, 3, 3), dtype=np.float32) * 3.0),
        F=_ArrayField(np.ones((1, 3, 3), dtype=np.float32) * 4.0),
        position_increment_residual_m=_ArrayField(
            np.asarray([[0.125, -0.25, 0.5]], dtype=np.float64)
        ),
    )
    return simulator, solid


def _rewrite_checkpoint_metadata(path: Path, **updates: object) -> None:
    with np.load(path, allow_pickle=False) as checkpoint:
        payload = {
            name: np.asarray(checkpoint[name]).copy()
            for name in checkpoint.files
        }
    metadata = json.loads(str(payload["__metadata__"]))
    metadata.update(updates)
    payload["__metadata__"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **payload)


class SquidCheckpointAtomicityTests(unittest.TestCase):
    def test_checkpoint_format_version_is_7(self):
        self.assertEqual(RUN_CHECKPOINT_VERSION, 7)

    def test_neo_position_increment_residual_round_trips(self):
        simulator, solid = _fixture()
        args = SimpleNamespace(solid_model="neo_hookean_mpm")
        expected = solid.position_increment_residual_m.to_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.npz"
            write_run_checkpoint(
                path,
                completed_step=1,
                step_count=2,
                full_pressure_waveform_steps=2,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
            )
            solid.position_increment_residual_m.from_numpy(
                np.zeros_like(expected)
            )

            load_run_checkpoint(
                path,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
                step_count=2,
                full_pressure_waveform_steps=2,
            )

        np.testing.assert_array_equal(
            solid.position_increment_residual_m.to_numpy(),
            expected,
        )

    def test_nonfinite_live_state_does_not_replace_a_valid_checkpoint(self):
        simulator, solid = _fixture()
        args = SimpleNamespace(solid_model="neo_hookean_mpm")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.npz"
            write_run_checkpoint(
                path,
                completed_step=1,
                step_count=2,
                full_pressure_waveform_steps=2,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
            )
            valid_bytes = path.read_bytes()
            solid.position_increment_residual_m.from_numpy(
                np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float64)
            )

            with self.assertRaisesRegex(ValueError, "must be finite"):
                write_run_checkpoint(
                    path,
                    completed_step=1,
                    step_count=2,
                    full_pressure_waveform_steps=2,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid,
                )

            self.assertEqual(path.read_bytes(), valid_bytes)

    def test_resume_history_requires_every_integer_step_in_order(self):
        rows = [
            {"step": 99, "time_s": 0.5},
            {"step": 2, "time_s": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "row 1 step"):
            checkpointing_module.validate_resume_history_checkpoint_alignment(
                rows,
                completed_step=2,
                checkpoint_time_s=1.0,
                dt_s=0.5,
            )

        rows[0]["step"] = 1.9
        with self.assertRaisesRegex(ValueError, "row 1 step"):
            checkpointing_module.validate_resume_history_checkpoint_alignment(
                rows,
                completed_step=2,
                checkpoint_time_s=1.0,
                dt_s=0.5,
            )

    def test_missing_marker_payload_does_not_partially_restore_other_state(self):
        simulator, solid = _fixture()
        args = SimpleNamespace(solid_model="neo_hookean_mpm")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.npz"
            write_run_checkpoint(
                path,
                completed_step=1,
                step_count=2,
                full_pressure_waveform_steps=2,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
            )
            simulator.time_s[None] = 999.0
            simulator.fluid.velocity.from_numpy(
                np.full((2, 2, 2, 3), 999.0, dtype=np.float32)
            )
            marker_field = _ArrayField(np.zeros((1, 3), dtype=np.float32))
            coupling = SimpleNamespace(
                markers=SimpleNamespace(
                    marker_count=1,
                    x_gamma_m=marker_field,
                    v_gamma_mps=_ArrayField(np.zeros((1, 3), dtype=np.float32)),
                    n_gamma=_ArrayField(np.zeros((1, 3), dtype=np.float32)),
                    A_gamma_m2=_ArrayField(np.zeros(1, dtype=np.float32)),
                )
            )

            with self.assertRaisesRegex(ValueError, "marker state"):
                load_run_checkpoint(
                    path,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid,
                    sharp_coupling_state=coupling,
                )

        self.assertEqual(simulator.time_s[None], 999.0)
        self.assertTrue(np.all(simulator.fluid.velocity.to_numpy() == 999.0))

    def test_invalid_completed_step_never_mutates_live_state(self):
        simulator, solid = _fixture()
        args = SimpleNamespace(solid_model="neo_hookean_mpm")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.npz"
            for invalid_completed_step in (True, 1.5, 2):
                with self.subTest(completed_step=invalid_completed_step):
                    write_run_checkpoint(
                        path,
                        completed_step=1,
                        step_count=2,
                        full_pressure_waveform_steps=2,
                        args=args,
                        simulator=simulator,
                        solid_mpm=solid,
                    )
                    _rewrite_checkpoint_metadata(
                        path,
                        completed_step=invalid_completed_step,
                    )
                    simulator.time_s[None] = 999.0
                    simulator.fluid.velocity.from_numpy(
                        np.full((2, 2, 2, 3), 999.0, dtype=np.float32)
                    )

                    with self.assertRaisesRegex(ValueError, "completed_step"):
                        load_run_checkpoint(
                            path,
                            args=args,
                            simulator=simulator,
                            solid_mpm=solid,
                            step_count=2,
                            full_pressure_waveform_steps=2,
                        )

                    self.assertEqual(simulator.time_s[None], 999.0)
                    self.assertTrue(
                        np.all(simulator.fluid.velocity.to_numpy() == 999.0)
                    )

    def test_nonintegral_metadata_identifiers_never_mutate_live_state(self):
        simulator, solid = _fixture()
        args = SimpleNamespace(solid_model="neo_hookean_mpm")
        invalid_metadata = (
            ("version", 5.5),
            ("grid_nodes", [2.5, 2, 2]),
            ("particle_count", True),
            ("requested_steps", 2.5),
            ("full_pressure_waveform_steps", 2.5),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.npz"
            for field, value in invalid_metadata:
                with self.subTest(field=field):
                    simulator.time_s[None] = 1.0
                    write_run_checkpoint(
                        path,
                        completed_step=1,
                        step_count=2,
                        full_pressure_waveform_steps=2,
                        args=args,
                        simulator=simulator,
                        solid_mpm=solid,
                    )
                    _rewrite_checkpoint_metadata(path, **{field: value})
                    simulator.time_s[None] = 999.0

                    with self.assertRaisesRegex(ValueError, field):
                        load_run_checkpoint(
                            path,
                            args=args,
                            simulator=simulator,
                            solid_mpm=solid,
                            step_count=2,
                            full_pressure_waveform_steps=2,
                        )

                    self.assertEqual(simulator.time_s[None], 999.0)

    def test_checkpoint_write_preserves_the_initialized_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mesh_path = root / "squid.step"
            config_path = root / "simulation_config.json"
            checkpoint_path = root / "restart.npz"
            mesh_path.write_bytes(b"mesh-A")
            config_path.write_text(
                json.dumps({"mesh_path": str(mesh_path)}),
                encoding="utf-8",
            )
            simulator, solid = _fixture()
            simulator.spec = _Spec(source_config_path=str(config_path))
            args = SimpleNamespace(
                solid_model="neo_hookean_mpm",
                source_config=str(config_path),
                disable_reduced_obstacles=False,
            )
            frozen_fingerprint = checkpoint_run_fingerprint(
                args=args,
                spec=simulator.spec,
                step_count=2,
                full_pressure_waveform_steps=2,
            )
            mesh_path.write_bytes(b"mesh-B")

            write_run_checkpoint(
                checkpoint_path,
                completed_step=1,
                step_count=2,
                full_pressure_waveform_steps=2,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
                frozen_run_fingerprint=frozen_fingerprint,
            )

            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                metadata = json.loads(str(checkpoint["__metadata__"]))
            current_fingerprint = checkpoint_run_fingerprint(
                args=args,
                spec=simulator.spec,
                step_count=2,
                full_pressure_waveform_steps=2,
            )
            self.assertEqual(metadata["run_fingerprint"], frozen_fingerprint)
            self.assertNotEqual(metadata["run_fingerprint"], current_fingerprint)
            with self.assertRaisesRegex(RuntimeError, "changed during initialization"):
                checkpointing_module.validate_frozen_checkpoint_run_fingerprint(
                    frozen_fingerprint,
                    args=args,
                    spec=simulator.spec,
                    step_count=2,
                    full_pressure_waveform_steps=2,
                )

            simulator.time_s[None] = 999.0
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_run_checkpoint(
                    checkpoint_path,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid,
                    step_count=2,
                    full_pressure_waveform_steps=2,
                )
            self.assertEqual(simulator.time_s[None], 999.0)

    def test_production_checkpoint_writes_receive_the_frozen_fingerprint(self):
        case_root = Path(__file__).resolve().parents[2] / "cases" / "squid_soft_robot"
        runner_source = (case_root / "runner.py").read_text(encoding="utf-8")
        step_loop_source = (case_root / "step_loop.py").read_text(encoding="utf-8")

        self.assertIn(
            "frozen_run_fingerprint = checkpoint_run_fingerprint(",
            runner_source,
        )
        revalidation = runner_source.index(
            "\n    validate_frozen_checkpoint_run_fingerprint("
        )
        self.assertGreater(
            revalidation,
            runner_source.index("source_config_water_obstacle_mask"),
        )
        self.assertLess(revalidation, runner_source.index("history_path ="))
        self.assertEqual(
            runner_source.count(
                "frozen_run_fingerprint=frozen_run_fingerprint"
            ),
            3,
        )
        self.assertEqual(
            step_loop_source.count(
                "frozen_run_fingerprint=resources.frozen_run_fingerprint"
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
