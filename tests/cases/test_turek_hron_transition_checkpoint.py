from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cases import turek_hron_fsi as turek


class _FakeNumpyField:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()

    def to_numpy(self) -> np.ndarray:
        return self.value.copy()

    def from_numpy(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()


def test_transition_checkpoint_fingerprint_ignores_run_length_only() -> None:
    short = replace(turek.TurekHronFsiConfig(), step_count=184)
    long = replace(short, step_count=220, flow_snapshot_interval_steps=20)
    changed_physics = replace(long, dt_s=0.004)

    fingerprint = turek._turek_hron_checkpoint_config_fingerprint
    assert fingerprint(short) == fingerprint(long)
    assert fingerprint(short) != fingerprint(changed_physics)


def test_transition_checkpoint_metadata_accepts_a_longer_run() -> None:
    capture_config = replace(turek.TurekHronFsiConfig(), step_count=184)
    replay_config = replace(capture_config, step_count=220)
    metadata = turek._turek_hron_transition_checkpoint_metadata(
        config=capture_config,
        preset="fsi1",
        completed_step=183,
        particle_count=1120,
        marker_count=100,
    )

    completed_step = turek._validate_turek_hron_transition_checkpoint_metadata(
        metadata=metadata,
        config=replay_config,
        preset="fsi1",
        particle_count=1120,
        marker_count=100,
    )

    assert completed_step == 183
    assert metadata["version"] == turek.TUREK_HRON_TRANSITION_CHECKPOINT_VERSION
    assert metadata["history_schema_version"] == turek.TUREK_HRON_HISTORY_SCHEMA_VERSION


def _complete_history_row(*, step: int, schema_version: int) -> dict[str, Any]:
    row = {field: 0 for field in turek.HISTORY_FIELDS}
    row["step"] = step
    row["history_schema_version"] = schema_version
    return row


def test_transition_checkpoint_history_rejects_old_or_incomplete_rows() -> None:
    old_row = _complete_history_row(step=1, schema_version=3)
    with pytest.raises(ValueError, match="history schema"):
        turek._validated_turek_hron_transition_checkpoint_history(
            [old_row],
            completed_step=1,
        )

    incomplete_row = _complete_history_row(
        step=1,
        schema_version=turek.TUREK_HRON_HISTORY_SCHEMA_VERSION,
    )
    incomplete_row.pop("solid_macro_accepted_time_s")
    with pytest.raises(ValueError, match="missing required fields"):
        turek._validated_turek_hron_transition_checkpoint_history(
            [incomplete_row],
            completed_step=1,
        )


def test_transition_checkpoint_history_and_csv_round_trip_schema_v4(
    tmp_path: Path,
) -> None:
    row = _complete_history_row(
        step=1,
        schema_version=turek.TUREK_HRON_HISTORY_SCHEMA_VERSION,
    )
    row["mpm_grid_out_of_bounds_particle_count"] = 7
    row["mpm_deformation_clamp_count"] = 11
    checkpoint_path = tmp_path / "step_000001_transition_checkpoint.npz"
    turek._write_turek_hron_transition_checkpoint(
        checkpoint_path,
        metadata={
            "version": turek.TUREK_HRON_TRANSITION_CHECKPOINT_VERSION,
            "history_schema_version": turek.TUREK_HRON_HISTORY_SCHEMA_VERSION,
            "history": [row],
        },
        arrays={"state": np.asarray([1.0], dtype=np.float32)},
    )

    loaded_metadata, _ = turek._load_turek_hron_transition_checkpoint(
        checkpoint_path
    )
    restored = turek._validated_turek_hron_transition_checkpoint_history(
        loaded_metadata["history"],
        completed_step=1,
    )
    csv_path = tmp_path / "history.csv"
    assert turek._flush_history_csv(csv_path, restored, header_written=False)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 1
    assert csv_rows[0]["history_schema_version"] == "4"
    assert csv_rows[0]["mpm_grid_out_of_bounds_particle_count"] == "7"
    assert csv_rows[0]["mpm_deformation_clamp_count"] == "11"


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"version": -1}, "version"),
        ({"preset": "fsi2"}, "preset"),
        ({"completed_step": 0}, "completed_step"),
        ({"particle_count": 1119}, "particle"),
        ({"marker_count": 99}, "marker"),
    ),
)
def test_transition_checkpoint_metadata_rejects_incompatible_state(
    override: dict[str, Any],
    message: str,
) -> None:
    config = replace(turek.TurekHronFsiConfig(), step_count=220)
    metadata = turek._turek_hron_transition_checkpoint_metadata(
        config=config,
        preset="fsi1",
        completed_step=183,
        particle_count=1120,
        marker_count=100,
    )

    with pytest.raises(ValueError, match=message):
        turek._validate_turek_hron_transition_checkpoint_metadata(
            metadata={**metadata, **override},
            config=config,
            preset="fsi1",
            particle_count=1120,
            marker_count=100,
        )


def test_transition_checkpoint_atomic_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "step_000183_transition_checkpoint.npz"
    metadata = {"version": 1, "completed_step": 183}
    arrays = {
        "fluid_velocity": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "solid_F": np.eye(3, dtype=np.float64)[None, :, :],
        "marker_v_gamma_mps": np.zeros((2, 3), dtype=np.float32),
    }

    written = turek._write_turek_hron_transition_checkpoint(
        checkpoint_path,
        metadata=metadata,
        arrays=arrays,
    )
    loaded_metadata, loaded_arrays = (
        turek._load_turek_hron_transition_checkpoint(checkpoint_path)
    )

    assert written == checkpoint_path
    assert loaded_metadata == metadata
    assert set(loaded_arrays) == set(arrays)
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded_arrays[name], expected)
        assert loaded_arrays[name] is not expected


def test_checkpoint_restore_validates_every_field_before_writing() -> None:
    target = SimpleNamespace(
        velocity=_FakeNumpyField(np.asarray([[1.0, 2.0]], dtype=np.float32)),
        pressure=_FakeNumpyField(np.asarray([3.0, 4.0], dtype=np.float64)),
    )
    velocity_before = target.velocity.to_numpy()
    pressure_before = target.pressure.to_numpy()

    with pytest.raises(ValueError, match="shape"):
        turek._restore_numpy_field_checkpoint_payload(
            target,
            {
                "fluid_velocity": np.asarray([[9.0, 8.0]], dtype=np.float32),
                "fluid_pressure": np.zeros((2, 1), dtype=np.float64),
            },
            names=("velocity", "pressure"),
            prefix="fluid",
        )

    np.testing.assert_array_equal(target.velocity.to_numpy(), velocity_before)
    np.testing.assert_array_equal(target.pressure.to_numpy(), pressure_before)


def _fake_transition_state_owners() -> tuple[Any, Any, Any, Any]:
    def fields(names: tuple[str, ...]) -> dict[str, Any]:
        return {
            name: _FakeNumpyField(np.asarray([float(index + 1)], dtype=np.float32))
            for index, name in enumerate(names)
        }

    fluid = SimpleNamespace(
        **fields(
            (
                "velocity",
                "velocity_prev",
                "pressure",
                "obstacle",
                "hibm_base_obstacle",
                "hibm_dynamic_solid_volume_obstacle",
                "hibm_dynamic_solid_volume_external_carve",
                "hibm_fresh_fluid_cell",
            )
        ),
        hibm_dynamic_solid_volume_enabled=True,
        _hibm_base_obstacle_initialized=True,
    )
    solid = SimpleNamespace(
        **fields(("x", "position_increment_residual_m", "v", "C", "F"))
    )
    markers = SimpleNamespace(
        **fields(("x_gamma_m", "v_gamma_mps", "n_gamma", "A_gamma_m2"))
    )
    boundary = SimpleNamespace(
        marker_pressure_neumann_gradient_field=_FakeNumpyField(
            np.asarray([7.0], dtype=np.float32)
        )
    )
    return fluid, solid, markers, boundary


def test_transition_checkpoint_restores_full_committed_state() -> None:
    source = _fake_transition_state_owners()
    payload = turek._turek_hron_transition_checkpoint_arrays(
        fluid=source[0],
        solid=source[1],
        markers=source[2],
        boundary=source[3],
    )
    target = _fake_transition_state_owners()
    target[0].hibm_dynamic_solid_volume_enabled = False
    target[0]._hibm_base_obstacle_initialized = False
    target[0].hibm_base_obstacle.value[:] = -5.0

    turek._restore_turek_hron_transition_checkpoint_arrays(
        fluid=target[0],
        solid=target[1],
        markers=target[2],
        boundary=target[3],
        payload=payload,
    )

    np.testing.assert_array_equal(
        target[0].hibm_base_obstacle.to_numpy(),
        source[0].hibm_base_obstacle.to_numpy(),
    )
    assert target[0]._hibm_base_obstacle_initialized is True
    assert target[0].hibm_dynamic_solid_volume_enabled is True
