from __future__ import annotations

from pathlib import Path

import numpy as np

from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (
    save_solver_npz_from_flow_snapshot,
)


def test_solver_npz_masks_canonical_hard_boundary_rows(tmp_path: Path) -> None:
    shape = (1, 2, 3)
    hard_row = (0, 0, 1)
    hard_component_mask = np.zeros(shape, dtype=np.int32)
    hard_component_mask[hard_row] = 0b100
    y = np.broadcast_to(
        np.asarray([0.005, 0.015], dtype=np.float64)[None, :, None],
        shape,
    )
    z = np.broadcast_to(
        np.asarray([0.02, 0.05, 0.08], dtype=np.float64)[None, None, :],
        shape,
    )
    snapshot = {
        "velocity": np.zeros(shape + (3,), dtype=np.float64),
        "pressure": np.zeros(shape, dtype=np.float64),
        "obstacle": np.zeros(shape, dtype=np.int32),
        # Reproduce the production snapshot: the legacy scalar ledger is
        # empty while the canonical component-face ledger owns a hard row.
        "velocity_dirichlet_boundary_active": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            shape, dtype=np.float64
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": (
            hard_component_mask
        ),
        "cell_center_y_m": y,
        "cell_center_z_m": z,
    }

    output_path = tmp_path / "solver_fields.npz"
    summary = save_solver_npz_from_flow_snapshot(
        output_path,
        snapshot,
        reverse_streamwise_axis=False,
    )

    assert summary["boundary_surrogate_cell_count"] == 1
    with np.load(output_path, allow_pickle=False) as fields:
        assert bool(fields["boundary_surrogate_mask"][0, 1])
        assert not bool(fields["fluid_mask"][0, 1])
        assert bool(fields["display_fluid_mask"][0, 1])


def test_span_mean_excludes_hard_boundary_slices_from_physical_values(
    tmp_path: Path,
) -> None:
    shape = (2, 2, 3)
    hard_component_mask = np.zeros(shape, dtype=np.int32)
    hard_component_mask[0, 0, 1] = 0b100
    velocity = np.zeros(shape + (3,), dtype=np.float64)
    velocity[0, :, :, 2] = 100.0
    velocity[1, :, :, 2] = 10.0
    pressure = np.zeros(shape, dtype=np.float64)
    pressure[0, :, :] = 1000.0
    pressure[1, :, :] = 10.0
    y = np.broadcast_to(
        np.asarray([0.005, 0.015], dtype=np.float64)[None, :, None],
        shape,
    )
    z = np.broadcast_to(
        np.asarray([0.02, 0.05, 0.08], dtype=np.float64)[None, None, :],
        shape,
    )
    snapshot = {
        "velocity": velocity,
        "pressure": pressure,
        "obstacle": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_active": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            shape, dtype=np.float64
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": (
            hard_component_mask
        ),
        "cell_center_y_m": y,
        "cell_center_z_m": z,
    }

    output_path = tmp_path / "solver_fields.npz"
    save_solver_npz_from_flow_snapshot(
        output_path,
        snapshot,
        streamwise_velocity_sign=1.0,
        reverse_streamwise_axis=False,
    )

    with np.load(output_path, allow_pickle=False) as fields:
        # One span slice remains physical, so this 2-D cell is comparable; its
        # values must come only from that slice, never from the hard surrogate.
        assert bool(fields["fluid_mask"][0, 1])
        assert bool(fields["boundary_surrogate_mask"][0, 1])
        assert fields["u"][0, 1] == 10.0
        assert fields["p"][0, 1] == 10.0
