from __future__ import annotations

from pathlib import Path

import numpy as np

from cases.squid_soft_robot.snapshots import (
    _vtk_point_data_values,
    _write_minimal_fluid_vti,
)


class _Field:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)

    def to_numpy(self) -> np.ndarray:
        return self._values.copy()


class _GradedFluid:
    def __init__(self) -> None:
        shape = (2, 2, 2)
        velocity = np.zeros((*shape, 3), dtype=np.float32)
        self.velocity = _Field(velocity)
        self.obstacle = _Field(np.zeros(shape, dtype=np.int32))
        self.divergence = _Field(np.zeros(shape, dtype=np.float32))
        self.cell_center_x_m = _Field(np.asarray([0.05, 0.20]))
        self.cell_center_y_m = _Field(np.asarray([0.10, 0.30]))
        self.cell_center_z_m = _Field(np.asarray([0.20, 0.70]))
        self.cell_width_x_m = _Field(np.asarray([0.10, 0.20]))
        self.cell_width_y_m = _Field(np.asarray([0.20, 0.20]))
        self.cell_width_z_m = _Field(np.asarray([0.40, 0.60]))


def test_failure_writer_uses_rectilinear_grid_for_graded_axes(
    tmp_path: Path,
) -> None:
    path = _write_minimal_fluid_vti(
        output_dir=tmp_path,
        step=3,
        fluid=_GradedFluid(),
    )

    assert path is not None
    assert path.suffix == ".vtr"
    text = path.read_text(encoding="utf-8")
    assert '<VTKFile type="RectilinearGrid"' in text
    assert '<VTKFile type="ImageData"' not in text
    assert 'Name="x_coordinates_m"' in text


def test_vtk_point_data_is_flattened_with_x_as_the_fastest_axis() -> None:
    values = np.empty((2, 2, 2), dtype=np.int32)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                values[i, j, k] = 100 * i + 10 * j + k

    np.testing.assert_array_equal(
        _vtk_point_data_values(values),
        np.asarray([0, 100, 10, 110, 1, 101, 11, 111]),
    )
