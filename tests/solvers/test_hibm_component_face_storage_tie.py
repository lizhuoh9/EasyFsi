import numpy as np
import pytest
import taichi as ti

from simulation_core import HibmMpmIbBoundaryConditions
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi, taichi_runtime_identity


@ti.kernel
def _probe_component_face_storage_tie(
    boundary: ti.template(),
    x_index: ti.i32,
    zpos: ti.f32,
    obstacle: ti.template(),
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    result_i32: ti.template(),
    result_f32: ti.template(),
):
    source = ti.Vector([x_index, 2, 2])
    center_x = cell_center_x_m[x_index]
    (
        valid,
        storage,
        alpha,
        error_code,
        pair_valid,
        pair_storage,
        pair_alpha,
    ) = boundary._select_canonical_component_face_storage_device(
        source,
        2,
        ti.Vector([center_x, 0.375, zpos]),
        ti.Vector([center_x, 0.75, zpos]),
        obstacle,
        0,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        4,
        4,
        4,
    )
    result_i32[0] = valid
    result_i32[1] = storage.x
    result_i32[2] = storage.y
    result_i32[3] = storage.z
    result_i32[4] = error_code
    result_i32[5] = pair_valid
    result_i32[6] = pair_storage.x
    result_i32[7] = pair_storage.y
    result_i32[8] = pair_storage.z
    result_f32[0] = alpha
    result_f32[1] = pair_alpha


def test_component_face_storage_pair_route_rejects_only_exact_tie() -> None:
    init_taichi(TaichiRuntimeConfig(arch="cuda", default_fp="f32", strict_arch=True))
    identity = taichi_runtime_identity()
    compiler = identity["compiler_configuration"]
    print(f"component-face-storage-tie runtime actual={identity['actual_arch']} cfg_optimization={compiler['cfg_optimization']} opt_level={compiler['opt_level']} advanced_optimization={compiler['advanced_optimization']}")

    boundary = object.__new__(HibmMpmIbBoundaryConditions)
    obstacle = ti.field(dtype=ti.i32, shape=(4, 4, 4))
    cell_face_x_m = ti.field(dtype=ti.f32, shape=5)
    cell_face_y_m = ti.field(dtype=ti.f32, shape=5)
    cell_face_z_m = ti.field(dtype=ti.f32, shape=5)
    cell_center_x_m = ti.field(dtype=ti.f32, shape=4)
    cell_center_y_m = ti.field(dtype=ti.f32, shape=4)
    cell_center_z_m = ti.field(dtype=ti.f32, shape=4)
    result_i32 = ti.field(dtype=ti.i32, shape=9)
    result_f32 = ti.field(dtype=ti.f32, shape=2)

    obstacle.fill(0)
    cell_face_x_m.from_numpy(np.asarray((0.0, 0.1, 0.4, 0.7, 1.0), dtype=np.float32))
    cell_center_x_m.from_numpy(np.asarray((0.05, 0.25, 0.55, 0.85), dtype=np.float32))
    uniform_faces = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float32)
    uniform_centers = np.asarray((0.125, 0.375, 0.625, 0.875), dtype=np.float32)
    for field in (cell_face_y_m, cell_face_z_m):
        field.from_numpy(uniform_faces)
    for field in (cell_center_y_m, cell_center_z_m):
        field.from_numpy(uniform_centers)

    assert 0.625 - 0.5 == 0.75 - 0.625
    assert (0.625 - 0.5) ** 2 == (0.75 - 0.625) ** 2
    cases = (
        (0, 0.625, (0, 2, 2), False, (-1, -1, -1)), (1, 0.625, (1, 2, 2), False, (-1, -1, -1)),
        (0, 0.6, (0, 2, 2), True, (0, 2, 2)), (1, 0.6, (1, 2, 2), True, (1, 2, 2)),
        (0, 0.65, (0, 2, 3), True, (0, 2, 3)), (1, 0.65, (1, 2, 3), True, (1, 2, 3)),
    )
    for x_index, zpos, storage_expected, pair_valid_expected, pair_storage_expected in cases:
        _probe_component_face_storage_tie(
            boundary,
            x_index,
            zpos,
            obstacle,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            result_i32,
            result_f32,
        )
        result = result_i32.to_numpy()
        assert int(result[0]) == 1
        assert tuple(int(value) for value in result[1:4]) == storage_expected
        assert int(result[4]) == 0
        assert float(result_f32[0]) == pytest.approx(2.0 / 3.0, abs=1.0e-6)
        assert int(result[5]) == int(pair_valid_expected)
        assert tuple(int(value) for value in result[6:9]) == pair_storage_expected
        expected_pair_alpha = 2.0 / 3.0 if pair_valid_expected else 0.0
        assert float(result_f32[1]) == pytest.approx(expected_pair_alpha, abs=1.0e-6)
