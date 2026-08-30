"""All-active-face owner selection without Python per-face geometry."""

import numpy as np
import pytest
import taichi as ti

from simulation_core.coupling.hibm_mpm.component_face_segment_assembly import RegisteredComponentFaceSegmentAssembler
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, taichi_runtime_identity


def _batch_case():
    assembler = RegisteredComponentFaceSegmentAssembler(
        grid_nodes=(2, 2, 1), marker_capacity=4,
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )
    positions = ti.Vector.field(3, dtype=ti.f32, shape=4)
    velocities = ti.Vector.field(3, dtype=ti.f32, shape=4)
    regions = ti.field(dtype=ti.i32, shape=4)
    segments = ti.Vector.field(3, dtype=ti.i32, shape=2)
    positions.from_numpy(np.asarray([(0, 0, 0), (1, 0, 0), (0, 0.4, 0), (1, 0.4, 0)], np.float32))
    velocities.from_numpy(np.asarray([(2, 0, 0), (4, 0, 0), (10, 0, 0), (12, 0, 0)], np.float32))
    regions.fill(0)
    segments.from_numpy(np.asarray([(0, 1, -1), (2, 3, -1)], np.int32))
    coordinates = {}
    for name, values in {
        "cell_face_x_m": (0, 0.5, 1), "cell_face_y_m": (-0.2, 0.2, 0.6),
        "cell_face_z_m": (0, 1), "cell_center_x_m": (0.25, 0.75),
        "cell_center_y_m": (-0.1, 0.1), "cell_center_z_m": (0.5,),
    }.items():
        field = ti.field(dtype=ti.f32, shape=len(values))
        field.from_numpy(np.asarray(values, np.float32))
        coordinates[name] = field
    arguments = {
        **coordinates, "inactive_axis": 2, "projection_segment_indices": segments,
        "projection_segment_count": 2, "marker_position_m": positions,
        "marker_velocity_mps": velocities, "marker_region_id": regions,
        "projection_vertex_count": 4,
    }
    assembler.clear_device_transaction()
    return assembler, arguments, regions, segments


def test_batch_owner_uses_mac_coordinates_and_ignores_author_multiplicity():
    assembler, arguments, regions, segments = _batch_case()
    assembler.face_raw_count[1, 1, 0] = (8, 1, 0)
    assembler.scan_registered_active_faces_device(**arguments)
    assert assembler.owner_valid[1, 1, 0, 0] == 1
    assert assembler.owner_target_mps[1, 1, 0, 0] == pytest.approx(3.0)
    assert assembler.owner_valid[0, 0, 0, 0] == 0
    # At the y-face both finite intercepts are equally close but distinct.
    assert assembler.owner_valid[1, 1, 0, 1] == 0
    assert assembler.owner_ambiguous[1, 1, 0, 1] == 1
    segments.from_numpy(np.asarray([(3, 2, -1), (1, 0, -1)], np.int32))
    assembler.face_raw_count[1, 1, 0] = (1, 1, 0)
    assembler.scan_registered_active_faces_device(**arguments)
    assert assembler.owner_target_mps[1, 1, 0, 0] == pytest.approx(3.0)
    assert assembler.owner_ambiguous[1, 1, 0, 1] == 1
    identity = taichi_runtime_identity()
    assert identity["actual_arch"] == "cuda"
    print(identity)


def test_batch_rescan_does_not_retain_old_validity_or_promote_farther_owner():
    assembler, arguments, regions, _ = _batch_case()
    assembler.face_raw_count[1, 1, 0] = (3, 0, 0)
    assembler.scan_registered_active_faces_device(**arguments)
    assert assembler.owner_valid[1, 1, 0, 0] == 1
    regions.from_numpy(np.asarray([-1, -1, 0, 0], np.int32))
    assembler.scan_registered_active_faces_device(**arguments)
    assert assembler.owner_valid[1, 1, 0, 0] == 0
    assert assembler.owner_blocked[1, 1, 0, 0] == 1
    assert assembler.owner_target_mps[1, 1, 0, 0] == pytest.approx(3.0)
