"""The captured step49 curved bridge is local despite leaving the P hull."""

import numpy as np
import pytest

from tests.solvers.test_registered_segment_path_audit import _configure_case, _cyclic, path_case


# Exact F32 marker 12/13/14/15 data, from the 53-array r36 step49 capture.
_POSITIONS = (
    (0.001500000013038516, 0.0004141739627812058, 0.04699907824397087),
    (0.0014999995473772287, 0.001227909349836409, 0.04700777679681778),
    (0.0014999988488852978, 0.002073799492791295, 0.046864718198776245),
    (0.0014999976847320795, 0.0030449454206973314, 0.04626474156975746),
)
_NORMALS = (
    (-6.394669327391966e-8, -0.0034771496430039406, -0.9999939203262329),
    (-2.8860119982709875e-7, -0.006719593890011311, -0.9999773502349854),
    (-1.0723873629103764e-6, -0.18644723296165466, -0.9824649691581726),
    (-2.3220507046062266e-6, -0.62154221534729, -0.7833806872367859),
)
_VELOCITIES = (
    (-4.350751936499364e-8, -0.002891208976507187, 0.000504234223626554),
    (-2.22963691953737e-7, -0.013297609984874725, 0.001976783387362957),
    (-5.639851679006824e-7, -0.046961285173892975, -0.023803753778338432),
    (-9.6940516414179e-7, -0.07521428167819977, -0.14368367195129395),
)
_FACE = (0.000375000003259629, 0.0006249999860301614, 0.04453124850988388)
_SOURCE = (0.000375000003259629, 0.0003124999930150807, 0.04453124850988388)
_ANCHOR = (0.000375000003259629, 0.002073799492791295, 0.046864718198776245)
_SAMPLE = (0.000375000003259629, -0.0006879815482534468, 0.04239438846707344)
_RAW_NORMAL = (0.0, -0.5255885720252991, -0.8507388234138489)
_RADIUS = (0.0012000000000000001, 0.003125, 0.0023437500000000003)


@pytest.mark.parametrize("inactive_axis", (0, 1, 2))
@pytest.mark.parametrize("reverse_storage", (False, True))
def test_captured_curved_bridge_is_certified_without_changing_raw_support(
    path_case, inactive_axis, reverse_storage,
):
    def rotate(value):
        return _cyclic((value[1], value[2], value[0]), inactive_axis)

    component_axis = (inactive_axis + 1) % 3
    edges = [(0, 1), (1, 2), (2, 3)]
    if reverse_storage:
        edges = [(second, first) for first, second in edges]
    scan = _configure_case(
        path_case, points=[rotate(row) for row in _POSITIONS] + [(0.0, 0.0, 0.0)] * 2,
        normals=[rotate(row) for row in _NORMALS] + [(0.0, 0.0, 0.0)] * 2,
        segments=edges, inactive_axis=inactive_axis, component_axis=component_axis,
        owner_face=rotate(_FACE), source_center=rotate(_SOURCE),
        source_anchor=rotate(_ANCHOR), source_sample=rotate(_SAMPLE),
    )
    path_case["velocities"].from_numpy(np.asarray(
        [rotate(row) for row in _VELOCITIES] + [(0.0, 0.0, 0.0)] * 2, np.float32,
    ))
    path_case["regions"].fill(202)
    assembler = path_case["assembler"]
    key = (0, 0, 0, component_axis)
    assembler.raw_route_primitive[key] = (3, 2, -1) if reverse_storage else (2, 3, -1)
    assembler.raw_route_weights[key] = (0.0, 1.0, 0.0) if reverse_storage else (1.0, 0.0, 0.0)
    assembler.raw_route_boundary_target_mps[key] = _VELOCITIES[2][1]
    assembler.raw_route_region[key] = 202
    assembler.raw_route_normal[key] = rotate(_RAW_NORMAL)
    assembler.scan_registered_active_faces_device(**scan)
    assert int(assembler.owner_segment_index[key]) == 0
    assert tuple(assembler.owner_segment[key]) == (0, 1)
    assembler.certify_active_raw_routes_device(
        expected_generation=31, support_available=1, support_anisotropic=1,
        strict_support_radius_xyz_m=rotate(_RADIUS), marker_normal_m=path_case["normals"],
        marker_role=path_case["roles"], **scan,
    )
    assert int(assembler._audited_owner_failure[key]) == 0
    assert int(assembler.raw_route_audit_failure[key]) == 0
    assert int(assembler.audit_valid[key]) == 1
    assert int(assembler.audit_raw_count[key]) == 1
    assert int(assembler.audit_rejection_count[None]) == 0
