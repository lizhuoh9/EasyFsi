"""CPU contract: cap wall velocity is the derivative of cap position."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import taichi as ti

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from simulation_core.coupling.hibm_mpm import core


@pytest.fixture(scope="module")
def cap():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, offline_cache=False)
    initialize = core.init_taichi
    core.init_taichi = lambda runtime=None: None
    try:
        markers = core.HibmMpmSurfaceMarkers(marker_capacity=8, projection_triangle_capacity=5)
    finally:
        core.init_taichi = initialize

    def rebuild(inactive_axis, component, sign, varying):
        permutation = np.roll(np.arange(3), inactive_axis)
        positions = np.array([
            [.5, .5, .625], [.5, .75, .625],
            [.5, .5, .375], [.5, .75, .375],
        ])[:, permutation]
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1]])[:, permutation]
        velocities = np.zeros((4, 3))
        velocities[:, component] = sign * np.array(
            [.125, .375, -.125, -.375] if varying else [.125] * 4
        )
        markers.load_markers(
            positions_m=positions, velocities_mps=velocities, normals=normals,
            areas_m2=(.25,) * 4, region_ids=(101, 101, 202, 202),
        )
        markers.configure_open_ribbon_tip_cap(
            primary_previous_marker_index=0, primary_tip_marker_index=1,
            secondary_previous_marker_index=2, secondary_tip_marker_index=3,
            cap_region_id=303, cap_area_m2=.25, inactive_axis=inactive_axis,
        )
        expected = np.stack((1.5 * velocities[1] - .5 * velocities[0],
                             1.5 * velocities[3] - .5 * velocities[2]))
        return markers.v_gamma_mps.to_numpy()[6:8], expected

    yield rebuild
    ti.reset()


CASES = [(axis, component, sign) for axis in range(3)
         for component in range(3) if component != axis for sign in (-1.0, 1.0)]


@pytest.mark.parametrize("inactive_axis,component,sign", CASES)
def test_cap_uniform_translation_control(cap, inactive_axis, component, sign):
    observed, expected = cap(inactive_axis, component, sign, False)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("inactive_axis,component,sign", CASES)
def test_cap_velocity_matches_extrapolated_position_derivative(cap, inactive_axis, component, sign):
    observed, expected = cap(inactive_axis, component, sign, True)
    np.testing.assert_array_equal(observed, expected)
