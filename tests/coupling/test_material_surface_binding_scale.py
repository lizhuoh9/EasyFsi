"""Unit changes cannot tighten a dimensionless partition-of-unity check."""

import numpy as np
import pytest

from simulation_core.coupling.hibm_mpm.material_surface_binding import (
    build_cartesian_material_surface_binding,
    interpolate_material_surface,
)


@pytest.mark.parametrize("scale_m", (1.0, 0.05, 1e-3, 1e-8))
@pytest.mark.parametrize("inactive_axis", range(3))
@pytest.mark.parametrize("reverse_particles", (False, True))
def test_unity_is_dimensionless_under_scaling_axis_and_particle_permutations(
    scale_m, inactive_axis, reverse_particles,
):
    particles = scale_m * np.asarray([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    ], dtype=np.float64)
    fractions = np.vstack((
        [[0.7652179349792353, 0.103475387248399]],
        np.random.default_rng(425).uniform(-0.49, 1.49, size=(64, 2)),
    ))
    markers = scale_m * np.column_stack((np.zeros(len(fractions)), fractions))
    axes = np.roll(np.arange(3), inactive_axis)
    particles, markers = particles[:, axes], markers[:, axes]
    if reverse_particles:
        particles = particles[::-1].copy()
    binding = build_cartesian_material_surface_binding(particles, markers, np.ones(4))
    row_l1 = np.sum(np.abs(binding.weights), axis=1)
    assert np.all(np.abs(binding.weights.sum(axis=1) - 1.0) <= 16 * np.finfo(float).eps * row_l1)
    np.testing.assert_allclose(
        interpolate_material_surface(binding, np.ones((4, 3))),
        np.ones((len(markers), 3)), rtol=0.0, atol=32 * np.finfo(float).eps,
    )
    np.testing.assert_allclose(
        interpolate_material_surface(binding, particles), markers,
        rtol=0.0, atol=32 * np.finfo(float).eps * scale_m,
    )


def _rounded_translated_box(axis, sign):
    lower, upper = 0.01469, 0.01769
    particles = np.zeros((4, 3), dtype=np.float32)
    particles[:, axis] = sign * (lower + (np.arange(4) + 0.5) * (upper - lower) / 4)
    walls = np.zeros((2, 3), dtype=np.float32)
    walls[:, axis] = sign * np.asarray((lower, upper))
    return particles, walls


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
def test_f32_physical_walls_are_admissible_after_translation_and_reflection(axis, sign):
    particles, walls = _rounded_translated_box(axis, sign)
    binding = build_cartesian_material_surface_binding(particles, walls, np.ones(4))
    np.testing.assert_allclose(
        interpolate_material_surface(binding, particles), walls,
        rtol=0.0, atol=32 * np.finfo(float).eps * np.max(np.abs(walls)),
    )


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("endpoint", (0, 1))
def test_translated_walls_beyond_total_input_roundoff_remain_outside(axis, sign, endpoint):
    particles, walls = _rounded_translated_box(axis, sign)
    wall = walls[endpoint: endpoint + 1].copy()
    direction = np.float32(sign * (-np.inf if endpoint == 0 else np.inf))
    for _ in range(4):
        wall[0, axis] = np.nextafter(wall[0, axis], direction, dtype=np.float32)
    with pytest.raises(ValueError, match="half-cell"):
        build_cartesian_material_surface_binding(particles, wall, np.ones(4))
