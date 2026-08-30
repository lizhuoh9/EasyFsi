"""Contract tests for fixed Cartesian material-to-surface reference stencils."""

import hashlib

import numpy as np
import pytest

from simulation_core.coupling.hibm_mpm.material_surface_binding import (
    build_cartesian_material_surface_binding,
    interpolate_material_surface,
    transpose_material_surface_loads,
)


def _cartesian_particles(axis_values, order=None):
    grid = np.array(np.meshgrid(*axis_values, indexing="ij"), dtype=np.float64)
    points = grid.reshape(3, -1).T
    if order is not None:
        points = points[np.asarray(order)]
    return points


def _planar_fixture():
    particles = _cartesian_particles(
        (
            np.array([0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([2.0, 3.0], dtype=np.float64),
        )
    )
    markers = np.array(
        [
            [0.0, -1.5, 1.5],
            [0.0, -0.25, 2.25],
            [0.0, 0.75, 3.5],
        ],
        dtype=np.float64,
    )
    masses = 0.5 + np.arange(particles.shape[0], dtype=np.float64)
    return particles, markers, masses


def _affine_values(points):
    linear = np.array(
        [[1.5, -0.75, 0.25], [-0.5, 2.0, 0.5], [0.2, -0.3, 1.25]],
        dtype=np.float64,
    )
    return points @ linear.T + np.array([0.7, -1.25, 2.0], dtype=np.float64)


def _f32_rounding_error_bound(value):
    quantized = np.float32(value)
    assert np.isfinite(quantized)
    lower = np.nextafter(quantized, np.float32(-np.inf), dtype=np.float32)
    upper = np.nextafter(quantized, np.float32(np.inf), dtype=np.float32)
    return 0.5 * max(abs(float(quantized - lower)), abs(float(upper - quantized)))


def _physical_f32_plane_grid(y_count, z_count):
    y_physical = np.linspace(0.04, 0.05, y_count, dtype=np.float64)
    z_physical = np.linspace(0.045, 0.048, z_count, dtype=np.float64)
    particles = _cartesian_particles(
        (
            np.array([np.float32(0.0)], dtype=np.float64),
            np.asarray(y_physical, dtype=np.float32).astype(np.float64),
            np.asarray(z_physical, dtype=np.float32).astype(np.float64),
        )
    )
    return particles, y_physical, z_physical


def _half_endpoint_uncertainty(values, at_lower_end):
    endpoint = 0 if at_lower_end else -1
    neighbor = 1 if at_lower_end else -2
    return 1.5 * _f32_rounding_error_bound(values[endpoint]) + 0.5 * _f32_rounding_error_bound(values[neighbor])


def test_planar_stencil_has_independent_expected_weights_and_affine_reproduction():
    particles, markers, masses = _planar_fixture()
    binding = build_cartesian_material_surface_binding(particles, markers, masses)

    assert binding.active_axes == (1, 2)
    assert binding.particle_indices.shape == (3, 8)
    np.testing.assert_array_equal(binding.stencil_sizes, np.array([4, 4, 4], dtype=np.int32))
    np.testing.assert_allclose(
        binding.weights[0, :4], np.array([2.25, -0.75, -0.75, 0.25]), atol=1.0e-15
    )
    np.testing.assert_allclose(binding.weights[:, 4:], 0.0, atol=0.0)
    np.testing.assert_allclose(
        interpolate_material_surface(binding, _affine_values(particles)),
        _affine_values(markers),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert binding.maximum_row_l1 <= 4.0
    assert binding.maximum_row_inverse_mass_gain <= 6.25 / masses.min()


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_axis_permutations_and_boundary_extrapolation_reproduce_affine_fields(axis, sign):
    axes = [np.array([0.25], dtype=np.float64) for _ in range(3)]
    axes[axis] = np.array([-2.0, -1.0, 0.0], dtype=np.float64)
    particles = _cartesian_particles(tuple(axes))
    marker = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
    marker[0, axis] = -2.5 if sign < 0.0 else 0.5
    binding = build_cartesian_material_surface_binding(particles, marker, np.ones(len(particles)))

    assert binding.active_axes == (axis,)
    assert binding.stencil_sizes.tolist() == [2]
    np.testing.assert_allclose(np.sort(binding.weights[0, :2]), np.array([-0.5, 1.5]))
    np.testing.assert_allclose(interpolate_material_surface(binding, _affine_values(particles)), _affine_values(marker))
    assert binding.maximum_row_l1 <= 2.0
    assert binding.maximum_row_inverse_mass_gain <= 2.5


def test_exact_translation_rotation_and_material_position_round_trip_and_partition_invariance():
    particles, markers, masses = _planar_fixture()
    binding = build_cartesian_material_surface_binding(particles, markers, masses)
    theta = 0.37
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([0.4, -0.1, 2.5])
    transformed = particles @ rotation.T + translation
    np.testing.assert_allclose(interpolate_material_surface(binding, transformed), markers @ rotation.T + translation)

    displacement = _affine_values(particles) * 0.01
    restored = particles + displacement - displacement
    np.testing.assert_allclose(interpolate_material_surface(binding, restored), markers, atol=2.0e-14)
    velocity = _affine_values(particles) * 0.03
    dt = 0.12
    final_one = particles + dt * velocity
    final_partitioned = particles + (dt / 3.0) * velocity + (2.0 * dt / 3.0) * velocity
    np.testing.assert_allclose(
        interpolate_material_surface(binding, final_one),
        interpolate_material_surface(binding, final_partitioned),
        atol=2.0e-14,
    )


def test_particle_and_velocity_permutation_preserves_physical_interpolation_but_changes_identity():
    particles, markers, masses = _planar_fixture()
    permutation = np.array([5, 3, 2, 0, 1, 4])
    original = build_cartesian_material_surface_binding(particles, markers, masses)
    permuted = build_cartesian_material_surface_binding(
        particles[permutation], markers, masses[permutation]
    )
    assert original.identity_sha256 != permuted.identity_sha256
    np.testing.assert_allclose(
        interpolate_material_surface(original, _affine_values(particles)),
        interpolate_material_surface(permuted, _affine_values(particles)[permutation]),
    )


def test_transpose_force_torque_and_virtual_power_conservation_on_deformed_positions():
    particles, markers, masses = _planar_fixture()
    binding = build_cartesian_material_surface_binding(particles, markers, masses)
    deformed_particles = particles + _affine_values(particles) * 0.04
    deformed_markers = interpolate_material_surface(binding, deformed_particles)
    rng = np.random.default_rng(192)
    forces = rng.normal(size=(len(markers), 3))
    velocities = rng.normal(size=(len(particles), 3))
    particle_forces = transpose_material_surface_loads(binding, forces)

    np.testing.assert_allclose(particle_forces.sum(axis=0), forces.sum(axis=0), atol=2.0e-14)
    np.testing.assert_allclose(
        np.cross(deformed_particles, particle_forces).sum(axis=0),
        np.cross(deformed_markers, forces).sum(axis=0),
        atol=2.0e-14,
    )
    assert np.sum(forces * interpolate_material_surface(binding, velocities)) == pytest.approx(
        np.sum(particle_forces * velocities), abs=2.0e-14
    )


def test_negative_weights_and_analytic_inverse_mass_bound_are_not_clipped():
    particles = _cartesian_particles((np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0])))
    markers = np.array([[-0.5, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([2.0, 4.0])
    binding = build_cartesian_material_surface_binding(particles, markers, masses)

    np.testing.assert_allclose(np.sort(binding.weights[0, :2]), [-0.5, 1.5])
    assert binding.maximum_row_l1 == pytest.approx(2.0)
    assert binding.maximum_row_inverse_mass_gain == pytest.approx(1.1875)
    assert binding.maximum_row_inverse_mass_gain <= 2.5 / masses.min()


@pytest.mark.parametrize(("y_count", "z_count"), [(12, 4), (256, 20)])
def test_physical_f32_plane_grids_admit_wall_and_cap_half_cell_markers(y_count, z_count):
    particles, y_physical, z_physical = _physical_f32_plane_grid(y_count, z_count)
    markers = np.array(
        [
            [
                0.0,
                y_physical[0] - 0.5 * (y_physical[1] - y_physical[0]),
                z_physical[0] - 0.5 * (z_physical[1] - z_physical[0]),
            ],
            [
                0.0,
                y_physical[-1] + 0.5 * (y_physical[-1] - y_physical[-2]),
                z_physical[-1] + 0.5 * (z_physical[-1] - z_physical[-2]),
            ],
        ],
        dtype=np.float64,
    )
    binding = build_cartesian_material_surface_binding(particles, markers, np.ones(len(particles)))

    np.testing.assert_allclose(
        interpolate_material_surface(binding, _affine_values(particles)),
        _affine_values(markers),
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_one_ulp_half_cell_uncertainty_has_explicit_adjusted_factor_bounds_without_clipping():
    particles, _, z_physical = _physical_f32_plane_grid(2, 4)
    z_stored = np.unique(particles[:, 2])
    lower_wall = z_physical[0] - 0.5 * (z_physical[1] - z_physical[0])
    marker = np.array([[0.0, particles[0, 1], lower_wall]], dtype=np.float64)
    binding = build_cartesian_material_surface_binding(particles, marker, np.ones(len(particles)))

    fraction = (lower_wall - z_stored[0]) / (z_stored[1] - z_stored[0])
    endpoint_uncertainty = _half_endpoint_uncertainty(z_stored, at_lower_end=True)
    fraction_uncertainty = endpoint_uncertainty / (z_stored[1] - z_stored[0])
    factor_l1 = abs(1.0 - fraction) + abs(fraction)
    factor_l2_squared = (1.0 - fraction) ** 2 + fraction**2
    assert fraction < -0.5
    assert factor_l1 <= 2.0 + 2.0 * fraction_uncertainty
    assert factor_l2_squared <= 2.5 + 4.0 * fraction_uncertainty + 2.0 * fraction_uncertainty**2
    nonzero_weights = binding.weights[0, np.abs(binding.weights[0]) > 1.0e-15]
    assert nonzero_weights.min() < -0.5
    assert nonzero_weights.max() > 1.5


def test_tiny_coordinate_nonuniformity_is_not_hidden_by_a_unit_scale_floor():
    particles = _cartesian_particles(
        (np.array([0.0, 1.0e-8, 2.1e-8]), np.array([0.0]), np.array([0.0]))
    )
    with pytest.raises(ValueError, match="uniform"):
        build_cartesian_material_surface_binding(
            particles, np.array([[1.0e-8, 0.0, 0.0]]), np.ones(len(particles))
        )


def test_singleton_axis_offset_within_f32_roundoff_still_fails_first_moment_reproduction():
    plane = float(np.float32(0.04))
    particles = _cartesian_particles(
        (np.array([plane]), np.array([0.04, 0.05]), np.array([0.045, 0.048]))
    )
    marker = np.array(
        [[plane + 0.25 * _f32_rounding_error_bound(plane), 0.045, 0.0465]], dtype=np.float64
    )
    with pytest.raises(ValueError, match="affine coordinate"):
        build_cartesian_material_surface_binding(particles, marker, np.ones(len(particles)))


def test_extremely_small_positive_mass_with_nonfinite_inverse_gain_fails_closed():
    particles = _cartesian_particles((np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0])))
    with pytest.raises(ValueError, match="inverse-mass"):
        build_cartesian_material_surface_binding(
            particles, np.array([[-0.5, 0.0, 0.0]]), np.array([np.nextafter(0.0, 1.0), 1.0])
        )


@pytest.mark.parametrize(
    ("particles", "markers", "masses", "match"),
    [
        (np.zeros((2, 3)), np.zeros((1, 3)), np.ones(2), "duplicate"),
        (np.array([[0, 0, 0], [0, 1, 0], [0, 3, 0]]), np.zeros((1, 3)), np.ones(3), "uniform"),
        (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.zeros((1, 3)), np.ones(3), "Cartesian"),
        (np.array([[0, 0, 0], [1, 0, 0]]), np.array([[1.500001, 0, 0]]), np.ones(2), "half"),
        (np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0.5, 1.0e-3, 0]]), np.ones(2), "inactive"),
        (np.array([[0, 0, 0], [1, 0, 0]]), np.zeros((1, 3)), np.array([1.0, 0.0]), "positive"),
        (np.array([[0, 0, 0], [np.nan, 0, 0]]), np.zeros((1, 3)), np.ones(2), "finite"),
    ],
)
def test_invalid_layout_geometry_mass_and_nonfinite_inputs_fail_closed(particles, markers, masses, match):
    with pytest.raises(ValueError, match=match):
        build_cartesian_material_surface_binding(
            np.asarray(particles, dtype=np.float64), np.asarray(markers, dtype=np.float64), np.asarray(masses)
        )


def test_buffers_are_copied_frozen_and_identity_covers_required_input_content():
    particles, markers, masses = _planar_fixture()
    binding = build_cartesian_material_surface_binding(particles, markers, masses)
    particles[:] = 99.0
    markers[:] = 99.0
    masses[:] = 99.0
    np.testing.assert_allclose(binding.reference_particle_positions_m[:, 0], 0.0)
    for buffer in (
        binding.particle_indices,
        binding.weights,
        binding.stencil_sizes,
        binding.reference_particle_positions_m,
        binding.reference_marker_positions_m,
        binding.particle_mass_kg,
    ):
        assert not buffer.flags.writeable
        with pytest.raises(ValueError):
            buffer.setflags(write=True)

    source_particles, source_markers, source_masses = _planar_fixture()
    baseline = build_cartesian_material_surface_binding(source_particles, source_markers, source_masses)
    changed_mass = build_cartesian_material_surface_binding(source_particles, source_markers, source_masses + 0.1)
    changed_markers = source_markers.copy()
    changed_markers[1, 2] += 0.01
    changed_marker = build_cartesian_material_surface_binding(
        source_particles, changed_markers, source_masses
    )
    assert baseline.identity_sha256 != changed_mass.identity_sha256
    assert baseline.identity_sha256 != changed_marker.identity_sha256
    assert len(baseline.identity_sha256) == hashlib.sha256().digest_size * 2


def test_input_and_output_shape_validation():
    particles, markers, masses = _planar_fixture()
    binding = build_cartesian_material_surface_binding(particles, markers, masses)
    with pytest.raises(ValueError, match="particle_values"):
        interpolate_material_surface(binding, np.zeros((len(particles), 2)))
    with pytest.raises(ValueError, match="marker_forces_n"):
        transpose_material_surface_loads(binding, np.zeros((len(markers), 2)))
